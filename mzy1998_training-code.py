# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import random
from scipy.io import wavfile
import IPython.display as ipd
import librosa.display
!pip install mir_eval

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from scipy import signal
import os

root = "../input/datasolo/solo/solo/"
solos = os.listdir(root)



fs = 44100
nb_frames, nb_features = 128, 1025


def process(solo, cut):
    dataset = {}
    tot = 0
    scalar_mean = np.zeros(nb_features)
    scalar_scal = np.zeros(nb_features)
    #cut = 10
    files = os.listdir(root + solo)
    print(solo)
    print(len(files))
    for file in files[:cut]:
        f = root + solo + "/" + file
        _, sig = wavfile.read(f)
        _, _, Zxx = signal.stft(sig, fs, nperseg=(nb_features - 1) * 2)
#         tot += Zxx.shape[1]
#         for i in range(Zxx.shape[1]):
#             scalar_mean += abs(Zxx[:, i])
#             scalar_scal += abs(Zxx[:, i]) ** 2
        dataset[file] = Zxx
#     scalar_mean /= tot
#     scalar_scal /= tot
#     scalar_scal -= scalar_mean ** 2
#     scalar_scal **= 0.5
#     plt.figure().clear()
#     plt.plot(scalar_mean)
#     plt.plot(scalar_scal)
#     plt.show()
    
    return dataset
# , scalar_mean, scalar_scal

print(solos)

# Any results you write to the current directory are saved as output.

dataset = {}
means = {}
scals = {}

for solo in solos:
    dataset[solo] = process(solo, 10)
def randomSolo():
    return solos[random.randint(0, len(solos) - 1)]
    
    
def genData(solo):
    lis = list(dataset[solo].keys())
    f = lis[random.randint(0, len(lis) - 1)]
    sig = dataset[solo][f]
    le = sig.shape[1]
    start = random.randint(0, le - nb_frames)
    sig = sig[:, start: start + nb_frames]
    return sig

def genMix(solo1, solo2):
    sig1 = genData(solo1)
    sig2 = genData(solo2)
    sig = sig1 + sig2
    return sig1, sig2, sig

# sig1, sig2, sig = genMix(solo1, solo2)

# _, restore = signal.istft(sig1, fs)
# ipd.display(ipd.Audio(restore, rate=fs))
# _, restore = signal.istft(sig2, fs)
# ipd.display(ipd.Audio(restore, rate=fs))
# _, restore = signal.istft(sig, fs)
# ipd.display(ipd.Audio(restore, rate=fs))

#f, axes = plt.subplots(2, 1, figsize=(12, 10))
#axes[0].pcolormesh(abs(sig))
#axes[1].pcolormesh(abs(sig1))
# def getMixStat(solo1):
#     num = 1000
#     tot = 0
#     scalar_mean = np.zeros(nb_features)
#     scalar_scal = np.zeros(nb_features)
#     mean2 = np.zeros(nb_features)
#     scal2 = np.zeros(nb_features)
#     for i in tqdm(range(num)):
#         solo2 = randomSolo()
#         _, sig2, sig = genMix(solo1, solo2)
#         tot += sig.shape[1]
#         for i in range(sig.shape[1]):
#             scalar_mean += abs(sig[:, i])
#             scalar_scal += abs(sig[:, i]) ** 2
#             mean2 += abs(sig2[:, i])
#             scal2 += abs(sig2[:, i]) ** 2
#     scalar_mean /= tot
#     scalar_scal /= tot
#     scalar_scal -= scalar_mean ** 2
#     scalar_scal **= 0.5
#     mean2 /= tot
#     scal2 /= tot
#     scal2 -= mean2 ** 2
#     scal2 **= 0.5
    
#     plt.figure().clear()
#     plt.plot(mean2)
#     plt.plot(scal2)
#     plt.show()
#     return scalar_mean, scalar_scal, mean2, scal2

# mean_all = {}
# scal_all = {}

# mean2 = {}
# scal2 = {}

# for solo in solos:
#     mean_all[solo], scal_all[solo], mean2[solo], scal2[solo] = getMixStat(solo)


import torch
import torch.optim as optim
use_cuda = torch.cuda.is_available()
torch.manual_seed(42)
from torch.nn import Module, LSTM, Linear, Parameter
import torch.nn.functional as F

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}





class Vanilla(Module):
    def __init__(
        self, nb_features, nb_frames, hidden_size=512, nb_layers=1
#         input_mean=None, input_scale=None
    ):
        super(Vanilla, self).__init__()

        self.hidden_size = hidden_size

#         self.input_mean = Parameter(
#             torch.from_numpy(np.copy(input_mean)).float()
#         )

#         self.input_scale = Parameter(
#             torch.from_numpy(np.copy(input_scale)).float(),
#         )
        self.batchnorm = torch.nn.BatchNorm1d(num_features=nb_features)
    
        self.encode_fc = Linear(
            nb_features, hidden_size
        )

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=nb_layers,
            bidirectional=False,
            batch_first=False
        )

        self.fc = Linear(
            in_features=hidden_size,
            out_features=nb_features
        )
        
#         self.output_scale = Parameter(
#             torch.ones(nb_features).float()
#         )

#         self.output_mean = Parameter(
#             torch.from_numpy(np.copy(output_mean)).float()
#         )


    def forward(self, x):
        nb_frames, nb_batches, nb_features = x.data.shape

#         x -= self.input_mean
#         x /= self.input_scale
        x = self.batchnorm(x.permute(0, 2, 1)).permute(0, 2, 1)
        # reduce input dimensionality
        x = self.encode_fc(x.reshape(-1, nb_features))

        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        x, state = self.lstm(x.reshape(nb_frames, nb_batches, self.hidden_size))

        x = self.fc(x.reshape(-1, self.hidden_size))

        x = x.reshape(nb_frames, nb_batches, nb_features)
        
        x = F.sigmoid(x)
        return x


from mir_eval.separation import bss_eval_sources

def test(num):
    model1.eval()
#     model2.eval()
    test_size = 1
    X = np.zeros((test_size, nb_frames, nb_features))
    Y1 = np.zeros((test_size, nb_frames, nb_features))
    Y2 = np.zeros((test_size, nb_frames, nb_features))
    #np.random.seed(42)

    cnt = 0
    sdr = (0, 0)
    for i in tqdm(range(num)):
        try:
            with torch.no_grad():
#                 solo2 = randomSolo()
                sig1, sig2, sig = genMix(solo1, solo2)

                X[0] = abs(sig).transpose()
                Y1[0] = abs(sig1).transpose()
                Y2[0] = abs(sig2).transpose()
                Yt1 = torch.tensor(Y1, dtype=torch.float32, device=device)
                Yt2 = torch.tensor(Y2, dtype=torch.float32, device=device)
                Xt = torch.tensor(X, dtype=torch.float32, device=device)
#                 Y2_hat = model2(Xt)
                Xt = torch.tensor(X, dtype=torch.float32, device=device)
                Y1_hat = model1(Xt)
                look1 = (Y1_hat.cpu().detach().numpy() > 0.5) * X[0]
#                 look2 = Y2_hat.cpu().detach().numpy()

                _, restore_sig_ = signal.istft(sig, fs)
                _, restore_sig1_ = signal.istft(sig1, fs)
                _, restore_sig2_ = signal.istft(sig2, fs)

                _, restore_look1_ = signal.istft(look1[0].transpose() * sig / abs(sig), fs)
#                 _, restore_look2 = signal.istft(look2[0].transpose() * sig / abs(sig), fs)
                restore_look2_ = restore_sig_ - restore_look1_


                restore_sig = np.reshape(restore_sig_, (1, -1))
                restore_sig1 = np.reshape(restore_sig1_, (1, -1))
                restore_sig2 = np.reshape(restore_sig2_, (1, -1))
                restore_look1 = np.reshape(restore_look1_, (1, -1))
                restore_look2 = np.reshape(restore_look2_, (1, -1))

                #sdr1 = bss_eval_sources(np.concatenate((restore_sig1, restore_sig2)), np.concatenate((restore_sig, (restore_sig + 10000)* 0.000000001)))[0]
                sdr2 = bss_eval_sources(np.concatenate((restore_sig1, restore_sig2)), np.concatenate((restore_look1, restore_look2)))[0]
                print(sdr2)
                if num == 1 or sdr2[0] + sdr2[1] > 30:
                    ipd.display(ipd.Audio(restore_sig_, rate=fs))
                    ipd.display(ipd.Audio(restore_sig1_, rate=fs))
                    ipd.display(ipd.Audio(restore_sig2_, rate=fs))
                    ipd.display(ipd.Audio(restore_look1_, rate=fs))
                    ipd.display(ipd.Audio(restore_look2_, rate=fs))
                sdr += sdr2
                cnt += 1

        except:
            pass
    print(sdr / cnt)

# model2 = Vanilla(
#     nb_features, nb_frames,
#     input_mean=mean_all,
#     input_scale=scal_all,
#     output_mean=mean2,
# ).to(device)
# for solo1 in solos:

for solo1 in solos:
    for solo2 in solos:
        if solo2 >= solo1: continue
        batch_size = 16
        model1 = Vanilla(
            nb_features, nb_frames,
        #     input_mean=mean_all[solo],
        #     input_scale=scal_all[solo],
        ).to(device)
    #     model1 = torch.load('../input/saxophoneviolin/saxophoneviolin')
        iters = 10000
        # iters = 40000 indeed
        optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        # optimizer2 = optim.Adam(model2.parameters(), lr=0.001)


        for it in tqdm(range(iters)):
            model1.train()
            #model2.train()
            X = np.zeros((batch_size, nb_frames, nb_features))
            Y1 = np.zeros((batch_size, nb_frames, nb_features))
            #Y2 = np.zeros((batch_size, nb_frames, nb_features))

            for k in range(batch_size):
        #         solo2 = randomSolo()
                sig1, sig2, sig = genMix(solo1, solo2)
                X[k] = abs(sig).transpose()
                Y1[k] = abs(sig1).transpose() > abs(sig2).transpose()
                #Y2[k] = abs(sig2).transpose()

            Xt = torch.tensor(X, dtype=torch.float32, device=device).permute(1, 0, 2)
            Yt1 = torch.tensor(Y1, dtype=torch.float32, device=device).permute(1, 0, 2)
            #Yt2 = torch.tensor(Y2, dtype=torch.float32, device=device).permute(1, 0, 2)

            optimizer1.zero_grad()
            Y_hat1 = model1(Xt)
            loss1 = criterion(Y_hat1, Yt1)
            loss1.backward()
            optimizer1.step()
        #     torch.save(model1, solo1)
        #         Xt = torch.tensor(X, dtype=torch.float32, device=device).permute(1, 0, 2)
        #         optimizer2.zero_grad()
        #         Y_hat2 = model2(Xt)
        #         loss2 = criterion(Y_hat2, Yt2)
        #         loss2.backward()
        #         optimizer2.step()
            if it % 1000 == 0:
                #print(sdr / cnt)
                torch.save(model1, solo1 + solo2)
        torch.save(model1, solo1 + solo2)
        
        test(20)
import torch
torch.save(model1, solo1 + solo2)
test(20)
# import os
# import numpy as np
# import json
# import time
# from scipy.io import wavfile
# from scipy import signal
# import torch
# from torch.nn import Module, LSTM, Linear, Parameter
# import torch.nn.functional as F
# from mir_eval.separation import bss_eval_sources
# import librosa

# fs = 44100
# nb_frames, nb_features = 128, 1025


# class Vanilla(Module):
#     def __init__(
#             self, nb_features, nb_frames, hidden_size=512, nb_layers=1
#     ):
#         super(Vanilla, self).__init__()

#         self.hidden_size = hidden_size
#         self.batchnorm = torch.nn.BatchNorm1d(num_features=nb_features)
#         self.encode_fc = Linear(
#             nb_features, hidden_size
#         )
#         self.lstm = LSTM(
#             input_size=hidden_size,
#             hidden_size=hidden_size,
#             num_layers=nb_layers,
#             bidirectional=False,
#             batch_first=False
#         )
#         self.fc = Linear(
#             in_features=hidden_size,
#             out_features=nb_features
#         )

#     def forward(self, x):
#         nb_frames0, nb_batches, nb_features0 = x.data.shape
#         assert nb_frames == nb_frames0
#         assert nb_features == nb_features0
#         x = self.batchnorm(x.permute(0, 2, 1)).permute(0, 2, 1)
#         # reduce input dimensionality
#         x = self.encode_fc(x.reshape(-1, nb_features))
#         # squash range ot [-1, 1]
#         x = torch.tanh(x)
#         # apply 3-layers of stacked LSTM
#         x, state = self.lstm(x.reshape(nb_frames, nb_batches, self.hidden_size))
#         x = self.fc(x.reshape(-1, self.hidden_size))
#         x = x.reshape(nb_frames, nb_batches, nb_features)
#         x = F.sigmoid(x)
#         return x


# def Test(OutputPath, OutputAudio, VideoPath, AudioPath, ImagePath, offline=True):
#     list = os.listdir(AudioPath)

#     print(list)
#     result = {}
#     time_list = []
#     for file in list:
#         if file[0] == '.': continue
#         if file == '.DS_Store': continue
#         fileRaw = file[:-4]
#         file_key = fileRaw + ".mp4"
#         result[file_key] = []
#         print("1")
#         start = time.clock()
#         solo1 = fileRaw.split('_')[0]
#         if solo1 == 'acoustic':
#             solo1 = 'acoustic_guitar'
#         solo2 = fileRaw.split('_')[-2]
#         if solo2 == 'guitar':
#             solo2 = 'acoustic_guitar'
#         print("2")

#         print((solo1, solo2))

#         if solo1 != 'acoustic_guitar' or solo2 != 'flute':
#             continue
#         modelname = solo1 + solo2 if solo1 < solo2 else solo2 + solo1
#         model = model1#torch.load("modelsNew/" + modelname, map_location='cpu')
#         model.eval()
#         print("3")
        
#         _, sig = wavfile.read(AudioPath + '/' + file)
#         sig12 = sep(sig, model)

#         print("4")


#         position = (0, 1) if solo1 < solo2 else (1, 0)

#         for i in range(2):
#             temp = {}
#             temp['audio'] = OutputAudio + '/' + fileRaw + '_gt' + str(i + 1) + '.wav'
#             temp['position'] = position[i]
#             if offline:
#                 librosa.output.write_wav(temp['audio'], sig12[i], fs, norm=True)
#             result[file_key].append(temp)
#         print("5")

#         sig1hat, sig2hat = sig12
#         _, sig1 = wavfile.read(AudioPath + '/' + fileRaw + '_gt1.wav')
#         _, sig2 = wavfile.read(AudioPath + '/' + fileRaw + '_gt2.wav')
#         print("6")

#         sig1 = np.reshape(sig1, (1, -1))
#         sig2 = np.reshape(sig2, (1, -1))
#         sig1hat = np.reshape(sig1hat, (1, -1))
#         sig2hat = np.reshape(sig2hat, (1, -1))
#         print("7")

#         print(bss_eval_sources(np.concatenate((sig1, sig2)), np.concatenate((sig1hat, sig2hat))))

#         time_list.append(time.clock() - start)

#     if offline:
#         with open(os.path.join(OutputPath, "result.json"), "w") as f:
#             json.dump(result, f, indent=4)
#     print("test time:", sum(time_list))



# def sep(sig, model):
#     _, _, Zxx = signal.stft(sig, fs, nperseg=(nb_features - 1) * 2)
#     sigY = np.zeros_like(Zxx)
#     test_size = (Zxx.shape[1] + nb_frames - 1) // nb_frames
#     X = np.zeros((test_size, nb_frames, nb_features))
#     for i in range(test_size):
#         st, ed = i * nb_frames, (i + 1) * nb_frames
#         if ed > Zxx.shape[1]:
#             st, ed = Zxx.shape[1] - nb_frames, Zxx.shape[1]
#         X[i] = abs(Zxx).transpose()[st: ed, :]

#     Xt = torch.tensor(X, dtype=torch.float32).permute(1, 0, 2)
#     # hhh = np.mean(np.mean(X, axis=0), axis=0)
#     # import matplotlib.pyplot as plt
#     # plt.plot(hhh)
#     # hhh = model._modules['batchnorm'].running_mean.detach().numpy()
#     # plt.plot(hhh)
#     # plt.show()
#     Y_hat = model(Xt)
#     tmp = Y_hat.cpu().detach().numpy() > 0.5

#     for i in range(test_size):
#         st, ed = i * nb_frames, (i + 1) * nb_frames
#         if ed > Zxx.shape[1]:
#             st, ed = Zxx.shape[1] - nb_frames, Zxx.shape[1]

#         sigY[:, st: ed] = (tmp[:, i, :] * Zxx[:, st: ed].transpose()).transpose()

#     sigY *= Zxx / abs(Zxx + 0.000001)
#     _, restore = signal.istft(sigY, fs)
#     restore = restore[:len(sig)]
#     return restore, sig - restore



# if __name__ == '__main__':
#     print(torch.__version__)
#     root = '../input/test25/testset25/testset25/'
#     ImagePath = root + "testimage"
#     VideoPath = root + "testvideo"
#     AudioPath = root + "gt_audio"
#     OutputPath = root + "result_json"
#     OutputAudio = root + "result_audio"

#     Test(OutputPath, OutputAudio, VideoPath, AudioPath, ImagePath)

# class Discriminator(Module):
#     def __init__(
#         self, nb_features, nb_frames, hidden_size=256, nb_layers=1, 
#         input_mean=None, input_scale=None
#     ):
#         super(Discriminator, self).__init__()
#         self.hidden_size = hidden_size
#         self.input_mean = Parameter(
#             torch.from_numpy(np.copy(input_mean)).float()
#         )
#         self.input_scale = Parameter(
#             torch.from_numpy(np.copy(input_scale)).float(),
#         )
#         self.encode_fc = Linear(
#             nb_features, hidden_size
#         )
#         self.lstm = LSTM(
#             input_size=hidden_size,
#             hidden_size=hidden_size,
#             num_layers=nb_layers,
#             bidirectional=False,
#             batch_first=False
#         )
#         self.fc = torch.nn.Sequential(
#             Linear(
#             in_features=hidden_size * nb_frames,
#             out_features=2),
#             torch.nn.Dropout(0.2))
            


#     def forward(self, x):
#         nb_frames, nb_batches, nb_features = x.data.shape
#         x -= self.input_mean
#         x /= self.input_scale
#         x = self.encode_fc(x.reshape(-1, nb_features))
#         x = torch.tanh(x)
#         x, state = self.lstm(x.reshape(nb_frames, nb_batches, self.hidden_size))
#         x = self.fc(x.permute(1, 0, 2).reshape(-1, self.hidden_size * nb_frames))
#         x = F.softmax(x)

#         return x

# model_D = Discriminator(
#     nb_features, nb_frames,
#     input_mean=0,
#     input_scale=1,
# ).to(device)

# batch_size = 64
# iters = 1000
# optimizer = optim.Adam(model_D.parameters(), lr=0.0001)
# criterion = torch.nn.CrossEntropyLoss()


# for it in tqdm(range(iters)):
#     model_D.train()
#     X = np.zeros((batch_size, nb_frames, nb_features))
#     Y = np.zeros((batch_size))
    
#     for k in range(batch_size):
#         index1 = np.random.randint(0, len(solos) - 1)
#         index2 = np.random.randint(0, len(solos) - 1)
#         solo1 = solos[index1]
#         solo2 = solos[index2]
#         coef1 = np.random.uniform(0.02, 1)
#         coef2 = np.random.uniform(0.02, 1)
#         sig1 = genData(solo1)
#         sig2 = genData(solo2)
#         sig = sig1 * coef1 + sig2 * coef2
#         X[k] = abs(sig).transpose()
#         if solo1 == 'violin' or solo2 == 'violin':
#             Y[k] = 1
    
#     Xt = torch.tensor(X, dtype=torch.float32, device=device).permute(1, 0, 2)
#     Yt = torch.tensor(Y, dtype=torch.long, device=device)
    
#     #print(Xt.shape[1])
#     #Xhatt = model1(Xt).permute(1, 0, 2);
#     #print(Xhatt.shape)
#     #XD = np.concatenate((X, Xhatt.detach().cpu().numpy()), axis=0)
#     #XDt = torch.tensor(XD, dtype=torch.float32, device=device).permute(1, 0, 2)
#     #print(XD.shape)
    
#     optimizer.zero_grad()
#     Y_hat = model_D(Xt)
    
#     loss = criterion(Y_hat, Yt)
#     loss.backward()
#     optimizer.step()
    
#     if it % 30 == 0:
#         print(loss)
#         print(np.sum(np.argmax(Y_hat.detach().cpu().numpy(), axis=-1) == Y) / batch_size)
    
# model_D.eval()
# test_size = 1
# X = np.zeros((test_size, nb_frames, nb_features))
# Y = np.zeros((test_size))

# solo1 = 'violin'
# solo2 = 'saxophone'

# for k in range(test_size):
    
#     _, _, sig = genMix(solo1, solo2)
#     X[k] = abs(sig).transpose()
#     #Y[k] = index
# _, restore_sig = signal.istft(sig, fs)
# ipd.display(ipd.Audio(restore_sig, rate=fs))
# Xt = torch.tensor(X, dtype=torch.float32, device=device).permute(1, 0, 2)
# #Yt = torch.tensor(Y, dtype=torch.long, device=device)
# Y_hat = model_D(Xt)
# print(Y_hat.detach().cpu().numpy())

# # sig1, sig2, sig = genMix(solo1, solo2)
# # x = np.reshape(abs(sig), -1)
# # plt.hist(x, bins=100, range = (0, 25))
# os.listdir('../input/test25/testset25/testset25/gt_audio')
# f = '../input/test25/testset25/testset25/gt_audio/saxophone_3_violin_3.wav'
# model1.eval()
# _, sig = wavfile.read(f)
# ipd.display(ipd.Audio(sig, rate=fs))
# _, _, Zxx = signal.stft(sig, fs, nperseg=(nb_features - 1) * 2)
# sigY = np.zeros_like(Zxx)
# test_size = (Zxx.shape[1] + nb_frames - 1) // nb_frames
# X = np.zeros((test_size, nb_frames, nb_features))
# for i in range(test_size):
#     st, ed = i * nb_frames, (i + 1) * nb_frames
#     if ed > Zxx.shape[1]:
#         st, ed = Zxx.shape[1] - nb_frames, Zxx.shape[1]
#     X[i] = abs(Zxx).transpose()[st: ed, :]
# Xt = torch.tensor(X, dtype=torch.float32, device=device).permute(1, 0, 2)

# Y_hat = model1(Xt)
# tmp = Y_hat.cpu().detach().numpy()

# print(tmp.shape)
# # for i in range(test_size):
# #     st, ed = i * nb_frames, (i + 1) * nb_frames
# #     if ed > Zxx.shape[1]:
# #         st, ed = Zxx.shape[1] - nb_frames, Zxx.shape[1]
# #     sigY[:, st: ed] = tmp[i, :, :].transpose()

# # sigY *= Zxx / abs(Zxx + 0.00001)
# # _, restore = signal.istft(sigY, fs)
# class GAN(Module):
#     def __init__(
#         self, nb_features, nb_frames, hidden_size=256, nb_layers=1, 
#         input_mean=None, input_scale=None, output_mean=None
#     ):
#         super(GAN, self).__init__()

#         self.hidden_size = hidden_size
#         self.input_mean = Parameter(torch.from_numpy(np.copy(input_mean)).float())
#         self.input_scale = Parameter(torch.from_numpy(np.copy(input_scale)).float(),)
#         self.output_scale = Parameter(torch.ones(nb_features).float())
#         self.output_mean = Parameter(torch.from_numpy(np.copy(output_mean)).float())
        
#         self.G = Module()
#         self.G.encode_fc = Linear(nb_features, hidden_size)
#         self.G.lstm = LSTM(
#             input_size=hidden_size,
#             hidden_size=hidden_size,
#             num_layers=nb_layers,
#             bidirectional=False,
#             batch_first=False
#         )
#         self.G.fc = Linear(
#             in_features=hidden_size,
#             out_features=nb_features
#         )
        
#         self.D = Module()
#         self.D.encode_fc = Linear(nb_features, hidden_size)
#         self.D.lstm = LSTM(
#             input_size=hidden_size,
#             hidden_size=hidden_size,
#             num_layers=nb_layers,
#             bidirectional=False,
#             batch_first=False
#         )
#         self.D.fc = Linear(
#             in_features=hidden_size * nb_frames,
#             out_features=2
#         )
#         self.criterion = torch.nn.CrossEntropyLoss()
#         self.MSE = torch.nn.MSELoss()
        
        
#     def netG(self, x0):
#         nb_frames, nb_batches, nb_features = x0.data.shape
#         x = x0 - self.input_mean
#         x /= self.input_scale
#         x = self.G.encode_fc(x.reshape(-1, nb_features))
#         x = torch.tanh(x)
#         x, state = self.G.lstm(x.reshape(nb_frames, nb_batches, self.hidden_size))
#         x = self.G.fc(x.reshape(-1, self.hidden_size))
#         x = x.reshape(nb_frames, nb_batches, nb_features)
#         x *= self.output_scale
#         x += self.output_mean
#         x = F.relu(x)
#         return x
    
#     def netD(self, x0):
#         nb_frames, nb_batches, nb_features = x0.data.shape
#         x = x0 - self.input_mean
#         x /= self.input_scale
#         x = self.D.encode_fc(x.reshape(-1, nb_features))
#         x = torch.tanh(x)
#         x, state = self.D.lstm(x.reshape(nb_frames, nb_batches, self.hidden_size))
#         x = self.D.fc(x.permute(1, 0, 2).reshape(-1, self.hidden_size * nb_frames))
#         x = F.softmax(x)
#         return x
    
#     def forward(self, real_x, mix, single):
#         self.fake_x = self.netG(mix)
#         self.single = single;
#         self.real_x = real_x
#         self.label = torch.autograd.Variable(torch.LongTensor(real_x.shape[1]).fill_(1), requires_grad=False).to(device)
    
#     def backward_D(self):
#         pred_fake = self.netD(self.fake_x.detach()) 
#         self.loss_D_fake = self.criterion(pred_fake, self.label*0)
#         pred_real = self.netD(self.real_x)
#         self.loss_D_real = self.criterion(pred_real, self.label*1)
#         self.loss_D = self.loss_D_fake + self.loss_D_real
#         self.loss_D.backward()
        
    
#     def backward_G(self):
#         pred_fake = self.netD(self.fake_x)
#         self.loss1 = self.MSE(self.single, self.fake_x)
#         self.loss2 = self.criterion(pred_fake, self.label*1) * 10000
#         self.loss_G =  self.loss1 + self.loss2
#         self.loss_G.backward()
        
# gan = GAN(
#     nb_features, nb_frames,
#     input_mean=mean_all,
#     input_scale=scal_all,
#     output_mean=mean1
# ).to(device)

# d_optim = torch.optim.Adam(gan.D.parameters(), lr=0.001)
# g_optim = torch.optim.Adam(gan.G.parameters(), lr=0.001)
# from tqdm import tqdm_notebook as tqdm

# batch_size = 32
# num = 0
# for i in tqdm(range(num)):
#     gan.train()
#     single = np.zeros((batch_size, nb_frames, nb_features))
#     mix = np.zeros((batch_size, nb_frames, nb_features))
#     real = np.zeros((batch_size, nb_frames, nb_features))
#     for k in range(batch_size):
#         sig1, sig2, sig = genMix(solo1, solo2)
#         single[k] = abs(sig1).transpose()
#         mix[k] = abs(sig).transpose()
#         sig1, sig2, sig = genMix(solo1, solo2)
#         real[k] = abs(sig1).transpose()
    
#     real = torch.tensor(real, dtype=torch.float32, device=device).permute(1, 0, 2)
#     mix = torch.tensor(mix, dtype=torch.float32, device=device).permute(1, 0, 2)
#     single = torch.tensor(single, dtype=torch.float32, device=device).permute(1, 0, 2)
    
#     gan.forward(real, mix, single)
    
#     d_optim.zero_grad()
#     gan.backward_D()
#     d_optim.step()

#     g_optim.zero_grad()
#     gan.backward_G()
#     g_optim.step()
    
#     if i % 100 == 0:
#         #print(gan.G.fc.weight)
#         #print(gan.D.fc.weight)
#         print(gan.loss1)
#         print(gan.loss2)

# from mir_eval.separation import bss_eval_sources

# def test(num):
#     gan.eval()
#     test_size = 1
#     X = np.zeros((test_size, nb_frames, nb_features))
#     Y1 = np.zeros((test_size, nb_frames, nb_features))
#     Y2 = np.zeros((test_size, nb_frames, nb_features))
#     #np.random.seed(42)

#     cnt = 0
#     sdr = (0, 0)
#     for i in tqdm(range(num)):
#             with torch.no_grad():
#                 sig1, sig2, sig = genMix(solo1, solo2)

#                 X[0] = abs(sig).transpose()
#                 Y1[0] = abs(sig1).transpose()
#                 Y2[0] = abs(sig2).transpose()
#                 Yt1 = torch.tensor(Y1, dtype=torch.float32, device=device)
#                 Yt2 = torch.tensor(Y2, dtype=torch.float32, device=device)
#                 Xt = torch.tensor(X, dtype=torch.float32, device=device)
#                 gan.forward(Xt, Xt, Yt1)
#                 Y1_hat = gan.fake_x
#                 Y2_hat = Y1_hat
#                 look1 = Y1_hat.cpu().detach().numpy()
#                 look2 = Y2_hat.cpu().detach().numpy()

#                 _, restore_sig = signal.istft(sig, fs)
#                 _, restore_sig1 = signal.istft(sig1, fs)
#                 _, restore_sig2 = signal.istft(sig2, fs)

#                 _, restore_look1 = signal.istft(look1[0].transpose() * sig / abs(sig), fs)
#                 _, restore_look2 = signal.istft(look2[0].transpose() * sig / abs(sig), fs)

#                 if num == 1:
#                     ipd.display(ipd.Audio(restore_sig, rate=fs))
#                     ipd.display(ipd.Audio(restore_sig1, rate=fs))
#                     ipd.display(ipd.Audio(restore_sig2, rate=fs))
#                     ipd.display(ipd.Audio(restore_look1, rate=fs))
#                     ipd.display(ipd.Audio(restore_look2, rate=fs))

#                 restore_sig = np.reshape(restore_sig, (1, -1))
#                 restore_sig1 = np.reshape(restore_sig1, (1, -1))
#                 restore_sig2 = np.reshape(restore_sig2, (1, -1))
#                 restore_look1 = np.reshape(restore_look1, (1, -1))
#                 restore_look2 = np.reshape(restore_look2, (1, -1))

#                 #sdr1 = bss_eval_sources(np.concatenate((restore_sig1, restore_sig2)), np.concatenate((restore_sig, (restore_sig + 10000)* 0.000000001)))[0]
#                 #sdr2 = bss_eval_sources(np.concatenate((restore_sig1, restore_sig2)), np.concatenate((restore_look1, restore_look2)))[0]
#                 #sdr += sdr2
#                 #cnt += 1

#                 #print((sdr2, sdr1))
#     #print(sdr / cnt)
# test(1)
# import cv2
# from tqdm import tqdm_notebook as tqdm
# import os
# import torch
# from PIL import Image
# from torchvision import models, transforms
# from torch.autograd import Variable
# from torch.nn import functional as F
# import numpy as np

# import json


# global net
# global normalize
# global preprocess
# global features_blobs
# global classes
# global weight_softmax
# labels_path='../input/labels/labels.json'
# idxs=[401,402,486,513,558,642,776,889]
# names=['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']

# # def returnCAM(feature_conv, weight_softmax, class_idx):
# #     size_upsample = (256, 256)
# #     bz, nc, h, w = feature_conv.shape
# #     output_cam = []
# #     for idx in class_idx:
# #         cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
# #         cam = cam.reshape(h, w)
# #         cam = cam - np.min(cam)
# #         cam_img = cam / np.max(cam)
# #         cam_img = np.uint8(255 * cam_img)
# #         output_cam.append(cv2.resize(cam_img, size_upsample))
# #     return output_cam

# def hook_feature(module, input, output):
#     global features_blobs
#     features_blobs=output.data.cpu().numpy()

# def load_model():
#     global net
#     global normalize
#     global preprocess
#     global features_blobs
#     global classes
#     global weight_softmax
#     net = models.densenet161(pretrained=True)
#     finalconv_name = 'features'
#     net.eval()
#     net._modules.get(finalconv_name).register_forward_hook(hook_feature)
#     params = list(net.parameters())
#     weight_softmax = np.squeeze(params[-2].data.numpy())
#     normalize = transforms.Normalize(
#        mean=[0.485, 0.456, 0.406],
#        std=[0.229, 0.224, 0.225]
#     )
#     preprocess = transforms.Compose([
#        transforms.Resize((224,224)),
#        transforms.ToTensor(),
#        normalize
#     ])
#     classes = {int(key):value for (key, value) in json.load(open(labels_path,'r')).items()}
#     if torch.cuda.is_available():
#         net=net.cuda()

# def get_CAM(imdir,savedir,imname, pos):
#     img_pil = Image.open(os.path.join(imdir,imname))
#     width, height = img_pil.size
#     halfW = width / 2
#     if halfW > height:
#         box = [(halfW - height) / 2, 0, (halfW - height) / 2 + height, height]
#     else:
#         box = [0, (height - halfW) / 2, halfW, (height - halfW) / 2 + halfW]
#     if pos == 1:
#         box[0] += halfW
#         box[2] += halfW
    
#     img_pil = img_pil.crop(box)
    
#     #print(img_pil.size)
#     img_tensor = preprocess(img_pil)
#     img_variable = Variable(img_tensor.unsqueeze(0))
#     if torch.cuda.is_available():
#         img_variable=img_variable.cuda()
    
#     logit = net(img_variable)
#     h_x = F.softmax(logit, dim=1).data.squeeze()
#     if torch.cuda.is_available():
#         h_x=h_x.cpu()
#     probs1 = h_x.numpy()
#     probs=[]
#     for i in range(0, 8):
#         #print('{:.3f} -> {}'.format(probs1[idxs[i]], names[i]))
#         '''
#         CAMs = returnCAM(features_blobs, weight_softmax, [idxs[i]])        
#         heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
#         result = heatmap * 0.3 + img * 0.5
#         cv2.imwrite(os.path.join(savedir,names[i],imname), result)
#         '''
#         probs.append(probs1[idxs[i]])
#     return probs

# def main():
#     root = '../input/test25/testset25/testset25/testimage/'
#     root = '../input/testset7/testset7/testset7/testimage/'
#     for file in os.listdir(root):
        
#         imdir= root + file
#         load_model()
#         imlist=os.listdir(imdir)
#         probsLeft=np.zeros([8])
#         probsRight=np.zeros([8])
#         for index, im in tqdm(enumerate(imlist)):
#             probs1=get_CAM(imdir,'results',im, 0)
#             probsLeft += np.array(probs1)
#             probs1=get_CAM(imdir,'results',im, 1)
#             probsRight += np.array(probs1)
#             x = sorted(enumerate(probsLeft), key=lambda x:-x[1])
#             y = sorted(enumerate(probsRight), key=lambda x:-x[1])
#             if index > 10 and x[0][1] > 10 * x[1][1] and y[0][1] > 10 * y[1][1]:
#                 break
#         print(probsLeft)
#         print(probsRight)
        
#         print(names[x[0][0]])
#         print(names[y[0][0]])
#         print(file)
        

# main()
# net = models.densenet161(pretrained=True)
# net.classifier = torch.nn.Linear(in_features=2208, out_features=8, bias=True)
# print(net.classifier)
