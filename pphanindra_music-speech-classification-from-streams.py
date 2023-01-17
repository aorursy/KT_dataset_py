import torch
torch.manual_seed(0)

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-whitegrid')

import matplotlib as mpl
mpl.rcParams['figure.figsize']=(15,3.4)
path = '../input/gtzan-musicspeech-collection'

bollywood_audio = '../input/musicspeechclassificationfromstreams'
import librosa

audio_datafile = path + '/music_wav/bagpipe.wav'
x, sf = librosa.load(audio_datafile)
print(f'shape: {x.shape}\nsampling freq: {sf}\ntime duration: len/sf = {x.shape[0]/sf} seconds')
plt.title('plot of a sound sinal from a .wav file')
plt.plot(x[:1024])
import IPython.display as ipd
ipd.Audio(audio_datafile)
import librosa.display
#plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sf)
plt.title('a music waveplot with librosa.display.waveplot')
speech = bollywood_audio + '/housefull_dialog2.wav'

x, sf = librosa.load(speech)

print(f'shape: {x.shape}\nsampling freq: {sf}\ntime duration: len/sf = {x.shape[0]/sf} seconds')
plt.title('plot of a sound sinal from a .wav file')
plt.plot(x[:2048]);
music = bollywood_audio + '/housefull_song1.wav'

x, sf = librosa.load(music)

print(f'shape: {x.shape}\nsampling freq: {sf}\ntime duration: len/sf = {x.shape[0]/sf} seconds')
plt.title('plot of a sound sinal from a .wav file')
plt.plot(x[:2048]);
speech_file = path + '/speech_wav/god.wav'

x, sf = librosa.load(speech_file)

print(f'shape: {x.shape}\nsampling freq: {sf}\ntime duration: len/sf = {x.shape[0]/sf} seconds')
plt.title('plot of a sound sinal from a .wav file')
plt.plot(x[:1024]);
import IPython.display as ipd
ipd.Audio(speech_file)
import librosa.display
#plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sf)
plt.title('a speech waveplot')
import os

class_names = ['music', 'speech']
data_dir = {'music' : path + '/music_wav', 
            'speech': path + '/speech_wav'}

wav_files = {cls_name: [] for cls_name in class_names}
for cls_name in class_names:
    folder = data_dir[cls_name]
    filelist = os.listdir(folder)
    for filename in filelist:
        if filename[-4:] == '.wav':
            wav_files[cls_name].append(os.path.join(folder, filename))
            
fileNames = os.listdir(bollywood_audio)
bolly_wav_files = [os.path.join(bollywood_audio, file) for file in fileNames]
print (bolly_wav_files)
print([len(wav_files[c]) for c in class_names])
wav_files['music'][:3]
np.random.seed(1)

file_list = {'train': [], 'val': []}
for class_id, c in enumerate(class_names):
    n_data = len(wav_files[c])
    rindx = np.random.permutation(n_data)
    n_validation = int(0.25*n_data)
    v_indx = rindx[:n_validation]
    t_indx = rindx[n_validation:]
    file_list['train'] += [ (wav_files[c][k], class_id) for k in t_indx ] 
    file_list['val'] += [ (wav_files[c][k], class_id) for k in v_indx ] 
#     file_list['']

bolly_tuples = []
for file in bolly_wav_files:
    if 'song' in file:
        bolly_tuples.append((file, 0))
    else:
        bolly_tuples.append((file, 1))

[ len(file_list[tv]) for tv in ['train', 'val'] ]
import torch
import numpy as np
import librosa # audio file manipulation

class MSDataset(torch.utils.data.Dataset):
    """ Music/Speech Classification 
        filelist: [(file_path, class_id)]
        sample_time: time duration to sample from .wav file
                     the sample is extracted somewhere in the middle of the whole sequence
                     similar to data augmentation
                     
         Validation dataset: the first segment of the sequence is used.
                             Another option is to apply several segments and accumulate multiple inferences
    """
    def __init__(self, filelist, sample_sec=5., is_train=True):
        self.filelist = filelist
        self.time_duration = sample_sec
        self.is_train = is_train
        
        _, sf = librosa.load(filelist[0][0])
        self.sf = sf
        self.n_features = int(self.time_duration * sf)
        
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, i):
        # 1. load the file
        # 2. sample a segment of the length from the whole seq
        # 3. return segment, id
        audio_file, class_id = self.filelist[i]
        x, sf = librosa.load(audio_file)

        if self.is_train:
            k = np.random.randint(low=0, high=x.shape[0]-self.n_features) # choose the start index
        else:
            k = 0
        
        x = torch.from_numpy(x[k:k+self.n_features]).reshape(1,-1)
        
        return x, class_id
    
    def load(self, audio_file):
        return librosa.load(audio_file)
ds = MSDataset(file_list['train'], sample_sec=5, is_train=True)
# c = file_list['train'][1]
# print(c)
x, label = ds[-1]

print('dataset length: ', len(ds), x.dtype, x.shape, label)#label.dtype, label.shape)

plt.title(f'Audio segment, randomly sampled. src len: {x.shape} label: {label}');
plt.plot(x.reshape(-1)); 
sample_sec = 2
batch_size = 32

data_loader = {tv: 
                   torch.utils.data.DataLoader(MSDataset(file_list[tv], sample_sec=sample_sec, is_train=tv=='train'),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               )
               for tv in ['train', 'val']}
#
data_loader
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_size = 17
        self.Conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=3)
        self.BatchNorm = nn.BatchNorm1d(out_channels)
        self.ELU = nn.ELU()
    

    def forward(self, x):
        x = self.Conv(x)
        x = self.BatchNorm(x)
        x = self.ELU(x)

        return x
    
    
class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.Hidden_1 = BasicBlock(in_channels, 100)  
        self.Hidden_2 = BasicBlock(100, out_channels)
        self.Pooling = nn.AdaptiveAvgPool1d(1)
        self.Flatten = nn.Flatten()
        self.activation = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        x = self.Hidden_1(x)
        x = self.Hidden_2(x)
        x = self.Pooling(x)
        x = self.Flatten(x)
        x = self.activation(x)
        
        return x
n_channels = 1
n_targets  = 2

model = MyModel(n_channels, n_targets)
model
def training_loop(n_epochs, optim, model, loss_fn, dl_train, dl_val, hist=None):
    if hist is not None:
        pass
    else:
        hist = {'tloss': [], 'tacc': [], 'vloss': [], 'vacc': []}
    best_acc = 0
    for epoch in range(1, n_epochs+1):
        tr_loss, tr_acc = 0., 0.
        n_data = 0
        for im_batch, label_batch in dl_train: # minibatch
            im_batch, label_batch = im_batch.to(device), label_batch.to(device)
            ypred = model(im_batch)
            loss_train = loss_fn(ypred, label_batch)
        
            optim.zero_grad()
            loss_train.backward()
            optim.step()
   
            # accumulate correct prediction
            tr_acc  += (torch.argmax(ypred.detach(), dim=1) == label_batch).sum().item() # number of correct predictions
            tr_loss += loss_train.item() * im_batch.shape[0]
            n_data  += im_batch.shape[0]
        #
        # statistics
        tr_loss /= n_data
        tr_acc  /= n_data
        #
        val_loss, val_acc = performance(model, loss_fn, dl_val)
        
        if epoch <= 5 or epoch % 10 == 0 or epoch == n_epochs:
             print(f'Epoch {epoch}, tloss {tr_loss:.2f} t_acc: {tr_acc:.4f}  vloss {val_loss:.2f}  v_acc: {val_acc:.4f}')
#         else:
#             if best_acc < val_acc:
#                 best_acc = val_acc
#                 print(' best val accuracy updated: ', best_acc)
        #
        # record for history return
        hist['tloss'].append(tr_loss)
        hist['vloss'].append(val_loss) 
        hist['tacc'].append(tr_acc)
        hist['vacc'].append(val_acc)
        
    print ('finished training_loop().')
    return hist
#

def performance(model, loss_fn, dataloader):
    model.eval()
    with torch.no_grad():
        loss, acc, n = 0., 0., 0.
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            ypred = model(x)
            loss += loss_fn(ypred, y).item() * len(y)
            p = torch.argmax(ypred, dim=1)
            acc += (p == y).sum().item()
            n += len(y)
        #
    loss /= n
    acc /= n
    return loss, acc
#
def plot_history(history):
    fig, axes = plt.subplots(1,2, figsize=(16,6))
    axes[0].set_title('Loss'); 
    axes[0].plot(history['tloss'], label='train'); axes[0].plot(history['vloss'], label='val')
    axes[0].legend()
    max_vacc = max(history['vacc'])
    axes[1].set_title(f'Acc. vbest: {max_vacc:.2f}')
    axes[1].plot(history['tacc'], label='train'); axes[1].plot(history['vacc'], label='val')
    axes[1].legend()
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model
in_channels = 1
n_targets = 2

model = MyModel(in_channels, n_targets).to(device)

# optim
learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# loss
criterion = nn.CrossEntropyLoss().to(device)

# history
history = None
history = training_loop(230, optimizer, model, criterion, data_loader['train'], data_loader['val'], history)
plot_history(history)
bolly_data_set = torch.utils.data.DataLoader(MSDataset(bolly_tuples, sample_sec=sample_sec, is_train=False),
                                               batch_size=batch_size,
                                               shuffle=False,
                                               )
def testBollywoodAudio(model, dataloader):
    model.eval()
    pred = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            ypred = model(x)
            p = torch.argmax(ypred, dim=1)
            print(p, y)
            pred.append(p)
    return pred

predictions = testBollywoodAudio(model, bolly_data_set)

pred_arr = predictions[0].cpu().numpy()
from prettytable import PrettyTable
t = PrettyTable(['File', 'Actual', 'Predicted'])

idx = 0
for (path, actual_label) in bolly_tuples:
    file_name = path.split('/')[-1]
#     print('File: {} Actual Label: {} Predicted Label: {}'.format(file_name, actual_label, pred_arr[idx]))
    t.add_row([file_name, actual_label, pred_arr[idx]])
    idx += 1

print(t)