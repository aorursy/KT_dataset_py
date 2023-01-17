# File Settings

ROOT_FOLDERS = ['/kaggle/input/spanish-single-speaker-speech-dataset/', '/kaggle/input/120h-spanish-speech/asr-spanish-v1-carlfm01/']

CSV_FILE_PATH_1 = ROOT_FOLDERS[0] + 'transcript.txt'

CSV_FILE_PATH_2 = ROOT_FOLDERS[1] + 'files.csv'

CSV_FILE_PATH = [CSV_FILE_PATH_1, CSV_FILE_PATH_2]



SAVE_RESULTS_PATH = '/kaggle/working/'

SAVE_MODELS_PATH = '/kaggle/working/'





  
# Systems Libraries

import os

import time



# For audio processing

import librosa

import librosa.display

import IPython as ipd



#from torchsummary import summary



# For data processing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For Neural networks

import torch

from torch import nn

import pickle as pkl

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader





# For visualization

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline





# For text processing

import string



# Scripts created by me:

import utils

import models

import textprocessor

import speechdataset



# Import Early Stop https://github.com/Bjarten/early-stopping-pytorch

#from pytorchtools import EarlyStopping



from datetime import datetime



ipd.display.Audio(filename='../input/spanish-single-speaker-speech-dataset/batalla_arapiles/batalla_arapiles_0010.wav')
(waveform, sample_rate) = librosa.load('../input/spanish-single-speaker-speech-dataset/batalla_arapiles/batalla_arapiles_0010.wav')

spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)

utils.plot_all(audio_data=waveform, spec=spectrogram, sr=sample_rate, file='batalla_arapiles_0010.wav')
ipd.display.Audio(filename='../input/120h-spanish-speech/asr-spanish-v1-carlfm01/audios/00041a31-2e68-444a-9a46-d8140b532d9c.wav')
(waveform, sample_rate) = librosa.load('../input/120h-spanish-speech/asr-spanish-v1-carlfm01/audios/00041a31-2e68-444a-9a46-d8140b532d9c.wav')

spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)

utils.plot_all(audio_data=waveform, spec=spectrogram, sr=sample_rate, file='00041a31-2e68-444a-9a46-d8140b532d9c.wav')




# The following code will be for collat_fn for the pytorch dataloader function    

def data_processing(audio_data):

    spectrograms = []

    labels = []

    input_lengths = []

    label_lengths = []

    #print("data processing")

    for (spec,label) in audio_data:

        #The spectrogram is in (128, 407) and (128, 355) for example but later on for padding the function expects (407, 128) and (355, 128). So we need to transpose the matrices.

        spectrograms.append(torch.Tensor(spec.transpose()))

        t = textprocessor.TextProcessor()

        label = torch.Tensor(t.text2int(text=label))

        labels.append(label)

        input_lengths.append(spec.shape[0]//2)

        label_lengths.append(len(label))

    #print("Start padding")

    spec_pad = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2,3)   #(batch, channel=1, features, time )

    label_pad = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    #print("Finish padding")

    return spec_pad, label_pad, input_lengths, label_lengths



def load_dataset(csv_file, root_dir, n_samples=10000, f_type='spec'):

  

  # Create the dataset and split into validation dataset and test dataset.

    total_dataset = speechdataset.SpanishSpeechDataSet(csv_files=csv_file, root_dir=root_dir, f_type=f_type, num_samples=n_samples)

    train_size = int(0.8 * len(total_dataset))

    val_test_size = len(total_dataset) - train_size

    train_dataset, val_test_dataset = torch.utils.data.random_split(total_dataset, [train_size, val_test_size])

    valid_size = int(0.9 * len(val_test_dataset))

    test_size = len(val_test_dataset) - valid_size

    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [valid_size, test_size])

    print("Total Training Dataset = {}, Valid Dataset = {} and Test Dataset = {}".format(len(train_dataset),len(val_dataset), len(test_dataset) ))

    print("Total = ", len(train_dataset) + len(val_dataset) + len(test_dataset))

    sample = train_dataset[0]

    if f_type =='spec':

        print("*****Showing spectrogram with label:**** \n")

        print(sample[1])

        utils.plot_spec(sample[0], title="Spectrogram")

    else:

        print("*****Showing MFCCs with label:**** \n")

        print(sample[1])

        utils.plot_mfccs(sample[0])

        

    return (train_dataset, val_dataset, test_dataset)







def create_data_loaders(train_dataset, val_dataset, test_dataset, kwargs, batch_size):

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True,drop_last=True, collate_fn=lambda x: data_processing(x), **kwargs )

    valid_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,shuffle=False,drop_last=True, collate_fn=lambda x: data_processing(x), **kwargs)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False,drop_last=True, collate_fn=lambda x: data_processing(x), **kwargs)

    return (train_loader, valid_loader, test_loader)


def train(n_epochs, train_loader, valid_loader, model, criterion, clip, device, lr, batch_size, save_model_path, save_pkl_path, model_name, show_every_n_batch=50):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 

                                            steps_per_epoch=int(len(train_loader)),

                                            epochs=n_epochs,

                                            anneal_strategy='linear')

    train_data_len = len(train_loader.dataset)

    valid_data_len = len(valid_loader.dataset)

    epoch_train_loss = 0

    epoch_val_loss = 0

    train_losses = []

    valid_losses = []

    print("#######################")

    print("#  Start Training    #")

    print("#######################")

    

    model.train()

    for e in range(n_epochs):

        t0 = time.time()

        #Initialize hidden state

        #h = model.init_hidden(batch_size, device)



        #batch loop

        running_loss = 0.0

        for batch_idx, _data in enumerate(train_loader, 1):

            specs, labels, input_lengths, label_lengths = _data

            specs, labels = specs.to(device), labels.to(device)

            #print(batch_idx)

            # Creating new variables for the hidden state, otherwise

            # we'd backprop through the entire training history

            #h = h.detach()

            # zero accumulated gradients

            model.zero_grad()

            # get the output from the model

            #output, h = model(specs, h)

            output = model(specs)

            output = F.log_softmax(output, dim=2)

            output = output.transpose(0,1)

            # calculate the loss and perform backprop

            loss = criterion(output, labels.float(), input_lengths, label_lengths)

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

            nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            scheduler.step()

            # loss stats

            running_loss += loss.item() * specs.size(0)

            if (batch_idx % 1000 == 0):

                output = output.transpose(1,0)

                #print("Training Batch Number: ", batch_idx)

                (decoded_preds, decoded_targets) = textprocessor.GreedyDecoder(output, labels, label_lengths)

                for j in range(len(decoded_preds)):

                    #print("****************************************************************************")

                    #print("Predicted -- {}".format(decoded_preds[j]))

                    #print("Utterance -- {}\n\n".format(decoded_targets[j]))

                    utils.write_to_csv('/kaggle/working/'+model_name+'_training_results.csv', decoded_preds[j], decoded_targets[j], running_loss, epoch=e+1)

                

        t_t = time.time() - t0



            

        ######################    

        # validate the model #

        ######################

        with torch.no_grad():

            model.eval() 

            tv = time.time()

            running_val_loss = 0.0

            for batch_idx_v, _data in enumerate(valid_loader, 1):

                specs, labels, input_lengths, label_lengths = _data

                specs, labels = specs.to(device), labels.to(device)

                #val_h = model.init_hidden(batch_size, device)

                #output, val_h = model(specs, val_h)

                output = model(specs)

                output = F.log_softmax(output, dim=2)

                output = output.transpose(0,1)

                val_loss = criterion(output, labels.float(), input_lengths, label_lengths)

                running_val_loss += val_loss.item() * specs.size(0)

                if (batch_idx_v % 200 == 0):

                    output = output.transpose(1,0)

                    #print("Validation Batch Number: ", batch_idx)

                    (decoded_preds, decoded_targets) = textprocessor.GreedyDecoder(output, labels, label_lengths)

                    for j in range(len(decoded_preds)):

                        #print("****************************************************************************")

                        #print("Predicted -- {}".format(decoded_preds[j]))

                        #print("Utterance -- {}\n\n".format(decoded_targets[j]))

                        utils.write_to_csv('/kaggle/working/'+model_name+'_validation_results.csv', decoded_preds[j], decoded_targets[j], running_val_loss, epoch=e+1)

            print("Epoch {}: Training took {:.2f} [s]\tValidation took: {:.2f} [s]\n".format(e+1, t_t, time.time() - tv))

                

                

        epoch_train_loss = running_loss / train_data_len

        epoch_val_loss = running_val_loss / valid_data_len

        train_losses.append(epoch_train_loss)

        valid_losses.append(epoch_val_loss)

        print('Epoch: {} Losses\tTraining Loss: {:.6f}\tValidation Loss: {:.6f}'.format(

                e+1, epoch_train_loss, epoch_val_loss))

        model.train()

        

        print("-------------------------------------------------------------------------------------------")

        print('Epoch {} took total {} seconds'.format(e+1, time.time() - t0))

        print("-------------------------------------------------------------------------------------------")



    with open(save_pkl_path, 'wb') as f:       #this will save the list as "results.pkl" which you can load in later 

        pkl.dump((epoch_train_loss, epoch_val_loss), f)

    utils.save_model(save_path=save_model_path, model=model)

    utils.save_checkpoint(save_path=save_model_path, model=model, optimizer=optimizer, epoch=e, loss=train_losses)

    return (model, train_losses, valid_losses)
def test_model(test_data, model, model_name,device, batch_size):

    model.eval()

    print("#######################")

    print("# Testing Model: {} #".format(model_name))

    print("#######################\n\n")

    test_cer, test_wer = [], []

    test_loss = 0.0

    #h = model.init_hidden(batch_size, device)

    with torch.no_grad():

        for batch_idx, _data in enumerate(test_data, 1):

            specs, labels, input_lengths, label_lengths = _data

            specs, labels = specs.to(device), labels.to(device)

            # initialize the hidden state

            # get the output of the rnn

            #output, _ = model(specs, h)

            output = model(specs) 

            output = output.transpose(0,1) #(time, batch,n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)

            #Input should be [batch, time, n_classes]

            output = output.transpose(1,0)

            (decoded_preds, decoded_targets) = textprocessor.GreedyDecoder(output, labels, label_lengths)

            test_loss += loss.item()* specs.size(0)

            #print(test_loss.item())

            for j in range(len(decoded_preds)):

                print("****************************************************************************")

                print("Predicted -- {}".format(decoded_preds[j]))

                print("Utterance -- {}\n\n".format(decoded_targets[j]))

                test_cer.append(textprocessor.cer(decoded_targets[j], decoded_preds[j]))

                test_wer.append(textprocessor.wer(decoded_targets[j], decoded_preds[j]))

                utils.write_to_csv('/kaggle/working/'+model_name + '_testing_results.csv', decoded_preds[j], decoded_targets[j], test_loss)

    avg_cer = sum(test_cer)/len(test_cer)

    avg_wer = sum(test_wer)/len(test_wer)

    print('Test set:Average CER: {:4f} Average WER: {:.4f}\n'.format(avg_cer, avg_wer)) 



    
use_cuda = torch.cuda.is_available()

if use_cuda:

    torch.manual_seed(7)

device = torch.device("cuda" if use_cuda else "cpu")
#input_size = 64

n_classes = 29

hidden_dim = 256

n_layers =1

clip=2 # gradient clipping

# If MFCC = 13 and 128 if Specs

f_type = 'spec'

if f_type == 'spec':

    n_feats = 64

else:

    n_feats = 13

lr = 1e-4





epochs = 30







batch_size = 32

conv_n_layers = 1

gru_n_layers = 2



# Total samples of audio. At least 12000, otherwise it won't work

n_samples = 90000



criterion = nn.CTCLoss(blank=28, zero_infinity=True)


kwargs={'num_workers': 4, 'pin_memory': True} if use_cuda else {}

(train_dataset, val_dataset, test_dataset) = load_dataset(csv_file=CSV_FILE_PATH, root_dir=ROOT_FOLDERS, n_samples=n_samples, f_type=f_type)

(train_loader, valid_loader, test_loader) = create_data_loaders(train_dataset, val_dataset, test_dataset, kwargs=kwargs, batch_size=batch_size)

    
conv_bi_gru = models.ASRConvBiGRU(in_channel=1, gru_input_size=512, hidden_dim=hidden_dim, n_layers=n_layers,

                                  n_feats=n_feats, n_classes=n_classes, conv_n_layers=conv_n_layers,

                                  gru_n_layers=gru_n_layers, drop_prob=0.2, bidir=True)

#conv_bi_gru.apply(models.weight_init)

conv_bi_gru.to(device)
print(models.count_parameters(conv_bi_gru))


model_name = 'Conv-Bi-GRU'

(conv_bi_gru_trained, bi_train_losses, bi_val_losses) = train(n_epochs=epochs, train_loader=train_loader,

                                                     valid_loader=valid_loader,

                                                     model=conv_bi_gru, criterion=criterion, clip=clip,

                                                     device=device, lr=lr, batch_size=batch_size,

                                                     model_name=model_name,

                                                     save_model_path=SAVE_RESULTS_PATH+"conv_bigru",

                                                     save_pkl_path=SAVE_MODELS_PATH + "training_conv_bigru_iteration.pkl")



test_model(test_loader, conv_bi_gru_trained, "Conv-BI-GRU",device, batch_size=batch_size)


fig = plt.figure(figsize=(10,5))

ax = plt.subplot(111)

box = ax.get_position()



ax.plot(bi_train_losses, 'g',label='Conv-BI-GRU-train losses')

ax.plot(bi_val_losses, 'm',label='Conv-BI-GRU-valid losses')





plt.xlabel('epochs')

plt.ylabel('loss')



ax.set_position([box.x0, box.y0 + box.height * 0.1,

                 box.width, box.height * 0.9])

# Put a legend above current axis

ax.legend(loc='upper center',fontsize='small', bbox_to_anchor=(0.5, 1.09),

          fancybox=True, shadow=True, ncol=4) 

# Limits for the Y axis

plt.show()


#conv_gru = models.ASRConvBiGRU(in_channel=1, gru_input_size=512, hidden_dim=hidden_dim, n_layers=n_layers,

#                              n_feats=n_feats, n_classes=n_classes, conv_n_layers=conv_n_layers,

#                              gru_n_layers=gru_n_layers, drop_prob=0.2, bidir=False)

#conv_gru.apply(models.weight_init)

#conv_gru.to(device)