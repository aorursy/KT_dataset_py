!pip install ../input/transformers/transformers-master/

!pip install keras_preprocessing
# My endeavor is to build a voice recognition model

# that takes into account a language peculiarities.



# For now model is not complete and even though the

# notebook runs with no errors, it does not work as

# intended. But I am developing it gradually.



# Thanks Kaggle for the Notebook to persue our

# crazy ideas :)



import pandas as pd

from scipy.io import wavfile

import os

import librosa

import matplotlib.pyplot as plt

import numpy as np

#from sklearn.metrics.pairwise import cosine_similarity

from keras_preprocessing.sequence import pad_sequences

from transformers import BertModel, BertConfig, BertTokenizer

from torch.utils.data import Dataset, DataLoader

from torch import Tensor, tanh, flatten, squeeze, mean, stft

import torch.nn as nn

import torch.nn.functional as F

from torch.optim import Adam

from torch.nn.utils.rnn import pad_sequence

from torchaudio import load

from torchaudio.transforms import MFCC, Resample

from torchaudio.functional import istft

import numpy as np
data_dir = '../input/common-voice'

audio_dir = 'cv-valid-train'

#/cv-valid-train'

sound_len = 10

bert_path = '../input/bert-base-uncased'

bert_vocab_path = 'vocab.txt'

bert_model_path = 'bert-base-uncased/pytorch_model.bin'

max_len_text = 30

window = 40

#n_mfcc=40
class VoiceInstance:

    

    def __init__(self, file, tokenizer, sound_len=10**10, text=None):

        

        self.file = file

        self.text = text

        self.tokenizer = tokenizer

        self._transform_audio()

        self.emb = self._get_embeddings()

        

    @staticmethod

    def pad_seq(sequence):

        return Tensor(np.pad(sequence, 0))

    

    def _transform_audio(self):

        

        waveform, rate = load(os.path.join(data_dir, self.file))

        new_rate = rate/100

        resampled = Resample(rate, new_rate)(waveform)

        self.stft = self._get_stft(resampled, new_rate)

        self.mfcc = self._get_mfcc(resampled, new_rate)

        

    def _get_mfcc(self, arr, sample_rate=22000):

        

        mfcc_tensor = MFCC(sample_rate, n_mfcc=window)

        return mfcc_tensor.forward(arr)

    

    def _get_stft(self, waveform, rate):

        return stft(waveform, int(rate))

    

    def _get_embeddings(self):

        #configuration = BertConfig()

        

        def remove_punkts(text):

            

            punkts = ['.', '?', '!', '\'', '"', '’', '‘', ',', ';']

            for punkt in punkts:

                text = str(self.text).replace(punkt, '')

            return text

        

        def tokenize_text(text):

            return self.tokenizer.tokenize(text)

        

        def get_inputs(tokens):

            return self.tokenizer.convert_tokens_to_ids(tokens)

        

        if self.text:

            self.text = remove_punkts(self.text)

            self.tokens = tokenize_text(self.text)

            emb = get_inputs(self.tokens)

            return emb
class VoiceDataset(Dataset):

    

    def __init__(self, info_frame, audio_dir, tokenizer):

        

        self.info_frame = info_frame

        self.audio_dir = audio_dir

        self.tokenizer = tokenizer

        self.load_data()

    

    def __len__(self):

        #gen = (x for x in self.instances)

        return len(self.instances)

    

    def __getitem__(self, item):

        return item

    

    def load_data(self):

        

        self.instances = []

        for i in self.info_frame.index:

            if self.info_frame.loc[i, 'text'] is not None:

                audio = VoiceInstance(file=os.path.join(self.audio_dir, self.info_frame.loc[i, 'filename']),

                                      text=self.info_frame.loc[i, 'text'], tokenizer=self.tokenizer)

                self.instances.append(audio)

        embs = []

        mfccs = []

        for each in self.instances:

            embs.append(each.emb)

            mfccs.append(each.mfcc)

        embs = pad_sequences(embs, maxlen=max_len_text, padding='pre', value=0)

        #mfccs = pad_sequences(mfccs, maxlen=max_len_text, padding='pre', value=0)

        for i in range(len(self.instances)):

            self.instances[i].emb = embs[i]

            #self.instances[i].mfcc = mfccs[i]

            #print(self.instances[i].emb)

        # print('max length:', np.max(np.array(self.instances)))
class VoiceModel(nn.Module):

    

    def __init__(self):

        

        super().__init__()

        self.lstm_enc = nn.LSTM(input_size=window, hidden_size=512, batch_first=True)

        self.lstm_dec = nn.LSTM(input_size=512, hidden_size=256)

        #self.padding = nn.ConstantPad1d()

        

        

        

        self.input = nn.GRU(input_size=window, bidirectional=True, hidden_size=512)

        self.gru = nn.GRU(input_size=512*2, bidirectional=False, hidden_size=512)

        self.flatten = nn.Flatten()

        #self.pool = nn.MaxPool1d(10)

        self.dense = nn.Linear(256, 128)

        self.dropout = nn.Dropout(0.3)

        self.linear_2 = nn.Linear(128, max_len_text)

        self.output = nn.Linear(max_len_text*100, 1)

    

    def forward(self, audio):

        #print(audio.shape)

        audio = Tensor(audio)

        #print(audio)

        #audio = nn.functional.pad(audio, (1, 0), mode='constant', value=0)

        #print(audio.shape)

        audio, hidden_enc = self.lstm_enc(audio.reshape(1, -1, window))

        audio = tanh(audio)

        audio, hidden_dec = self.lstm_dec(audio)

        audio = tanh(audio)

        audio = F.relu(self.dense(audio))

        audio = self.flatten(audio)

        # that's to see if it works, later I will figure an appropriate value

        max_sound_len = 3000

        audio = nn.functional.pad(audio, (max_sound_len-audio.shape[-1], 0), mode='constant', value=0)

        audio = self.output(audio)

        print(audio.shape)

        

        return audio
def main():

    

    #info_frame = pd.read_csv('../input/common-voice/cv-valid-train.csv')

    

    batch_size = 16

    epochs = 10

    model = VoiceModel()

    criterion = nn.CosineSimilarity(dim=0)

    #criterion.requres_grad = True

    optimizer = Adam(model.parameters(), lr=0.001)

    info_frame = pd.read_csv(os.path.join(data_dir, 'cv-valid-train.csv'))[:50]	# .loc[25:75, :]

    

    tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_path, bert_vocab_path), return_tensors='pt')

    #model = BertModel.from_pretrained(os.path.join(bert_path, bert_model_path))

    

    real_texts = []

    lengths = []

    for i in info_frame.index:

        tokens = tokenizer.tokenize(info_frame.loc[i, 'text'])

        real_texts.append(tokenizer.convert_tokens_to_ids(tokens))

        lengths.append(len(real_texts[i]))

        

    print(info_frame.columns)

    #print(lengths)

    

    

    dataset = VoiceDataset(info_frame=info_frame, audio_dir=audio_dir, tokenizer=tokenizer)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #print(dataset.instances[0].fft)

    

    for epoch in range(epochs):

        running_loss = 0

        i=0

        for batch in dataset.instances:

            

            optimizer.zero_grad()

            #print(batch.fft) # torch.Size([1, 40, 10])

            #print(batch.stft.shape)

            output = model(batch.mfcc)

            #print(output, batch.emb)

            #loss = criterion(batch.emb, real_texts[i])

            #print(batch.emb) #, real_texts[i])

            #print('models output shape:', output.shape, '\nlabels shape:', Tensor(batch.emb).shape)

            #loss = criterion(output, Tensor(batch.emb))

            

            #loss.mean().backward()

            #optimizer.step()

            #running_loss += loss.item()

            #print(running_loss)

            i += 1

        # print(loss.mean())

        

    model.eval()

    

    test_frame = pd.read_csv(os.path.join(data_dir, 'cv-valid-train.csv'))[50:75]

    test_frame.reset_index()

    print(test_frame.index)

    

    #real_embs = []

    for text, file in zip(test_frame['text'], test_frame['filename']):

        tokens = tokenizer.tokenize(text)

        real_emb = tokenizer.convert_tokens_to_ids(tokens)

        #lengths.append(len(real_texts[i]))

        print('real embedding:', real_emb)

        test_instance = VoiceInstance(file='cv-valid-train/'+file, 

                                      tokenizer=tokenizer)

        pred = model(test_instance.mfcc)

        print('predicted:', pred)

            

if __name__ == '__main__':

    main()