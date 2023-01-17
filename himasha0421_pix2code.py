# WandB – Install the W&B library

!pip install --upgrade wandb

import wandb
!wandb login b77985991c4b2c51ccc6af56fdfd2515dac6016a
# WandB – Initialize a new run

wandb.init(project="pix2code")
import requests



url = "https://www.floydhub.com/api/v1/resources/YrepYFsJ8om77SivtHpVHH?content=true&download=true&rename=floydhub-datasets-pix2code-1"

target_path = 'xlrd-0.9.4.tar.gz'



response = requests.get(url, stream=True)

if response.status_code == 200:

    with open(target_path, 'wb') as f:

        f.write(response.raw.read())
!mkdir data

import gc

del response ; gc.collect()
import tarfile

my_tar = tarfile.open('xlrd-0.9.4.tar.gz')

my_tar.extractall('data') # specify which folder to extract to

my_tar.close()
import os

import cv2

import torch

import torch.nn.functional as F

import torch.nn as nn

from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from torch.utils.data import SubsetRandomSampler

from torchvision import transforms

import torch.optim as optim

from PIL import Image
del my_tar ; gc.collect()
image_files = []

html_files = []

Transforms =transforms.Compose([

                    transforms.ToTensor() ,

                    transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])

                ])



def load_doc(file_path):

    file = open(file_path , 'r')

    text = file.read()

    file.close()

    return text



def blue_evaluation(text_data , gui_data ):

    html_seq =[]

    gui_seq= []

    for html_file in text_data :

        html_seq.append(text_process(html_file))

        

    for gui_file in gui_data :

        img_tensor = Transforms(gui_file)

        gui_seq.append(img_tensor)

    

    gui_seq = torch.stack(gui_seq)

        

    return html_seq , gui_seq

    



def text_process(file):

    """

    text = load_doc(file_path)

    text = '<START> ' + text + ' <END>'

    syntax = " ".join(text.split())

    #replace the commas with ' ,' syntax

    syntax = syntax.replace(',', ' ,')

    """

    return file.split()



def image_process(img_path):

    img_bgr = cv2.imread(img_path)

    img_rgb = cv2.cvtColor(img_bgr , cv2.COLOR_BGR2RGB)

    return img_rgb
#data paths

DS_TRAIN_PATH = "data/train/"

DS_VAL_PATH = "data/eval/"
import matplotlib.pyplot as plt

import os

import numpy as np



def show_img(im, figsize=None, ax=None):

    if not ax: fig,ax = plt.subplots(figsize=figsize)

    ax.imshow(im)

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    return ax



# Read a file and return a string

def load_doc(filename):

    file = open(filename, 'r')

    text = file.read()

    file.close()

    return text



def load_data(data_dir):

    text = []

    images = []

    # Load all the files and order them

    all_filenames = os.listdir(data_dir)

    all_filenames.sort()

    for filename in (all_filenames):

        if filename[-3:] == "npz":

            # Load the images already prepared in arrays

            image = np.load(data_dir+filename)

            images.append(image['features'])

        else:

            # Load the boostrap tokens and rap them in a start and end tag

            syntax = '<START> ' + load_doc(data_dir+filename) + ' <END>'

            # Seperate all the words with a single space

            syntax = ' '.join(syntax.split())

            # Add a space after each comma

            syntax = syntax.replace(',', ' ,')

            text.append(syntax)

    images = np.array(images, dtype=float)

    return images, text



# Get images and text

train_features, train_texts = load_data(DS_TRAIN_PATH)

valid_features , valid_texts = load_data(DS_VAL_PATH)
import random

import numpy as np

idx_list = list(np.arange(len(train_features)))

rand_idx  = np.random.choice(idx_list)

blue_gui_list = train_features[rand_idx:rand_idx+2]

blue_html_list = train_texts[rand_idx:rand_idx+2]
blue_html_seq , blue_gui_seq = blue_evaluation(blue_html_list , blue_gui_list)
#define the vocabulary

words=[]

word_count=[]

for html_path in train_texts :

    syntax = text_process(html_path)

    sen_len = len(syntax)

    words.extend(syntax)

    word_count.append(sen_len)
#define the couunter object 

from collections import Counter

from collections import OrderedDict

word_counter = Counter(words)

word_count_sorted = sorted(word_counter.items() , key=lambda x : x[1] , reverse=True)

ordered_dict = OrderedDict(word_count_sorted)

ordered_dict["<PAD>"]=1

ordered_dict["<UNK>"]=1
from torchtext.vocab import Vocab



html_vocab = Vocab(ordered_dict, min_freq=0 , specials=("<PAD>" ,"<UNK>") , specials_first=True)

vocab = html_vocab.itos
word_2_idx = {word : idx  for idx , word in enumerate(vocab)}

idx_2_word = {idx : word  for idx , word in enumerate(vocab)}
import numpy as np

max_seq_len = max(word_count)

def pad_seqence(sequence , max_len):

    seq=[]

    for word in sequence :

        idx = word_2_idx[word]

        seq.append(idx)

    

    pad_seq = np.zeros(max_len , dtype=np.uint8)

    pad_seq[-len(seq):] = seq

    

    return pad_seq
#main data reader

max_len = max(word_count)

MAX_LEN = 48

def token_target(text):

    outter_token =[]

    outter_target =[]

    for html_pth in (text):

        """

        1 step each html sentence create the token list

        2 step each toekn create the input sequence with token list and target syntax

        3 step for each token and target syntax take the approriate image file name



        """

        inner_token =[]

        inner_target=[]

        syntax = text_process(html_pth)

        for i_step in range(1, len(syntax)):

            in_seq  , out_seq = syntax[:i_step] , syntax[i_step]

            pad_in_seq = pad_seqence(in_seq , max_len)

            out_seq = word_2_idx[out_seq]

            pad_in_seq = pad_in_seq[-(MAX_LEN):]

            inner_token.append(pad_in_seq)

            inner_target.append(out_seq)

            

        outter_token.append(inner_token)

        outter_target.append(inner_target)

        

    return outter_token , outter_target

        

train_token_set , train_target_set = token_target(train_texts)

eval_token_set , eval_target_set  = token_target(valid_texts)
train_token_set = np.array(train_token_set)

train_target_set = np.array(train_target_set)

eval_token_set = np.array(eval_token_set)

eval_target_set = np.array(eval_target_set)
from torch.utils.data import TensorDataset

def my_collate(batch):

    

    gui_set = batch[0]['gui']

    token_set = batch[0]['token']

    target_set = batch[0]['target']



    html_token_data=   [ data.clone() for data in    token_set  ]

    html_target_data = [ torch.tensor(data)  for data in   target_set ]

    seq_len = len(html_target_data)

    gui_rgb = [ gui_set.clone() ]

    

    batch_32 = int(np.ceil(seq_len/32))

    

    #gui_rgb = [ torch.stack(gui_rgb[i:i+16])  for i in range(batch_32) ]

    html_token_data = [ torch.stack(html_token_data[i:i+32]) for i in range(batch_32) ]

    html_target_data = [ torch.stack(html_target_data[i:i+32]) for i in range(batch_32) ]   

    

    gui_rgb =torch.stack(gui_rgb)

    #html_token_data =torch.stack(html_token_data)

    #html_target_data=torch.stack(html_target_data)

    

    

    

    sample ={

        'gui':gui_rgb ,

        'token': html_token_data ,

        'target': html_target_data

        }

 



    return sample
class Pix2CodeData(Dataset):

    def __init__(self , html_token_set , html_target_set , gui_path_set):

        self.html_token_set = html_token_set

        self.html_target_set = html_target_set

        self.gui_path_set = gui_path_set

        self.transforms =transforms.Compose([

                            transforms.ToTensor() ,

                            transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])

                            ])

        

    def __len__(self):

        return len(self.html_target_set)

    def __getitem__(self , index):

        """

        1 . take the html token data

        2 . take the html target data

        3 . take the gui path and take the rgb image

        """

        html_token_data = self.html_token_set[index]

        html_target_data = self.html_target_set[index]

        gui_rgb = self.gui_path_set[index]

        

        #transform the image

        gui_rgb = self.transforms(gui_rgb)

        html_token_data = torch.LongTensor(html_token_data)

        

        sample ={

            'gui':gui_rgb ,

            'token': html_token_data ,

            'target': html_target_data

        }

        

        return sample
#define the data loaders



train_data_set = Pix2CodeData(train_token_set , train_target_set  , train_features)

eval_data_set = Pix2CodeData(eval_token_set , eval_target_set , valid_features)



train_loader = DataLoader(train_data_set , batch_size=1 , shuffle=True , collate_fn=my_collate)

valid_loader = DataLoader(eval_data_set , batch_size=1 , shuffle=True  , collate_fn=my_collate)
del train_features ; gc.collect()

del valid_features ; gc.collect()

del train_texts ; gc.collect()

del valid_texts ; gc.collect()
data = next(iter(train_loader))
print("Sample GUI tensor shape : ", data['gui'][0].shape)

print("Sample Token tensor shape : ", data['token'][0].shape)

print("Sample Target tensor shape : ",data['target'][0].shape)
import matplotlib.pyplot as plt

mean = [0.485, 0.456, 0.406] 

std = [0.229, 0.224, 0.225]

def viz_samples(gui):

    """

    1 step convert to numpy array

    2 step convert the channels

    3 renormalize the gui

    """

    gui = gui.numpy()

    gui = np.transpose(gui , (1,2,0))

    gui = gui*std + mean

    plt.imshow(gui)

    plt.title("GUI interface")

    plt.show()
gui_data = data['gui'][0]

viz_samples(gui_data)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
del data ; gc.collect()
class Pix2Code(nn.Module):

    def __init__(self , in_channels , out_dim , dropout ,hidden_token , hidden_decoder , lstm_layers , MAX_LEN):

        super(Pix2Code , self).__init__()

        self.in_channels = in_channels

        self.out_dim = out_dim

        self.dropout = dropout

        self.hidden_token = hidden_token

        self.hidden_decoder = hidden_decoder

        self.lstm_layers= lstm_layers

        self.MAX_LEN = MAX_LEN

        

        self.conv_32 = nn.Conv2d(in_channels=in_channels , out_channels=32 , 

                                 kernel_size=3 , stride=1 , padding=1)

        self.bn_32 = nn.BatchNorm2d(32)

        self.conv_32_down = nn.Conv2d(32 , 32 , kernel_size=3 , stride=2 , padding=1)

        self.bn_32_down = nn.BatchNorm2d(32)

        

        self.conv_64 = nn.Conv2d(32 , 64 , kernel_size=3 , stride=1 , padding=1)

        self.bn_64 = nn.BatchNorm2d(64)

        self.conv_64_down = nn.Conv2d(64 , 64 , kernel_size=3 , stride=2 , padding=1)

        self.bn_64_down = nn.BatchNorm2d(64)

        

        self.conv_128 = nn.Conv2d(64 , 128 , kernel_size=3 , stride=1 , padding=1)

        self.bn_128 = nn.BatchNorm2d(128)

        self.dense_size = 128 * 64 * 64

        

        self.fc_gui1 = nn.Linear(self.dense_size , 1024)

        self.fc_gui2 = nn.Linear(1024 , 1024)

        

        #****************************** Token Encoder *********************************************************

        self.token_embed = nn.Embedding(len(vocab) , 100)

        self.token_lstm = nn.LSTM(100 , self.hidden_token , num_layers=self.lstm_layers , 

                                  dropout = self.dropout , batch_first = True)

        

        #**************************** Decoder ******************************************************************

        self.decoder_in = 1024 + self.hidden_token

        self.decoder_lstm = nn.LSTM(self.decoder_in , self.hidden_decoder , num_layers=self.lstm_layers ,

                                    dropout= self.dropout , batch_first=True)

        

        self.fc_decoder = nn.Linear(self.hidden_decoder , len(vocab))

        self.softmax = nn.Softmax(dim=-1)

        

        

    def forward(self , gui , token ):

        

        hidden_token = self.init_token_hidden(token.shape[0])

        hidden_decoder = self.init_decoder_hidden(token.shape[0])

        

        # gui feature extraction 

        gui_x = F.relu(self.bn_32(self.conv_32(gui)))

        gui_x = F.relu(self.bn_32_down(self.conv_32_down(gui_x)))

        gui_x = F.relu(self.bn_64(self.conv_64(gui_x)))

        gui_x = F.relu(self.bn_64_down(self.conv_64_down(gui_x)))

        gui_x = F.relu(self.bn_128(self.conv_128(gui_x)))

        

        gui_x = gui_x.view(-1 , self.dense_size)

        gui_x = F.relu(self.fc_gui1(gui_x))

        gui_x = F.dropout(gui_x ,p=0.4)

        gui_x = F.relu(self.fc_gui2(gui_x))

        gui_x = gui_x.view(1 , 1024)

        

        gui_x = gui_x.repeat(token.shape[0] , 1 )

        

        gui_x = F.dropout(gui_x , p=0.4)

        

        #repeat the vector

        gui_x = gui_x.unsqueeze(1)

        gui_repeat = gui_x.repeat(1 , self.MAX_LEN , 1)      # (batch size , 1024) --> (batch size , MAX LEN , 1024)

        

        # encoder of the token features using embeddings and lstms



        token_embed = self.token_embed(token.data)

        token_lstm , _ = self.token_lstm(token_embed , hidden_token)



        decoder_in = torch.cat([token_lstm , gui_repeat] , dim=-1)

     

        # decoder output prediction

        decoder_lstm, _ = self.decoder_lstm(decoder_in , hidden_decoder)

        decoder_lstm = decoder_lstm[:,-1,:]

        decoder_fc  = decoder_lstm.contiguous().view(-1,self.hidden_decoder)

       

        decoder_fc = F.dropout(decoder_fc , p=0.5)

        decoder_out = self.fc_decoder(decoder_fc)

        

        return decoder_out

        

        

        

    def init_token_hidden(self , batch_size):

        weight = next(self.parameters()).data

        

        hidden =(weight.new(self.lstm_layers , batch_size , self.hidden_token).zero_().to(device) , 

                 weight.new(self.lstm_layers , batch_size , self.hidden_token).zero_().to(device) )

        

        return hidden

    

    def init_decoder_hidden(self,batch_size):

        weight = next(self.parameters()).data

        

        hidden = ( weight.new(self.lstm_layers , batch_size , self.hidden_decoder).zero_().to(device) , 

                   weight.new(self.lstm_layers , batch_size , self.hidden_decoder).zero_().to(device)  )

        

        return hidden
def word_into_idx(text):

    #split the text

    words = text.split()

        

    return words



# generate a description for an image

def generate_desc(model, photo, max_length):

    photo = photo.unsqueeze(0).float().to(device)

    # seed the generation process

    in_text = '<START> '

    # iterate over the whole length of the sequence

    print('\nPrediction---->\n\n<START> ', end='')

    for i in range(150):

        # integer encode input sequence

        sequence = word_into_idx(in_text)

        if(len(sequence)>=max_length):

            sequence = sequence[-(max_length):]

        #pad sequence

        sequence = pad_seqence(sequence , max_length)

        seq_tensor = torch.tensor(sequence).unsqueeze(0).to(device)

        seq_tensor = seq_tensor.long()

       

        # predict next word

        with torch.no_grad():

            yhat = model.forward(photo, seq_tensor)

        yhat_softmax = F.softmax(yhat , dim=-1)

        yhat_max = torch.argmax(yhat_softmax)



        # map integer to word

        word = idx_2_word[yhat_max.item()]

        # stop if we cannot map the word

        if word is None:

            break

        # append as input for generating the next word

        in_text += word + ' '

        # stop if we predict the end of the sequence

        print(word + ' ', end='')

        if word == '<END>':

            break

    return in_text
from nltk.translate.bleu_score import corpus_bleu

from nltk.translate.bleu_score import SmoothingFunction

smoothie = SmoothingFunction().method4

# Evaluate the skill of the model

def evaluate_model(model, texts, photos, max_length):

    actual, predicted = list(), list()

    # step over the whole set

    for i in range(len(texts)):

        viz_samples(photos[i])

        yhat = generate_desc(model , photos[i], max_length)

        # store actual and predicted

        text = " ".join(texts[i])

        print('\n\nReal---->\n\n' + text)

        actual.append(text)

        predicted.append(yhat.split())

    # calculate BLEU score

    bleu = corpus_bleu(actual, predicted , smoothing_function=smoothie )

    return bleu, actual, predicted
#keep track of the configurations

config = wandb.config  

config.in_channels =3 

config.out_dim =len(vocab) 

config.dropout = 0.3 

config.hidden_token = 128

config.hidden_decoder =512

config.lstm_layers =2

config.MAX_LEN = 48

config.EPOCHS=20

config.grad_clip = 3.0

config.learning_rate = 0.0001



pix2code =  Pix2Code(in_channels= config.in_channels , out_dim=config.out_dim , 

                    dropout=config.dropout , hidden_token=config.hidden_token , 

                    hidden_decoder=config.hidden_decoder , lstm_layers=config.lstm_layers , MAX_LEN=config.MAX_LEN)



pix2code.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.RMSprop(pix2code.parameters() , lr=config.learning_rate)
from collections import deque

import gc



mean_train_loss = deque(maxlen=100)

mean_val_loss = deque(maxlen=100)

total_train_loss = []

total_val_loss = []

print_every = 250

blue_score =0

#define the training loop and wandb logging

#wandb.watch(pix2code, log="all")

step = 0

for i_epoch in range(config.EPOCHS) :

    pix2code.train()

    epoch_train_loss = 0

    epoch_val_loss = 0

    

    

    for idx , data in enumerate(train_loader) :

        

        gui = data['gui']

        token = data['token']

        target = data['target']

        

        for i_batch in range(len(token)):

        

            gui_i = gui.float().to(device)

            token_i = token[i_batch].to(device)

            target_i = target[i_batch].to(device)



            target_pred = pix2code(gui_i , token_i)

            loss = criterion(target_pred , target_i)



            #reset the optimizer

            optimizer.zero_grad()



            #backprop the loss

            loss.backward()



            #clip the exploiding gradinets

            torch.nn.utils.clip_grad_norm_(pix2code.parameters(), config.grad_clip)



            #optimize the model

            optimizer.step()



            epoch_train_loss += loss.to('cpu').detach().item()



            mean_train_loss.append(loss.to('cpu').detach().item())

            total_train_loss.append(loss.to('cpu').detach().item())

        



        if((idx+1)%print_every==0):

            step += 1

            print("Epoch {} Step {} train loss {:.6f}".format(i_epoch ,step , np.mean(mean_train_loss)))

            wandb.log({

                "Steps":step ,

                "Train Loss":np.mean(mean_train_loss)})



        

    pix2code.eval()

    with torch.no_grad():

        for idx , data in enumerate(valid_loader) :

    

            gui = data['gui']

            token = data['token']

            target = data['target']

            

            for i_batch in range(len(token)):

    

                gui_i = gui.float().to(device)

                token_i = token[i_batch].to(device)

                target_i = target[i_batch].to(device)



                target_pred = pix2code(gui_i , token_i)

                loss = criterion(target_pred , target_i)



                epoch_val_loss += loss.to('cpu').detach().item()



                mean_val_loss.append(loss.to('cpu').detach().item())

                total_val_loss.append(loss.to('cpu').detach().item())

                

         

        print("Epoch {} Val loss {:.6f}".format(i_epoch , np.mean(mean_val_loss)))

        wandb.log({

            "Epoch":i_epoch ,

            "Val Loss":np.mean(mean_train_loss)})

     

    #do some model evaluation based on the belu scores

    # Eval on the first 10 samples

    

    if((i_epoch +1 )%8==0):

        bleu, actual, predicted = evaluate_model( pix2code , blue_html_seq, blue_gui_seq, MAX_LEN)

        print("BLUE score: ", bleu)

        if (bleu > blue_score):

            # Save model to wandb

            #torch.save(pix2code.state_dict(), os.path.join(wandb.run.dir, 'pix2code.pt'))

            blue_score = bleu

    

    wandb.log({

        "Train Loss":np.mean(mean_train_loss) ,

        "Test Loss": np.mean(mean_val_loss) ,

        "Blue Score": blue_score})