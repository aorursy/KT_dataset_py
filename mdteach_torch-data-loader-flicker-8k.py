#location of the data 

data_location =  "../input/flickr8k"

!ls $data_location
#reading the text data 

import pandas as pd

caption_file = data_location + '/captions.txt'

df = pd.read_csv(caption_file)

print("There are {} image to captions".format(len(df)))

df.head(7)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



#select any index from the whole dataset 

#single image has 5 captions

#so, select indx as: 1,6,11,16...

data_idx = 11



#eg path to be plot: ../input/flickr8k/Images/1000268201_693b08cb0e.jpg

image_path = data_location+"/Images/"+df.iloc[data_idx,0]

img=mpimg.imread(image_path)

plt.imshow(img)

plt.show()



#image consits of 5 captions,

#showing all 5 captions of the image of the given idx 

for i in range(data_idx,data_idx+5):

    print("Caption:",df.iloc[i,1])

#imports 

import os

from collections import Counter

import spacy

import torch

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader,Dataset

import torchvision.transforms as T



from PIL import Image
#using spacy for the better text tokenization 

spacy_eng = spacy.load("en")



#example

text = "This is a good place to find a city"

[token.text.lower() for token in spacy_eng.tokenizer(text)]
class Vocabulary:

    def __init__(self,freq_threshold):

        #setting the pre-reserved tokens int to string tokens

        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}

        

        #string to int tokens

        #its reverse dict self.itos

        self.stoi = {v:k for k,v in self.itos.items()}

        

        self.freq_threshold = freq_threshold

        

    def __len__(self): return len(self.itos)

    

    @staticmethod

    def tokenize(text):

        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    

    def build_vocab(self, sentence_list):

        frequencies = Counter()

        idx = 4

        

        for sentence in sentence_list:

            for word in self.tokenize(sentence):

                frequencies[word] += 1

                

                #add the word to the vocab if it reaches minum frequecy threshold

                if frequencies[word] == self.freq_threshold:

                    self.stoi[word] = idx

                    self.itos[idx] = word

                    idx += 1

    

    def numericalize(self,text):

        """ For each word in the text corresponding index token for that word form the vocab built as list """

        tokenized_text = self.tokenize(text)

        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]    
#testing the vicab class 

v = Vocabulary(freq_threshold=1)



v.build_vocab(["This is a good place to find a city"])

print(v.stoi)

print(v.numericalize("This is a good place to find a city here!!"))
class FlickrDataset(Dataset):

    """

    FlickrDataset

    """

    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):

        self.root_dir = root_dir

        self.df = pd.read_csv(caption_file)

        self.transform = transform

        

        #Get image and caption colum from the dataframe

        self.imgs = self.df["image"]

        self.captions = self.df["caption"]

        

        #Initialize vocabulary and build vocab

        self.vocab = Vocabulary(freq_threshold)

        self.vocab.build_vocab(self.captions.tolist())

        

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self,idx):

        caption = self.captions[idx]

        img_name = self.imgs[idx]

        img_location = os.path.join(self.root_dir,img_name)

        img = Image.open(img_location).convert("RGB")

        

        #apply the transfromation to the image

        if self.transform is not None:

            img = self.transform(img)

        

        #numericalize the caption text

        caption_vec = []

        caption_vec += [self.vocab.stoi["<SOS>"]]

        caption_vec += self.vocab.numericalize(caption)

        caption_vec += [self.vocab.stoi["<EOS>"]]

        

        return img, torch.tensor(caption_vec)
#defing the transform to be applied

transforms = T.Compose([

    T.Resize((224,224)),

    T.ToTensor()

])
def show_image(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated
#testing the dataset class

dataset =  FlickrDataset(

    root_dir = data_location+"/Images",

    captions_file = data_location+"/captions.txt",

    transform=transforms

)







img, caps = dataset[0]

show_image(img,"Image")

print("Token:",caps)

print("Sentence:")

print([dataset.vocab.itos[token] for token in caps.tolist()])
class CapsCollate:

    """

    Collate to apply the padding to the captions with dataloader

    """

    def __init__(self,pad_idx,batch_first=False):

        self.pad_idx = pad_idx

        self.batch_first = batch_first

    

    def __call__(self,batch):

        imgs = [item[0].unsqueeze(0) for item in batch]

        imgs = torch.cat(imgs,dim=0)

        

        targets = [item[1] for item in batch]

        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        return imgs,targets
#writing the dataloader

#setting the constants

BATCH_SIZE = 4

NUM_WORKER = 1



#token to represent the padding

pad_idx = dataset.vocab.stoi["<PAD>"]



data_loader = DataLoader(

    dataset=dataset,

    batch_size=BATCH_SIZE,

    num_workers=NUM_WORKER,

    shuffle=True,

    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)

)
#generating the iterator from the dataloader

dataiter = iter(data_loader)



#getting the next batch

batch = next(dataiter)



#unpacking the batch

images, captions = batch



#showing info of image in single batch

for i in range(BATCH_SIZE):

    img,cap = images[i],captions[i]

    caption_label = [dataset.vocab.itos[token] for token in cap.tolist()]

    eos_index = caption_label.index('<EOS>')

    caption_label = caption_label[1:eos_index]

    caption_label = ' '.join(caption_label)                      

    show_image(img,caption_label)

    plt.show()