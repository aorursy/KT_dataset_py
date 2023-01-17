import pickle

import string

import numpy as np

import torch

import torch.nn as nn

import torchvision.models as models

import torchvision.transforms as transforms

from torchtext.data.utils import get_tokenizer

from PIL import Image

from collections import Counter

from matplotlib import pyplot as plt

%matplotlib inline
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def get_activation(name, _dict):

    def hook(model, input, output):

        _dict[name] = output.detach()

    

    return hook
def get_transforms():

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    return transforms.Compose([transforms.ToTensor(), normalize])
def load_captions(path):

    """

    Input:

            - path: path to .txt file in which captions are stored

    Output:

            - list: a list of all captions

    """

    

    # A list to store all captions for image

    captions_list = []

    

    # Open the file containing captions

    with open(path, "r") as file:

        

        #Iterate through each line, append each caption to the list

        for line in file.readlines():

            

            # Extract caption by splitting the line into words

            words = line.strip("\n").split()

            

            # Join the words to form a caption, append to the list

            caption = ' '.join(words[1:])

            captions_list.append(caption)

    return captions_list
def preprocess_caption(caption):

    """

    Input:

            - caption: raw caption to pre-process

    Output:

            - list: list of tokens; tokenized caption

    """

    

    # Tokenizer - it will automatically convert words to lower case and tokenize them

    tokenizer = get_tokenizer("basic_english")

    

    # Removing punctuations from caption

    caption = "".join([char for char in caption if char not in string.punctuation])

        

    # Tokenizing caption

    tokenized_caption = tokenizer(caption)

    

    return tokenized_caption
def build_vocabulary(captions_list, min_freq = 5):

    """

    Input:

            - list: a list containing all captions

            - min_freq: minimum count of a word to be part of the vocabulary

    Output:

            - dict: word mapping / vocabulary

    """

    

    # Frequency counter

    word_freq = Counter()

    

    # Pre-process the caption and update the frequency counter

    for caption in captions_list:

        

        # Preprocessing caption

        tokenized_caption = preprocess_caption(caption)

        

        # Update freq count

        word_freq.update(tokenized_caption)

    

    # Create word mapping / vocabulary

    words = [w for w in word_freq.keys() if word_freq[w] >= min_freq]

    word_map = {k: v + 1 for v, k in enumerate(words)}

    word_map['<unk>'] = len(word_map) + 1

    word_map['<start>'] = len(word_map) + 1

    word_map['<end>'] = len(word_map) + 1

    word_map['<pad>'] = 0

        

    return word_map
tokens_path = "../input/flicker8k-image-captioning/Flickr8k_text/Flickr8k.token.txt"

captions_list = load_captions(path = tokens_path)



### Build Vocabulary from captions

word_map = build_vocabulary(captions_list = captions_list, min_freq = 5)
import pickle

pickle.dump(word_map, open('word_map_min5.pkl', 'wb'))
class NICModel(nn.Module):

    def __init__(self, vocab_size, emb_dim, hidden_units):

        super(NICModel, self).__init__()

        self.vocab_size = vocab_size

        self.emb_dim = emb_dim

        self.hidden_units = hidden_units

        self.dropout = nn.Dropout(p = 0.5)

        self.softmax = nn.Softmax(dim = -1)

        self.batchnorm = nn.BatchNorm1d(num_features = self.emb_dim)

        self.embedding = nn.Embedding(num_embeddings = self.vocab_size, embedding_dim = self.emb_dim)

        self.dense1 = nn.Linear(in_features = 2048, out_features = self.emb_dim, bias = False)

        self.dense2 = nn.Linear(in_features = self.hidden_units, out_features = self.vocab_size)

        self.lstm = nn.LSTM(input_size = self.emb_dim, hidden_size = self.hidden_units, batch_first = True)

    

    def forward(self, inputs, mode = "train"):

        input1 = inputs[0]

        input2 = inputs[1]

        x_img = self.dropout(input1)

        x_img = self.dense1(x_img)

        x_img = self.batchnorm(x_img)

        x_img = x_img.unsqueeze(dim = 1)

        x_text = self.embedding(input2)

        x_text = self.dropout(x_text)

        _, h = self.lstm(x_img)

        A, _ = self.lstm(x_text, h) 

        outputs = self.dense2(A)

        outputs = outputs.view(-1, self.vocab_size)

        return outputs
class NIC_Encoder(NICModel):

    

    def __init__(self, vocab_size, emb_dim, hidden_units):

        super(NIC_Encoder, self).__init__(vocab_size, emb_dim, hidden_units)

    def forward(self, x_img):

        x_img = self.dense1(x_img)

        x_img = self.batchnorm(x_img)

        x_img = x_img.unsqueeze(dim = 1)

        

        # Encode Image

        A, h = self.lstm(x_img)

        

        return A, h
class NIC_Decoder(NICModel):

    

    def __init__(self, vocab_size, emb_dim, hidden_units):

        super(NIC_Decoder, self).__init__(vocab_size, emb_dim, hidden_units)

    

    def forward(self, x_text, h):

        x_text = self.embedding(x_text)

        A, h = self.lstm(x_text, h)

        A = self.dense2(A)

        A = self.softmax(A)

        return A, h
def inference_greedy(cnn_model, nic_encoder, nic_decoder, img_path, id_to_word, transforms, max_length = 40):

    activation = {}

    cnn_model.Mixed_7c.register_forward_hook(get_activation(name = "Mixed_7c", _dict = activation))

    pool = nn.AdaptiveAvgPool2d((1, 1))

    img = Image.open(img_path)

    img = img.resize((299, 299))

    img_tensor = transforms(img)

    img_tensor = torch.unsqueeze(img_tensor, dim = 0)

    img_tensor = img_tensor.to(device)

    output = cnn_model(img_tensor)

    features = activation["Mixed_7c"]

    features = pool(features)

    features = features.view(1, 2048)

    _, h = nic_encoder(features)

    start = 2993

    x_text = torch.tensor(start, device = device).view(1,1)

    output = torch.zeros((1, max_length), device = device)

    for i in range(max_length):

        A, h = nic_decoder(x_text, h)

        A = torch.argmax(A, dim = 2)

        if id_to_word[A.item()] == "<end>":

            break

        output[:, i] = A.item()

        x_text = A

    output = [id_to_word[key] for key in output.cpu().numpy()[0] if id_to_word[key] != "<pad>"]

    output = " ".join(output)    

    return output
inception_v3 = models.inception_v3(pretrained = True, progress = True)
nic_encoder = NIC_Encoder(vocab_size = len(word_map), emb_dim = 512, hidden_units = 512)

nic_encoder.load_state_dict(torch.load("../input/flicker8k-image-captioning/nic_weights/nic_500_epochs_0.01_lr_no_l2.pth", 

                                                map_location = device))

nic_decoder = NIC_Decoder(vocab_size = len(word_map), emb_dim = 512, hidden_units = 512)

nic_decoder.load_state_dict(torch.load("../input/flicker8k-image-captioning/nic_weights/nic_500_epochs_0.01_lr_no_l2.pth", 

                                                map_location = device))

inception_v3.eval()

inception_v3.to(device)

nic_encoder.eval()

nic_encoder.to(device)

nic_decoder.eval()

nic_decoder.to(device)

id_to_word = {v:k for k,v in word_map.items()}
img_path = "../input/flicker8k-image-captioning/Flickr8k_Dataset/Flicker8k_Dataset/1000268201_693b08cb0e.jpg"

img = Image.open(img_path)

caption = inference_greedy(cnn_model = inception_v3, nic_encoder = nic_encoder, nic_decoder = nic_decoder,

                           img_path = img_path, id_to_word = id_to_word, transforms = get_transforms(), max_length = 40)



plt.imshow(img)



print(caption)