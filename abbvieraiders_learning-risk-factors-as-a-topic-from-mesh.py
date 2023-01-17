! pip install pyphen

import pandas
import math
import torch
import json
import re
import pyphen
import string
from IPython.display import clear_output
from numpy import mean
from glob import glob
from torch import nn
from torch import optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
class RegexTokenizer:
    def __init__(self, regex=re.compile(r'\w+|\. |\.$|\b\?|\b\!'), lower=False):
        self.regex = regex
        self.lower = lower

    def tokenize(self, text):
        text = text.lower() if self.lower else text
        return self.regex.findall(text)


class SyllableTokenizer:
    def __init__(self):
        self.dic = pyphen.Pyphen(lang='en')

    def tokenize(self, word):
        return ["#" + s for s in self.dic.inserted(word).split("-")]


class CharacterTokenizer:
    def __init__(self):
        pass

    def tokenize(self, word):
        return ['#' + char for char in list(word.replace('#', ''))]


class FirstPassTokenizer:
    def __init__(self, frequency_dict, min_freq, sentence=RegexTokenizer(), word=SyllableTokenizer()):
        self.sentence = sentence
        self.word = word
        self.frequency_dict = frequency_dict
        self.min_freq = min_freq

    def tokenize(self, text):
        zeroth_pass = self.sentence.tokenize(text)
        tokens = []
        for potential_token in zeroth_pass:
            try:
                assert self.frequency_dict[potential_token] >= self.min_freq
                tokens.append(potential_token)
            except (AssertionError, KeyError):
                tokens += self.word.tokenize(potential_token)
        return tokens


class TextTokenizer:
    def __init__(self, frequency_dict,  min_freq, first_pass, character=CharacterTokenizer()):
        self.frequency_dict = frequency_dict
        self.min_freq = min_freq
        self.character = character
        self.first_pass = first_pass
        self.vocabulary = [token for token in frequency_dict.keys() if frequency_dict[token] >= min_freq]

    def tokenize(self, text):
        first_pass = self.first_pass.tokenize(text)
        tokens = []
        for potential_token in first_pass:
            try:
                assert potential_token in self.vocabulary
                tokens.append(potential_token)
            except (AssertionError, KeyError):
                tokens += self.character.tokenize(potential_token)
        return tokens

class IntegerCoder:
    def __init__(self, tokenizer, pad_string="[[PAD]]", cls_string="[[CLS]]"):
        self.tokenizer = tokenizer
        self.pad_string = pad_string
        self.cls_string = cls_string
        self.vocabulary = tokenizer.vocabulary + ['#{}'.format(char) for char in string.printable]
        self.vocabulary.sort()
        self.decode_dict = dict(enumerate(self.vocabulary))
        if pad_string is not None:
            self.decode_dict[max(self.decode_dict.keys()) + 1] = pad_string
        if cls_string is not None:
            self.decode_dict[max(self.decode_dict.keys()) + 1] = cls_string
        self.encode_dict = {v: k for k, v in self.decode_dict.items()}
        self.vocab_size = len(list(self.encode_dict.keys()))

    def encode_word(self, word):
        try:
            return self.encode_dict[word]
        except KeyError:
            return self.vocab_size

    def encode(self, obj):
        if type(obj) == list:
            return [self.encode(item) for item in obj]
        elif type(obj) == str:
            return self.encode_word(obj)
        else:
            raise TypeError(
                "Argument to self.encode must be a string, list of strings, or list of lists of strings, etc.")

    def decode_integer(self, integer):
        return self.decode_dict[integer]

    def decode(self, obj):
        if type(obj) == list:
            return [self.decode(item) for item in obj]
        elif type(obj) == int:
            return self.decode_integer(obj)
        else:
            raise TypeError("Argument to self.decode must be an int, list of ints, list of lists of ints, etc.")
            
with open('../input/frequency-dictionaries-for-biomedical-tokenizer/text_frequencies.json', 'r') as f:
    text_frequencies = json.load(f)

with open('../input/frequency-dictionaries-for-biomedical-tokenizer/first_pass_tokenizer_frequencies.json', 'r') as f:
    first_pass_frequencies = json.load(f)

first_pass_tokenizer = FirstPassTokenizer(text_frequencies, min_freq=10000)

tokenizer = TextTokenizer(first_pass_frequencies, min_freq=10000, first_pass=first_pass_tokenizer)
    
text_coder = IntegerCoder(tokenizer)
def sequentialize(text, seq_length):
    tokens = tokenizer.tokenize(text)
    if len(tokens) >= seq_length:
        return tokens[0:seq_length]
    else:
        return tokens + [text_coder.pad_string] * (seq_length - len(tokens))


def numericalize(text, seq_length=256):
    return torch.tensor(text_coder.encode(sequentialize(str(text), seq_length)))


print(tokenizer.tokenize("This is a test of the numericalization function."))
print(numericalize("This is a test of the numericalization function.", 15))
class MeshSamples(Dataset):
    def __init__(self, directory, subdirectory, X_col='abstract', y_col=None):
        self.directory = directory
        self.subdirectory = subdirectory
        self.X_col = X_col
        self.y_col = y_col if y_col else self.directory
        self.files = glob("../input/data-for-learning-from-mesh-risk-factors/{}/{}/*.csv".format(self.directory, self.subdirectory))
        self.dfs = [pandas.read_csv(f) for f in self.files]
        self.lengths = [len(df) for df in self.dfs]
        self.length = sum(self.lengths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        a, b = self.get_coordinates(index)
        df = self.dfs[a]
        X = numericalize(df[self.X_col][b])
        y = df[self.y_col][b]
        return X, y
        
    def get_coordinates(self, index):
        total = 0
        for i, length in enumerate(self.lengths):
            old_total = total
            total += length
            if index < total:
                return i, index - old_total
        return None
        
    
ms = MeshSamples('risk_factors', 'train')
dl = DataLoader(ms, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)
next(iter(dl))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, nhead, nhid, nlayers, dropout=0):
        super(TransformerClassifier, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size
        self.output = nn.Linear(embedding_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)
        encoded = self.transformer_encoder(src)
        cls = encoded.mean(1)
        output = self.sigmoid(self.output(cls)).flatten()
        return output

num_types = len(text_coder.vocabulary) + 2
tc = TransformerClassifier(num_types, 50, 1, 10, 1).to('cuda')
print(tc)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(tc.parameters(), lr=0.01)
losses = []
accuracies = []
num_batches = 500

for i, batch in enumerate(dl):
    if i < num_batches:
        X = batch[0].to('cuda')
        y = batch[1].float().to('cuda')
        y_pred = tc.forward(X)
        loss = criterion(y_pred,y)
        losses.append(loss.item())
        accuracy = (((y_pred >= 0.5) & (y >= 0.5)).sum() + ((y_pred < 0.5) & (y < 0.5)).sum()).item()
        accuracy = accuracy / y.shape[0]
        accuracies.append(accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clear_output()
        print("Batch {} loss {} accuracy {} cumacc {}".format(i + 1, loss.item(), accuracy, mean(accuracies)))
    else:
        break

torch.save(tc, 'tc.pt')
torch.save(tc.state_dict(), 'tc.pth')
test_ms = MeshSamples('risk_factors', 'test')
test_dl = DataLoader(test_ms, batch_size=dl.batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_accuracies = []

for i, batch in enumerate(test_dl):
    clear_output()
    print(i)
    X = batch[0].to('cuda')
    y = batch[1].float().to('cuda')
    y_pred = tc.forward(X)
    accuracy = (((y_pred >= 0.5) & (y >= 0.5)).sum() + ((y_pred < 0.5) & (y < 0.5)).sum()).item() / test_dl.batch_size
    test_accuracies.append(accuracy)

print(mean(test_accuracies))
cord = pandas.read_csv('../input/cord19-abstracts/cord.txt', sep='\t')

scores = []

for i, abstract in enumerate(cord['abstract']):
    clear_output()
    print(i)
    numericalized = numericalize(abstract)
    reshaped = numericalized.view(1, numericalized.shape[0]).to('cuda')
    scores.append(tc.forward(reshaped).item())
        
cord['risk_factors'] = scores

cord.to_csv('cord_enhanced.csv')
[title for title in cord.sort_values('risk_factors', ascending=False).head(1000)['title'] if 'COVID' in title]