import tensorflow as tf



# Get the GPU device name.

device_name = tf.test.gpu_device_name()



# The device name should look like the following:

if device_name == '/device:GPU:0':

    print('Found GPU at: {}'.format(device_name))

else:

    raise SystemError('GPU device not found')
import torch



# If there's a GPU available...

if torch.cuda.is_available():    



    # Tell PyTorch to use the GPU.    

    device = torch.device("cuda")



    print('There are %d GPU(s) available.' % torch.cuda.device_count())



    print('We will use the GPU:', torch.cuda.get_device_name(0))



# If not...

else:

    print('No GPU available, using the CPU instead.')

    device = torch.device("cpu")
!pip install transformers
import numpy as np

import pandas as pd

import torch

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)

df.head()
df.columns
df[1].value_counts()
negative=df.loc[df[1]==0][:1000]

positive=df.loc[df[1]==1][:1000]

df=pd.DataFrame(columns=[0,1])

df=df.append(negative)

df=df.append(positive)

df=df.sample(frac=1)

df.head()
df.shape
import transformers
from transformers import DistilBertTokenizer



# Load the DistillBERT tokenizer.

print('Loading DistillBERT tokenizer...')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)# convertiing evry input to lower case
# Print the original sentence.

print(' Original: ', df[0][0])



# Print the sentence split into tokens.

print('Tokenized: ', tokenizer.tokenize(df[0][0]))



# Print the sentence mapped to token ids.

print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(df[0][0])))
model, pretrained_weights = (transformers.DistilBertModel, 'distilbert-base-uncased')
model = model.from_pretrained(pretrained_weights)
tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
# Padding

max_len = 0

for i in tokenized.values:

    if len(i) > max_len:

        max_len = len(i)



padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
print(padded.shape)

# Print the original sentence.

print(' Original: ', df[0][0])

print('\n')

# Print the sentence split into tokens.

print('Tokenized: ', tokenizer.tokenize(df[0][0]))

print('\n')

# Print the sentence mapped to token ids.

print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(df[0][0])))

print('\n')

# padded sentence

print(padded[0])
attention_mask = np.where(padded != 0, 1, 0)

attention_mask.shape
df.shape
input_ids = torch.tensor(padded)   # converting into torch tensors

attention_mask = torch.tensor(attention_mask) # converting into torch tensors
with torch.no_grad():

    last_hidden_states = model(input_ids, attention_mask=attention_mask)
features = last_hidden_states[0][:,0,:].numpy()
labels = df[1] # getting labels
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
print(type(train_features))

print(type(test_features))
print(type(train_labels))

print(type(test_labels))
train_labels = train_labels.to_numpy() 

test_labels=test_labels.to_numpy()
train_labels=train_labels.astype('int')
test_labels=test_labels.astype('int')
lr_clf = LogisticRegression(C=5.2, max_iter=2000)

lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)