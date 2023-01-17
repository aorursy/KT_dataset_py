# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install transformers
import transformers

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch

import numpy as np

import pandas as pd

import seaborn as sns

from pylab import rcParams

import matplotlib.pyplot as plt

import re

from tqdm import tqdm

from string import punctuation

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from matplotlib import rc

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, f1_score

from collections import defaultdict

from textwrap import wrap

from torch import nn, optim

from torch.utils.data import Dataset, DataLoader

from sklearn.utils import resample

from tqdm import tqdm

import warnings



%matplotlib inline

%config InlineBackend.figure_format='retina'

warnings.filterwarnings('ignore')



sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



stop_words = set(stopwords.words('english'))

punctuation = punctuation + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
df = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')

df.head()
df['target'].value_counts()
# Checing for the null values

df.isnull().sum(axis = 0)
# Contraction Dictionary for the expansion

contractions_dict = {

    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",

    "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",

    "doesn’t": "does not", "don't": "do not", "don’t": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not",

    "haven't": "have not", "he'd": "he had", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",

    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "i'd": "i would", "i'd've": "i would have",

    "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",

    "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not","might've": "might have",

    "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",

    "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",

    "shan't": "shall not","sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",

    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",

    "shouldn't've": "should not have", "so've": "so have", "so's": "so is", "that'd": "that would", "that'd've": "that would have",

    "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would",

    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",

    "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",

    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

    "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

    "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",

    "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",

    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y’all": "you all", "y'all'd": "you all would",

    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",

    "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", "ain’t": "am not", "aren’t": "are not",

    "can’t": "cannot", "can’t’ve": "cannot have", "’cause": "because", "could’ve": "could have", "couldn’t": "could not", "couldn’t’ve": "could not have",

    "didn’t": "did not", "doesn’t": "does not", "don’t": "do not", "don’t": "do not", "hadn’t": "had not", "hadn’t’ve": "had not have",

    "hasn’t": "has not", "haven’t": "have not", "he’d": "he had", "he’d’ve": "he would have", "he’ll": "he will", "he’ll’ve": "he will have",

    "he’s": "he is", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will", "how’s": "how is", "i’d": "i would", "i’d’ve": "i would have",

    "i’ll": "i will", "i’ll’ve": "i will have", "i’m": "i am", "i’ve": "i have", "isn’t": "is not", "it’d": "it would", "it’d’ve": "it would have",

    "it’ll": "it will", "it’ll’ve": "it will have", "it’s": "it is", "let’s": "let us", "ma’am": "madam", "mayn’t": "may not",

    "might’ve": "might have", "mightn’t": "might not", "mightn’t’ve": "might not have", "must’ve": "must have", "mustn’t": "must not",

    "mustn’t’ve": "must not have", "needn’t": "need not", "needn’t’ve": "need not have", "o’clock": "of the clock",

    "oughtn’t": "ought not", "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have",

    "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have", "she’s": "she is",

    "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have", "so’ve": "so have", "so’s": "so is",

    "that’d": "that would", "that’d’ve": "that would have", "that’s": "that is", "there’d": "there would", "there’d’ve": "there would have",

    "there’s": "there is", "they’d": "they would", "they’d’ve": "they would have", "they’ll": "they will", "they’ll’ve": "they will have",

    "they’re": "they are", "they’ve": "they have", "to’ve": "to have", "wasn’t": "was not", "we’d": "we would", "we’d’ve": "we would have",

    "we’ll": "we will", "we’ll’ve": "we will have", "we’re": "we are", "we’ve": "we have", "weren’t": "were not", "what’ll": "what will",

    "what’ll’ve": "what will have", "what’re": "what are", "what’s": "what is", "what’ve": "what have", "when’s": "when is",

    "when’ve": "when have", "where’d": "where did", "where’s": "where is", "where’ve": "where have", "who’ll": "who will",

    "who’ll’ve": "who will have", "who’s": "who is", "who’ve": "who have","why’s": "why is", "why’ve": "why have", "will’ve": "will have",

    "won’t": "will not", "won’t’ve": "will not have", "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have",

    "y’all": "you all", "y’all": "you all", "y’all’d": "you all would", "y’all’d’ve": "you all would have", "y’all’re": "you all are",

    "y’all’ve": "you all have", "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will", "you’ll’ve": "you will have",

    "you’re": "you are", "you’re": "you are", "you’ve": "you have"

}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
# Function to clean the html from the Questions

def cleanhtml(raw_html):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, '', raw_html)

    return cleantext



# Function expand the contractions if there's any

def expand_contractions(s, contractions_dict=contractions_dict):

    def replace(match):

        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, s)



# Function to preprocess the questions

def preprocessing(question):

    global question_sent

    

    # Converting to lowercase

    question = question.str.lower()

    

    # Removing the HTML

    question = question.apply(lambda x: cleanhtml(x))

    

    # Removing the email ids

    question = question.apply(lambda x: re.sub('\S+@\S+','', x))

    

    # Removing The URLS

    question = question.apply(lambda x: re.sub("((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?",'', x))

    

    # Mapping the contractions

    question = question.apply(lambda x: expand_contractions(x))

    

    # Stripping the possessives

    question = question.apply(lambda x: x.replace("'s", ''))

    question = question.apply(lambda x: x.replace('’s', ''))

    question = question.apply(lambda x: x.replace("\'s", ''))

    question = question.apply(lambda x: x.replace("\’s", ''))

    

    # Removing the Trailing and leading whitespace and double spaces

    question = question.apply(lambda x: re.sub(' +', ' ',x))

    

    # Removing punctuations from the question

    question = question.apply(lambda x: ''.join(word for word in x if word not in punctuation))

    

    # Removing the Trailing and leading whitespace and double spaces again as removing punctuation might lead to a white space

    question = question.apply(lambda x: re.sub(' +', ' ',x))

    

    # Removing the Stopwords

    # question = question.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    

    return question
df['processed_text'] = preprocessing(df['question_text'])



# Lemmatization using WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

df['processed_text'] = df['processed_text'].apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split()))



df['processed_text'] = df['processed_text'].astype('str')

df = resample(df, random_state = RANDOM_SEED)



df.head(20)
# Bert Parameters

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

MAX_LEN = 160

BATCH_SIZE = 16

EPOCHS = 1
# Bert Tokenizer

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)



# Bert Tokenizer with example

sample_txt = 'When was I last outside? I am stuck at home for 2 weeks.'



tokens = tokenizer.tokenize(sample_txt)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')

print(f'   Tokens: {tokens}')

print(f'Token IDs: {token_ids}')



encoding = tokenizer.encode_plus(

  sample_txt,

  max_length=32,

  add_special_tokens=True, # Add '[CLS]' and '[SEP]'

  return_token_type_ids=False,

  pad_to_max_length=True,

  return_attention_mask=True,

  return_tensors='pt',  # Return PyTorch tensors

)

encoding.keys()
# Using the Bert tokenizer for encoding the questions

token_lens = []

for txt in tqdm(df.processed_text):

    tokens = tokenizer.encode(txt, max_length=512, truncation = True)

    token_lens.append(len(tokens))
# Distribution of tokens

sns.distplot(token_lens)

plt.xlim([0, 256]);

plt.xlabel('Token count');
# Preparing the dataset with input ids and attention_masks

class GPquestionDataset(Dataset):

    

    def __init__(self, questions, targets, tokenizer, max_len):

        self.questions = questions

        self.targets = targets

        self.tokenizer = tokenizer

        self.max_len = max_len

    def __len__(self):

        return len(self.questions)

    def __getitem__(self, item):

        question = str(self.questions[item])

        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(

          question,

          add_special_tokens=True,

          max_length=self.max_len,

          return_token_type_ids=False,

          pad_to_max_length=True,

          return_attention_mask=True,

          return_tensors='pt',

          truncation=True  

        )

        return {

          'question_text': question,

          'input_ids': encoding['input_ids'].flatten(),

          'attention_mask': encoding['attention_mask'].flatten(),

          'targets': torch.tensor(target, dtype=torch.long)

      }
# Splitting the dataset for training, validation and testing

df_train, df_test = train_test_split(

  df,

  test_size = 0.4,

  random_state = RANDOM_SEED

)

df_val, df_test = train_test_split(

  df_test,

  test_size = 0.6,

  random_state = RANDOM_SEED

)

print ("The shape of the training dataset : ", df_train.shape)

print ("The shape of the validation dataset : ", df_val.shape)

print ("The shape of the testing dataset : ", df_test.shape)
# Creating the data loader for training, validation and testing

def create_data_loader(df, tokenizer, max_len, batch_size):

    ds = GPquestionDataset(

      questions = df.processed_text.to_numpy(),

      targets = df.target.to_numpy(),

      tokenizer = tokenizer,

      max_len = max_len

    )

    return DataLoader(

      ds,

      batch_size = batch_size,

      num_workers = 0

    )



train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)

val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)



data = next(iter(train_data_loader))

data.keys()



print (len(train_data_loader))

print (len(val_data_loader))

print (len(test_data_loader))
print(data['input_ids'].shape)

print(data['attention_mask'].shape)

print(data['targets'].shape)
# Using the Bert Model 'bert-base-cased'

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)



last_hidden_state, pooled_output = bert_model(

  input_ids=encoding['input_ids'],

  attention_mask=encoding['attention_mask']

)



print (last_hidden_state.shape)
bert_model.config.hidden_size
pooled_output.shape
# Model

class QuestionClassifier(nn.Module):

    def __init__(self, n_classes):

        super(QuestionClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        self.drop = nn.Dropout(p=0.3)

        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):

        _, pooled_output = self.bert(

          input_ids=input_ids,

          attention_mask=attention_mask

        )

        output = self.drop(pooled_output)

        return self.out(output)



model = QuestionClassifier(2)

model = model.to(device)
input_ids = data['input_ids'].to(device)

attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length

print(attention_mask.shape) # batch size x seq length



model(input_ids, attention_mask)



optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(

  optimizer,

  num_warmup_steps=0,

  num_training_steps=total_steps

)

loss_fn = nn.CrossEntropyLoss().to(device)
# Training function

def train_epoch(

    model,

    data_loader,

    loss_fn,

    optimizer,

    device,

    scheduler,

    n_examples

):  

    model = model.train()

    losses = []

    correct_predictions = 0

    for d in data_loader:

        input_ids = d["input_ids"].to(device)

        attention_mask = d["attention_mask"].to(device)

        targets = d["targets"].to(device)

        outputs = model(

          input_ids=input_ids,

          attention_mask=attention_mask

        )

        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)

        losses.append(loss.item())

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)
# Evaluation

def eval_model(model, data_loader, loss_fn, device, n_examples):

    model = model.eval()

    losses = []

    correct_predictions = 0

    with torch.no_grad():

        for d in data_loader:

            input_ids = d["input_ids"].to(device)

            attention_mask = d["attention_mask"].to(device)

            targets = d["targets"].to(device)

            outputs = model(

              input_ids=input_ids,

              attention_mask=attention_mask

            )

            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)
%%time

history = defaultdict(list)

best_accuracy = 0

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')

  print('-' * 50)

  train_acc, train_loss = train_epoch(

    model,

    train_data_loader,

    loss_fn,

    optimizer,

    device,

    scheduler,

    len(df_train)

  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(

    model,

    val_data_loader,

    loss_fn,

    device,

    len(df_val)

  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')

  print()

  history['train_acc'].append(train_acc)

  history['train_loss'].append(train_loss)

  history['val_acc'].append(val_acc)

  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:

    torch.save(model.state_dict(), 'best_model_state.bin')

    best_accuracy = val_acc
# The accuracy of the model

test_acc, _ = eval_model(

  model,

  test_data_loader,

  loss_fn,

  device,

  len(df_test)

)



print (test_acc)
# Predictions

def get_predictions(model, data_loader):

    model = model.eval()

    question_texts = []

    predictions = []

    prediction_probs = []

    real_values = []

    with torch.no_grad():

        for d in data_loader:

            texts = d["question_text"]

            input_ids = d["input_ids"].to(device)

            attention_mask = d["attention_mask"].to(device)

            targets = d["targets"].to(device)

            outputs = model(

              input_ids=input_ids,

              attention_mask=attention_mask

            )

            _, preds = torch.max(outputs, dim=1)

            question_texts.extend(texts)

            predictions.extend(preds)

            prediction_probs.extend(outputs)

            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()

    prediction_probs = torch.stack(prediction_probs).cpu()

    real_values = torch.stack(real_values).cpu()

    return question_texts, predictions, prediction_probs, real_values



y_question_texts, y_pred, y_pred_probs, y_test = get_predictions(

  model,

  test_data_loader

)
# First 10 predictions and text

i = 0

for t, pred, prob in zip(y_question_texts, y_pred, y_pred_probs):

    print (t, end = "   ")

    print (pred, end = "   ")

    print (prob)

    i+=1

    if i == 10:

        break
# Classification_report

print(classification_report(y_test, y_pred))
# Confusion matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print ("True Negative : ", tn)

print ("False Positive : ", fp)

print ("False Negative : ", fn)

print ("true positive : ", tp)
# f1_score

print (f1_score(y_test, y_pred))