!pip install transformers==3.0.2
from abc import ABC
import random
import multiprocessing

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def read_fake_news_raw_data(train_test_ratio=0.9, train_valid_ratio=0.8, max_samples=None):
    raw_data_path = '../input/real-and-fake-news-dataset/news.csv'
    df_raw = pd.read_csv(raw_data_path)

    if max_samples:
        df_raw = df_raw.head(max_samples)

    # Prepare columns
    df_raw['label'] = (df_raw['label'] == 'FAKE').astype('int')
    df_raw['titletext'] = df_raw['title'] + ". " + df_raw['text']

    df_train, df_test = train_test_split(df_raw, train_size=train_test_ratio, random_state=1)
    df_valid = pd.DataFrame()
    if train_valid_ratio < 1:
        df_train, df_valid = train_test_split(df_train, train_size=train_valid_ratio, random_state=1)
        
    print('train, valid, test:', df_train.shape, df_valid.shape, df_test.shape)
    return df_train, df_valid, df_test


def report(y_pred, y_true):
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    print(confusion_matrix(y_true, y_pred, labels=[1,0]))
    

def run_experiments(models, max_samples=None, train=True):
    df_train, df_valid, df_test = read_fake_news_raw_data(max_samples=max_samples)
    for model in models:
        print(model)
        if train:
            model.train(df_train, df_valid)
        y_preds = model.predict(df_test)
        report(y_preds, df_test['label'])
class TextDataset(torch.utils.data.Dataset):

    def __init__(self, df, pretrained_model_name='bert-base-uncased', max_len=512):
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.inputs = []
        self.labels = torch.from_numpy(np.asarray(df['label'], dtype=np.int32)).long()
        for titletext in df['titletext']:
            split = titletext.split(maxsplit=512)
            text = ' '.join(split[:512])
            inputs = tokenizer(titletext, max_length=max_len, pad_to_max_length=True, return_tensors='pt')
            self.inputs.append(inputs)
        
    def __getitem__(self, idx):
        item = self.inputs[idx]
        return item['input_ids'][0], item['attention_mask'][0], item['token_type_ids'][0], self.labels[idx]
        
    def __len__(self):
        return len(self.inputs)


class BertModel:
    
    def __init__(self, pretrained_model_name='bert-base-uncased', freeze_bert=False, max_len=512):
        self.num_epochs = 5
        self.freeze_bert = freeze_bert
        self.max_len = max_len
        self.pretrained_model_name = pretrained_model_name

    def train(self, df_train, df_valid):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model = BertForSequenceClassification.from_pretrained(self.pretrained_model_name).to(self.device)
        self._model.bert.requires_grad = not self.freeze_bert
        
        train_loader = self._create_data_loader(df_train)
        valid_loader = self._create_data_loader(df_valid)
        
        optimizer = optim.Adam(self._model.parameters(), lr=2e-5)
        best_valid_loss = float("Inf")

        for epoch in range(self.num_epochs):
            self._model.train()
            train_loss = 0.0
            valid_loss = 0.0

            for batch in train_loader:
                loss, _ = self.model_forward(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self._model.eval()
            with torch.no_grad():                    
                for batch in valid_loader:
                    loss, _ = self.model_forward(batch)
                    valid_loss += loss.item()

            train_loss = train_loss / len(train_loader)
            valid_loss = valid_loss / len(valid_loader)

            print((f'Epoch [{epoch+1}/{self.num_epochs}] '
                   f'Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f}'))

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self._model.state_dict(), 'checkpoint.pt')
            else:
                # restore the best weights and quit
                self._model.load_state_dict(torch.load('checkpoint.pt', map_location=self.device))
                break
        
    def predict(self, df_test):
        y_probs = self.predict_probs(df_test)
        return torch.argmax(y_probs, 1).tolist()
    
    def predict_probs(self, df_test):
        test_loader = self._create_data_loader(df_test)
        outputs = []
        # the model doesn't output probabilities, because Softmax is hidden in the loss function for better performance
        softmax = nn.Softmax(dim=1)

        self._model.eval()
        with torch.no_grad():
            for batch in test_loader:
                _, output = self.model_forward(batch)
                output = softmax(output)
                outputs.append(output)
        return torch.cat(outputs, dim=0).cpu()
    
    def model_forward(self, batch):
        input_ids, attention_mask, token_type_ids, labels = batch
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids =token_type_ids.to(self.device)
        labels = labels.to(self.device)
        
        loss, text_fea = self._model(input_ids, attention_mask, token_type_ids, labels=labels)[:2]
        return loss, text_fea

    def _create_data_loader(self, df):
        dataset = TextDataset(df, self.pretrained_model_name, max_len=self.max_len)
        return DataLoader(dataset, batch_size=8, shuffle=False)
class SciKitClassifierModel(ABC):
    
    def train(self, df_train, df_valid):
        df_train = pd.concat([df_train, df_valid])
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=100000)
        self.vectorizer.fit_transform(df_train['titletext'])
        X_train = self.vectorize(df_train)
        y_train = df_train['label']
        self.model.fit(X_train, y_train)
        
    def predict(self, df_test):
        X_test = self.vectorize(df_test)
        return self.model.predict(X_test)
        
    def predict_probs(self, df_test):
        X_test = self.vectorize(df_test)
        return self.model.predict_proba(X_test)
    
    def vectorize(self, df):
        return self.vectorizer.transform(df['titletext'])


class PACModel(SciKitClassifierModel):

    def __init__(self):
        self.model = PassiveAggressiveClassifier(max_iter=50)
        
    def predict_probs(self, df_test):
        X_test = self.vectorize(df_test)
        return self.model._predict_proba_lr(X_test)
    

class SVMModel(SciKitClassifierModel):
    
    def __init__(self):
        self.model = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
        

class GradientBoostingModel(SciKitClassifierModel):
    
    def __init__(self):
        self.model = GradientBoostingClassifier()
        

class Ensemble:
    
    def __init__(self, models):
        self.models = models
    
    def train(self, df_train, df_valid):
        for model in self.models:
            model.train(df_train, df_valid)
        
    def predict(self, df_test):
        y_probs = self.predict_probs(df_test)
        return np.argmax(y_probs, 1)
        
    def predict_probs(self, df_test):
        predictions = np.stack([np.array(model.predict_probs(df_test)) for model in self.models])
        return np.mean(predictions, axis=0)
class GloveModel:
    
    def __init__(self, max_len=1024):
        self.max_len = max_len
        self.tokenizer = text.Tokenizer(num_words=None)
    
    def train(self, df_train, df_valid):
        self.tokenizer.fit_on_texts(list(df_train['titletext']) + list(df_valid['titletext']))
        
        X_train = self._vectorize(df_train)
        X_valid = self._vectorize(df_valid)
        y_train = np_utils.to_categorical(df_train['label'])
        y_valid = np_utils.to_categorical(df_valid['label'])
        
        self.model = self._init_model(y_train.shape[1])
        earlystop = EarlyStopping(monitor='val_loss', restore_best_weights=True)
        # note that keras automatically uses GPU if available
        self.model.fit(X_train, y=y_train, batch_size=256, epochs=10, verbose=1, 
                       validation_data=(X_valid, y_valid), callbacks=[earlystop])
    
    def predict(self, df_test):
        y_probs = self.predict_probs(df_test)
        return np.argmax(y_probs, 1)
    
    def predict_probs(self, df_test):
        X_test = self._vectorize(df_test)
        return self.model.predict(X_test)
    
    def _vectorize(self, df):
        seq = self.tokenizer.texts_to_sequences(df['titletext'])
        return sequence.pad_sequences(seq, maxlen=self.max_len)
        
    def _init_embedding(self):
        # in kaggle notebooks, just hit "Add data" in upper right and search for glove.840B.300d.pkl
        embeddings_index = np.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl', allow_pickle=True)

        word_index = self.tokenizer.word_index
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
    
    def _init_model(self, output_shape):
        embedding_matrix = self._init_embedding()
        model = Sequential()
        model.add(Embedding(len(self.tokenizer.word_index) + 1, 300, weights=[embedding_matrix], 
                            input_length=self.max_len, trainable=True))
        model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)))

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(output_shape))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
# classic models
classic_models = [
    PACModel(),
    GradientBoostingModel(), 
    SVMModel(),
]
run_experiments(classic_models)
# GloVe models
# there are some issues with cleaning up GPU memory after Keras uses it, so let's run this in a subprocess
def run_glove_models():
    glove_models = [
        GloveModel(max_len=512),
        GloveModel(),
    ]
    run_experiments(glove_models)
p = multiprocessing.Process(target=run_glove_models)
p.start()
p.join()
# BERT models
bert_models = [
    BertModel(max_len=128),
    BertModel(freeze_bert=True),
    BertModel('bert-base-cased'),
    BertModel(),
]
run_experiments(bert_models)
# ensambles
ensemble_models = [
    Ensemble(classic_models[:2] + bert_models[-1:]),
    Ensemble(classic_models[:2] + bert_models[-2:]),
]
run_experiments(ensemble_models, train=False)