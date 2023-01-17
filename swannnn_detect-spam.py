!pip install ktrain
import numpy as np 
import pandas as pd 
import re
import ktrain
from ktrain import text
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
TESTSIZE = 0.3
PATH = '/kaggle/input/spam-text-message-classification/'
TRAIN = 'SPAM text message 20170820 - Data.csv'
data = pd.read_csv(PATH+TRAIN)
print(data.shape)
data.head()
def cleaner(sentence):
    sentence = str(sentence).lower()
    sentence = re.sub(r"[^a-zA-Z ]+",'',sentence)
    sentence = re.sub(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+','',sentence)
    cleaned = ''''''
    for word in sentence.split():
        if word not in STOPWORDS:
            cleaned = "{} {}".format(cleaned, word)
    return cleaned

def df_cleaner(df, column):
    df.replace(np.nan,"", regex=True,inplace=True)
    df[column] = df[column].apply(lambda x: cleaner(x))
    return df
data = df_cleaner(data, "Message")
data.head()
data = pd.concat([data, data.Category.astype('str').str.get_dummies()], axis=1, sort=False)
data = data[['Message','ham','spam']]
data.head()
%%time
n = data.shape[0]*TESTSIZE
train = data.loc[:n]
test  = data.loc[n:]
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(train, 
                                                                   'Message', # name of column containing review text
                                                                   label_columns=['ham','spam'],
                                                                   maxlen=75, 
                                                                   max_features=100000,
                                                                   preprocess_mode='bert',
                                                                   val_pct=0.1)
model = text.text_classifier('bert', (x_train, y_train) , preproc=preproc)
learner = ktrain.get_learner(model, 
                             train_data=(x_train, y_train), 
                             val_data=(x_test, y_test), 
                             batch_size=32)
learner.fit_onecycle(2e-5, 2)
p = ktrain.get_predictor(learner.model, preproc)
test['predicted_value'] = test['Message'].apply(lambda x: p.predict(x))
test
