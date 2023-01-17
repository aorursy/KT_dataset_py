!pip install ktrain
import numpy as np 
import pandas as pd 
import re
from nltk.corpus import stopwords
import ktrain
from ktrain import text
STOPWORDS = stopwords.words('english')
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print(test.shape)
test.head()
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
print(train.shape)
train.tail()
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
data = df_cleaner(train, "text")
data.head()
df = data[['text','target']]
df = pd.concat([df, df.target.astype('str').str.get_dummies()], axis=1, sort=False)
df = df[['text','1','0']]
df.rename(columns={'1':'disaster', '0':'not_disaster'}, inplace=True)
df.head()
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(df, 
                                                                   'text', # name of column containing review text
                                                                   label_columns=['disaster', 'not_disaster'],
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
test_cleaned = df_cleaner(test,"text")
test_cleaned.head()
test_cleaned['predicted_target'] = test_cleaned['text'].apply(lambda x: p.predict(x))
test_cleaned.head(20)
sub = test_cleaned[['id','predicted_target']]
sub.rename(columns={'predicted_target':'target'}, inplace=True)
sub['target'] = sub['target'].apply(lambda x: 1 if x=='disaster' else 0)
sub
sub.to_csv("/kaggle/working/submission.csv", index=False)