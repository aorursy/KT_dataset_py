# install ktrain on Google Colab
!pip3 install ktrain
# import ktrain and the ktrain.text modules
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
ktrain.__version__
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download('wordnet')
from nltk.tokenize import WordPunctTokenizer 
from nltk.stem import WordNetLemmatizer 
import json
import tensorflow as tf
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df=pd.read_excel(r'../input/tamildata/Tamil-Codemixed_offensive_data_Training-Tweet-HL (1).xlsx')
df.columns = ['id', 'tweet','label','english']
df.shape
df.head(5)
df['label']=df['label'].str.upper() 
test=pd.read_csv('../input/testtamil/tamil_test - tamil_test.csv')
test.head(3)
df1, df2 = train_test_split(df, test_size=0.1, random_state=42)
print(df1.shape)
print(df2.shape)
(x_train,  y_train), (x_test, y_test), preprocess    = text.texts_from_df(train_df=df1,
                   text_column='tweet',
                   label_columns='label',
                   val_df=df2,
                   maxlen=120,
                   preprocess_mode='bert' )
print(x_train[0].shape)
print(y_train[0].shape)
model= text.text_classifier(name='bert',
                            train_data=(x_train,  y_train), 
                            preproc=preprocess)
learner= ktrain.get_learner(model=model,
                            train_data=(x_train,  y_train), 
                            val_data=(x_test,  y_test), 
                            batch_size=6)
learner.fit_onecycle(3e-5, 3)
predictor=ktrain.get_predictor(learner.model,preprocess)
test=pd.read_csv('../input/testtamil/tamil_test - tamil_test.csv')
test.head()
te=test['text'].to_list()
type(te)
yas=predictor.predict(te)
print(yas[:5])
print(len(yas))
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(classification_report(yas,df2.label))

predictor.predict('ada paavingala naa 1991 kids da ipo thaanda enaku marriage ku ponu pakaranga')
test['label']=yas
test.to_csv('Bert.csv')




###########MULTILINGUAL
MODEL_NAME = 'bert-base-multilingual-cased'
t1 = text.Transformer(MODEL_NAME, maxlen=120, class_names=['NOT','OFF'])

trn = t1.preprocess_train(df1.tweet.to_list(), df1.label.to_list())
len(trn)

val = t1.preprocess_test(df2.tweet.to_list(), df2.label.to_list())

len(val)
model1 = t1.get_classifier()

learner1 = ktrain.get_learner(model1, train_data=trn, val_data=val, batch_size=6)

learner1.fit_onecycle(3e-5, 1)
predictor1=ktrain.get_predictor(learner1.model,preproc=t1)
yas1=predictor1.predict(te)
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(classification_report(yas1,df2.label))

