import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import nltk
import gc
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv("../input/train_main.csv")
test = pd.read_csv("../input/test_main.csv")
print('Shape of train dataset:{}'.format(train.shape))
print('Shape of test dataset:{}'.format(test.shape))
train['train_or_test'] = 'train'
test['train_or_test'] = 'test'
df = pd.concat([train,test], sort=False)
print('Combined df shape:{}'.format(df.shape))
del train, test
gc.collect()
df.head()
df['tare'].nunique()
df.describe()
df.groupby('tare').describe()
df['length'] = df['name'].apply(len)
df.head()
df['length'].plot(bins=50,kind='hist')
df.length.describe()
df[df['length']==184]['name'].iloc[0]
import string

name = 'Тестовое сообщение, обратите внимание на пункутацию'

# Проверка, есть ли пунктуация в тексте
nopunc = [char for char in name if char not in string.punctuation]

# Соединение символов для их перевода в текст.
nopunc = ''.join(nopunc)
print(nopunc)
from nltk.corpus import stopwords
stopwords.words('russian')[0:10]
#Дробим текст в лист для следующей операции
nopunc.split()
clean_name = [word for word in nopunc.split() if word.lower() not in stopwords.words('russian')]
print(clean_name)
def text_process(name):
    """
    1.Убираем всю пунктуацию
    2.Убираем все stopwords(повторяющиеся слова)
    3.Возвращаем чистый текст
    """
    # Проверка, есть ли пунктуация в тексте
    nopunc = [char for char in name if char not in string.punctuation]
    
    # Соединение символов для их перевода в текст
    nopunc = ''.join(nopunc)
    
    #Убираем все stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('russian')]
df.head()
df['name'].head(5).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
#Может занять некоторое время
bow_transformer = CountVectorizer(analyzer=text_process).fit(df['name'])

# Вывести слова в нашем словаре
print(len(bow_transformer.vocabulary_))
df['name'][2]
name4 = df[df['train_or_test']=='train'].iloc[3]['name']
print(name4)
train_df = df[df['train_or_test']=='train']
test_df = df[df['train_or_test']=='test']
bow4 = bow_transformer.transform([name4])
print(bow4)
print(bow4.shape)
names_test_bow = df[df['train_or_test']=='test']
names_train_bow = df[df['train_or_test']=='train']
names_train_bow = bow_transformer.transform(names_train_bow['name'])
names_test_bow = bow_transformer.transform(names_test_bow['name'])
print('Shape of Train Sparse Matrix:{}'.format(names_train_bow.shape))
print('Shape of Test Sparse Matrix:{}'.format(names_test_bow.shape))
print('Amount of Non-Zero occurences in Train:{}'.format(names_train_bow.nnz))
print('Amount of Non-Zero occurences in Test:{}'.format(names_test_bow.nnz))
train_sparsity = (100.0 * names_train_bow.nnz / (names_train_bow.shape[0] * names_train_bow.shape[1]))
test_sparsity = (100.0 * names_test_bow.nnz / (names_test_bow.shape[0] * names_test_bow.shape[1]))
print('Sparsity of Train: {}'.format(round(train_sparsity)))
print('Sparsity of Test: {}'.format(round(test_sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer_train = TfidfTransformer().fit(names_train_bow)
tfidf_transformer_test = TfidfTransformer().fit(names_test_bow)
tfidf4 = tfidf_transformer_train.transform(bow4)
print(tfidf4)
names_train_tfidf = tfidf_transformer_train.transform(names_train_bow)
names_test_tfidf = tfidf_transformer_train.transform(names_test_bow)
print(names_train_tfidf.shape)
print(names_test_tfidf.shape)
tare_train = df[df['train_or_test']=='train']['tare']
tare_train.head()
from sklearn.naive_bayes import MultinomialNB
tare_assign_model = MultinomialNB().fit(names_train_tfidf, train_df['tare'])
print('predicted:', tare_assign_model.predict(tfidf4)[0])
print('expected:', df.tare[3])
from sklearn.model_selection import train_test_split

name_train, name_test, tare_train, tare_test = \
train_test_split(train_df['name'],train_df['tare'], test_size=0.2)

print(len(name_train), len(name_test), len(name_train) + len(name_test))
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # текст в числовые векторы
    ('tfidf', TfidfTransformer()),  # числовые векторы с помощью TFIDF
    ('classifier', MultinomialNB()),  # тренировка на векторах TF-IDF с Наивным Байесовским классификатором
])
pipeline.fit(name_train,tare_train)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
predictions = pipeline.predict(name_test)
print(classification_report(predictions,tare_test))
print(accuracy_score(predictions,tare_test))
test_preds = pipeline.predict(test_df['name'])
test = df.loc[df.train_or_test=='test', :]
train = df.loc[~(df.train_or_test=='test'), :]
print('Train shape:{}, Test shape:{}'.format(train.shape, test.shape))
sub = test.loc[:,['id','name','tare']]
sub['tare'] = test_preds
sub['id'] = sub.id.astype(int)
sub['name'] = test_df['name']
sub.to_csv('submission_r.csv', index=False)
sub.head()
