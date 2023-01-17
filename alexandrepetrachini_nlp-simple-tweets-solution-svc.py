import pandas as pd

import numpy as np

import nltk

from nltk import word_tokenize

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score

import re

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from collections import Counter

from sklearn.metrics import confusion_matrix

from sklearn import svm



clf3 = svm.SVC(random_state= 0,kernel = 'linear', C=1)







def reduce_lengthening(text):

    pattern = re.compile(r"(.)\1{2,}")

    return pattern.sub(r"\1\1", text)



clf = RandomForestRegressor(n_estimators = 200,max_depth = 50,random_state = 0, n_jobs = -1,min_samples_leaf = 10,criterion='mse')

clf2 = LogisticRegression(random_state = 0)





param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

clf2 = GridSearchCV(LogisticRegression(max_iter = 1000), param_grid)























vectorizer = TfidfVectorizer()



nltk.download("stopwords")



nltk.download('maxent_ne_chunker')



nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('words')

nltk.download('wordnet')



data_tr = pd.read_csv("../input/nlp-getting-started/test.csv")

data_test = pd.read_csv("../input/nlp-getting-started/train.csv")











y = data_tr.iloc[:,4].values

data_tr = data_tr.drop(labels=['target','location','keyword'], axis=1) 

data_test = data_test.drop(labels=['location','keyword'], axis=1) 



data_tr = data_tr.dropna()

data_test = data_test.dropna()















def format(data_train):

    spec_chars = ["!",'"',"#","%","&","'","(",")",

                  "*","+",",","-",".","/",":",";","<",

                  "=",">","?","@","[","\\","]","^","_",

                  "`","{","|","}","~","–"]

    

    porter = PorterStemmer()

    wordnet_lemmatizer = WordNetLemmatizer()

    

    

    for char in spec_chars:

        data_train['text'] = data_train['text'].str.replace(char, '')

      

    

    

    for k in data_train['text']:

        output = re.sub("[^0-9a-zA-Z]+", ' ', k)

        output = re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b"," ",k)

        data_train['text'].replace(k,value = output,inplace=True)   

    

        

    somer = 0  

    for k in data_train['text']:

        somer += 1

        k_edit = (k.lower()).split(' ')

        for l in k_edit:

            if 'http' in l:

                del(k_edit[k_edit.index(l)])

            elif l in stopwords.words('english'):

                del(k_edit[k_edit.index(l)])

            else:

                l = reduce_lengthening(l)

                #l = spell.correction(l)

                l = wordnet_lemmatizer.lemmatize(l, pos="v")

        k_edit = ' '.join(k_edit)        

        data_train['text'].replace(k,value = k_edit,inplace=True)   

        print("Success "+str(somer))

        

    

    return data_train





data_tr = format(data_tr)

data_test = format(data_test)



docs = data_tr['text']

docs2 = data_test['text']



X_all = pd.concat([data_tr['text'], data_test['text']])



vectorizer.fit(X_all)



X = vectorizer.transform(docs)

X_final = vectorizer.transform(docs2)

print(X_final.shape)



X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.01,random_state = 0,stratify = y)





print(X_train.shape,X_test.shape)



clf3.fit(X_train,y_train)

preds = clf3.predict(X_final)



result = [int((round(x))) for x in preds]

f1_score(result,y_test)



confusion_matrix(result,y_test)



df_final = data_test[['id','text']]

se = pd.Series(preds)

df_final['text'] = se

df_final.columns = ['id','target']



df_final.to_csv('submission.csv',index = False)







    
