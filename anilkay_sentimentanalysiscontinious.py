# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/turkish-movie-sentiment-analysis-dataset/turkish_movie_sentiment_dataset.csv")

data.head()
data.shape
len(set(data["film_name"]))
import string 

string.punctuation
data["comment_no_punc"]=data['comment'].str.replace("["+string.punctuation+"]",'')
data_no_punc_v2=data["comment_no_punc"].str.replace("[\\n\\t]","")
data_no_punc_v3=data_no_punc_v2.str.strip()
from nltk.corpus import stopwords

turkish_stopwords=stopwords.words("turkish")

turkish_stopwords.append("bir")

turkish_stopwords.append("iki")

turkish_stopwords.append("üç")

turkish_stopwords.extend([

"benim",

"beri",

"beş",

"bile",

"bilhassa",

"bin",

"biraz",

"birçoğu",

"birisi",

"şey",

 "bi",

  "zaten",  

"böylece"])
#turkish_stopwords

from stop_words import get_stop_words

turkish_stopwords.extend(get_stop_words("tr"))
turkish_stopwords
data_no_punc_v4=data_no_punc_v3.apply(lambda x: ' '.join([item for item in x.split() if item not in turkish_stopwords]))
data_clean=pd.DataFrame(

 {

     "comments":data_no_punc_v4,

     "film_name":data["film_name"],

     "point":data["point"]

 }

)
data_clean.to_csv("clean_data.csv")
x=data_clean["film_name"]

y=data_clean["point"]
y=(data_clean["point"].str.replace(",",".")).astype(float)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(max_features=1000)

X_train=vectorizer.fit_transform(x_train)

X_test=vectorizer.transform(x_test)
from sklearn.decomposition import PCA

pca=PCA(n_components=100)

X_train_pca=pca.fit_transform(X_train.toarray())

X_test_pca=pca.transform(X_test.toarray())
import sklearn.metrics as metrik

from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor()

regressor.fit(X_train_pca,y_train)

ypred=regressor.predict(X_test_pca)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
#from sklearn.svm import SVR

#svr=SVR()

#svr.fit(X_train_pca,y_train)

#ypred=svr.predict(X_test_pca)

#metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)
data_clean["point"]=(data_clean["point"].str.replace(",",".")).astype(float)

data_clean.to_csv("clean_data.csv")
from sklearn.decomposition import PCA

pca=PCA(n_components=2)

X_train_2_dimension=pca.fit_transform(X_train.toarray())

X_test_2_dimension=pca.transform(X_test.toarray())

visual=pd.DataFrame(X_train_2_dimension)

visual.columns=["bir","iki"]

visual["point"]=data_clean["point"]
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(16,12))

sns.scatterplot(data=visual,x="bir",y="iki",hue="point",hue_norm=(0, 5))
import math

possibleidealK=math.sqrt(x_train.shape[0])/2

possibleidealK
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor(n_neighbors=int(possibleidealK))

knn.fit(X_train_pca,y_train)

ypred=knn.predict(X_test_pca)

metrik.mean_absolute_error(y_pred=ypred,y_true=y_test)