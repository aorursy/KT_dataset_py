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
import matplotlib.pyplot as plt

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import  linear_model
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import utils
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # fancy statistics plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import geopandas as gpd
import scipy
from scipy.optimize import curve_fit
import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import seaborn as sns
from plotly.subplots import make_subplots
%matplotlib inline


import plotly.tools as tls
import cufflinks as cf
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
dataset
print("Satır Sayısı:\n",dataset.shape[0:])
print("Sütun Adlari:\n",dataset.columns.tolist())
print("Veri Tipleri:\n",dataset.dtypes)
print(dataset.shape)
print(dataset.head(10))
plt.figure(figsize=(15,5))
sns.barplot(x=dataset['Date'], y=dataset['Deaths'])
plt.xticks(rotation= 90)
plt.xlabel('Date')
plt.ylabel('Deaths')
plt.title("Günlük Ölen Sayısı")
plt.figure(figsize=(15,5))
sns.barplot(x=dataset['Date'], y=dataset['Confirmed'])
plt.xticks(rotation= 90)
plt.xlabel('Date')
plt.ylabel('Confirmed')
plt.title("Günlük Hastalanan Sayısı")
plt.figure(figsize=(15,5))
sns.barplot(x=dataset['Date'], y=dataset['Recovered'])
plt.xticks(rotation= 90)
plt.xlabel('Date')
plt.ylabel('Recovered')
plt.title("Günlük İyileşen Sayısı")

dataset.hist()
pyplot.show()
data_birlesim=dataset.groupby(['Country/Region']).agg({'Confirmed':'sum','Deaths':'sum'}).sort_values(["Confirmed"],ascending=False).reset_index()
data_birlesim.head(20)
fig = px.pie(data_birlesim.head(20),
             values="Confirmed",
             names="Country/Region",
             title="Ülkelere Göre Hasta Sayıları",
             template="seaborn")
fig.update_traces(rotation=45, pull=0.05, textinfo='label+value')
fig.show()
f,ax1 = plt.subplots(figsize =(30,15))
sns.pointplot(x=dataset.Date,y=dataset.Deaths,color='red')
sns.pointplot(x=dataset.Date,y=dataset.Confirmed,color='green')
plt.text(0,3500,'Confirmed',color='green',fontsize =10,style = 'italic')
plt.text(0,1000,' Deaths',color='red',fontsize = 10,style = 'italic')
plt.xlabel('Tarih')
plt.ylabel('İnsan sayısı')
plt.xticks(rotation= 90)
plt.title('Dünyada ölen ve hastaların karşılaştırılması')
plt.grid()
dataset.isnull().sum()
dataset.dropna()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
dataset['Country_en']= label_encoder.fit_transform(dataset['Country/Region'])

dataset
dataset=dataset.drop(columns ='Province/State')
dataset =dataset.drop(columns ='WHO Region')
dataset=dataset.drop(columns ='Lat')
dataset=dataset.drop(columns ='Long')
dataset

dataset

array = dataset.values
X = array[:,0:5]
y = array[:,0:1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.15, random_state=1)

print("Dataframe boyutu: ",dataset.shape)
print("Eğitim verisi boyutu: ",X_train.shape, Y_train.shape)
print("Test verisi boyutu: ",X_validation.shape, Y_validation.shape)
# type error için target typesı "Label Encoder" ile  multiclassa çevirdim.(Target=Y_train)
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)
print(utils.multiclass.type_of_target(y))
print(utils.multiclass.type_of_target(Y_train.astype('int')))
print(utils.multiclass.type_of_target(encoded))

lab_enc = preprocessing.LabelEncoder()
Y_train = lab_enc.fit_transform(Y_train)
print(utils.multiclass.type_of_target(Y_train))

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
#Decision Trees
cellTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(cellTree) # it shows the default parameters
  #I fit the data with the training
cellTree.fit(X_train,Y_train)
  #now predictions
yhat_dt = cellTree.predict(X_validation)

  #Accuracy evaluation
acc = metrics.accuracy_score(Y_validation, yhat_dt)
print('karar agaci icin accuracy: ',acc)

#karar agaci icin confusion matrix ve metrik degerler
cellTree_dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik
cv_scores_dt = cross_val_score(cellTree_dt, X,y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores_dt)
print('cv_scores mean:{}'.format(np.mean(cv_scores_dt)))
from sklearn.metrics import classification_report
prec_dt = classification_report(yhat_dt,Y_validation)
print(prec_dt)
#call the models
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 3)
#fit the models
neigh = knn_model.fit(X_train,Y_train)
#predict the mode;
yhatknn=neigh.predict(X_validation)

  #Accuracy evaluation
accknn = metrics.accuracy_score(Y_validation, yhatknn)
print('en yakin komsular icin accuracy',accknn)

#knn=3 icin confusion matrix ve metrik degerler
knn_knn = KNeighborsClassifier(n_neighbors = 3)
#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik
cv_scores_knn = cross_val_score(knn_knn, X,y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores_knn)
print('cv_scores mean:{}'.format(np.mean(cv_scores_knn)))

#knn scores
from sklearn.metrics import classification_report
prec_knn = classification_report(yhatknn,Y_validation)
print(prec_knn)
#gaussian NB 
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
#call the models
gnb = GaussianNB()
  #fit the model
gnb.fit(X_train, Y_train) 
  #predict
yhatgnb = gnb.predict(X_validation)
accgnb = metrics.accuracy_score(Y_validation, yhatgnb)
print('gaussian naive bayes icin accuracy',accgnb)


#gaussian naive bayes icin confusion matrix ve metrik degerler
clf_gnb = GaussianNB()
#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik
cv_scores_gnb = cross_val_score(clf_gnb, X,y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores_gnb)
print('cv_scores mean:{}'.format(np.mean(cv_scores_gnb)))

#klasifikasyon tablosu
from sklearn.metrics import classification_report
prec_gnb = classification_report(yhatgnb,Y_validation)
print(prec_gnb)