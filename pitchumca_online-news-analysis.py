# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import gc

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

# Feature Extraction with RFE

from pandas import read_csv



char =  pd.read_csv('../input/OnlineNewsPopularity.csv')



categorical_list = []

numerical_list = []

for i in char.columns.tolist():

    if char[i].dtype=='object':

        categorical_list.append(i)

    else:

        numerical_list.append(i)

print('Number of categorical features:', str(len(categorical_list)))

print('Number of numerical features:', str(len(numerical_list)))

from sklearn.preprocessing import Imputer

char[numerical_list] = Imputer(strategy='median').fit_transform(char[numerical_list])

char.describe(include='all')

#count missing values

char.isna().sum()

char.isna().sum()

char.url.describe()



plt.figure(figsize=(20,12))

sns.heatmap(char.corr(), annot=True)

plt.show()

gc.collect()

char = pd.get_dummies(char, drop_first=True)

print(char.shape)

#X = char[:,0:61]

#Y = char[:,61]

#feature_name = X.columns.tolist()

chars_X = char.iloc[:, :-1]

chars_y = char.iloc[:, -1]

feature_name = chars_X.columns.tolist()

data=char



def cor_selector(X, y):

    cor_list = []

    # calculate the correlation with y for each feature

    for i in X.columns.tolist():

        cor = np.corrcoef(X[i], y)[0, 1]

        cor_list.append(cor)

    # replace NaN with 0

    cor_list = [0 if np.isnan(i) else i for i in cor_list]

    # feature name

    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-100:]].columns.tolist()

    # feature selection? 0 for not select, 1 for select

    cor_support = [True if i in cor_feature else False for i in feature_name]

    return cor_support, cor_feature

cor_support, cor_feature = cor_selector(chars_X,chars_y )

print(str(len(cor_feature)), 'selected features')

import numpy as np

import pylab as pl

from sklearn import svm, datasets



# import some data to play with



import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

%matplotlib inline

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

char =  pd.read_csv('../input/OnlineNewsPopularity.csv')

#char[char['title_sentiment_polarity'] == char['title_sentiment_polarity'].min()]

df =  pd.read_csv('../input/OnlineNewsPopularity.csv')

df = df.iloc[:,:-1]

t=.8 

df.drop('url', inplace=True, axis=1)

t = int(t*len(df)) 



# Train dataset 

df= pd.get_dummies(df, drop_first=True)

X_train = df[:t] 



y_train = df[:t]  



# Test dataset 



X_test = df[t:] 



y_test = df[t:]

array=df.values

X = array[:,0:61]

Y = array[:,58]

linear = LinearRegression().fit(X_train,y_train) 

#model = LogisticRegression()

#print(linear)

#rfe = RFE(model, 2)

#fit = rfe.fit(X_train,y_train)

#print(df.shares([X_train.data[4]]))

from sklearn.datasets import load_svmlight_file

from sklearn import svm

#X_train, y_train = load_svmlight_file("/path-to-file/train.txt")

#X_test, y_test = load_svmlight_file("/path-to-file/test.txt")

#clf = svm.SVC(kernel='linear')

#clf.fit(X_train, y_train)

#print (clf.score(X_test,y_test))             

#print("fit.n_features_:%d, Num Features" % (fit.n_features,))

#print('Num Features: %d') % fit.n_features_

#print("Selected Features: %s") % fit.support_

#print("Feature Ranking: %s") % fit.ranking

#data = pd.concat([Y,df.iloc[:,0:61]],axis=1)

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Put this when it's called

from sklearn.model_selection import train_test_split

from sklearn.model_selection import learning_curve

from sklearn.model_selection import validation_curve

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

def draw_missing_data_table(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data

df=pd.read_csv('../input/OnlineNewsPopularity.csv')

draw_missing_data_table(df)

df.dtypes

df.drop('url', axis=1, inplace=True)

df.head()

df = pd.get_dummies(df, drop_first=True)  # To avoid dummy trap

df.head()
