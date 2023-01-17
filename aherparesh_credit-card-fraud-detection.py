# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/abstract-data-set-for-credit-card-fraud-detection/creditcardcsvpresent.csv')

data.head()
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
datanew = reduce_mem_usage(data)
datanew.drop(['Transaction date'], axis = 1, inplace = True)
datanew.describe()
df= datanew.corr()

df.head()
heat_map = sb.heatmap(df)
datanew.columns
data.head()
dataf=datanew.dropna()
datanew.isnull()


dataf=pd.get_dummies(datanew)

dataf
dataf.columns

X= dataf.iloc[:,:14]

Y=dataf[['isFradulent_Y']]
X.shape
Y.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state = 101, test_size=0.3)
from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer(max_features=1000, binary=True)



X_train_vect = vect.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

model.fit(X_train, y_train)

model.score(X_train, y_train)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predicted))

print('\n')

print(classification_report(y_test, predicted))

model.score(X_train,y_train)
from sklearn.ensemble import GradientBoostingClassifier

model1= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

model1.fit(X_train, y_train)

predicted= model1.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predicted))

print('\n')

print(classification_report(y_test, predicted))

model1.score(X_train,y_train)
from sklearn.naive_bayes import MultinomialNB



nb = MultinomialNB()



nb.fit(X_train, y_train)



nb.score(X_train, y_train)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predicted))

print('\n')

print(classification_report(y_test, predicted))

nb.score(X_train,y_train)
TestValue=dataf.iloc[0,:13]

TestValue