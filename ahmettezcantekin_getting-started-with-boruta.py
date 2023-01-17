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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df=pd.read_csv('/kaggle/input/beginner-dataset-v2/beginner_level_dataset_2/census_income.csv', sep=';')
df.head(5)
#Information about the dataset

df.info()
#Replace question marks with nan value

df[df == '?'] = np.nan
df.info()
# There are missing values in some columns

df.isnull().sum()
#Column names in the dataset

df.columns
#Strip spaces in the dataset

df.rename(columns=lambda x: x.strip(),inplace=True)
#Impute missing values with mode

for col in ['sex', 'capital-gain', 'capital-loss','hours-per-week','native-country','income' ]:

    df[col].fillna(df[col].mode()[0], inplace=True)
df.isnull().sum()
categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

df[categorical] = df[categorical].astype(str)
class MultiColumnLabelEncoder:

    def __init__(self,columns = None):

        self.columns = columns # array of column names to encode



    def fit(self,X,y=None):

        return self # not relevant here



    def transform(self,X):

        '''

        Transforms columns of X specified in self.columns using

        LabelEncoder(). If no columns specified, transforms all

        columns in X.

        '''

        output = X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output



    def fit_transform(self,X,y=None):

        return self.fit(X,y).transform(X)
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

df=MultiColumnLabelEncoder(columns = categorical).fit_transform(df)
X = df.drop(['income'], axis=1)

y = df['income']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)



X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print('Logistic Regression accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)



print('Random Forest accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy

from datetime import datetime

import pandas as pd
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
X = df.drop(['income'], axis=1).values

y = df['income'].values
rfc = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced', max_depth=6)

boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)

start_time = timer(None)

boruta_selector.fit(X, y)

timer(start_time)
# number of selected features

print ('\n Number of selected features:')

print (boruta_selector.n_features_)
feature_df = pd.DataFrame(df.drop(['income'], axis=1).columns.tolist(), columns=['features'])

feature_df['rank']=boruta_selector.ranking_

feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)

print ('\n Top %d features:' % boruta_selector.n_features_)

feature_df2=feature_df.head(boruta_selector.n_features_)

feature_df2.shape
feature_list=feature_df2['features'].to_list()

feature_list
feature_list.append('income')
df_boruta=df[feature_list]
df_boruta.info()
df_boruta.head()
X = df_boruta.drop(['income'], axis=1)

y = df_boruta['income']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)



X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print('Logistic Regression accuracy score with Boruta features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)



print('Random Forest accuracy score with Boruta features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))