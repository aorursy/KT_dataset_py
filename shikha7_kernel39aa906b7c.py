# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

df=pd.read_csv('/kaggle/input/horse-colic/horse.csv')

df=df.drop(['nasogastric_reflux_ph','abdomo_protein','abdomo_appearance','abdomen','lesion_2','lesion_3','cp_data'],axis=1)

df["outcome"]= df["outcome"].replace('euthanized', 'died') 



#treating NaN

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):



    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]



            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],



            index=X.columns)



        return self



    def transform(self, X, y=None):



        return X.fillna(self.fill)





xt = DataFrameImputer().fit_transform(df)



#encoding

label_encoder = LabelEncoder()

xt.iloc[:,0] = label_encoder.fit_transform(xt.iloc[:,0]).astype('float64')

xt.iloc[:,1] = label_encoder.fit_transform(xt.iloc[:,1]).astype('float64')

xt.iloc[:,6] = label_encoder.fit_transform(xt.iloc[:,6]).astype('float64')

xt.iloc[:,7] = label_encoder.fit_transform(xt.iloc[:,7]).astype('float64')

xt.iloc[:,8] = label_encoder.fit_transform(xt.iloc[:,8]).astype('float64')

xt.iloc[:,9] = label_encoder.fit_transform(xt.iloc[:,9]).astype('float64')

xt.iloc[:,10] = label_encoder.fit_transform(xt.iloc[:,10]).astype('float64')

xt.iloc[:,11] = label_encoder.fit_transform(xt.iloc[:,11]).astype('float64')

xt.iloc[:,12] = label_encoder.fit_transform(xt.iloc[:,12]).astype('float64')

xt.iloc[:,13] = label_encoder.fit_transform(xt.iloc[:,13]).astype('float64')

xt.iloc[:,14] = label_encoder.fit_transform(xt.iloc[:,14]).astype('float64')

xt.iloc[:,15] = label_encoder.fit_transform(xt.iloc[:,15]).astype('float64')

xt.iloc[:,19] = label_encoder.fit_transform(xt.iloc[:,19]).astype('float64')





X = xt.drop("outcome",axis=1)

y = xt["outcome"] 





X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))



from sklearn.ensemble import RandomForestClassifier

clf1=RandomForestClassifier(max_features=18,random_state=0)

clf1.fit(X_train,y_train)

y_pred1 = clf1.predict(X_test)

print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(clf1.score(X_test, y_test)))



from sklearn.ensemble import GradientBoostingClassifier

clf2=GradientBoostingClassifier(random_state=0)

clf2.fit(X_train,y_train)

y_pred2 = clf2.predict(X_test)

print('Accuracy ofGradient Boosting classifier on test set: {:.2f}'.format(clf2.score(X_test, y_test)))