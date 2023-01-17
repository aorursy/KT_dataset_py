# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler,LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load data, view columns

data = pd.read_csv('../input/HR_comma_sep.csv')

data.columns



#no null,nice

data.isnull().sum()



#Checking for enumerable features

data['sales'].unique()

data['left'].unique()

data['salary'].unique()



data['sales'] = LabelEncoder().fit_transform(data['sales'])

data['left'] = LabelEncoder().fit_transform(data['left'])

data['salary'] = LabelEncoder().fit_transform(data['salary'])



data.head()
data = data.sample(frac=1).reset_index(drop=True)

X = data.drop(['left'],axis=1).values

y = data['left'].values



X = MinMaxScaler().fit_transform(X)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=2)

X_train.shape
rf = RandomForestClassifier()

rf.fit(X_train,y_train)

rep = classification_report(y_test,rf.predict(X_test))

rep
#Checking the feature importances

rf.feature_importances_

#Try taking first 5

X_train,X_test,y_train,y_test = train_test_split(X[:,1:5],y,test_size=0.25,random_state=2)

rf.fit(X_train,y_train)

rep = classification_report(y_test,rf.predict(X_test))

rep
#F1 drops with PCA

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=2)

from sklearn.decomposition import PCA

pca = PCA(n_components=6)

Xr_train = pca.fit_transform(X_train)

Xr_test = pca.fit_transform(X_test)

rf = RandomForestClassifier()

rf.fit(Xr_train,y_train)

rep = classification_report(y_test,rf.predict(Xr_test))

rep