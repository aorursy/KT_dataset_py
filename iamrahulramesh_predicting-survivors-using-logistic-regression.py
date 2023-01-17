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
import numpy as np

import pandas as pd

import seaborn as sb



import matplotlib.pyplot as plt

from pandas import Series,DataFrame

from pylab import rcParams

import sklearn

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_predict



from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score,recall_score
%matplotlib inline

rcParams['figure.figsize']= 5,4

sb.set_style('whitegrid')
titanic_training =pd.read_csv('../input/titanic/train.csv')



titanic_training.head()
print(titanic_training.info())
sb.countplot(x="Survived",data = titanic_training,palette ='hls')
titanic_training.isnull().sum()
titanic_training.describe()
titanic_training.columns
titanic_data = titanic_training.drop(['Name','Ticket','Cabin'],axis =1)

titanic_data.head()
sb.boxplot(x='Parch',y='Age',data = titanic_data,palette = 'hls')
parch_group = titanic_data.groupby(titanic_data['Parch'])

pd.DataFrame(parch_group)
parch_group.mean()
def age_approx(cols):

    Age = cols[0]

    Parch =cols[1]

    if pd.isnull(Age):

        if Parch == 0:

            return 32

        elif Parch == 1:

            return 24

        elif Parch == 2:

            return 17

        elif Parch == 3:

            return 33

        elif Parch == 4:

            return 44

        elif Parch == 5:

            return 39

        elif Parch == 6:

            return 43

        else:

            return 30

    else:

        return Age
titanic_data['Age'] = titanic_data[['Age','Parch']].apply(age_approx,axis = 1)

titanic_data.isnull().sum()
titanic_data.dropna(inplace = True)

titanic_data.reset_index(inplace = True,drop = True)

titanic_data.info()
titanic_data.head()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

gender_cat = titanic_data['Sex']

gender_encoded = label_encoder.fit_transform(gender_cat)

gender_encoded[0:5]
gender_df= pd.DataFrame(gender_encoded,columns = ['Male_gender'])

gender_df.head()
embarked_cat = titanic_data['Embarked']

embarked_encoded = label_encoder.fit_transform(embarked_cat)

embarked_encoded[0:100]
from sklearn.preprocessing import OneHotEncoder

binary_encoder = OneHotEncoder(categories = 'auto')

embarked_1hot = binary_encoder.fit_transform(embarked_encoded.reshape(-1,1))

embarked_1hot_mat =embarked_1hot.toarray()

embarked_df = pd.DataFrame(embarked_1hot_mat,columns = ['C','Q','S'])

embarked_df.head()
titanic_data.drop(['Sex','Embarked'],inplace = True,axis = 1)

titanic_data.head()
titanic_dmy = pd.concat([titanic_data,gender_df,embarked_df],verify_integrity = True,axis =1).astype(float)

titanic_dmy.head()
sb.heatmap(titanic_dmy.corr())
titanic_dmy.drop(['Pclass','Fare'],axis =1,inplace = True)

titanic_dmy.head()
titanic_dmy.info()
X_train,X_test,y_train,y_test = train_test_split(titanic_dmy.drop(['Survived'],axis=1),

                                                 titanic_dmy['Survived'],test_size =0.2,random_state =200)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
pd.DataFrame(X_train)
pd.DataFrame(y_train)
logreg = LogisticRegression(solver = 'liblinear')

logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test,y_pred))
y_train_pred = cross_val_predict(logreg,X_train,y_train,cv = 5)

confusion_matrix(y_train,y_train_pred)
precision_score(y_train,y_train_pred)
titanic_dmy[863:864]
test_passenger = np.array([866,42,0,0,0,0,0,1]).reshape(1,-1)

print(logreg.predict(test_passenger))

print(logreg.predict_proba(test_passenger))
titanic_dmy[364:365]
test_passenger_2 = np.array([366,30,0,0,1,0,0,1]).reshape(1,-1)

print(logreg.predict(test_passenger_2))

print(logreg.predict_proba(test_passenger_2))