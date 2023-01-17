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
from scipy.sparse.csr import csr_matrix
df_train , df_test = pd.read_csv("/kaggle/input/titanic/train.csv") , pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.head()
df_train["Embarked"].value_counts()
df_test.head()
df_train.info()
df_test.info()
corr_matrix = df_train.corr()

corr_matrix["Survived"]
useless_Features = ['Name' , 'Cabin' , 'PassengerId' , 'Ticket'] # Cabin has too many null values

df_train.drop(useless_Features,axis = 1,inplace=True) ; df_test.drop(useless_Features,axis = 1,inplace = True)
df_test.head(1)

df_train.head(1)
df_train.info()

df_train.dropna(subset =[ 'Embarked'] , axis = 0 ,inplace = True) 
df_test.dropna(subset = ['Embarked'] , axis = 0 ,inplace = True)
df_train.info()
df_train.head()
df_train.hist(figsize = (20,15))
print(df_test['Pclass'].value_counts)

df_train.columns
y_train =df_train['Survived'].copy()
x_num_train = df_train.loc[:,['Age' , 'Fare']]
x_num_test = df_test.loc[: , ['Age' , 'Fare']]
x_cat_train = df_train.drop(['Age' , 'Fare' , 'Survived'], axis = 1)
x_cat_test = df_test.drop(['Age' , 'Fare' ], axis = 1)
#imports needed for preprocessing
#x_num_train will get fitted by imputer to replace Nans in age
from sklearn.impute import SimpleImputer
#x_cat_train will go through OneHotEncoding , and Num hrough scaler
from sklearn.preprocessing import OneHotEncoder , StandardScaler
#Pipeline
from sklearn.pipeline import Pipeline

print(x_cat_train.shape , x_num_train.shape)
def preprocess(num_train ,num_test, cat_train , cat_test):
    imputer = SimpleImputer()
    scaler = StandardScaler()
    x_train = imputer.fit_transform(num_train)
    x_train = scaler.fit_transform(x_train)
    x_test = imputer.fit_transform(num_test)
    x_test = scaler.fit_transform(x_test)
    
    categorical = np.concatenate((cat_train,cat_test))
    one_hot = OneHotEncoder(sparse=False)
    x_cat = one_hot.fit_transform(categorical)
    
    
    return np.concatenate((x_train,x_cat[:num_train.shape[0],:]),axis = 1) , np.concatenate((x_test,x_cat[num_train.shape[0]:,:]),axis = 1)

X_train , X_test = preprocess(x_num_train,x_num_test,x_cat_train,x_cat_test)


assert( X_train.shape[1] == X_test.shape[1])

    
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate , cross_val_score


def model(X_train,y_train,X_test,epochs = 1000 , learning_rate = 0.005 ,model = 'linear',C=1.0 , kernel = 'linear'):
    
    
    if model == 'linear':
        classifier = LogisticRegression(max_iter= epochs)
    else:
        classifier = SVC(max_iter=epochs ,C=C  , kernel = kernel)
    
    classifier.fit(X_train,y_train)
    cv = cross_validate(classifier ,X_train,y_train,cv = 5)
    
    predictions = classifier.predict(X_test)
    
    print('Train set Accuracy: ', np.sum(cv['test_score'])/5)
    return predictions
    
    
pred_linear = model(X_train,y_train,X_test)
pred_svc = model(X_train,y_train,X_test , model='svc' , epochs = 2000 , kernel = 'poly' , C= 0.5)
ID = np.arange(892,892+418)
np.reshape(ID,(418,1))
np.reshape(pred_svc,(418,1))

np.vstack((ID,pred_svc))
my_submission = pd.DataFrame({'Id':ID , 'Survived':pred_svc})
my_submission.to_csv('submission.csv',index = False)
