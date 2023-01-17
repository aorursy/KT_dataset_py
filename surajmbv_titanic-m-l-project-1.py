# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import  LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data= pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

test_data= pd.read_csv("/kaggle/input/titanic/test.csv")
#checking missing values
test_data.isnull().sum()
#to avoid skewing distribution we are feature engineering Parch and SibSp to cube root since this right skewed.
    #in train data
train_data['Parch_mod'] = train_data['Parch'].apply(lambda x: np.cbrt(x))
train_data['SibSp_mod'] = train_data['SibSp'].apply(lambda x: np.cbrt(x))
    #in test data
test_data['Parch_mod'] = test_data['Parch'].apply(lambda x: np.cbrt(x))
test_data['SibSp_mod'] = test_data['SibSp'].apply(lambda x: np.cbrt(x))

#working with missing values 
#predicting age which has null values
#creating df which has no null values
data_without_null = train_data[['Survived','Pclass','Sex','Parch_mod','SibSp_mod','Fare','Embarked','Age']].dropna()
data_without_null['Sex'] = data_without_null['Sex'].apply(lambda x : 1 if x =='male' else 0 )
data_without_null['Embarked'] = data_without_null['Embarked'].apply(lambda x : 1 if x =='S' else ( 2 if x=='C' else 3))

#creating df which has age as null values
data_with_null = train_data.loc[:,['Survived','Pclass','Sex','Parch_mod','SibSp_mod','Fare','Embarked','Age']]

#dropping embarked na 
data_with_null = data_with_null.dropna(subset=['Embarked'])

data_with_null['Sex'] = data_with_null['Sex'].apply(lambda x : 1 if x =='male' else 0 )
data_with_null['Embarked'] = data_with_null['Embarked'].apply(lambda x : 1 if x =='S' else ( 2 if x=='C' else 3))

miss_X_train = data_without_null.iloc[:,:7]
miss_y_train = data_without_null.iloc[:,7]
miss_X_test  = data_with_null.iloc[:,:7] 

#fiting with available data
clf = LinearRegression().fit(miss_X_train,miss_y_train)
agepred = pd.DataFrame(clf.predict(miss_X_test),columns=['Age'])
data_with_null['Age'].fillna(agepred['Age'],inplace=True)
cleaned_train_data = data_with_null
data_with_null.isnull().sum()
#normalizing cleaned train_data
cleaned_train_data.head()
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(data_without_null.iloc[:,1:8])
y_train = data_without_null.iloc[:,0]

#transforming test data
test_data_trans = test_data.loc[:,['Pclass','Sex','Parch_mod','SibSp_mod','Fare','Embarked','Age']]
test_data_trans['Sex'] = test_data_trans['Sex'].apply(lambda x : 1 if x =='male' else 0 )
test_data_trans['Embarked'] = test_data_trans['Embarked'].apply(lambda x : 1 if x =='S' else ( 2 if x=='C' else 3))

#handling missing values in test data
test_data_trans['Age'].replace(np.NaN, test_data_trans['Age'].mean(),inplace=True)
test_data_trans['Fare'].replace(np.NaN, test_data_trans['Fare'].mean(),inplace=True)
X_test_scaled  = scaler.transform(test_data_trans)


#fitting data using random forrest
clf_2 = RandomForestClassifier(n_estimators=200,max_depth=5,max_features=6, random_state=0, min_samples_leaf = 2)
clf_2.fit(X_train_scaled, y_train)
y_pred=clf_2.predict(X_test_scaled)
y=pd.DataFrame(y_pred,index=test_data['PassengerId'],columns=['Survived'])
y.reset_index(inplace=True)
y.to_csv('Titanic.csv',index=False)
print("Success")

sample_leaf_options = [1,2,3,4,5]
maximum_features =[1,2,3,4,5,6]
for leaf_size in sample_leaf_options :
    for features in maximum_features:
        model = RandomForestClassifier(n_estimators = 200, n_jobs = -1,random_state =0, min_samples_leaf = leaf_size, max_features=features)
        model.fit(X_train_scaled, y_train)
        print("\n Leaf size :", leaf_size)
        print("Max Feature :",features )
        print ("AUC - ROC : ", roc_auc_score(y_train,model.predict(X_train_scaled)))
        print ("Accuracy : ", clf_2.score(X_train_scaled,y_train))
        
