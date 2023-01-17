import pandas as pd
dataset_train=pd.read_csv('../input/train.csv')
dataset_test=pd.read_csv('../input/test.csv')
dataset_train.head()
dataset_test.head()
dataset_train.isnull().values.any()
dataset_test.isnull().values.any()
dataset_train.info()
dataset_test.info()
dataset_train.describe()
dataset_test.describe()
dataset_train.shape
dataset_test.shape
dataset_train.head()
X=dataset_train.iloc[:,1:-1].values
X.shape
y=dataset_train.iloc[:,-1].values
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0)
X_train.shape
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=54,random_state=101,min_samples_split=4,criterion='entropy')
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
rf.score(X_test,y_test)
dataset_test.head()
X_new=dataset_test.iloc[:,1:].values
X_new.shape
X_new=sc.transform(X_new)
y_new=rf.predict(X_new)
y_new.shape
upload=pd.DataFrame(y_new,dataset_test['Id'])
upload.head()
upload.to_csv('Submisssion_new.csv')





