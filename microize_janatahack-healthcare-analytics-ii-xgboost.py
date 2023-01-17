import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder



import xgboost as XGB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('../input/av-janatahack-healthcare-hackathon-ii/Data/train.csv',index_col=['case_id'])

test=pd.read_csv('../input/av-janatahack-healthcare-hackathon-ii/Data/test.csv',index_col=['case_id'])

train.shape,test.shape
# Number of Null Values

display(train.isna().sum())

display(test.isna().sum())



# Unique Values in Each Column

display(train.nunique())

display(test.nunique())
# Viewing 1st five rows of training dataset

train.head()
# displaying Unique Values in each columns

for col in train.columns:

    print(col,":",train[col].unique())

    print("___________________________")

print("  ")    

print("  ")

print("*****************************")

print("*****************************")

print("*****************************")

print("  ")    

print("  ")

for col in test.columns:

    print(col,":",test[col].unique())

    print("___________________________")
# Selecting Categorical Columns for Label Encoding

cat_train=train.select_dtypes(include=['object'])

cat_col=list(cat_train.columns)

cat_col_test=['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']
# Label encoding

le=LabelEncoder()

for cat_column in cat_col_test:

    train[cat_column]=le.fit_transform(train[cat_column])

    test[cat_column]=le.transform(test[cat_column])

train['Stay']=le.fit_transform(train['Stay'])

print(le.inverse_transform([0,1,2,3,4,5,6,7,8,9,10,]))
# Datatype of Train Dataset

train.info()
X_train[0]
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.optimizers import Adam



# intializing X and Y 

X=train.drop(['Stay'],axis=1)

y=train['Stay']



# train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



X_train=X_train.to_numpy()



model=Sequential()

model.add(Dense(1200,activation='tanh',input_shape=(16,)))

model.add(Dropout(0.4))

model.add(Dense(600,activation='tanh'))

model.add(Dropout(0.4))

model.add(Dense(300,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(11,activation='softmax'))



model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=5,validation_split=0.2)
test.shape
# intializing X and Y 

X=train.drop(['Stay'],axis=1)

y=train['Stay']



# train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



# XGB Classifier

clf=XGB.XGBClassifier(learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,)

clf.fit(X_train,y_train)



# Predict the values

y_predict=clf.predict(X_test)

test['Stay']=clf.predict(test)



test['Stay1']=test['Stay']



test['Stay']=test['Stay'].map({0:'0-10',1:'11-20',2:'21-30',3:'31-40',4:'41-50',5:'51-60',6:'61-70',7:'71-80',8:'81-90',9:'91-100',10:'More than 100 Days'})



# Accuracy Score

acc=accuracy_score(y_test,y_predict)

print(acc)
# output file

test['Stay'].head()

test['Stay'].to_csv('output.csv')