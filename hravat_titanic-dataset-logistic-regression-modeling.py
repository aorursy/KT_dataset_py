##Possible improvments
##New feature for median age on Sex Pclass and title for missing age values
##Inclusion of fare in featured add mean care value
##missing emarked value with highest embarked frequencies
##Map first letter of cabin and use label encoder for it missing values areeassigned U
## use matplotlib to plot survived with cabin
## preprocess ticket extract ticket prefix of three letters
## if not present use XXX
#### Check If Logistic regresssion CV can be used

## Data Processing 
import numpy as np 
import scipy as sp
import pandas as pd
import math
## read data set
df_titanic_train=pd.read_csv('../input/train.csv')
df_titanic_train
df_titanic_train_mod=df_titanic_train
df_titanic_train_mod
## Assign numbers 
df_titanic_train_mod['Sex'].replace('female',0,inplace=True)
df_titanic_train_mod['Sex'].replace('male',1,inplace=True) 
df_titanic_train_mod['Embarked'].replace('S',1,inplace=True) 
df_titanic_train_mod['Embarked'].replace('C',2,inplace=True)
df_titanic_train_mod['Embarked'].replace('Q',3,inplace=True) 
##df_titanic_train_mod['Title'] = np.nan
##avg_fare=df_titanic_train_mod['Fare'].mean()
##print(avg_fare)
##df_titanic_train_mod
##df_titanic_train_mod['Fare'].replace(NaN,avg_fare,inplace=True)

for index,rows in df_titanic_train_mod.iterrows():
    start_pos=int(rows['Name'].index(','))
    end_pos=int(rows['Name'].index('.'))
    title=rows['Name'][start_pos+2:end_pos]
    df_titanic_train_mod.loc[index,'Title'] = title.strip()

df_titanic_train_mod    
    
from sklearn import preprocessing    

le = preprocessing.LabelEncoder()
le.fit(df_titanic_train_mod['Title'])
list(le.classes_)
df_titanic_train_mod['Title']=le.transform(df_titanic_train_mod['Title'])

df_titanic_train_mod

X =df_titanic_train_mod.loc[:,['Pclass','Sex','Embarked','Age','Title','SibSp','Parch']]
y =df_titanic_train_mod.loc[:,['Survived']]

## Prepare featurer and label matrices
X_NumPy_Mat= X.as_matrix()
Y_NumPy_Mat= y.as_matrix()
np.shape(X_NumPy_Mat)
np.shape(Y_NumPy_Mat)

##Build Logistic Regression Model with sk learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score





## Handle Missing values
my_imputer = Imputer()
X_NumPy_Mat_NM = my_imputer.fit_transform(X_NumPy_Mat)
np.shape(X_NumPy_Mat_NM)

scaler = StandardScaler()
X_scale = scaler.fit_transform(X_NumPy_Mat_NM)
np.shape(X_scale)

##Split the set into training and test set
X_scale_train,X_scale_test,Y_scale_train,Y_scale_test = train_test_split(X_scale,Y_NumPy_Mat,test_size=0.2,random_state=78)


##Display size of original training and test set 
np.shape(X_NumPy_Mat_NM)
np.shape(X_scale_train)
np.shape(X_scale_test)
np.shape(X)

##Develop the model
C_list=[0.0001,0.0003,0.001,0.003,0.01,0.01,0.1,0.3,1,3]
accuracy=0
c_final=0
accuracy_1=0

for c in C_list:
    model = LogisticRegression(penalty='l2',C=c,solver='liblinear')
    Y_scale_train_ravel=Y_scale_train.ravel()
    model.fit(X_scale_train,Y_scale_train_ravel)
    Y_scale_test_ravel=Y_scale_test.ravel()
    accuracy_1=accuracy_score(Y_scale_test_ravel,model.predict(X_scale_test))
    if  accuracy_1 > accuracy:  
        accuracy=accuracy_1                     
        c_final=c

    print ('C is',c,'Accuracy is ',accuracy_score(Y_scale_test_ravel,model.predict(X_scale_test)))


print(accuracy)
print(c_final)                                


####Pick max accuracy

model = LogisticRegression(penalty='l2',C=c_final,solver='liblinear')
Y_scale_train_ravel=Y_scale_train.ravel()
model.fit(X_scale_train,Y_scale_train_ravel)
    
                                
y_predict=model.predict(X_scale_test)
y_predict


##Preperation of test data


df_titanic_test=pd.read_csv('../input/test.csv')
df_titanic_test
df_titanic_test_mod=df_titanic_test
df_titanic_test_mod
## Assign numbers 
df_titanic_test_mod['Sex'].replace('female',0,inplace=True)
df_titanic_test_mod['Sex'].replace('male',1,inplace=True) 
df_titanic_test_mod['Embarked'].replace('S',1,inplace=True) 
df_titanic_test_mod['Embarked'].replace('C',2,inplace=True)
df_titanic_test_mod['Embarked'].replace('Q',3,inplace=True) 
df_titanic_train_mod["Title"] = np.nan



for index,rows in df_titanic_test_mod.iterrows():
    start_pos=int(rows['Name'].index(','))
    end_pos=int(rows['Name'].index('.'))
    title=rows['Name'][start_pos+2:end_pos]
    df_titanic_test_mod.loc[index,'Title'] = title.strip()

df_titanic_test_mod    

le.fit(df_titanic_test_mod['Title'])
df_titanic_test_mod['Title']=le.transform(df_titanic_test_mod['Title'])



df_titanic_test_mod
X_test =df_titanic_test_mod.loc[:,['Pclass','Sex','Embarked','Age','Title','SibSp','Parch']]
X_test

##Making predictions 
##Handle Missing values on the pridctct 
X_predict_NM = my_imputer.fit_transform(X_test)
X_predict_NM
X_predict_scale = scaler.fit_transform(X_predict_NM)
X_predict_scale
y_predict_test=model.predict(X_predict_scale)
##Disply shapes of 
np.shape(y_predict_test)
np.shape(X_predict_scale)
df_titanic_test_mod
df_y_predict=pd.DataFrame(y_predict_test)
df_y_predict=df_y_predict.rename(columns = {0:'Survived'})
df_y_predict
final_predict=df_titanic_test_mod.join(df_y_predict)
final_predict_new = final_predict[['PassengerId','Survived']]
final_predict_new 

final_predict_new.to_csv('hravat_titanic_pred_titl.csv',sep=',')