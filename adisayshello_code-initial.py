# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
import keras
from sklearn.ensemble import RandomForestClassifier
import csv
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
gender_data=pd.read_csv('../input/gender_submission.csv')
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
#Filling in for missing data in age
mean_age=train_data["Age"].mean()
std_age=train_data["Age"].std()
null_age_entries=train_data["Age"].isnull().sum()

generated_ages=np.random.randint(mean_age - std_age, mean_age + std_age, size = null_age_entries)

train_data["Age"][np.isnan(train_data["Age"])] = generated_ages
print(generated_ages)
gt = (train_data.values)[:,1]
data = np.delete(train_data.values,1,axis=1)

print(data[0:4,:])
print(data.shape)
cat_feature= data[:,3]
#print(cat_feature)
#np.delete(a, [2,3], axis=1)
lb = LabelBinarizer().fit(cat_feature)
cat_feature = lb.transform(cat_feature)
#Y_test = lb.transform(Y_test)
#Removing features with no information
data=np.delete(data,[2,7,9],axis=1)

#Label Encoding Sex,Embarked
emb=LabelEncoder().fit_transform(data[:,7].astype(str))
gend = LabelEncoder().fit_transform(data[:,2].astype(str))

#Replacing encoded data into the training data
data[:,2]=gend
data[:,7]=emb
#Printing Formatted Data
print(data[500:505,:])
print(data.shape)
print(gt.shape)
model = Sequential()
    
model.add(Dense(1000,input_shape=(8,),kernel_initializer='glorot_normal',activation='sigmoid'))
#model.add(Dense(10,kernel_initializer='glorot_normal',activation='sigmoid'))
#model.add(Dense(10,kernel_initializer='glorot_normal',activation='sigmoid'))
model.add(Dense(1,kernel_initializer='glorot_normal',activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer = 'adam',metrics = ['accuracy'])
print(model.summary())

model.fit(data, gt, validation_data=(data, gt), batch_size=4, epochs=100, verbose=1)
pred=model.predict(data)
print(np.round(pred,decimals=0))
print(gt)
#Filling in for missing data in age
mean_age=test_data["Age"].mean()
std_age=test_data["Age"].std()
null_age_entries=test_data["Age"].isnull().sum()

generated_ages=np.random.randint(mean_age - std_age, mean_age + std_age, size = null_age_entries)

test_data["Age"][np.isnan(test_data["Age"])] = generated_ages

print(test_data.iloc[0,:])
test_data=test_data.drop(['Name','Ticket','Cabin'],axis=1)
mean_fare = test_data["Fare"].mean()
std_fare=test_data["Fare"].std()
null_fare_entries=test_data["Fare"].isnull().sum()

generated_fare=np.random.randint(mean_fare - std_fare, mean_fare + std_fare, size = null_fare_entries)
test_data["Fare"][np.isnan(test_data["Fare"])] = generated_fare


test=test_data.values
#test=np.delete(test,[2,7,9],axis=1)

#Encoding Labels
#Label Encoding Sex,Embarked
emb=LabelEncoder().fit_transform(test[:,7].astype(str))
gend = LabelEncoder().fit_transform(test[:,2].astype(str))

#Replacing encoded data into the training data
test[:,2]=emb
test[:,7]=gend

result=model.predict(test)
#print(np.round(result))
pass_id=test_data.iloc[:,0].values
output=np.column_stack((pass_id,np.round(result)))
#print((output).astype(int))
np.savetxt('nn_output.csv',output,delimiter=',')
random_forest = RandomForestClassifier(n_estimators=100)


if((np.isnan(data.astype(int))).any == True ): print('Failed')
else: print('k')
if((np.isnan(gt.astype(int))).any == True ): print('Failed')
else: print('k')

random_forest.fit(data.astype(int), gt.astype(int))


Y_pred = random_forest.predict(test)
print(Y_pred)
#np.savetxt('forest_out.csv',Y_pred,delimiter=',')
consolidated_output= np.column_stack((pass_id,Y_pred))
frame=pd.DataFrame(consolidated_output,columns=['PassengerID','Survived'])
frame
output_to_write=frame.values
#np.savetxt('forest_out.csv',output_to_write,delimiter=',')
frame.to_csv('forest_out.csv', index=False, header=True, sep=',')
#print(data[0:5,:])

#print((test_data["Embarked"].isnull().sum()))
#print(test_data.iloc[0][:])
#test_data=np.delete(test_data,[2,7,9],axis=1)
print(test[0:10,:])
