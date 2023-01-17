import pandas as pd
#reading the dataset
dataset = pd.read_csv('../input/data_fin.csv')

X_var = dataset.iloc[:,3:17].values
Y_var = dataset.iloc[:,18:19].values
Xdf = dataset.iloc[:,3:17]
Ydf = dataset.iloc[:,18:19]

#converting the quantitative variables(input features alone) to catogorical variables
from sklearn.model_selection import train_test_split
#reading the whole dataset by giving the train_size as 0.99
Xtrain_samp,Xtest_samp,Ytrain_samp,Ytest_samp = train_test_split(X_var,Y_var,train_size=0.99,random_state=42)

X=Xtrain_samp
perp_age = Ytrain_samp


Xcate1=[0,1,3,5,6,7,9,10,11]
Xcate2=[0,1,3,9,10,11]

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in Xcate1:
    label=LabelEncoder()
    #i=int(i)
    X[:,i]=label.fit_transform(X[:,i])

temp=[]    
for i in Xcate2:
    temp.append(max(X[:,i:i+1]))

maxs=[]  
for i in range (0,len(temp)):
    maxs.append(temp[i][0])

one = OneHotEncoder(categorical_features=[0,1,3,9,10,11])
X =  one.fit_transform(X).toarray()


indep_var = pd.DataFrame(data=X[0:,0:])
#print(indep_var)

del_v = [0,40,60,66,69,80]

for i in del_v:
    del indep_var[i]

# Split the data for training and testing
from sklearn.model_selection import train_test_split
split_test_size = 0.10
x_train,x_test,y_train,y_test = train_test_split(indep_var,perp_age,test_size=split_test_size,random_state = 42)

#Preprocessing the data ( Removing the string variables in my target variable perpetrator age)
da1 = []
da2 = []
import numpy as np
for i in range(0,np.shape(y_train)[0]):
    
    da1.append(y_train[i][0])
    
for i in range(0,np.shape(da1)[0]):
    if da1[i] == ' ':
        da1[i] = 0
        

for i in range(0,np.shape(da1)[0]):
    da2.append(int(da1[i]))
    
# Applying MultiLinear Regression to predict the perpetrator Age
from sklearn import  linear_model
from sklearn.metrics import  r2_score
regr = linear_model.LinearRegression()
regr.fit(x_train,da2)
y_pred = regr.predict(x_test)
print('The accuracy of predicting perpetrator Age is : %.2f' % r2_score(y_test,y_pred))
    


import pandas as pd

dataset = pd.read_csv('../input/data_fin.csv')

#always dataframe starts with 0
X = dataset.iloc[:,3:17].values
Y = dataset.iloc[:,17:18].values
Xdf = dataset.iloc[:,3:17]
Ydf = dataset.iloc[:,17:18] # perpetrator sex

from sklearn.model_selection import train_test_split
Xtrain_samp,Xtest_samp,Ytrain_samp,Ytest_samp = train_test_split(X,Y,train_size=0.99,random_state=42)

X=Xtrain_samp
perp_sex=Ytrain_samp


Xcate1=[0,1,3,5,6,7,9,10,11]
Xcate2=[0,1,3,9,10,11]

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in Xcate1:
    label=LabelEncoder()
    i=int(i)
    X[:,i]=label.fit_transform(X[:,i])

temp=[]    
for i in Xcate2:
    temp.append(max(X[:,i:i+1]))

maxs=[]  
for i in range (0,len(temp)):
    maxs.append(temp[i][0])


one = OneHotEncoder(categorical_features=[0,1,3,9,10,11])
X =  one.fit_transform(X).toarray()


indep_var = pd.DataFrame(data=X[0:,0:])

del_v = [0,40,60,66,69,80]

for i in del_v:
    del indep_var[i]

# target variable ( perpetrator gender) conversion(Quantitative to catagorical)

Ycate1 = [0]
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in Ycate1:
    label=LabelEncoder()
    i=int(i)
    perp_sex[:,i]=label.fit_transform(perp_sex[:,i])

temp=[]    
for i in Ycate1:
    temp.append(max(perp_sex[:,i:i+1]))
#starts with 0
maxs=[]  
for i in range (0,len(temp)):
    maxs.append(temp[i][0])


one = OneHotEncoder(categorical_features=[0])
perp_sex =  one.fit_transform(perp_sex).toarray()


dep_var = pd.DataFrame(data=perp_sex[0:,0:])
del dep_var[0]


split_test_size = 0.20
x_train,x_test,y_train,y_test = train_test_split(indep_var,dep_var,test_size=split_test_size,random_state = 42)

print('Predicting the Perpetrator Gender using K-neighbors Classifier.')
print('This Might take few minutes...')
# k nearest neighbour

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=5)
knc.fit(x_train,y_train)
y_pred= knc.predict(x_test)

from sklearn.metrics import accuracy_score
print('The accuracy of predicting perpetrator Gender is : %.2f' % accuracy_score(y_test, y_pred))


#PERFORMANCE METRICS

from sklearn.metrics import accuracy_score , roc_auc_score , f1_score , recall_score , precision_score
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred)
f1_score(y_test, y_pred, average='weighted')  
recall_score(y_test, y_pred, average='macro') 
precision_score(y_test, y_pred, average='macro')  

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, y_pred)

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='macro')

#Loss metrics
from sklearn.metrics import hamming_loss
print('The Loss in predicting perpetrator Gender is : %.2f' % hamming_loss(y_test, y_pred))

from sklearn.metrics import log_loss
log_loss(y_test, y_pred)

from sklearn.metrics import zero_one_loss
zero_one_loss(y_test, y_pred)
import pandas as pd
dataset = pd.read_csv('../input/data_fin.csv')

#always dataframe starts with 0
X = dataset.iloc[:,3:17].values
Y = dataset.iloc[:,19:].values
Xdf = dataset.iloc[:,3:17]
Ydf = dataset.iloc[:,19:] # relationship

from sklearn.model_selection import train_test_split
Xtrain_samp,Xtest_samp,Ytrain_samp,Ytest_samp = train_test_split(X,Y,train_size=0.99,random_state=42)

X=Xtrain_samp
rel=Ytrain_samp


Xcate1=[0,1,3,5,6,7,9,10,11]
Xcate2=[0,1,3,9,10,11]

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in Xcate1:
    label=LabelEncoder()
    i=int(i)
    X[:,i]=label.fit_transform(X[:,i])

temp=[]    
for i in Xcate2:
    temp.append(max(X[:,i:i+1]))

maxs=[]  
for i in range (0,len(temp)):
    maxs.append(temp[i][0])


one = OneHotEncoder(categorical_features=[0,1,3,9,10,11])
X =  one.fit_transform(X).toarray()


indep_var = pd.DataFrame(data=X[0:,0:])


del_v = [0,40,60,66,69,80]

for i in del_v:
    del indep_var[i]


# target variable ( perpetrator relationship) conversion

Ycate1 = [0]
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in Ycate1:
    label=LabelEncoder()
    i=int(i)
    rel[:,i]=label.fit_transform(rel[:,i])

temp=[]    
for i in Ycate1:
    temp.append(max(rel[:,i:i+1]))
#starts with 0
maxs=[]  
for i in range (0,len(temp)):
    maxs.append(temp[i][0])


one = OneHotEncoder(categorical_features=[0])
rel =  one.fit_transform(rel).toarray()


dep_var = pd.DataFrame(data=rel[0:,0:])
del dep_var[0]


split_test_size = 0.30

x_train,x_test,y_train,y_test = train_test_split(indep_var,dep_var,test_size=split_test_size,random_state = 42)

print('Predicting the Perpetrator Relationship using K-neighbors Classifier.')
print('This Might take few minutes...')

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=25)
knc.fit(x_train,y_train)
y_pred= knc.predict(x_test)

from sklearn.metrics import accuracy_score
print('The accuracy of predicting perpetrator Gender is : %.2f' % accuracy_score(y_test, y_pred))




from sklearn.metrics import accuracy_score , roc_auc_score , f1_score , recall_score , precision_score
accuracy_score(y_test, y_pred)
roc_auc_score(y_test, y_pred)
f1_score(y_test, y_pred, average='weighted')  
recall_score(y_test, y_pred, average='micro') 
precision_score(y_test, y_pred, average='weighted')  

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, y_pred)

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1','class 2', 'class 3','class 4',
                'class 5','class 6','class 7','class 8','class 9','class 10',
                'class 11','class 12','class 13','class 14','class 15','class 16',
                'class 17','class 18','class 19','class 20','class 21','class 22',
                'class 23','class 24','class 25','class 26']
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='macro')

#Loss
from sklearn.metrics import hamming_loss
print('The Loss in predicting perpetrator relationship is : %.2f' % hamming_loss(y_test, y_pred))

from sklearn.metrics import log_loss
log_loss(y_test, y_pred)

from sklearn.metrics import zero_one_loss
zero_one_loss(y_test, y_pred)

import pandas as pd
dataset = pd.read_csv('../input/data_fin.csv')

#always dataframe starts with 0
X = dataset.iloc[:,3:17].values
Y = dataset.iloc[:,17:18].values
Xdf = dataset.iloc[:,3:17]
Ydf = dataset.iloc[:,17:18] # perpetrator sex

from sklearn.model_selection import train_test_split
Xtrain_samp,Xtest_samp,Ytrain_samp,Ytest_samp = train_test_split(X,Y,train_size=0.99,random_state=42)

X=Xtrain_samp
perp_sex=Ytrain_samp


Xcate1=[0,1,3,5,6,7,9,10,11]
Xcate2=[0,1,3,9,10,11]

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in Xcate1:
    label=LabelEncoder()
    i=int(i)
    X[:,i]=label.fit_transform(X[:,i])

temp=[]    
for i in Xcate2:
    temp.append(max(X[:,i:i+1]))

maxs=[]  
for i in range (0,len(temp)):
    maxs.append(temp[i][0])


one = OneHotEncoder(categorical_features=[0,1,3,9,10,11])
X =  one.fit_transform(X).toarray()


indep_var = pd.DataFrame(data=X[0:,0:])

del_v = [0,40,60,66,69,80]

for i in del_v:
    del indep_var[i]

# target variable conversion

Ycate1 = [0]
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in Ycate1:
    label=LabelEncoder()
    i=int(i)
    perp_sex[:,i]=label.fit_transform(perp_sex[:,i])

temp=[]    
for i in Ycate1:
    temp.append(max(perp_sex[:,i:i+1]))
#starts with 0
maxs=[]  
for i in range (0,len(temp)):
    maxs.append(temp[i][0])


one = OneHotEncoder(categorical_features=[0])
perp_sex =  one.fit_transform(perp_sex).toarray()


dep_var = pd.DataFrame(data=perp_sex[0:,0:])
del dep_var[0]


split_test_size = 0.30

x_train,x_test,y_train,y_test = train_test_split(indep_var,dep_var,test_size=split_test_size,random_state = 42)

# Converting the train and test data into matrices 
X_train = x_train.as_matrix(columns=None)
Y_train = y_train.as_matrix(columns=None)
X_test = x_test.as_matrix(columns=None)
Y_test = y_test.as_matrix(columns=None)

# Applying Neural Networks using ( Keras)
from keras.models import Sequential

model = Sequential()
from keras.layers import Dense,Flatten

#input layer
model.add(Dense(units=98, activation='sigmoid', input_dim=98))


model.add(Dense(18, activation='sigmoid'))


#output layer
model.add(Dense(units=2, activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=19, batch_size=200)
model.predict(X_test)
score = model.evaluate(X_test, Y_test,verbose=1)

print('The Loss and Accuracy  predicting perpetrator Gender using Neural Networks is :')

print(score)

import pandas as pd
dataset = pd.read_csv('../input/data_fin.csv')

#always dataframe starts with 0
X = dataset.iloc[:,3:17].values
Y = dataset.iloc[:,19:].values
Xdf = dataset.iloc[:,3:17]
Ydf = dataset.iloc[:,19:] # relationship

from sklearn.model_selection import train_test_split
Xtrain_samp,Xtest_samp,Ytrain_samp,Ytest_samp = train_test_split(X,Y,train_size=0.99,random_state=42)

X=Xtrain_samp
rel=Ytrain_samp


Xcate1=[0,1,3,5,6,7,9,10,11]
Xcate2=[0,1,3,9,10,11]

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in Xcate1:
    label=LabelEncoder()
    i=int(i)
    X[:,i]=label.fit_transform(X[:,i])

temp=[]    
for i in Xcate2:
    temp.append(max(X[:,i:i+1]))

maxs=[]  
for i in range (0,len(temp)):
    maxs.append(temp[i][0])


one = OneHotEncoder(categorical_features=[0,1,3,9,10,11])
X =  one.fit_transform(X).toarray()


indep_var = pd.DataFrame(data=X[0:,0:])


del_v = [0,40,60,66,69,80]

for i in del_v:
    del indep_var[i]


# target variable conversion

Ycate1 = [0]
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

for i in Ycate1:
    label=LabelEncoder()
    i=int(i)
    rel[:,i]=label.fit_transform(rel[:,i])

temp=[]    
for i in Ycate1:
    temp.append(max(rel[:,i:i+1]))
#starts with 0
maxs=[]  
for i in range (0,len(temp)):
    maxs.append(temp[i][0])


one = OneHotEncoder(categorical_features=[0])
rel =  one.fit_transform(rel).toarray()


dep_var = pd.DataFrame(data=rel[0:,0:])
del dep_var[0]

split_test_size = 0.30

x_train,x_test,y_train,y_test = train_test_split(indep_var,dep_var,test_size=split_test_size,random_state = 42)

X_train = x_train.as_matrix(columns=None)
Y_train = y_train.as_matrix(columns=None)
X_test = x_test.as_matrix(columns=None)
Y_test = y_test.as_matrix(columns=None)

from keras.models import Sequential

model = Sequential()
from keras.layers import Dense

#input layer
model.add(Dense(units=98, activation='sigmoid', input_dim=98))
#Hidden layer
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
#output layer
model.add(Dense(units=27, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=9, batch_size=100)
model.predict(X_test)
score = model.evaluate(X_test, Y_test,verbose=1)

print('The Loss and Accuracy predicting perpetrator Relationship using Neural Networks is :')
print(score)


