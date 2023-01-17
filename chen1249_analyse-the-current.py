import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as plt 

from sklearn import linear_model

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import svm,tree, ensemble,model_selection
train_pc1 = pd.read_csv("../input/test_data1536128871.84_screen2.csv", header=None)

print(train_pc1.shape)

train_pc1.head()
train_pc2 = pd.read_csv("../input/test_data1536128871.84_screen2.csv", header=None)

print(train_pc2.shape)

train_pc2.head()
train_tempsen=pd.read_csv("../input/test_data1536128871.84_screen2.csv",header=None)

print(train_tempsen.shape)

train_tempsen.head()
testdata=pd.read_csv("../input/test_data1536128871.84_screen2.csv",header=None)

print(testdata.shape)

testdata.head()
train_pc1.rename(columns={0:"Timestamp",1:"MaxCurrent",2:"EffCurrent"},inplace=True)

train_pc2.rename(columns={0:"Timestamp",1:"MaxCurrent",2:"EffCurrent"},inplace=True)

train_tempsen.rename(columns={0:"Timestamp",1:"MaxCurrent",2:"EffCurrent"},inplace=True)

testdata.rename(columns={0:"Timestamp",1:"MaxCurrent",2:"EffCurrent"},inplace=True)
train_pc1.head()
train_pc2.head()
train_tempsen.head()
testdata.head()
train_pc1.isnull().sum()
train_pc2.isnull().sum()
train_tempsen.isnull().sum()
linesPC1 = train_pc1.plot.line(x='Timestamp', y='EffCurrent',title='PC Screen 1')  

linesPC2 = train_pc2.plot.line(x='Timestamp', y='EffCurrent',title='PC Screen 2')

linestempsen = train_tempsen.plot.line(x='Timestamp', y='EffCurrent',title='Sensor')

linemain = testdata.plot.line(x='Timestamp', y='EffCurrent',title='main')
train_pc1['Time Period']=''



for index,row in train_pc1.iterrows():

    if(index==0):

        train_pc1['Time Period'][0]=0

    else:

        train_pc1['Time Period'][index]=train_pc1['Timestamp'][index]-train_pc1['Timestamp'][index-1]

train_pc1.head()
train_pc1['PC Screen1']=''

train_pc1['EffCurrent_Diff'] = train_pc1['EffCurrent'].diff()

for index,row in train_pc1.iterrows():

    if (train_pc1['EffCurrent'][index] == 0):

        train_pc1['PC Screen1'][index] = 0 

    else:

        if (train_pc1['EffCurrent_Diff'][index]== 0):

            train_pc1['PC Screen1'][index] = 2

        else:

            train_pc1['PC Screen1'][index] = 1

train_pc1.head()
train_pc1.drop('EffCurrent_Diff',axis=1,inplace=True)

train_pc1.head()
train_pc2['Time Period']=''



for index,row in train_pc2.iterrows():

    if(index==0):

        train_pc2['Time Period'][0]=0

    else:

        train_pc2['Time Period'][index]=train_pc2['Timestamp'][index]-train_pc2['Timestamp'][index-1]

        

train_pc2['PC Screen2']=''

train_pc2['EffCurrent_Diff'] = train_pc2['EffCurrent'].diff()

for index,row in train_pc2.iterrows():

    if (train_pc2['EffCurrent'][index] == 0):

        train_pc2['PC Screen2'][index] = 0 

    else:

        if (train_pc2['EffCurrent_Diff'][index]== 0):

            train_pc2['PC Screen2'][index] = 2

        else:

            train_pc2['PC Screen2'][index] = 1

train_pc2.drop('EffCurrent_Diff',axis=1,inplace=True)

train_pc2.head()
train_tempsen['Time Period']=''

for index,row in train_tempsen.iterrows():

    if(index==0):

        train_tempsen['Time Period'][0]=0

    else:

        train_tempsen['Time Period'][index]=train_tempsen['Timestamp'][index]-train_tempsen['Timestamp'][index-1]

train_tempsen['Temp sensor']=''

for index,row in train_tempsen.iterrows():

    if (train_tempsen['EffCurrent'][index] != 0):

        train_tempsen['Temp sensor'][index] = 1 

    else:

        train_tempsen['Temp sensor'][index] = 0

train_tempsen.head()
linesPC1 = train_pc1.plot.line(x='Timestamp', y='PC Screen1',title='PC Screen 1')

linesPC2 = train_pc2.plot.line(x='Timestamp', y='PC Screen2',title='PC Screen 2')

linesSensor = train_tempsen.plot.line(x='Timestamp', y='Temp sensor',title='Sensor')
models={'logit':'','svm':'','rforest':''}

X = pd.DataFrame(train_pc1[['MaxCurrent','EffCurrent','Time Period']])

y = pd.DataFrame(train_pc1['PC Screen1'])

y = y.astype('int')

# Split the Dataset into Train and Test RANDOMLY

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

##Logistic Regression

regularization=[0.001,0.01,0.1,1,10,100,1000]

print("Logistic Regression")

scores=[]

for c in regularization:

    logit1=linear_model.LogisticRegression(C=c)

    logit1.fit(X_train,y_train)

    print("Accuracy for C= ",c," is= ", logit1.score(X_test,y_test))

    scores.append(logit1.score(X_test, y_test))



c=regularization[np.argmax(scores)]

logit1=linear_model.LogisticRegression(C=c)

logit1.fit(X_train,y_train)

print("With C= ",c," Score is:",logit1.score(X_test, y_test))



models['logit']=logit1.score(X_test, y_test)

y_pred=logit1.predict(X_test)

    

labels = [0,1,2]

print("Accuracy= ",round(logit1.score(X_test,y_test)*100,1),"%")



##Support Vector Machines

print("Support Vector Machines")

scores=[]

for c in regularization:

    svc1= svm.LinearSVC(C=c)

    svc1.fit(X_train,y_train)

    print("Accuracy for C= ",c," is= ", svc1.score(X_test,y_test))

    scores.append(svc1.score(X_test,y_test))



c=regularization[np.argmax(scores)]

svc1=svm.LinearSVC(C=c)

svc1.fit(X_train,y_train)

y_pred_svm=svc1.predict(X_test)



print("Accuracy= ",round(svc1.score(X_test,y_test)*100,1),"%")

    

models['svm']=svc1.score(X_test, y_test)



#Boosting with Random Forest

print("Boosting")

rforest1 = ensemble.RandomForestClassifier(max_features = 3, oob_score = True, random_state = 2018)

rforest1.fit(X_train, y_train)



print("Accuracy: ",rforest1.score(X_test,y_test))

models['rforest']=rforest1.score(X_test, y_test)



##Running the model on the test Data



model=max(models,key=models.get)

print("Best Model is : ",model)
models={'logit':'','svm':'','rforest':''}

X = pd.DataFrame(train_pc2[['MaxCurrent','EffCurrent','Time Period']])

y = pd.DataFrame(train_pc2['PC Screen2'])

y = y.astype('int')

# Split the Dataset into Train and Test RANDOMLY

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

##Logistic Regression

regularization=[0.001,0.01,0.1,1,10,100,1000]

print("Logistic Regression")

scores=[]

for c in regularization:

    logit2=linear_model.LogisticRegression(C=c)

    logit2.fit(X_train,y_train)

    print("Accuracy for C= ",c," is= ", logit2.score(X_test,y_test))

    scores.append(logit2.score(X_test, y_test))



c=regularization[np.argmax(scores)]

logit2=linear_model.LogisticRegression(C=c)

logit2.fit(X_train,y_train)

print("With C= ",c," Score is:",logit2.score(X_test, y_test))



models['logit']=logit2.score(X_test, y_test)

y_pred=logit2.predict(X_test)

    

labels = [0,1,2]

print("Accuracy= ",round(logit2.score(X_test,y_test)*100,1),"%")



##Support Vector Machines

print("Support Vector Machines")

scores=[]

for c in regularization:

    svc2= svm.LinearSVC(C=c)

    svc2.fit(X_train,y_train)

    print("Accuracy for C= ",c," is= ", svc2.score(X_test,y_test))

    scores.append(svc2.score(X_test,y_test))



c=regularization[np.argmax(scores)]

svc2=svm.LinearSVC(C=c)

svc2.fit(X_train,y_train)

y_pred_svm=svc2.predict(X_test)



print("Accuracy= ",round(svc2.score(X_test,y_test)*100,1),"%")

    

models['svm']=svc2.score(X_test, y_test)



#Boosting with Random Forest

print("Boosting")

rforest2 = ensemble.RandomForestClassifier(max_features = 3, oob_score = True, random_state = 2018)

rforest2.fit(X_train, y_train)



print("Accuracy: ",rforest2.score(X_test,y_test))

models['rforest']=rforest2.score(X_test, y_test)



##Running the model on the test Data



model=max(models,key=models.get)

print("Best Model is : ",model)
models={'logit':'','svm':'','rforest':''}

X = pd.DataFrame(train_tempsen[['MaxCurrent','EffCurrent','Time Period']])

y = pd.DataFrame(train_tempsen['Temp sensor'])

y = y.astype('int')

# Split the Dataset into Train and Test RANDOMLY

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

##Logistic Regression

regularization=[0.001,0.01,0.1,1,10,100,1000]

print("Logistic Regression")

scores=[]

for c in regularization:

    logit=linear_model.LogisticRegression(C=c)

    logit.fit(X_train,y_train)

    print("Accuracy for C= ",c," is= ", logit.score(X_test,y_test))

    scores.append(logit.score(X_test, y_test))



c=regularization[np.argmax(scores)]

logit=linear_model.LogisticRegression(C=c)

logit.fit(X_train,y_train)

print("With C= ",c," Score is:",logit.score(X_test, y_test))



models['logit']=logit.score(X_test, y_test)

y_pred=logit.predict(X_test)

    

labels = [0,1,2]

print("Accuracy= ",round(logit.score(X_test,y_test)*100,1),"%")



##Support Vector Machines

print("Support Vector Machines")

scores=[]

for c in regularization:

    svc= svm.LinearSVC(C=c)

    svc.fit(X_train,y_train)

    print("Accuracy for C= ",c," is= ", svc.score(X_test,y_test))

    scores.append(svc.score(X_test,y_test))



c=regularization[np.argmax(scores)]

svc=svm.LinearSVC(C=c)

svc.fit(X_train,y_train)

y_pred_svm=svc.predict(X_test)



print("Accuracy= ",round(svc.score(X_test,y_test)*100,1),"%")

    

models['svm']=svc.score(X_test, y_test)



#Boosting with Random Forest

print("Boosting")

rforest = ensemble.RandomForestClassifier(max_features = 3, oob_score = True, random_state = 2018)

rforest.fit(X_train, y_train)



print("Accuracy: ",rforest.score(X_test,y_test))

models['rforest']=rforest.score(X_test, y_test)



##Running the model on the test Data



model=max(models,key=models.get)

print("Best Model is : ",model)
testdata['Time Period']=''

for index,row in testdata.iterrows():

    if(index==0):

        testdata['Time Period'][0]=0

    else:

        testdata['Time Period'][index]=testdata['Timestamp'][index]-testdata['Timestamp'][index-1]

testdata.head()
testdata
y_pc1_pred=rforest1.predict(pd.DataFrame(testdata.iloc[:,1:4]))

y_pc2_pred=logit2.predict(pd.DataFrame(testdata.iloc[:,1:4]))

y_sensor_pred=logit.predict(pd.DataFrame(testdata.iloc[:,1:4]))

y_pc1_pred
frame=pd.DataFrame(list(zip(y_pc1_pred,y_pc2_pred,y_sensor_pred)))

frame.rename(columns={0:'PC Screen1',1:'PC Screen2',2:'Temperature Sensor'},inplace=True)

testdata=pd.concat([testdata,frame],axis=1,sort=False)

testdata
testdata.replace(0,'OFF',inplace=True)

testdata.replace(1,'ON',inplace=True)

testdata.replace(2,'IDLE',inplace=True)

testdata.head()

testdata
testdata.groupby('PC Screen1').size()
testdata.groupby('PC Screen2').size()
testdata.groupby('Temperature Sensor').size()