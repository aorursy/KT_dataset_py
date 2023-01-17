import pandas as pd
import numpy as np
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/gender_submission.csv")
print(train.shape,test.shape)
train.describe()

test.describe()
trainData = train.drop(["PassengerId","Name","Age","Ticket","Cabin"],axis=1,inplace = False)
testData = test.drop(["PassengerId","Name","Age","Ticket","Cabin"],axis=1,inplace = False)

trainData.describe(), trainData.shape, testData.shape
trainData.fillna(axis=1,inplace = False,method = 'pad')
testData= testData.fillna(axis=1,inplace = False ,method = 'pad')
ytrain=trainData.Survived.values
xtrain=trainData.drop('Survived', axis=1).values
xtest = testData.values
from sklearn.preprocessing import OneHotEncoder
xtrain.shape, ytrain.shape,xtest.shape
xtest
for i in range(len(xtrain)):
    if xtrain[:,1][i]=='male':
        xtrain[:,1][i]=1
    else:
        xtrain[:,1][i]=0
        
for i in range(len(xtest)):
    if xtest[:,1][i]=='male':
        xtest[:,1][i]=1
    else:
        xtest[:,1][i]=0


xtest
fare = 4
fareMean=xtrain[:,fare].mean()
fareStd=xtrain[:,fare].std()

for i in range(len(xtrain)):
    xtrain[:,fare][i]= (xtrain[:,fare][i] - fareMean)/ (fareStd*1.0)


fare = 4
fareMeanTest=xtest[:,fare].mean()
fareStdTest=xtest[:,fare].std()

for i in range(len(xtest)):
    xtest[:,fare][i]= (xtest[:,fare][i] - fareMean)/ (fareStd*1.0) # scales it according to the train data

xtrain,xtest
      
      
dummies = pd.get_dummies(train.Embarked).values
dummiesTest = pd.get_dummies(test.Embarked).values
dummies = dummies[:,:-1]
dummiesTest = dummiesTest[:,:-1]
dummies.shape, dummiesTest.shape
xtrain_onehot= xtrain[:,:-1]
xtest_onehot = xtest[:,:-1]
xtrain_final = np.append(xtrain_onehot,dummies,axis=1)
xtest_final = np.append(xtest_onehot, dummiesTest,axis=1)
xtrain_final.shape,ytrain.shape,xtest_final.shape
from sklearn.svm import SVC

clf = SVC(kernel = "rbf",gamma='scale')
from sklearn.model_selection import train_test_split
# xtrain1,xtest1,ytrain1,ytest1=train_test_split(xtrain_final,ytrain,random_state=5)
# clf1 = SVC(kernel = "rbf",gamma='scale')
# clf1.fit(xtrain1,ytrain1)
# clf1.score(xtest1,ytest1)


# #The above is a test script for accuracy, I got an accuracy of 82%, 
# #and test accuracy of 78%, hence it is clearly not over fitting

clf.fit(xtrain_final,ytrain)
ypred = clf.predict(xtest_final)
ypredDF = pd.DataFrame(ypred)
ypredDF.to_csv("submission.csv",sep=",",index_label=['PassengerId'],header=['Survived'],index=True)
temp = pd.read_csv('submission.csv')
temp.PassengerId = temp.PassengerId + 892
# submission
temp.to_csv("submission.csv",sep=",",header=['PassengerId','Survived'],index=False)
