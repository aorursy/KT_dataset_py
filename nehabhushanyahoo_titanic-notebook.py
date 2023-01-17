import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")
genderOutput = pd.read_csv("../input/gender_submission.csv")
testData = pd.merge(testData,genderOutput,on=testData.PassengerId)
testData.info()
trainData.head(5)

trainData.shape
trainData.drop(axis=1, labels= ["Cabin"] ,inplace= True)
trainData.info()
m = trainData.Age[trainData.Age.notnull()].mean()
#Removed Age with values NaN with with mean of all the ages in training data set
trainData.Age[trainData.Age.isnull()] = m
plt.hist(trainData.Age,density= True)
trainData.info()
trainData.dropna(inplace=True)
trainData.info()
trainData.drop(labels=["Ticket"],axis=1,inplace= True)
trainData.dropna(axis=0,inplace=True)
trainData.info()
features = trainData.iloc[:,[2,4,5,6,7,8,9]]
label= trainData.iloc[:,1]
features.head(5)
features.info()
from sklearn.preprocessing import LabelEncoder
labelEn = LabelEncoder()
features.Sex = labelEn.fit_transform(features.Sex)
features.Sex.unique()
features.info()
features.Embarked = labelEn.fit_transform(features.Embarked)
features.Embarked.unique()
features.info()
features.corr()
plt.imshow(features.corr(),)
plt.colorbar()
#determine importance of feature
from sklearn.decomposition import PCA
pca = PCA(n_components= 7)

pca_component = pca.fit(features)
pca.explained_variance_ratio_
plt.bar(["PC1","PC2","PC3","PC4","PC5","PC6","PC7"],pca.explained_variance_ratio_)
f = features.iloc[:,[0,1]]
f.info()
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_model = log_reg.fit(f,label)

log_model.score(f,label)
testData.info()
features_test = testData.iloc[:,[1,3]]
label_test = testData.iloc[:,12]
features_test.info()
from sklearn.preprocessing import LabelEncoder
labelEn_t = LabelEncoder()
features_test.Sex = labelEn_t.fit_transform(features_test.Sex)
#features_test.Embarked = labelEn_t.fit_transform(features_test.Embarked)
log_model.score(features_test,label_test)
pred = log_model.predict(features_test)
predDf = pd.DataFrame(pred)
genderOutput = pd.concat([genderOutput,predDf],axis=1)



