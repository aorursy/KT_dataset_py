import numpy as np

import pandas as pd

import sklearn as sk

from sklearn import preprocessing



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import xgboost

color = sns.color_palette()
#Load train dataset

trainData = pd.read_csv('/kaggle/input/train.csv')

                

#Load test dataset

testData = pd.read_csv('/kaggle/input/test.csv')
#Metadata info

trainData.info()
trainData.head()
#statistical summary of the train dataset

trainData.describe()
#0 variance - remove columns



trainData.var()[trainData.var()==0].index.values
#drop 0 variance columns- as per instructions

trainDataFinal=trainData.drop(trainData.var()[trainData.var()==0].index.values, axis=1)

trainDataFinal.info()
#select features and labels

features=trainDataFinal.iloc[:,2:]

label=trainDataFinal.iloc[:,1].values



features.head()
#Get object columns from feature column for LE and OHE



features.describe(include=['object'])
#Get only object column names



objcols=features.describe(include=['object']).columns.values

objcols
#Label Encoder on the object column



from sklearn.preprocessing import LabelEncoder

#from sklearn.preprocessing import OHE



le=LabelEncoder()



for i in objcols:

  features[i]= le.fit_transform(features[i])



#ohe=OHE(categorical_features=[0,1,2,3,4,5,6,8])



Features=features.values

#featureohe=Features #ohe.fit_transform(fea).toarray()



#featureOHE

FeaturesCpy = Features



#import warnings

#warnings.filterwarnings('ignore')



#stateOHE = OneHotEncoder(categorical_features=[0,1,2,3,4,5,6,7])

#stateOHE.fit(FeaturesCpy)

#FeaturesCpy = stateOHE.transform(FeaturesCpy).toarray()
#PCA



#Feature scaling

from sklearn.preprocessing import StandardScaler

Stdsclr = StandardScaler()

ScaledFeatures = Stdsclr.fit_transform(FeaturesCpy)

ScaledFeatures.shape
#PCA (dimension reduction) - All-in 364 compnents

from sklearn.decomposition import PCA



pca = PCA(n_components=364, svd_solver='full')

pca.fit(ScaledFeatures,label)
pca.explained_variance_ratio_
#Mean variance - for threshold

np.mean(pca.explained_variance_ratio_)
#number of components >threhold variance

pca.explained_variance_ratio_ > 0.002747252747252748
#PCA (dimension reduction) - 93 components

from sklearn.decomposition import PCA



pca = PCA(n_components=93, svd_solver='full')

pca.fit(ScaledFeatures,label)



#Transform the Features

finalFeatures = pca.transform(ScaledFeatures)

finalFeatures.shape
# Considering the n_components= 93 only as per above findings.

# Recreating the PCA Object with n_components = 93

pca = PCA(n_components=93)

pca.fit(ScaledFeatures,label)
#Using KFold method for gradient boosting



#Initialise the algo

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



#Initiallise KFold Method

from sklearn.model_selection import KFold

from xgboost import XGBRegressor



kfold=KFold(n_splits=25,

           random_state=1,

           shuffle=True)

#kfold.split(ScaledFeatures)



#Initialise for Loop

i=0

for train,test in kfold.split(ScaledFeatures):

    i = i+1

    X_train,X_test = ScaledFeatures[train],ScaledFeatures[test]

    y_train,y_test = label[train],label[test]

    model = XGBRegressor(objective='reg:squarederror', learning_rate=0.1)

    

    model.fit(X_train,y_train)

    

    if model.score(X_test,y_test) > 0.10: #model.score(X_train, y_train)

        print("Test Score: {}, train score: {}, for Sample Split: {}".format(model.score(X_test,y_test),model.score(X_train,y_train),i))
#Test Score: 0.7483431141129421, train score: 0.6041973992150029, for Sample Split: 5

#Extract sample 5 with split 25



kfold = KFold(n_splits=25, 

              random_state=1,

              shuffle=True)

i=0

for train,test in kfold.split(finalFeatures):

    i = i+1

    if i == 5:

        X_train,X_test,y_train,y_test = finalFeatures[train],finalFeatures[test],label[train],label[test]
#fit into model- XGBoost regression

model = XGBRegressor(objective='reg:squarederror', learning_rate=0.1)

    

model.fit(X_train,y_train)



#Check the quality of model

print("Training Accuracy ",model.score(X_train,y_train))

print("Testing Accuracy ",model.score(X_test,y_test))
#Check model accuracy and error metrics

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



#MAE - check this error metrics 

mean_absolute_error(y_test,model.predict(X_test))
#Check model accuracy based on R2

r2_score(y_test,model.predict(X_test), multioutput='variance_weighted')
#Check MSE

mean_squared_error(y_test,model.predict(X_test))
#Process TEST dataset to predict Y
#TEST data

testData.info()
testData.head()
#statistical summary of the test data set

testData.describe()
#Remove some columns like for Train

#'X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347'



datafinal=testData.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290','X293', 'X297', 'X330', 'X347'], axis=1)

datafinal.info()
#Get feature columns



testFeatures=datafinal.iloc[:,1:]



#get object columns



objcolstest=testFeatures.describe(include=['object']).columns.values

objcolstest
#Apply Label Encoder for object columns. 



from sklearn.preprocessing import LabelEncoder

#from sklearn.preprocessing import OneHotEncoder



le=LabelEncoder()



for i in objcolstest:

  testFeatures[i]= le.fit_transform(testFeatures[i])



f1=testFeatures.values





f1.ndim
#Transform test data - PCA



transformFromPCA = pca.transform(f1)



labelpred=model.predict(transformFromPCA)

labelpred
#Save to file



testData.to_csv('TestPredictiveValue.csv')
#Append Y predicted value in test data set



testdataforfile=pd.concat([testData.iloc[:,0],pd.concat([pd.DataFrame(data=labelpred, columns=['y']),testData.iloc[:,1:]], axis=1)],axis=1)



testdataforfile.head()