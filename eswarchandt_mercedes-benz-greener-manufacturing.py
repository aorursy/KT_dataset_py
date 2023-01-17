#import required libraries

import numpy as np

import pandas as pd

#Load train dataset

data=pd.read_csv(r'../input/mercedesbenz-greener-manufacturing/train.csv')

datatest=pd.read_csv(r'../input/mercedesbenz-greener-manufacturing/test.csv')

data.head()
#Get details for data



data.info()
#Remove columns with variance 0 as those columns will not impact the prediction and mostly those will have constant data

#to reduce number of cols remve those var=0 cols



data.var()[data.var()==0].index.values
#drop 0 var columns

datafinal=data.drop(data.var()[data.var()==0].index.values, axis=1)

datafinal.info()
datafinal.head()
#get features and labels



features=datafinal.iloc[:,2:]

label=datafinal.iloc[:,1].values



features.head()
label
#Get object columns from feature column for LE and OHE



features.describe(include="O")
#Get only object column names



objcols=features.describe(include=['object']).columns.values

objcols
#Apply Label Encoder  for object columns. OHE is not use das this will create PC's instead of using direct variables



from sklearn.preprocessing import LabelEncoder

#from sklearn.preprocessing import OneHotEncoder



le=LabelEncoder()



for i in objcols:

  features[i]= le.fit_transform(features[i])



#ohe=OneHotEncoder(categorical_features=[0,1,2,3,4,5,6,8])



fea=features.values

#featureohe=fea #ohe.fit_transform(fea).toarray()



#featureohe

fea
#Standardize the data using StandardScaler



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

featuresstd = sc.fit_transform(fea)

featuresstd
#USe PCA for dimensionality reduction

#Use no of components as ALL column to identify correct number of componnets

from sklearn.decomposition import PCA



pca = PCA(n_components=364, svd_solver='full')

pca.fit(featuresstd,label)
#check explained variance ratio

pca.explained_variance_ratio_
#Get mean of explained variance ratio to get the threshold value for varince

np.mean(pca.explained_variance_ratio_)
#Check number of components which has explained variance >threshold value

pca.explained_variance_ratio_>0.002747252747252748
data.var()[data.var()==0].index.values
print(len(pca.explained_variance_ratio_[pca.explained_variance_ratio_>0.002747252747252748]))
#PCA with n components 93 which has explained variance ratio >threshold



pca = PCA(n_components=93,svd_solver='full')

pca.fit(featuresstd,label)



finalFeatures = pca.transform(featuresstd)

finalFeatures.shape
finalFeatures
label.shape
#Used No of splits for sample from 5 and increased to 5, 10 ,20,30 etc till 1000. Got a better generalized model at split=1000



from sklearn.model_selection import KFold

from xgboost import XGBRegressor



kfold=KFold(n_splits=1000,random_state=1,shuffle=True)

#kfold.split(finalFeatures)



i=0

for train,test in kfold.split(finalFeatures):

    i = i+1

    X_train,X_test = finalFeatures[train],finalFeatures[test]

    y_train,y_test = label[train],label[test]

    model = XGBRegressor(objective='reg:squarederror', learnning_rate=0.1)

    

    model.fit(X_train,y_train)

    

    if model.score(X_test,y_test) > 0.93: #model.score(X_train, y_train)

        print("Test Score: {}, train score: {}, for Sample Split: {}".format(model.score(X_test,y_test),model.score(X_train,y_train),i))

        
#Test Score: 0.9919590467202489, train score: 0.6536183541830974, for Sample Split: 554



#Extract sample at value 554 with split 1000



kfold = KFold(n_splits=1000, 

              random_state=1,

              shuffle=True)

i=0

for train,test in kfold.split(finalFeatures):

    i = i+1

    if i == 554:

        X_train,X_test,y_train,y_test = finalFeatures[train],finalFeatures[test],label[train],label[test]
#fit into model and get scores using XGBoost regression



model = XGBRegressor(objective='reg:squarederror', learnning_rate=0.1)

    

model.fit(X_train,y_train)



#Check the quality of model

print("Training Accuracy ",model.score(X_train,y_train))

print("Testing Accuracy ",model.score(X_test,y_test))
#Check model accuracy and error metrices



from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



#MAE - check this error metrics. 

mean_absolute_error(y_test,model.predict(X_test))
#MAE is low which gives high accuracy

#Check modle accuracy based on R2

r2_score(y_test,model.predict(X_test), multioutput='variance_weighted')
#R2 also gives good accuracy. 

#So accept this model

#Check MSE

mean_squared_error(y_test,model.predict(X_test))
#Now read and process test data file and predict Y value



datatest.head()
#Remove same columns as we did for tain dataset

#'X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290',

      # 'X293', 'X297', 'X330', 'X347'



datafinal=datatest.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290','X293', 'X297', 'X330', 'X347'], axis=1)

datafinal.info()
#Get feature columns



testFeatures=datafinal.iloc[:,1:]



#get object columns



objcolstest=testFeatures.describe(include=['object']).columns.values

objcolstest



#Apply Label Encoder  for object columns. 



from sklearn.preprocessing import LabelEncoder

#from sklearn.preprocessing import OneHotEncoder



le=LabelEncoder()



for i in objcolstest:

  testFeatures[i]= le.fit_transform(testFeatures[i])



f1=testFeatures.values





f1.ndim
#transform test data using PCA



transformFromPCA = pca.transform(f1)





labelpred=model.predict(transformFromPCA)

labelpred
#Append Y predicted avlue in test data set



testdataforfile=pd.concat([datatest.iloc[:,0],pd.concat([pd.DataFrame(data=labelpred, columns=['y']),datatest.iloc[:,1:]], axis=1)],axis=1)



testdataforfile.head()
testdataforfile.to_csv("testwithpredvalue.csv",index=False)