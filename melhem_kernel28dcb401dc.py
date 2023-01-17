# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold

from sklearn.metrics import precision_recall_fscore_support

from sklearn.svm import SVC

from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np 
test = pd.read_csv("../input/healthcare-dataset-stroke-data/test_2v.csv")

train = pd.read_csv("../input/healthcare-dataset-stroke-data/train_2v.csv")
ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='most_frequent')

ImputedX = ImputedModule.fit(train)

train = ImputedX.transform(train)

df_train =  pd.DataFrame(train)

#print(df_train.head(5))

#print(df_train.shape)



ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='most_frequent')

ImputedX = ImputedModule.fit(test)

test = ImputedX.transform(test)

df_test =  pd.DataFrame(test)

#print(df_test.head(5))

#print(df_test.shape)
enc1  = LabelEncoder()

enc1.fit(df_train[1])



#print('classed found : ' , list(enc1.classes_))

#print('equivilant numbers are : ' ,enc1.transform(df_train[1]) )



df_train["newGender"] = enc1.transform(df_train[1])

#df_train.drop([1], axis=1)

del df_train[1]



#print('Updates dataframe is : \n' ,df_train )

#print('Inverse Transform  : ' ,list(enc1.inverse_transform([1,0,2,1,1,0,0])))

#--------

enc5  = LabelEncoder()

enc5.fit(df_train[5])



#print('classed found : ' , list(enc5.classes_))

#print('equivilant numbers are : ' ,enc5.transform(df_train[5]) )



df_train["Marrid"] = enc5.transform(df_train[5])

#df_train.drop([5], axis=1)

del df_train[5]





#print('Updates dataframe is : \n' ,df_train )

#print('Inverse Transform  : ' ,list(enc5.inverse_transform([1,0,2,1,1,0,0])))

#-------

enc6  = LabelEncoder()

enc6.fit(df_train[6])



#print('classed found : ' , list(enc6.classes_))

#print('equivilant numbers are : ' ,enc6.transform(df_train[6]) )



df_train["Work"] = enc6.transform(df_train[6])

#df_train = df_train.drop(6)

del df_train[6]



#print('Updates dataframe is : \n' ,df_train )

#print('Inverse Transform  : ' ,list(enc6.inverse_transform([1,0,2,1,1,0,0])))

#----------

enc7  = LabelEncoder()

enc7.fit(df_train[7])



#print('classed found : ' , list(enc7.classes_))

#print('equivilant numbers are : ' ,enc7.transform(df_train[7]) )



df_train["Resident"] = enc7.transform(df_train[7])

#df_train = df_train.drop(7)

del df_train[7]



#print('Updates dataframe is : \n' ,df_train )

#print('Inverse Transform  : ' ,list(enc7.inverse_transform([1,0,2,1,1,0,0])))

#----------

enc10  = LabelEncoder()

enc10.fit(df_train[10])



#print('classed found : ' , list(enc10.classes_))

#print('equivilant numbers are : ' ,enc10.transform(df_train[10]) )



df_train["Smoking"] = enc10.transform(df_train[10])

#df_train = pd.drop(10)

del df_train[10]



#print('Updates dataframe is : \n' ,df_train )

#print('Inverse Transform  : ' ,list(enc10.inverse_transform([1,0,2,1,1,0,0])))

#-----------------TestData------------------------------------------------------

enc1  = LabelEncoder()

enc1.fit(df_test[1])



#print('classed found : ' , list(enc1.classes_))

#print('equivilant numbers are : ' ,enc1.transform(df_test[1]) )



df_test["newGender"] = enc1.transform(df_test[1])

#df_test.drop([1], axis=1)

del df_test[1]



#print('Updates dataframe is : \n' ,df_test )

#print('Inverse Transform  : ' ,list(enc1.inverse_transform([1,0,2,1,1,0,0])))

#--------

enc5  = LabelEncoder()

enc5.fit(df_test[5])



#print('classed found : ' , list(enc5.classes_))

#print('equivilant numbers are : ' ,enc5.transform(df_test[5]) )



df_test["Marrid"] = enc5.transform(df_test[5])

#df_test.drop([5], axis=1)

del df_test[5]





#print('Updates dataframe is : \n' ,df_test )

#print('Inverse Transform  : ' ,list(enc5.inverse_transform([1,0,2,1,1,0,0])))

#-------

enc6  = LabelEncoder()

enc6.fit(df_test[6])



#print('classed found : ' , list(enc6.classes_))

#print('equivilant numbers are : ' ,enc6.transform(df_test[6]) )



df_test["Work"] = enc6.transform(df_test[6])

#df_test = df_test.drop(6)

del df_test[6]



#print('Updates dataframe is : \n' ,df_test )

#print('Inverse Transform  : ' ,list(enc6.inverse_transform([1,0,2,1,1,0,0])))

#----------

enc7  = LabelEncoder()

enc7.fit(df_test[7])



#print('classed found : ' , list(enc7.classes_))

#print('equivilant numbers are : ' ,enc7.transform(df_test[7]) )



df_test["Resident"] = enc7.transform(df_test[7])

#df_test = df_test.drop(7)

del df_test[7]



#print('Updates dataframe is : \n' ,df_test )

#print('Inverse Transform  : ' ,list(enc7.inverse_transform([1,0,2,1,1,0,0])))

#----------

enc10  = LabelEncoder()

enc10.fit(df_test[10])



#print('classed found : ' , list(enc10.classes_))

#print('equivilant numbers are : ' ,enc10.transform(df_test[10]) )



df_test["Smoking"] = enc10.transform(df_test[10])

#df_test = pd.drop(10)

del df_test[10]

#-------------------





#print('Updates dataframe is : \n' ,df_test )

#print('Inverse Transform  : ' ,list(enc10.inverse_transform([1,0,2,1,1,0,0])))



df_train = pd.DataFrame(df_train)

df_train["Stroke"] = df_train[11]

del df_train[11]

#print(df_train["Stroke"])
#print(df_test.head(5))

#print(df_test.shape)



#print(df_train.head(5))

#df_train = pd.DataFrame(df_train)



#print(df_train.head(2))
df_trainX = df_train.iloc[:,0:10]

print(df_trainX.shape)



df_trainY = df_train.iloc[:,11:]

print(df_trainY.shape)

#df_train = pd.DataFrame(df_train)

#df_trainY =(df_train.iloc[:,11:])

#print(df_trainY.head(5))

#df_trainX = (df_train[[1,2,3,4,5,6,7,8,9,10,]])

#print(df_trainX)

#df_trainX = pd.DataFrame(df_trainX)

#df_trainY = pd.DataFrame(df_trainY)

#print(df_trainX.iloc[:5,:5])
X_train, X_test, y_train, y_test = train_test_split(df_trainX, df_trainY, test_size=0.30, random_state=44, shuffle =True)



#Splitted Data

print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)

#print(y_test)

lab_enc = preprocessing.LabelEncoder()

training_scores_encoded = lab_enc.fit_transform(y_train)



lab_enc = preprocessing.LabelEncoder()

training_scores_encoded1 = lab_enc.fit_transform(y_test)



SVCModel = SVC(kernel= 'sigmoid',# it can be also linear,poly,sigmoid,precomputed

               max_iter=50000,C=0.00001,gamma='auto_deprecated',degree=100)



SVCModel.fit(X_train, training_scores_encoded)



#Calculating Details

print('SVCModel Train Score is : ' , SVCModel.score(X_train, training_scores_encoded))

print('SVCModel Test Score is : ' , SVCModel.score(X_test, training_scores_encoded1))

print('----------------------------------------------------')



DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='entropy',max_depth= 20,random_state=500) #criterion can be entropy

DecisionTreeClassifierModel.fit(X_train, training_scores_encoded)



#Calculating Details

print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, training_scores_encoded))

print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, training_scores_encoded1))

#print('DecisionTreeClassifierModel Classes are : ' , DecisionTreeClassifierModel.classes_)

#print('DecisionTreeClassifierModel feature importances are : ' , DecisionTreeClassifierModel.feature_importances_)

print('----------------------------------------------------')



#loading models for Voting Classifier

LRModel_ = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=33)

RFModel_ = RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=1, random_state=33)

KNNModel_ = KNeighborsClassifier(n_neighbors= 4 , weights ='uniform', algorithm='auto')



#loading Voting Classifier

VotingClassifierModel = VotingClassifier(estimators=[('LRModel',LRModel_),('RFModel',RFModel_),('KNNModel',KNNModel_)], voting='hard')





DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',max_depth= 9,random_state=500) #criterion can be entropy



SVCModel = SVC(kernel= 'sigmoid',# it can be also linear,poly,sigmoid,precomputed

               max_iter=50000,C=0.00001,gamma='auto_deprecated',degree=100)





#KFold Splitting data

FoldNo  = 10

kf = KFold(n_splits=FoldNo, random_state=44, shuffle =True)



score_DTC = 0

score_VCM = 0

score_SVC = 0 





#KFold Data

X = df_trainX

y = df_trainY

lab_enc = preprocessing.LabelEncoder()

for train_index, test_index in kf.split(X):

    print('--------------FrontFold-----------------')

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    

    X_train = pd.DataFrame(X_train)

    y_train = pd.DataFrame(y_train)

    

    X_test = pd.DataFrame(X_test)

    y_test = pd.DataFrame(y_test)

    

    y_train = lab_enc.fit_transform(y_train)

    y_test = lab_enc.fit_transform(y_test)

    

    

    print('========================================')

    

    

    #DTC Model

    

    DecisionTreeClassifierModel.fit(X_train,y_train)

    print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))

    print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))

    

    y_pred = DecisionTreeClassifierModel.predict(X_test)

    

    score_DTC = score_DTC + DecisionTreeClassifierModel.score(X_test, y_test)

    

    

    #Calculating Confusion Matrix

    CM1 = confusion_matrix(y_test, y_pred)

    print('Confusion Matrix is : \n', CM1)



   

    

    print('========================================')

    

    

    #SVC Model

    

    SVCModel.fit(X_train,y_train)

    print('SVCModel Train Score is : ' , SVCModel.score(X_train,y_train))

    print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))

    

    y_predect = SVCModel.predict(X_test)

    

    score_SVC = score_SVC + SVCModel.score(X_test, y_test)



    #Calculating Confusion Matrix

    CM2 = confusion_matrix(y_test, y_predect)

    print('Confusion Matrix is : \n', CM2)



    print('========================================')

    

    

    # Voting Model 

    

    VotingClassifierModel.fit(X_train, y_train)

    print('VotingClassifierMode Train Score is : ' , VotingClassifierModel.score(X_train,y_train))

    print('VotingClassifierMode Test Score is : ' , VotingClassifierModel.score(X_test, y_test))

    

    y_predected = VotingClassifierModel.predict(X_test)

    

    score_VCM = score_VCM + VotingClassifierModel.score(X_test, y_test)

    

    #Calculating Confusion Matrix

    CM3 = confusion_matrix(y_test, y_predected)

    print('Confusion Matrix is : \n', CM3)

     

    F1Score = f1_score(y_test, y_predected, average='micro') #it can be : binary,macro,weighted,samples

    print('F1 Score is : ', F1Score)

    print('========================================')



    print('-------------------EndFold---------------------------------')

    

print("the total score for score_DTC", score_DTC/FoldNo)

print("the total score for score_VCM", score_VCM/FoldNo)

print("the total score for score_SVC", score_SVC/FoldNo)