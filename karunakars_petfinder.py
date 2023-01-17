#Packages required for the model

#For Data Handling and cleaning.
import os
import pandas as pd
import numpy as np

#for Machine learning and computations of kappa score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score

# files and directories used . please change the path to desired path

#print('Within the train directory: \n{}\n'.format(os.listdir("Downloads//all//")))

#working with train.csv . Please replace desired input path in the variable Input_path before running the script

Input_path = '../input/train/'
data = pd.read_csv(Input_path +"train.csv", na_values=['no info','.'])

#dropping NA value rows in columns.

data = data.dropna(how ='any',axis=0)
    
#dropping columns the optional columns .

dt1 = data.drop(['Name','State','RescuerID','Description','PetID'],axis=1)

#print(dt1.head())
dtyp = dt1.dtypes.astype(str).to_dict()

print(dtyp)

#Importing test data . Please specify the desired path here before running the script

Ipath = '../input/test/'
datatest = pd.read_csv(Ipath +"test.csv", na_values=['no info','.'])

#dropping optional columns from test data set
datatest1 = datatest.drop(['Name','State','RescuerID','Description','PetID'],axis=1)

#Printing datatypes    
dtyptest = datatest1.dtypes.astype(str).to_dict()


#Selecting Attributes

X_attributes=dt1.iloc[:,0:18]



#Labels
y_target=dt1.iloc[:,18]

#Preparing Classification model
Classification = RandomForestClassifier()

#Grid
forestgrid = {
    
    'bootstrap': [True],
    'max_depth': [None],
    'max_features': ['auto'],
    'min_samples_leaf': [5,10,15],
    'min_samples_split': [5,10,15],
    'n_estimators': [50,100,150]
}

#K fold cross validation 
gridoper = GridSearchCV(estimator = Classification, 
                           param_grid = forestgrid, 
                           cv = 7, 
                           verbose = 1,
                           n_jobs = -1)

#Fitting the model
gridoper.fit(X_attributes, y_target)

#Identifying best parameters for each model
print("*********************************************")
print('Best parameters:', gridoper.best_params_)

print("*********************************************")

#Kappa score
print('Quadratic weighted kappa score: ', cohen_kappa_score(gridoper.predict(X_attributes), 
                                y_target, weights='quadratic'))

print('***********Predictions on train data*********')

y_pred = gridoper.predict(X_attributes)
print('***********Predictions on train data*********')


#Predictions on testdataset
y_predtestn = gridoper.predict(datatest1)


submission_df = pd.DataFrame(data={"PetID":datatest["PetID"], 
                                   "AdoptionSpeed": y_predtestn})

submission_df.head()
submission_df.to_csv('submission.csv',index=False)
print("************submission*************")
print(submission_df)

print("************submission*************")
     
#writing data to file. please change this to desired path before running the script
'''
writepath = '../input/test/'
wr = open(writepath+"submission.csv", "w")
wr.write(str(submission_df))      
wr.close()
'''
#Check outputfile kaggleoutput.txt in respective folders




