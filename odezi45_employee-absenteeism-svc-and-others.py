# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm, tree
import xgboost
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle
%matplotlib inline
# load the preprocessed CSV data
data = pd.read_excel('/kaggle/input/employee-absenteeism/Absenteeism_at_work_Project.xls')
#display some information about data
data.info()
#upon examination, there will be no need for the ID collumn, so it will be dropped
data = data.drop(['ID'], axis = 1)
# Quick check on the 'Reason for absence column'
sorted(data['Reason for absence'].unique())
## Given the below meaning for the categoy of CIDs I will grup for better repesentation ##
'''
I Certain infectious and parasitic diseases
II Neoplasms
III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
IV Endocrine, nutritional and metabolic diseases
V Mental and behavioural disorders
VI Diseases of the nervous system
VII Diseases of the eye and adnexa
VIII Diseases of the ear and mastoid process
IX Diseases of the circulatory system
X Diseases of the respiratory system
XI Diseases of the digestive system
XII Diseases of the skin and subcutaneous tissue
XIII Diseases of the musculoskeletal system and connective tissue
XIV Diseases of the genitourinary system
XV Pregnancy, childbirth and the puerperium
XVI Certain conditions originating in the perinatal period
XVII Congenital malformations, deformations and chromosomal abnormalities
XVIII Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified
XIX Injury, poisoning and certain other consequences of external causes
XX External causes of morbidity and mortality
XXI Factors influencing health status and contact with health services.

And 7 categories without (CID) patient follow-up (22), medical consultation (23), blood donation 
(24), laboratory examination (25), unjustified absence (26), physiotherapy (27), dental consultation (28).'''

### the grouping is as follows
'''
1- 14 are various diseases
15 -17 : pregnancy and given birth related
18-21: poisons or diseases not elsewere categorise
22 and above : light reason or less serious reasons

'''
### then apply the function below to the dataframe


def Reason(data):
    if data['Reason for absence'] < 15 :
        d = 'R_Known'
    elif data['Reason for absence'] >= 15 and data['Reason for absence'] <= 17  :
        d = 'R_Preg_Birth'
    elif data['Reason for absence'] >= 18 and data['Reason for absence'] <= 21  :
        d = 'R_Pois_unclass'
    elif data['Reason for absence'] == 22:
        d = 'R_NotSerious'
    else:
        d = 'R_NotSerious'
    return d

data['ReasonGroups'] = data.apply(Reason, axis=1)

# so lets drop the Reasons for absence column because is no more useful
data = data.drop(['Reason for absence'], axis = 1)

#lets get dummies
R_dummies = pd.get_dummies(data['ReasonGroups'])

#lets merge it
data = pd.concat([data, R_dummies], axis = 1)

data = data.drop(['ReasonGroups'], axis = 1)
data.head(10)
# data.isnull() # shows a df with the information whether a data point is null 
# Since True = the data point is missing, while False = the data point is not missing, we can sum them
# This will give us the total number of missing values feature-wise
data.isnull().sum()
# As we can see there are quite a few missing data and for a datset with just 740 rows
# it is not a good idea to drop this rows with missiing data so different method will be used to fill the columns
# so let take a look at the spread of each column

# Visulazing the distibution of the data for every feature
data.hist(linewidth=1, histtype='stepfilled', facecolor='g', figsize=(20, 20));
#from the chat above we can make some assumption on how best to fill the missing data
# 1. Transportation expense ,Age, Distance from Residence to Work,Service time, Work load Average/day,Hit 
#    target,Weight, Height,Body mass index 
#    we will fill with there mean value
#2.  Disciplinary failure,Education,Son,Social drinker ,Social smoker, Pet ,Absenteeism time in hours  
#    we will fill with 0
#3.  Month of absence also have values from 0 - 12 and a missing, since we know there are only 12 month in a year, 
#    we can infer that there is an issue with the data, will deal with this later

#so lets start by filling the 2nd list of variables
data[['Month of absence','Disciplinary failure','Education','Son','Social drinker' ,'Social smoker', 'Pet' ,'Absenteeism time in hours']] = data[['Month of absence','Disciplinary failure','Education','Son','Social drinker' ,'Social smoker', 'Pet' ,'Absenteeism time in hours']].fillna(0)

#then we proceed with the rest
data = data.fillna(data.mean())

data.isnull().sum()
###
# lets drop rows where the month is not a recognised value e.g. btw 1 and 12
data = data[data['Month of absence']>0]
data.head(10)
#then we proceed with the rest
data.isnull().sum()

#lets look at the basic description of the data
data.describe()
# Data looks fairly OK to me. No obvious error
# so let compute our Target which will be from 'Absenteeism time in hours'
# This taret will be ctegorical and I have decided to use median rather than chosing an abitrary cut off, which might 
# make the data unbalanced. this reason for this is because we have very few rows of data.
targets = np.where(data['Absenteeism time in hours'] > data['Absenteeism time in hours'].median(),1,0)
#let do a quick check if targets is balanced in the data
targets.sum() / targets.shape[0]
# trgets looks good to me because it shows a ratio of 55/45

# create a Series in the original data frame that will contain the targets for the regression
data['Absenteeism'] = targets
#drop the old absenteesim column
adata = data.drop(['Absenteeism time in hours'], axis = 1)
adata = adata.reset_index(drop=True)
# check what happened
# maybe manually see how the targets were created
adata
#targets.shape
# Create a variable that will contain the inputs (everything without the targets)
unscaled_inputs = adata.iloc[:,:-1]
targets = adata['Absenteeism']


# define scaler as an object
absenteeism_scaler = StandardScaler()
# create the Custom Scaler class

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler(copy,with_mean,with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    
    # the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
# choose the columns to scale
# we later augmented this code and put it in comments
# select the columns to omit
columns_to_omit = ['R_Known', 'R_NotSerious', 'R_Pois_unclass', 'R_Preg_Birth','Education']
# create the columns to scale, based on the columns to omit
# use list comprehension to iterate over the list
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]
# declare a scaler object, specifying the columns you want to scale
absenteeism_scaler = CustomScaler(columns_to_scale)
# fit the data (calculate mean and standard deviation); they are automatically stored inside the object 
absenteeism_scaler.fit(unscaled_inputs)
# standardizes the data, using the transform method 
# in the last line, we fitted the data - in other words
# we found the internal parameters of a model that will be used to transform data. 
# transforming applies these parameters to our data
# note that when you get new data, you can just call 'scaler' again and transform it in the same way as now
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
#scaled_inputs= scaled_inputs.dropna()
# the scaled_inputs are now an ndarray, because sklearn works with ndarrays
scaled_inputs
# check how this method works
train_test_split(scaled_inputs, targets)
# declare 4 variables for the split
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, #train_size = 0.75, 
                                                                            test_size = 0.25, random_state = 20)
# check the shape of the train inputs and targets
print (x_train.shape, y_train.shape)
# check the shape of the test inputs and targets
print (x_test.shape, y_test.shape)
#Now, we will create an array of Classifiers and append different classification models to our array
classifiers = [] 

mod1 = xgboost.XGBClassifier()
classifiers.append(mod1)
mod2 = svm.SVC()
classifiers.append(mod2)
mod3 = RandomForestClassifier()
classifiers.append(mod3)
mod4 = LogisticRegression()
classifiers.append(mod4)
mod5 = KNeighborsClassifier(3)
classifiers.append(mod5)
mod6 = AdaBoostClassifier()
classifiers.append(mod6)
mod7= GaussianNB()
classifiers.append(mod7)
#Lets fit the models into anarray

for clf in classifiers:
    clf.fit(x_train,y_train)
    y_pred= clf.predict(x_test)
    y_tr = clf.predict(x_train)
    acc_tr = accuracy_score(y_train, y_tr)
    acc = accuracy_score(y_test, y_pred)
    mn = type(clf).__name__
    
    print(clf)
    print("Accuracy of trainset %s is %s"%(mn, acc_tr))
    print("Accuracy of testset %s is %s"%(mn, acc))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of testset %s is %s"%(mn, cm))

# So we stick with SVC as the best model in this case

SVCclf = svm.SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
SVCclf.fit(x_train,y_train)
ypredtrain = SVCclf.predict(x_train)
y_pred = SVCclf.predict(x_test)
acc_tr = accuracy_score(y_train, ypredtrain)
acc = accuracy_score(y_test, y_pred)
print(acc_tr, acc)

# pickle the model file
with open('model', 'wb') as file:
    pickle.dump(SVCclf, file)
param = [{'C': [1, 10, 20], 'gamma': ['scale', 'auto'], 'kernel': ['rbf','linear']}]

# Create a classifier object with the classifier and parameter candidates
svcclf = GridSearchCV(estimator=svm.SVC(), param_grid=param)

# Train the classifier on data1's feature and target data
svcclf.fit(x_train, y_train)  

# View the accuracy score
print('Best score for data1:', svcclf.best_score_) 


# View the best parameters for the model found using grid search
print('Best C:',svcclf.best_estimator_.C) 
print('Best Kernel:',svcclf.best_estimator_.kernel)
print('Best Gamma:',svcclf.best_estimator_.gamma)
# inteesting result but SVC & Adaboost model seems to be doing very well, 
#So let see if we can improve on accuracy by tunning some parameters, otherwise we stick to the default parameters
#lets start with Adaboost

#Creating a grid of hyperparameters
boost = AdaBoostClassifier(base_estimator=None)
parameters = {'n_estimators': (50,100,150,200),
              'learning_rate': (0.1,0.5,1, 2)}
adab = GridSearchCV(boost, parameters)
adab.fit(x_train, y_train)
score = adab.best_score_
param = adab.best_params_
print(score)
print(param)
# create a logistic regression object
rr = LogisticRegression()
# fit our train inputs
# that is basically the whole training part of the machine learning
rr.fit(x_train,y_train)
rr.score(x_train,y_train)
# save the names of the columns in an ad-hoc variable
feature_name = unscaled_inputs.columns.values
df = pd.DataFrame (columns=['Feature name'], data = feature_name)
# add the coefficient values to the df
df['Coefficient'] = np.transpose(rr.coef_)
# move all indices by 1
df.index = df.index + 1
# add the intercept at index 0
df.loc[0] = ['Intercept', rr.intercept_[0]]
# sort the df by index
df = df.sort_index()
# create a new Series called: 'Odds ratio' which will show the.. odds ratio of each feature
df['Odds_ratio'] = np.exp(df.Coefficient)
# sort the table according to odds ratio
df.sort_values('Odds_ratio', ascending=False)
