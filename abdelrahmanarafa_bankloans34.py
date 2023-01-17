#import basic libraries to handle data set and plot graphs

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import chi2 , f_classif 

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.model_selection import GridSearchCV
#read data set

loans = pd.read_csv('../input/bank-loans/Bank Loans.csv')

loans.head()
loans.dropna(how='all',inplace=True)

print("NUmber of data rows are :    ",loans.shape[0])

print("NUmber of data columns are : ",loans.shape[1])

print(loans.info())
column_names=list(loans.columns)

column_len=len(column_names)

print('The column names are:',column_names )

print( "Number of clolumns are: ", column_len)



for c in range(0,column_len):

  U=loans[column_names[c]].unique()

  n=len(loans[column_names[c]].unique())

  print("there are " ,n,  "      unique value   ")

  print('the unique values of:   ', column_names[c], "        are           ", U)

  print("-------------------------------------------------------------------------------------------------------------------")
loans.isnull().sum()


ax = loans['Credit Score'].plot.kde(ind=range(int(loans['Credit Score'].min()),int(loans['Credit Score'].max())))
loans['Number of Credit Problems']=loans['Number of Credit Problems'].replace([2,3,4,5,6,7,8,9,10,11,12,15],3)

loans.boxplot(by ='Number of Credit Problems', column =['Credit Score'], grid = False, figsize=(30,30)) 
#loans['Credit Score']=loans['Credit Score'].fillna(value=loans['Credit Score'].mode()[0],inplace=True)

from sklearn.impute import SimpleImputer





imp = SimpleImputer(missing_values = np.nan, strategy ='most_frequent')

loans['Credit Score'] = imp.fit_transform(pd.DataFrame(loans['Credit Score']))
loans['Credit Score'].isnull().sum()


#ax = loans['Annual Income'].plot.kde(ind=range(int(loans['Annual Income'].min()),int(loans['Annual Income'].max())))

z =loans['Annual Income'].value_counts()

zd=pd.DataFrame(z)

sns.distplot(loans['Annual Income']);

print(zd)

loans['Years in current job']=loans['Years in current job'].str.extract('(\d+)').astype(float)

ax = loans['Years in current job'].plot.kde(ind=range(int(loans['Years in current job'].min()),int(loans['Years in current job'].max())))
X = loans.drop(['Number of Credit Problems'], axis=1, inplace=False)

y = loans['Number of Credit Problems']


from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

object_column=['Loan Status','Term','Home Ownership','Purpose']

for col in object_column:

        le = preprocessing.LabelEncoder()

        X[col] = le.fit_transform(X[col])

        l=list(le.classes_)

        print('classed found : ' , l)

        print('Update data is : \n' ,X[col] )
loans.info()
X.describe().T
problems = y.value_counts()

p=pd.DataFrame(problems)

print(p)

print(y.unique())

p.plot.pie(y="Number of Credit Problems",figsize=(7,7),autopct='%1.1f%%')

print(y.describe())
from sklearn.impute import SimpleImputer





imp = SimpleImputer(missing_values = np.nan, strategy ='most_frequent')

loans['Years in current job'] = imp.fit_transform(pd.DataFrame(loans['Years in current job']))
from sklearn.impute import SimpleImputer





imp = SimpleImputer(missing_values = np.nan, strategy ='most_frequent')

loans['Annual Income'] = imp.fit_transform(pd.DataFrame(loans['Annual Income']))
loans.dropna(subset=['Maximum Open Credit','Bankruptcies','Tax Liens'],inplace=True)

loans.drop(['Months since last delinquent'], axis=1,inplace=True)

loans.isnull().sum()
X = loans.drop(['Number of Credit Problems','Customer ID','Loan ID'], axis=1, inplace=False)
X.info
#print('Original X Shape is ' , X.shape)

FeatureSelection = SelectPercentile(score_func = f_classif, percentile=20) # score_func can = f_classif

X = FeatureSelection.fit_transform(X, y)



#showing X Dimension 

#print('X Shape is ' , X.shape)

#print('Selected Features are : ' , FeatureSelection.get_support())
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

X = scaler.fit_transform(X)



#showing data

#print('X \n' , X[:10])

#print('y \n' , y[:10])
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 44,shuffle =True)
RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=10,max_depth=3,random_state=None) #criterion can be also : entropy 

RandomForestClassifierModel.fit(X_train, y_train)



#Calculating Details

print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))

print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))

print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)

print('----------------------------------------------------')

#Calculating Prediction

y_pred = RandomForestClassifierModel.predict(X_test)

y_pred_prob = RandomForestClassifierModel.predict_proba(X_test)

print('Predicted Value for RandomForestClassifierModel is : ' , y_pred[:10])

print('Prediction Probabilities Value for RandomForestClassifierModel is : ' , y_pred_prob[:10])



#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

# precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’,sample_weight=None)



PrecisionScore = precision_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Precision Score is : ', PrecisionScore)



#----------------------------------------------------

#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2  

# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)



RecallScore = recall_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Recall Score is : ', RecallScore)
SelectedModel = RandomForestClassifier()

SelectedParameters = { 

            "n_estimators"      : [100,200,300],

            "max_depth"      : [1,2,3],

            "min_samples_split" : [2,4,8],

            "bootstrap": [True, False],

            }



# #=======================================================================

GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)

GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]



# Showing Results

print('All Results are :\n', GridSearchResults )

print('Best Score is :', GridSearchModel.best_score_)

print('Best Parameters are :', GridSearchModel.best_params_)

print('Best Estimator is :', GridSearchModel.best_estimator_)
SVCModel = SVC(kernel= 'poly',# it can be also linear,poly,sigmoid,precomputed

               max_iter=10,C=1.0,gamma='auto')

SVCModel.fit(X_train, y_train)



#Calculating Details

print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))

print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))

print('----------------------------------------------------')



#Calculating Prediction

y_pred = SVCModel.predict(X_test)

print('Predicted Value for SVCModel is : ' , y_pred[:10])

#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

# precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’,sample_weight=None)



PrecisionScore = precision_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Precision Score is : ', PrecisionScore)



#----------------------------------------------------

#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2  

# recall_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)



RecallScore = recall_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples

print('Recall Score is : ', RecallScore)
SelectedModel = SVC(gamma='auto')

SelectedParameters = {'kernel':('poly', 'rbf'), 'C':[1,2,3,4,5]}

GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2,return_train_score=True)

GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]



# Showing Results

print('All Results are :\n', GridSearchResults )

print('Best Score is :', GridSearchModel.best_score_)

print('Best Parameters are :', GridSearchModel.best_params_)

print('Best Estimator is :', GridSearchModel.best_estimator_)
