from datetime import datetime, timedelta,date

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from IPython.display import Image

from IPython.core.display import HTML 



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import mean_squared_error



import sklearn.metrics as metrics

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



from sklearn.cluster import KMeans
TelecomChurn = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#first few rows

TelecomChurn.head()
# data summary

print("Data dimension:",TelecomChurn.shape)

TelecomChurn.info()
# encoding the churn variable into 0 and 1

TelecomChurn['Churn'] = TelecomChurn['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

TelecomChurn.head()
# changing TotalCharges column from object to float

TelecomChurn['TotalCharges'] = TelecomChurn['TotalCharges'].apply(pd.to_numeric, downcast='float', errors='coerce')

print(TelecomChurn.dtypes)
# check for null and total observations related to it

null_columns=TelecomChurn.columns[TelecomChurn.isnull().any()]

TelecomChurn[null_columns].isnull().sum()
# drop na variables

TelecomChurn = TelecomChurn.dropna()

TelecomChurn.shape
# summary description of the numeric variables of the dataset

TelecomChurn[['tenure', 'MonthlyCharges', 'TotalCharges']].describe()
# to check the number of unique values in each of the columns

for col in list(TelecomChurn.columns):

    print(col, TelecomChurn[col].nunique())
# calculate the proportion of churn vs non-churn

TelecomChurn['Churn'].mean()
# calculate the proportion of churn by gender

churn_by_gender = TelecomChurn.groupby(by='gender')['Churn'].sum() / TelecomChurn.groupby(by='gender')['Churn'].count() * 100.0

print('Churn by gender:',churn_by_gender)
# calculate the proportion of churn by contract

churn_by_contract = TelecomChurn.groupby(by='Contract')['Churn'].sum() / TelecomChurn.groupby(by='Contract')['Churn'].count() * 100.0

print('Churn by contract:',churn_by_contract)
# calculate the proportion of churn by payment method

churn_by_payment = TelecomChurn.groupby(by='PaymentMethod')['Churn'].sum() / TelecomChurn.groupby(by='PaymentMethod')['Churn'].count() * 100.0

print('Churn by payment method:',churn_by_payment)

pd.DataFrame(churn_by_payment)
# figure

ax = churn_by_payment.plot(

    kind='bar',

    color='skyblue',

    grid=False,

    figsize=(10, 7),

    title='Churn Rates by Payment Methods'

)



ax.set_xlabel('Payment Methods')

ax.set_ylabel('Churn rate (%)')



plt.show()
# proportion of churn by gender and contract

churn_gendercontract = TelecomChurn.groupby(['gender', 'Contract'])['Churn'].sum()/TelecomChurn.groupby(['gender', 'Contract'])['Churn'].count()*100

churn_gendercontract
# keep gender in row and contract by column

churn_gendercontract1 = churn_gendercontract.unstack('Contract').fillna(0)

churn_gendercontract1 
# figure

ax = churn_gendercontract1.plot(

    kind='bar', 

    grid= False,

    figsize=(10,7)

)



ax.set_title('Churn rates by Gender & Contract Status')

ax.set_xlabel('Gender')

ax.set_ylabel('Churn rate (%)')



plt.show()
# observations by citizen type

TelecomChurn['SeniorCitizen'].value_counts()
# Total observations by citizen type, contract and tech support 

TelecomChurn.groupby(['SeniorCitizen','Contract','TechSupport'])['Churn'].count()
# proportion of churn by gender and contract

churn_citizentechcontract = TelecomChurn.groupby(['SeniorCitizen','Contract','TechSupport'])['Churn'].sum()/TelecomChurn.groupby(['SeniorCitizen','Contract','TechSupport'])['Churn'].count()*100

churn_citizentechcontract
# keep gender and payment method in row and contract by column

churn_citizentechcontract1 = churn_citizentechcontract.unstack(['TechSupport']).fillna(0)

churn_citizentechcontract1
# figure

ax = churn_citizentechcontract1.plot(

    kind='bar', 

    grid= False,

    figsize=(10,7)

)



ax.set_title('Churn rates by Citizen Type, Tech Support & Contract Status')

ax.set_xlabel('Non-Senior:"0"   Senior:"1"')

ax.set_ylabel('Churn rate (%)')



plt.xticks()

plt.show()
# summary of tenure, monthly charges and total charges

TelecomChurn[['tenure','MonthlyCharges','TotalCharges']].describe()
# plot a histogram

plt.hist(TelecomChurn['tenure'], bins= 100, alpha=0.5,)

plt.title('Frequency Distribution by Tenure')

plt.xlabel('Tenure')

plt.ylabel('Frequency')

plt.show()





plt.hist(TelecomChurn['tenure'], cumulative=1, density =True, bins= 100)

plt.title('Cumulative Frequency Distribution by Tenure')

plt.xlabel('Tenure')

plt.ylabel('Cumulative Frequency Distribution')

plt.show()
# proportion of churn by tenure

churn_monthlycharges = TelecomChurn.groupby(by = 'tenure')['Churn'].mean().reset_index()

churn_monthlycharges



plt.figure(figsize=(10,7))

plt.scatter(churn_monthlycharges.tenure, churn_monthlycharges.Churn)

plt.title('Churn Rate by Tenure')

plt.xlabel('Tenure')

plt.ylabel('Churn Rate')

plt.show()
# proportion of churn by MonthlyCharges

churn_monthlycharges = TelecomChurn.groupby(by = 'MonthlyCharges')['Churn'].mean().reset_index()

plt.figure(figsize=(6,5))

plt.scatter(churn_monthlycharges.MonthlyCharges, churn_monthlycharges.Churn)

plt.title('Churn Rate by Monthly Charges')

plt.xlabel('Monthly Charges')

plt.ylabel('Churn Rate')

plt.show()



# proportion of churn by TotalCharges

churn_totalcharges = TelecomChurn.groupby(by = 'TotalCharges')['Churn'].mean().reset_index()

plt.figure(figsize=(6,5))

plt.scatter(churn_totalcharges.TotalCharges, churn_totalcharges.Churn)

plt.title('Churn Rate by Total Charges')

plt.xlabel('Total Charges')

plt.ylabel('Churn Rate')

plt.show()
# finding correlations

corrdata = TelecomChurn[['Churn','tenure','MonthlyCharges','TotalCharges']]

corr = corrdata.corr()

# plot the heatmap

sns.heatmap(corr,cmap="coolwarm",

        xticklabels=corrdata.columns,

        yticklabels=corrdata.columns,annot=True)
# segmenting based on data type and pre-processing

#customer id col

Id_col     = ['customerID']

#Target column. y should be an array

target_col = ["Churn"]

y = (TelecomChurn[target_col]).values.ravel()

# cluster column 

cluster_col = ["tenure"]

#categorical columns with categories less than 6

cat_cols   = TelecomChurn.nunique()[TelecomChurn.nunique() < 6].keys().tolist()

cat_cols   = [x for x in cat_cols if x not in target_col]

print(cat_cols)

#Binary columns with 2 values

bin_cols   = TelecomChurn.nunique()[TelecomChurn.nunique() == 2].keys().tolist()

print(bin_cols)

#Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]

print(multi_cols)

# continuous column

cont_col = ["tenure","MonthlyCharges"]

print(cont_col)

print(y)
#Label encoding Binary columns

le = LabelEncoder()

binary = TelecomChurn[bin_cols]

print(binary.shape) 

print(binary.info())

binary.head()

for i in bin_cols :

    binary[i] = le.fit_transform(binary[i])
# multi-label categorical columns

dummy_vars = pd.get_dummies(TelecomChurn[multi_cols])

print(dummy_vars.shape)

print(dummy_vars.info())
#Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(TelecomChurn[cont_col])

scaled = pd.DataFrame(TelecomChurn,columns= cont_col)

scaled.shape

print(scaled.info())
# creating a dataset to combine pre-processed variables

X = pd.concat([binary,scaled,dummy_vars], axis = 1)

# drop churn variable from the X dataset

X = X.drop(['Churn'],axis=1)

print(X.shape)

print(X.info())
# import machine learning libraries

from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

import xgboost as xgb



# creating the function

# XGBoost and SVC functions are used while modeling and are thus not presented here

logreg = LogisticRegression(solver='lbfgs', max_iter = 10000)

DT = DecisionTreeClassifier()

rfc = RandomForestClassifier()

knn = KNeighborsClassifier()
# recursive feature extraction for the top 15 features

rfe = RFE(rfc, 10)

rfe = rfe.fit(X, y)

print(rfe.support_)

print(rfe.ranking_)



#identifying columns for RFE

rfe_data = pd.DataFrame({"rfe_support" :rfe.support_,

                       "columns" : [i for i in X.columns if i not in Id_col + target_col],

                       "ranking" : rfe.ranking_,

                      })



# extract columns as a list

rfe_var = rfe_data[rfe_data["rfe_support"] == True]["columns"].tolist()



rfe_data
# select a subset of variables for the dataframe based on RFE method

X1 = X[rfe_var]
# running a logistic regression

# copy the dataset 

X2 = X1

# manually add intercept

X2['intercept'] = 1.0;

#X2.head()

logit_model=sm.Logit(y,X2)

result=logit_model.fit()

print(result.summary2())
# create a train and test set with the new selected variables

Xtrain, Xtest, ytrain,ytest = train_test_split(X1,y,test_size = 0.2,random_state = 111)
print('Ratio of churn in the training sample:',ytrain.mean())

print('Ratio of churn in the training sample:',ytest.mean())
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(estimator=logreg, param_grid = parameters,cv = 10,scoring = 'accuracy')

grid.fit(Xtrain,ytrain)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_)
def result(X_test,y_test):



    y_test_pred = grid.predict(X_test)

    print('Accuracy score:{:.2f}'.format(accuracy_score(y_test, y_test_pred)))

    print(                                                                                )



    confusionmat_data = pd.DataFrame({'y_Predicted': y_test_pred,'y_Actual': y_test},columns=['y_Actual','y_Predicted'])

    confusion_matrix = pd.crosstab(confusionmat_data['y_Actual'], confusionmat_data['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    print('Confusion Matrix:\n {}\n'.format(confusion_matrix))

    print(                                                                               )



    class_report = classification_report(y_test, y_test_pred)

    print('Classification report:\n {}\n'.format(class_report))

    print(                                                                               )



    mse = mean_squared_error(y_test, y_test_pred)

    rmse = np.sqrt(mse)

    print('Mean-squared error:\n {}\n'.format(rmse))



    # predict probabilities

    #probs = grid.predict_proba(X_test)

    #probs = grid.predict(X_test)



    fpr, tpr, threshold = metrics.roc_curve(y_test, y_test_pred)

    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

# result

result(Xtest,ytest)
parameters = {'min_samples_split': [10,100,1000,10000],

              'max_depth':[2,5,10,100,150,200,250]}



grid = GridSearchCV(estimator= DT, param_grid = parameters,cv = 10,scoring = 'accuracy')

grid.fit(Xtrain,ytrain)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_)
grid.best_estimator_.feature_importances_

for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,Xtrain.columns),reverse= True)[:5]:

    print(name, importance)

    

featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = Xtrain.columns)

featureimp_plot.nlargest(5).plot(kind='barh')  
# result

result(Xtest,ytest)
parameters = {'n_estimators': [1,5,10,100,200],'min_samples_split': [10,100,1000,10000],'max_depth':[2,5,10,100,150,200,250]}



grid = GridSearchCV(estimator= rfc, param_grid = parameters,cv = 10,scoring = 'accuracy')

grid.fit(Xtrain,ytrain)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_)
grid.best_estimator_.feature_importances_

for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,Xtrain.columns),reverse= True)[:5]:

    print(name, importance)

    

featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = Xtrain.columns)

featureimp_plot.nlargest(5).plot(kind='barh')   
# result

result(Xtest,ytest)
#building the model & printing the score

parameter = {

'max_depth': [1,5,10,15],

'n_estimators': [50,100,150,300],

'learning_rate': [0.01, 0.1, 0.3],

}



grid = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic'), param_grid = parameter, cv= 5, scoring='balanced_accuracy')

grid.fit(Xtrain,ytrain)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_)
featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = Xtrain.columns)

featureimp_plot.nlargest(5).plot(kind='barh') 
# result

result(Xtest,ytest)
parameter = {'C': [5,10, 100]}

grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid = parameter, cv= 4)



grid.fit(Xtrain,ytrain)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_)
# result

result(Xtest,ytest)
from imblearn.over_sampling import SMOTE
#Split train and test data

smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(X,y,

                                                                         test_size = .25 ,

                                                                         random_state = 111)



#oversampling minority class using smote

os = SMOTE(random_state = 0)

os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)

os_smote_X = pd.DataFrame(data = os_smote_X,columns= X.columns)

os_smote_Y = pd.DataFrame(data = os_smote_Y,columns= ['Churn'])



print(os_smote_X.shape)

print(os_smote_Y.shape)
rfe = RFE(rfc, 10)

rfe = rfe.fit(X, y)

print(rfe.support_)

print(rfe.ranking_)



#identified columns Recursive Feature Elimination

rfe_data = pd.DataFrame({"rfe_support" :rfe.support_,

                       "columns" : [i for i in X.columns if i not in Id_col + target_col],

                       "ranking" : rfe.ranking_,

                      })

selected_cols = rfe_data[rfe_data["rfe_support"] == True]["columns"].tolist()



rfe_data

print(selected_cols)
# calculate the proportion of churn is now equal to the no churn

os_smote_Y.mean()
#train and test data under SMOTE

train_smoterfe_X = os_smote_X[selected_cols]

train_smoterfe_Y = os_smote_Y.values.ravel()

test_smoterfe_X  = smote_test_X[selected_cols]

test_smoterfe_Y  = smote_test_Y
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(estimator=logreg, param_grid = parameters,cv = 10,scoring = 'accuracy')

grid.fit(train_smoterfe_X,train_smoterfe_Y)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_)

# result

result(test_smoterfe_X,test_smoterfe_Y)
parameters = {'min_samples_split': [10,100,1000,10000],

              'max_depth':[2,5,10,100,150,200,250]}



grid = GridSearchCV(estimator= DT, param_grid = parameters,cv = 10,scoring = 'accuracy')

grid.fit(train_smoterfe_X,train_smoterfe_Y)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_) 



grid.best_estimator_.feature_importances_

for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,train_smoterfe_X.columns),reverse= True)[:5]:

    print(name, importance)

    

featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = train_smoterfe_X.columns)

featureimp_plot.nlargest(5).plot(kind='barh')   
# result

result(test_smoterfe_X,test_smoterfe_Y)
parameters = {'n_estimators': [1,5,10,100,200],'min_samples_split': [10,100,1000,10000],'max_depth':[2,5,10,100,150,200,250]}



grid = GridSearchCV(estimator= rfc, param_grid = parameters,cv = 10,scoring = 'accuracy')

grid.fit(train_smoterfe_X,train_smoterfe_Y)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_)
grid.best_estimator_.feature_importances_

for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,train_smoterfe_X.columns),reverse= True)[:5]:

    print(name, importance)

    

featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = train_smoterfe_X.columns)

featureimp_plot.nlargest(5).plot(kind='barh')   
# result

result(test_smoterfe_X,test_smoterfe_Y)
#building the model & printing the score

parameter = {

'max_depth': [1,5,10,15],

'n_estimators': [50,100,150,300],

'learning_rate': [0.01, 0.1, 0.3],

}



grid = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic'), param_grid = parameter, cv= 5, scoring='balanced_accuracy')

grid.fit(train_smoterfe_X,train_smoterfe_Y)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_)
grid.best_estimator_.feature_importances_

for name, importance in sorted(zip(grid.best_estimator_.feature_importances_,train_smoterfe_X.columns),reverse= True)[:5]:

    print(name, importance)

    

featureimp_plot = pd.Series(grid.best_estimator_.feature_importances_, index = train_smoterfe_X.columns)

featureimp_plot.nlargest(5).plot(kind='barh')  
# result

result(test_smoterfe_X,test_smoterfe_Y)
parameter = {'C': [1,5,10, 100]}

grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid = parameter, cv= 4)

grid.fit(train_smoterfe_X,train_smoterfe_Y)



print("Best score: %0.3f" % grid.best_score_)

print("Best parameters:", grid.best_params_)
# result

result(test_smoterfe_X,test_smoterfe_Y)