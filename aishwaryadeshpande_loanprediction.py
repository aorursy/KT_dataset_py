# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)


test = pd.read_csv("../input/loanprediction/test_lAUu6dG.csv")

train = pd.read_csv("../input/loanprediction/train_ctrUa4K.csv")
train.head()
#Catergorical Variables

plt.figure(1) 

plt.subplot(221) 

train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender') 

plt.subplot(222) 

train['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 

plt.subplot(223) 

train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 

plt.subplot(224) 

train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 

plt.show()
#Ordinal Variables

plt.figure(1)

plt.subplot(131)

train['Dependents'].value_counts().plot.bar(figsize=(24,6), title= 'Dependents') 

plt.subplot(132)

train['Education'].value_counts().plot.bar(figsize=(24,6), title= 'Education') 

plt.subplot(133)

train['Property_Area'].value_counts().plot.bar(figsize=(24,6), title= 'Property_Area') 
#Quantitative Variables

plt.figure(1)

plt.subplot(131)

train['ApplicantIncome'].value_counts().plot.hist(figsize=(24,6), title= 'ApplicantIncome') 

plt.subplot(132)

train['LoanAmount'].value_counts().plot.hist(figsize=(24,6), title= 'LoanAmount') 

plt.subplot(133)

train['CoapplicantIncome'].value_counts().plot.hist(figsize=(24,6), title= 'CoapplicantIncome') 
#Catergorical Variables 

cat_var = ['Gender','Married','Self_Employed','Credit_History','Dependents','Education','Property_Area']

for col in cat_var:

    Gender=pd.crosstab(train[col],train['Loan_Status']) 

    Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

    plt.show()
quant_arr = ['ApplicantIncome','LoanAmount','CoapplicantIncome']

for col in quant_arr:

    train.groupby('Loan_Status')[col].mean().plot.bar()

    plt.xlabel(col)

    plt.show()
train.corr()
def survival_stacked_bar(variable):

    approved=train[train["Loan_Status"]==1][variable].value_counts()/len(train["Loan_Status"]==1)

    NotApproved=train[train["Loan_Status"]==0][variable].value_counts()/len(train["Loan_Status"]==0)

    data=pd.DataFrame([approved,NotApproved])

    data.index=["Approved","NotApproved"]

    data.plot(kind="bar",stacked=True,title="Percentage")

    return data.head()
missing_data = train.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print("")
def replceWithMode(array,data):

    for col in array: 

        data[col].replace(np.nan,data[col].mode()[0],inplace = True)

mode_array = ['Self_Employed','Dependents','Credit_History','Loan_Amount_Term','Married','Gender']

replceWithMode(mode_array,train)

replceWithMode(mode_array,test)



train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)
train['LoanAmount_log'] = np.log(train['LoanAmount']+1) 

train['LoanAmount_log'].hist(bins=20) 

test['LoanAmount_log'] = np.log(test['LoanAmount']+1)
train["Loan_Amount_Term"] = train["Loan_Amount_Term"]/12

test["Loan_Amount_Term"] = test["Loan_Amount_Term"]/12
#Creating total income with applicants income and coapplicants income 

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome'] 

test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']



train['Total_Income_log'] = np.log(train['Total_Income']+1)

test['Total_Income_log'] = np.log(test['Total_Income']+1)
train['LoanAmount_log'] = np.log(train['LoanAmount']*1000+1)

test['LoanAmount_log'] = np.log(test['LoanAmount']*1000+1)
#Creating EMI

train['EMI']=train['LoanAmount']/train['Loan_Amount_Term'] 

test['EMI']=test['LoanAmount']/test['Loan_Amount_Term'] 

train['EMI']
train['Balance Income']=train['Total_Income']-(train['EMI']*1000) 

test['Balance Income']=test['Total_Income']-(test['EMI']*1000)
#train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 

#test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
#Binary Data

loan_map = {'Y':1,'N':0}

train['Loan_Status'] = train['Loan_Status'].map(loan_map)
train[['Loan_Status']] = train[['Loan_Status']].apply(pd.to_numeric)

train['Dependents'].replace('3+',3,inplace = True)

test['Dependents'].replace('3+',3,inplace = True)

train[['Dependents']] = train[['Dependents']].apply(pd.to_numeric)

test[['Dependents']] = test[['Dependents']].apply(pd.to_numeric)
X = train.drop(["Loan_Status"],axis=1)

X = X.drop(["Loan_ID"],axis=1)



#X = pd.get_dummies(X)

colnames = ['Gender','Married','Education','Self_Employed','Property_Area']

for col in colnames:

    X[col] = pd.factorize(X[col])[0]

y = train[['Loan_Status']]
X_test_data = test.drop(["Loan_ID"],axis=1)

test_colnames = ['Gender','Married','Education','Self_Employed','Property_Area']

for col in test_colnames:

    X_test_data[col] = pd.factorize(X_test_data[col])[0] 

#X_test_data = pd.get_dummies(X_test_data)

X_test_data.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.ensemble import AdaBoostClassifier #For Classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



def submission(model,feat):        

    Y_pred = model.predict( X_test_data[feat])

    Y_df = pd.DataFrame(test['Loan_ID'])

    Y_df['Loan_Status'] = Y_pred

    loan = {1:'Y',0:'N'}

    Y_df['Loan_Status'] = Y_df['Loan_Status'].map(loan)

    return Y_df



def confusion_matrix_model(model_used,x_test):

    cm=confusion_matrix(y_test,model_used.predict(x_test))

    col=["Predicted Not Approved","Predicted Approved"]

    cm=pd.DataFrame(cm)

    cm.columns=["Predicted Not Approved","Predicted Approved"]

    cm.index=["Actual Not Approved","Actual Approved"]

    cm[col]=np.around(cm[col].div(cm[col].sum(axis=1),axis=0),decimals=2)

    return cm



x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)



def ModelDevelopement(model,feat):

    fit = model.fit(x_train[feat],y_train)

    pred = fit.predict(x_test[feat])

    print(confusion_matrix_model(log_reg,x_test[feat]))

    print(accuracy_score(y_test,pred))

    fit2 =  model.fit(X[feat],y)

    df = submission(fit2,feat)

    return pred,df

# Create a random forest classifier

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

# Train the classifier

clf.fit(X,y)

names = X.columns



print ("Features sorted by their score:")

RF_feat = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names),reverse=True))
RF_feat
pd.DataFrame(X_res)
feat = ["Credit_History","Balance Income","ApplicantIncome","Total_Income_log","LoanAmount_log","Property_Area","Dependents"]

#feat = ["Credit_History","Balance Income","Total_Income","ApplicantIncome","EMI","Dependents","Property_Area_Semiurban","LoanAmount","CoapplicantIncome"]

log_reg=LogisticRegression(C = 4,penalty = 'l2')

log_pred,log_sub_df = ModelDevelopement(log_reg,feat)

create_download_link(log_sub_df)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)

{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(x_train[feat], y_train)
rf_random.best_params_
RF = RandomForestClassifier(n_estimators = 800,min_samples_split = 10, random_state = 0, n_jobs = -1,min_samples_leaf = 4,max_depth = 50,max_features = 'sqrt',bootstrap = 'True') 

RF_pred,RF_sub_df = ModelDevelopement(RF,feat)

create_download_link(RF_sub_df)
xgb = XGBClassifier(n_estimators=35, max_depth=3,learning_rate = 0.1)

xgb_pred,xgb_sub_df = ModelDevelopement(xgb,feat)

create_download_link(xgb_sub_df)


dt = DecisionTreeClassifier() 

adB = AdaBoostClassifier(n_estimators=300, base_estimator=dt,learning_rate=0.1) 

adB_pred,adB_sub_df = ModelDevelopement(adB,feat)

create_download_link(adB_sub_df)
from sklearn import svm

 

svm = svm.SVC(kernel='linear', C=1, gamma=1) 

svm_pred,svm_sub_df = ModelDevelopement(svm,feat)

create_download_link(svm_sub_df)
import sklearn.exceptions
from mlxtend.classifier import StackingClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC



estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('svr', make_pipeline(StandardScaler(),LinearSVC(random_state=42)))]

stack = StackingClassifier(classifiers=[RF, xgb],meta_classifier=log_reg)

stack_pred,stack_sub_df = ModelDevelopement(stack,feat)

create_download_link(stack_sub_df)
from sklearn import model_selection

from mlxtend.classifier import EnsembleVoteClassifier



eclf = EnsembleVoteClassifier(clfs=[log_reg,xgb,RF], weights=[1,1,1])

eclf_pred,eclf_sub_df = ModelDevelopement(eclf,feat)

create_download_link(svm_sub_df)
final_pred = 0.3*log_pred+ 0.1*adB_pred+0.4*RF_pred+0.2*xgb_pred

for i in range(len(final_pred)):

    if(final_pred[i]  < 0.5):

        final_pred[i] = 0

    else:

        final_pred[i]= 1

print(accuracy_score(y_test,final_pred))

print(confusion_matrix(y_test,final_pred))