# Main libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Processing libraries

from sklearn.preprocessing import StandardScaler



# Model libraries

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB



# Testing libraries

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, learning_curve

from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report, roc_curve, auc





# Other useful things

# View all columns of dataframes

pd.options.display.max_columns = None



# View all cell outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)

# Load the data

d = pd.read_csv('../input/train.csv')

loan_default = d['loan_default']
# Looking at the data

d.head(8)

d.info()

d.describe()

d.describe(include=['O'])
# Look at data distribution, count values, visualisations (to decide what variables to use for the model)

# Look at the Starter: Loan Default Prediction for data analysis.
# Here I'm not going into details about visualisations

# Look at data distribution, count values, visualisations (to decide what variables to use for the model)

# Look at the Starter: Loan Default Prediction for data analysis

# I've decided based on things above how I would deal with data.



# Must do - checklist:

#    - create new variables

#    - pca dimentionality reduction if possible

#    - drop irrelevant variables

#    - fill nulls (mean, median, 0, depending on the data)

#    - deal with categorical - posibly create dummy variables

#    - normilise/standadise (int and floats)

#    - balance the data set (by the dependant variable)

#    - check for multicolinearity (posibly drop variables)

# Create new variables (combine old ones)

d['PRI.percent_default'] = d['PRI.OVERDUE.ACCTS'].divide(d['PRI.ACTIVE.ACCTS'], fill_value=0)

d['PRI.percent_default'][d['PRI.percent_default'] == np.inf] = 0



d['PRI.disbursed_to_outstanding'] = d['PRI.DISBURSED.AMOUNT'].divide(d['PRI.CURRENT.BALANCE'], fill_value=0)

d['PRI.disbursed_to_outstanding'][d['PRI.disbursed_to_outstanding'] == np.inf] = 0



d['SEC.percent_default'] = d['SEC.OVERDUE.ACCTS'].divide(d['PRI.ACTIVE.ACCTS'], fill_value=0)

d['SEC.percent_default'][d['SEC.percent_default'] == np.inf] = 0



d['SEC.disbursed_to_outstanding'] = d['SEC.DISBURSED.AMOUNT'].divide(d['SEC.CURRENT.BALANCE'], fill_value=0)

d['SEC.disbursed_to_outstanding'][d['SEC.disbursed_to_outstanding'] == np.inf] = 0



d['SEC_PRI.sanctioned'] = d['SEC.SANCTIONED.AMOUNT'].divide(d['PRI.SANCTIONED.AMOUNT'], fill_value=0)

d['SEC_PRI.sanctioned'][d['SEC_PRI.sanctioned'] == np.inf] = 0



d = d.round(3)
# Do pca (dimentionality reduction): 'branch_id', 'supplier_id', 'manufacturer_id'

# will be done in the later versions.
# Drop some irrelevant variables

d = d.drop(['UniqueID', 'branch_id', 'supplier_id', 'manufacturer_id',

            'Date.of.Birth', 'DisbursalDate', 'State_ID',

            'Current_pincode_ID', 'Employee_code_ID', 

            'MobileNo_Avl_Flag',  'PERFORM_CNS.SCORE.DESCRIPTION', 

            'PRI.OVERDUE.ACCTS', 'PRI.ACTIVE.ACCTS',  'PRI.DISBURSED.AMOUNT', 

            'PRI.CURRENT.BALANCE', 'SEC.OVERDUE.ACCTS', 'PRI.ACTIVE.ACCTS', 

            'SEC.DISBURSED.AMOUNT', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 

            'PRI.SANCTIONED.AMOUNT', 'PRI.NO.OF.ACCTS', 'SEC.NO.OF.ACCTS',

            'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH', 'loan_default'], axis=1)
# Dealing with null values, for numerical:

for col in d.columns:

    if d[col].dtype != object:

        d[col]=d[col].fillna(d[col].mean())
# Create dummies - for better model outcomes

d = pd.get_dummies(d)



# Data Normalisation

names = d.columns

scaler = StandardScaler()

scaled_d = scaler.fit_transform(d)

d = pd.DataFrame(scaled_d, columns=names)
d['loan_default'] = pd.Series(loan_default)



# Class balance

ld = d[d.loan_default == 1]

no_ld = d[d.loan_default == 0]



balanced_d = pd.concat([ld.sample(int(len(no_ld)*2/3), replace = True), no_ld] )

x = balanced_d.iloc[:,:-1]

y = balanced_d.iloc[:,-1:]
# Check for multicollinearity

#d.corr() #less representative way

plt.figure(figsize=(30,15))

sns.heatmap(d[d.columns].corr(),cmap="BrBG",annot=True)

plt.show();
# Drop values to avoid multicolinearity

x = x.drop(['asset_cost', 'Aadhar_flag'], axis=1)
# Random state

rs = 2



# Split the data to check which algorithms learn better (later on we can check )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rs)



# look at the shape of the data (many problems can arise from wrong shape)

x_train.shape

y_test.shape

# List of classifiers:

classifiers = [

LogisticRegression(random_state = rs),

DecisionTreeClassifier(random_state=rs),

RandomForestClassifier(n_estimators = 10, random_state=rs),

ExtraTreesClassifier(random_state=rs),

AdaBoostClassifier((DecisionTreeClassifier(random_state=rs)), random_state=rs, learning_rate=0.1),

GradientBoostingClassifier(random_state=rs),

GaussianNB(),

MLPClassifier(random_state=rs),

LinearDiscriminantAnalysis(),

QuadraticDiscriminantAnalysis()

]



# List of results that will occure:

clf_name = [] # names of the clasifiers

model_results = pd.DataFrame.copy(y_test) #resulting of prediction from the models



kfold = StratifiedKFold(n_splits=10) #cross-validation

cv_results = [] # scores from cross validation

cv_acc = [] # mean accuracy from cross validation, need to maximize

cv_std = [] # standard deviation from cross validation, need to minimise



cnfm = [] #confusion matrix

clr = [] #clasification report

roc_auc = [] #roc curve:

roc_tpr = []

roc_fpr = []

# Training the algorithms and results

for clf in classifiers:

    name = clf.__class__.__name__

    clf_name.append(name)

    

    #fitting and predictions

    model = clf.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    model_results[name] = y_pred

    

    #accuracy and log loss

    cv_results.append(cross_val_score(clf, x_train, y_train, scoring = "accuracy",cv = kfold))

    acc = round(accuracy_score(y_test, y_pred), 4) #need to maximise

    train_pred = clf.predict_proba(x_test)

    ll = round(log_loss(y_test, train_pred), 4) #need to minimise

    print(f'Accuracy: {acc} \t Log Loss: {ll} \t ---> {name} ')

    

    #confusion matrix, clasification report, roc curve

    cnfm.append(confusion_matrix(y_test, y_pred))

    clr.append(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_pred, y_test)

    roc_auc.append(auc(fpr, tpr))

    roc_tpr.append(tpr)

    roc_fpr.append(fpr)

    



for i in cv_results:

    cv_acc.append(i.mean())

    cv_std.append(i.std())

# Cross validation accuracy results graph

cv_res = pd.DataFrame({"CrossValMeans":cv_acc, "CrossValerrors": cv_std,"Algorithm":clf_name})



plt.figure(figsize=(15,8));

sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set2",orient = "h",**{'xerr':cv_std});

plt.xlabel("Mean Accuracy");

plt.title("Cross validation scores");

plt.grid();

plt.show;
# Confusion matrixes (not-normalized confusion matrix)

plt.figure(figsize=(10,10))

for i in range(len(classifiers)):

    plt.subplot(5,2,i+1) #adjust this acourding to the number of algorithms

    sns.heatmap(cnfm[i], annot=True, fmt="d",cmap="Blues");

    plt.xlabel('Predicted');

    plt.ylabel('Actual');

    plt.title(clf_name[i]);

plt.show;

#Clasification reports

for i in range(len(classifiers)):

    print (f"{clf_name[i]} Clasification Report:" );

    print (clr[i]);
# ROC Curve

plt.figure(figsize=(15,8))

for i in range(len(classifiers)):

    cm = ['red', 'blue', 'orange', 'green', 'pink', 'yellow', 'lightgreen', 'black', 'purple', 'lightblue'] #add more colours for more algorithms

    plt.plot(roc_fpr[i], roc_tpr[i], c=cm[i], lw=1, label=f'{clf_name[i]}: area = {roc_auc[i]}0.2f)')

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    plt.xlabel('False Positive Rate');

    plt.ylabel('True Positive Rate');

    plt.title('ROC curve: Receiver Operating Characteristic');

    plt.legend(loc="lower right");

plt.show;
# Best model tuning (tuning Extra trees)

# GridSearch

#ET_param_gs = {"max_depth": [None],

#              "max_features": [1, 3, 10],

#              "min_samples_split": [2, 3, 10],

#              "min_samples_leaf": [1, 3, 10],

#              "bootstrap": [False],

#              "n_estimators" :[100,300],

#              "criterion": ["gini"]}



# 4n_jobs for faster processing

#ET_gs = GridSearchCV(ExtraTreesClassifier(), param_grid = ET_param_gs, cv=kfold, n_jobs=4, scoring="accuracy", verbose = 1)



#models = [ET_gs]



#for model in models:

#    model.fit(x_train, y_train)

#    best_model = model.best_estimator_

#    score_bm = model.best_score_



#score_bm
best_model = ExtraTreesClassifier(random_state=rs)
# Load the test data

d2 = pd.read_csv('../input/test_bqCt9Pv.csv')

UniqueID = d2['UniqueID'] #save this variable for correct format of the submision





# Create new variables (combine old ones)

d2['PRI.percent_default'] = d2['PRI.OVERDUE.ACCTS'].divide(d2['PRI.ACTIVE.ACCTS'], fill_value=0)

d2['PRI.percent_default'][d2['PRI.percent_default'] == np.inf] = 0



d2['PRI.disbursed_to_outstanding'] = d2['PRI.DISBURSED.AMOUNT'].divide(d2['PRI.CURRENT.BALANCE'], fill_value=0)

d2['PRI.disbursed_to_outstanding'][d2['PRI.disbursed_to_outstanding'] == np.inf] = 0



d2['SEC.percent_default'] = d2['SEC.OVERDUE.ACCTS'].divide(d2['PRI.ACTIVE.ACCTS'], fill_value=0)

d2['SEC.percent_default'][d2['SEC.percent_default'] == np.inf] = 0



d2['SEC.disbursed_to_outstanding'] = d2['SEC.DISBURSED.AMOUNT'].divide(d2['SEC.CURRENT.BALANCE'], fill_value=0)

d2['SEC.disbursed_to_outstanding'][d2['SEC.disbursed_to_outstanding'] == np.inf] = 0



d2['SEC_PRI.sanctioned'] = d2['SEC.SANCTIONED.AMOUNT'].divide(d2['PRI.SANCTIONED.AMOUNT'], fill_value=0)

d2['SEC_PRI.sanctioned'][d2['SEC_PRI.sanctioned'] == np.inf] = 0



d2 = d2.round(3)





# Drop the irrelevant columns

d2 = d2.drop(['UniqueID', 'branch_id', 'supplier_id', 'manufacturer_id','Aadhar_flag',

            'Date.of.Birth', 'DisbursalDate', 'State_ID', 'asset_cost',

            'Current_pincode_ID', 'Employee_code_ID', 

            'MobileNo_Avl_Flag',  'PERFORM_CNS.SCORE.DESCRIPTION', 

            'PRI.OVERDUE.ACCTS', 'PRI.ACTIVE.ACCTS',  'PRI.DISBURSED.AMOUNT', 

            'PRI.CURRENT.BALANCE', 'SEC.OVERDUE.ACCTS', 'PRI.ACTIVE.ACCTS', 

            'SEC.DISBURSED.AMOUNT', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 

            'PRI.SANCTIONED.AMOUNT', 'PRI.NO.OF.ACCTS', 'SEC.NO.OF.ACCTS',

            'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH'], axis=1)





# Dealing with null values, for numerical:

for col in d2.columns:

    if d2[col].dtype != object:

        d2[col]=d2[col].fillna(d2[col].mean())

        

        



# Create dummies - for better model outcomes

d2 = pd.get_dummies(d2)



# Data Normalisation

names = d2.columns

scaler = StandardScaler()

scaled_d = scaler.fit_transform(d2)

d2 = pd.DataFrame(scaled_d, columns=names)

#train the model on the whole dataset and produce results:

model = best_model.fit(x, y)

loan_default_pred = pd.Series(model.predict(d2), name='loan_default')

# Results (#loan_default = d2['loan_default'])

d3 = pd.concat([UniqueID, loan_default_pred], axis=1)

d3.to_csv('submision.csv', index=False)
