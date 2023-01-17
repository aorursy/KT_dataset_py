# Basics

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Preprocssing

import missingno as msno

from sklearn.preprocessing import StandardScaler, MinMaxScaler, binarize



# Model Selection 

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV



# Models

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

# Ensemble

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier



# Metrics

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score



# Feature Selection

from sklearn.feature_selection import SelectKBest, chi2



# Warnings

import warnings as ws

ws.filterwarnings('ignore')
# Load dataset

data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head()
# Summary

def summary(data):

    df = {

     'Count' : data.shape[0],

     'NA values' : data.isna().sum(),

     '% NA' : round((data.isna().sum()/data.shape[0]) * 100, 2),

     'Unique' : data.nunique(),

     'Dtype' : data.dtypes,

     'min' : round(data.min(),2),

     '25%' : round(data.quantile(.25),2),

     '50%' : round(data.quantile(.50),2),

     'mean' : round(data.mean(),2),

     '75%' : round(data.quantile(.75),2),   

     'max' : round(data.max(),2)

    } 

    return(pd.DataFrame(df))



print('Shape is :', data.shape)

summary(data)
data.hist(figsize = (10,10))

plt.show()
# Target Variables

data['quality'].value_counts()
# Convert Target variable into binary

bins = [2,6.5, 8]

labels = ['Bad','Good']

data['quality'] = pd.cut(data['quality'], bins = bins, labels = labels)



data['quality'].value_counts()
col_names = data.drop('quality', axis = 1).columns.tolist()



plt.figure(figsize = (15,10))

i=0

for col in col_names:

    plt.subplot(3,4, i+1)

    plt.grid(True, alpha = 0.5)

    sns.kdeplot(data[col][data['quality'] == 'Bad'], label = 'Bad Quality')

    sns.kdeplot(data[col][data['quality'] == 'Good'], label = 'Good Quality')

    plt.title(col + ' vs Quality', size = 15)

    plt.xlabel(col, size = 12)

    plt.ylabel('Density')    

    plt.tight_layout()

    i+=1

plt.show()
X = data.drop('quality', axis = 1)

Y = data['quality'].replace({'Bad':0, 'Good' : 1})



x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('SVM', SVC()))

models.append(('RF', RandomForestClassifier()))

models.append(('ADA', AdaBoostClassifier()))

models.append(('GB', GradientBoostingClassifier()))
models
def model_selection(X,Y):

    acc_results = []

    auc_results = []

    names = []



    # Set Table

    col = ['Model Name','ROC AUC Mean','ROC AUC Std','ACC Mean', 'AUC Std']

    model_results = pd.DataFrame(columns = col)



    i = 0

    for name, model in models:

        kfold = KFold(n_splits = 10, random_state = 7)



        cv_acc_results = cross_val_score(model, X,Y, cv = kfold, scoring = 'accuracy')

        cv_auc_results = cross_val_score(model, X,Y, cv = kfold, scoring =  'roc_auc')



        acc_results.append(cv_acc_results)

        auc_results.append(cv_auc_results)

        names.append(name)



        model_results.loc[i] = [name, cv_auc_results.mean(),cv_auc_results.std(), cv_acc_results.mean(), cv_acc_results.std()]

        i+=1



    model_results = model_results.sort_values(['ROC AUC Mean'], ascending = False)     



    # View Model Results

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)

    sns.boxplot(x = names, y = acc_results)

    plt.title('Accuracy Score')



    plt.subplot(1,2,2)

    sns.boxplot(x = names, y = auc_results)

    plt.title('AUC Score')

    plt.show()

    

    return(model_results)
model_selection(x_train, y_train)
def model_validation(model,x_test,y_test,thr = 0.5) :

    

    y_pred_prob = model.predict_proba(x_test)[:,1]

    y_pred = binarize(y_pred_prob.reshape(1,-1), thr)[0]

    

    cnf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize = (10,3))

    plt.subplot(1,2,1)

    sns.heatmap(cnf_matrix, annot = True, fmt = 'g')

    plt.title('Confusion Matrix')

    plt.xlabel('Predicted label')

    plt.ylabel('Actual label')



    fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)

    plt.subplot(1,2,2)

    sns.lineplot(fpr, tpr)

    plt.plot([0,1],[0,1], 'r--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.show()



    

    print('Classification Report :')

    print('===' * 20)

    print(classification_report(y_test, y_pred))



    score = tpr - fpr

    opt_threshold = sorted(zip(score,threshold))[-1][1]

    print('='*20)

    print('Area Under Curve', roc_auc_score(y_test,y_pred))

    print('Accuracy', accuracy_score(y_test,y_pred))

    print('Optimal Threshold : ',opt_threshold)

    print('='*20)
param_grid = {

    'bootstrap': [True,False],

    'max_depth': [10, 50, 100],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [10,100, 200, 300, 1000]

}



rf = RandomForestClassifier()

grid = GridSearchCV(rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 1)



grid.fit(x_train, y_train)

grid.best_params_
model_validation(grid, x_test, y_test)
# Final Model

final_model = grid.best_estimator_

final_model.fit(x_train, y_train)



model_validation(final_model,x_test,y_test, 0.106)