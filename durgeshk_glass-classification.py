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
#import the dataset as a dataframe

df=pd.read_csv('/kaggle/input/glass/glass.csv')
df.head()
df.describe()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(df['Type'])

plt.show()
#we will generate synthetic samples using smote

sampling = {1: 76, 2: 76, 3: 76,5:76,6:76,7:76}
 #Visualize the Data

df.hist(bins=50, figsize=(15,15))

plt.figure()

plt.show()
df.isnull().values.any()

#no NaN value in columns
# kernel density plot for analysis

def plot_skew_kurt(column_name):

    from scipy import stats

    from scipy.stats import skew,norm

    sns.distplot(df[column_name],fit=norm);

    plt.ylabel =('Frequency')

    plt.title = (column_name+' Distribution');

    #Get the fitted parameters used by the function

    (mu, sigma) = norm.fit(df[column_name]);

    #QQ plot

    fig = plt.figure()

    res = stats.probplot(df[column_name], plot=plt)

    plt.show()

    print(column_name+" skewness: %f" % df[column_name].skew())

    print(column_name +" kurtosis: %f" % df [column_name].kurt())
columns_features=df.columns[:-1].tolist()

def plot_skew(columns_features):

    for item in columns_features:

        plot_skew_kurt(item)
plot_skew(columns_features)
plt.figure()

sns.pairplot(df,hue='Type')

plt.show()
import numpy as np

def outlier_detec(feature):

    data_mean, data_std,data_median = np.mean(df[feature]), np.std(df[feature]),np.median(df[feature])

    cut_off = data_std * 3

    #alculating the cut-off for identifying outliers as more than 3 standard deviations from the mean

    lower, upper = data_mean - cut_off, data_mean + cut_off

    # identify outliers

    print("data_mean, data_std,data_median",data_mean, data_std,data_median)

    outliers = [x for x in df[feature] if x < lower or x > upper]

    return outliers
for item in df.columns:

    print(item,"--->",outlier_detec(item))
#function for log transforming the skewed data.

def log_transform(feature):

    df[feature] = np.log1p(df[feature])

    plot_skew_kurt(feature)
#transforming the data

for item in columns_features:

    log_transform(item)
#Checking correlation between features

corr = df[columns_features].corr()

plt.figure(figsize=(11,11))

sns.heatmap(corr, annot=True,

           xticklabels= columns_features, yticklabels= columns_features, alpha = 0.7,   cmap= 'coolwarm')

plt.show()
from sklearn.model_selection import (train_test_split,

                                     cross_val_score, GridSearchCV)

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#to scale the data

sc = StandardScaler()
X_data = df[columns_features] 

y_data = df.Type

from collections import Counter

#since out target data is imbalanced,  approach is to either oversample or undersample

# since the data is very less< i will be doing oversampling

from imblearn.over_sampling import RandomOverSampler

from imblearn.datasets import make_imbalance

ros = RandomOverSampler(sampling_strategy=sampling)

X_data, y_data = ros.fit_resample(X_data, y_data)

X_data = sc.fit_transform(X_data)

X, X_test, y, y_test = train_test_split(X_data, y_data, test_size = 0.15 , random_state = 42)
#using this code from some other developers. I forgot from where here I had taken this. *sorry* 

def plot_pie(y):

    target_stats = Counter(y)

    labels = list(target_stats.keys())

    sizes = list(target_stats.values())

    explode = tuple([0.1] * len(target_stats))

    def make_autopct(values):

        def my_autopct(pct):

            total = sum(values)

            val = int(round(pct * total / 100.0))

            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

        return my_autopct



    fig, ax = plt.subplots()

    ax.pie(sizes, explode=explode, labels=labels, shadow=True,

           autopct=make_autopct(sizes))

    ax.axis('equal')
print('Information of the glass data set after making it '

      'balanced by cleaning sampling: \n sampling_strategy={} \n y: {}'

      .format(sampling, Counter(y_data)))

plot_pie(y_data)
#intializing the classifiers

rfc=RandomForestClassifier(random_state=42)

# gbc=GradientBoostingClassifier(random_state=42)

logreg=LogisticRegression(random_state=42)

svc=SVC(random_state=42)

knn=KNeighborsClassifier()
rfc.fit(X,y)

#to find the important feature for the predictions

rfc_features=rfc.feature_importances_

rfc_features
#creating param grids for GridSearch to find best possible hyperparameters

import numpy as np

param_grid_rfc = { 

    'n_estimators': [200, 500],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,7,8],

    'criterion' :['gini', 'entropy']

}

# param_grid_gbc = {

#     "learning_rate": [0.01, 0.05, 0.15, 0.2],

#     "criterion": ["friedman_mse",  "mae"],

#     "n_estimators":[200,500]

# }

param_grid_logreg = {

    "C":np.logspace(-3,3,7),

    "solver":['lbfgs', 'liblinear', 'sag']

}

param_grid_svc = {

    'kernel': ['rbf'], 

    'gamma': [1e-2, 1e-3, 1e-4, 1e-5,'auto'],

    'C': np.logspace(-3,3,7),

    'decision_function_shape':('ovo','ovr'),

    'shrinking':(True,False)

}

param_grid_knn = {

    'n_neighbors': [4,7,9,11], 

    'weights': ['uniform','distance'],

    'metric':['euclidean','manhattan']

}
from sklearn.model_selection import GridSearchCV

#cross validation with hyperparameter tuning

def grid_searchCV(estimators,param_grids):

    return GridSearchCV(estimator=estimators, param_grid=param_grids, cv= 5)
#fetch CV for all models

cv_rfc = grid_searchCV(rfc,param_grid_rfc)

# cv_gbc = grid_searchCV(gbc,param_grid_gbc)

cv_logreg = grid_searchCV(logreg,param_grid_logreg)

cv_svc = grid_searchCV(svc,param_grid_svc)

cv_knn=grid_searchCV(knn,param_grid_knn)
#fitting on whole training-data since there is a separate test-data available

cv_rfc.fit(X, y)
cv_logreg.fit(X, y)

cv_svc.fit(X, y)

cv_knn.fit(X, y)
print("Tuned RFC Parameters: {}".format(cv_rfc.best_params_)) 

print("Best RFC Parameters score is {}".format(cv_rfc.best_score_))



# print("Tuned GBC Parameters: {}".format(cv_gbc.best_params_)) 

# print("Best GBC Parameters score is {}".format(cv_gbc.best_score_))



print("Tuned LogReg Parameters: {}".format(cv_logreg.best_params_)) 

print("Best LogReg Parameters score is {}".format(cv_logreg.best_score_))



print("Tuned SVC Parameters: {}".format(cv_svc.best_params_)) 

print("Best SVC Parameters score is {}".format(cv_svc.best_score_))



print("Tuned Knn Parameters: {}".format(cv_knn.best_params_)) 

print("Best knn Parameters score is {}".format(cv_knn.best_score_))
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,roc_auc_score

#creating function for create classification report, model accuracy score and confusion matrix

def model_performance(ypred,y_test,model_name,model,X_test):

    cnf=confusion_matrix(ypred,y_test)

    print("confusion Matrix: ",cnf)

    score=model.score(X_test,y_test)

    print("accuracy-Score of ",model_name,": ",score)

    print("classification_report of ",model_name,": ",classification_report(y_test, ypred))

    return score
ypred_rfc=cv_rfc.predict(X_test)

# ypred_gbc=cv_gbc.predict(X_test)

ypred_logreg=cv_logreg.predict(X_test)

ypred_svc=cv_svc.predict(X_test)

ypred_knn=cv_knn.predict(X_test)
#Classification Report of the models

rfc_score=model_performance(ypred_rfc,y_test,'Random-Forest',cv_rfc,X_test)

# gbc_score=model_performance(ypred_gbc,y_test,'Gradient-Boost',cv_gbc,X_test)

logreg_score=model_performance(ypred_logreg,y_test,'Logistic-Regression',cv_logreg,X_test)

svc_score=model_performance(ypred_svc,y_test,'Support Vector C',cv_svc,X_test)

knn_score=model_performance(ypred_knn,y_test,'K-nearest neighbors',cv_knn,X_test)
results= pd.DataFrame({'Models':['Logistic', 'RandomForrest','SVC','KNN'],

                       'Score':[str('%.2f' % (logreg_score*100))+' %',str('%.2f' % (rfc_score*100))+' %'

                                ,str('%.2f' % (svc_score*100))+' %' ,

                                str('%.2f' % (knn_score*100))+' %' ]})

results.index=np.arange(1,len(results)+1)
results
print("prediction precision rate:",round(cv_rfc.score(X_test,y_test),2)*100)

result=cv_rfc.predict(X_test)

print("predicted:",result)

myarray = np.asarray(y_test.tolist())

print("original type:",myarray)