

#load packages

import pandas as pd

import numpy as np 

#import scipy as sp #mathematics functions

#import IPython

#from IPython import display #printing nice

#import sklearn #machine learning algorithms

#import random #misc libraries

#import time #misc libraries



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)



# List the files in the input directory

import os

print(os.listdir("../input"))
#Common Model Algorithms

#from sklearn import linear_model, svm, neighbors, tree, naive_bayes

#from sklearn import ensemble, discriminant_analysis, gaussian_process

#from xgboost import XGBClassifier



#Common Model Helpers

#from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#from sklearn import feature_selection, model_selection, metrics



#Visualization

#import matplotlib as mpl

#import matplotlib.pyplot as plt

#import matplotlib.pylab as pylab

import seaborn as sns

#from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

#%matplotlib inline

#mpl.style.use('ggplot')

#sns.set_style('white')

#pylab.rcParams['figure.figsize'] = 12,8

print('-'*25)

print("Loading OK")
data = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv')

data_copy = data.copy(deep = True)

#preview data

print (data.info())

print(data.head())
print("TRUE Potive in data:")

print (data['Biopsy'].value_counts())

#Check nulls

data = data.replace('?', np.nan)

print('Columns with null values:\n', data.isnull().sum())
#drop columns with too many nulls

data = data.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])

print('Columns with null values:\n', data.isnull().sum())
#drop all raws with missing values because median can give a big bias

#data = data.dropna()

#data.info()

#print('Columns with null values:\n', data.isnull().sum())

#turn data into numeric type for computation

data = data.convert_objects(convert_numeric=True) 

data.info()



data_numeric=['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies',

              'Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives (years)',

              'IUD (years)','STDs (number)']

data_cat=['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis',

          'STDs:cervical condylomatosis','STDs:vaginal condylomatosis',

          'STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',

          'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV',

          'STDs:Hepatitis B','STDs:HPV']

Target=['Hinselmann','Schiller','Citology','Biopsy']



#analayzing the data, exploring mean and mode for each category

for feature in data[data_numeric]:

    print(feature)

    data[feature].fillna((data[feature].mean()), inplace=True)

    print(data[feature].value_counts())

    print(round(data[feature].mean(),2))

    

for feature in data[data_cat]:

    print(feature)

    data[feature].fillna((data[feature].mode()[0]), inplace=True)

    print(data[feature].value_counts())

    print(round(data[feature].mode(),2))
data.info()

data.describe()
print("TRUE Potive in data:")

print (data['Biopsy'].value_counts())
#check uniques to see if there are abnormal data

#for col in data:

#    print (col,':\n',data[col].sort_values(ascending=False).unique())



#check validation of bool features

for index,row in data.iterrows():

    if(row['Smokes']==0 and (row['Smokes (years)']>0 or row['Smokes (packs/year)']>0)):

        print('Data Failure: not smoking:',index,row['Smokes'],row['Smokes (years)'],row['Smokes (packs/year)'])

    elif(row['Smokes']==1 and (row['Smokes (years)']==0 or row['Smokes (packs/year)']==0)):

        print(row)

    if(row['Hormonal Contraceptives']==0 and row['Hormonal Contraceptives (years)']>0):

        print(row)

    elif(row['Hormonal Contraceptives']==1 and row['Hormonal Contraceptives (years)']==0):

        print(row)

    if(row['IUD']==0 and row['IUD (years)']>0):

        print(row)

    elif(row['IUD']==1 and row['IUD (years)']==0):

        print(row)

    if(row['STDs']==0 and row['STDs (number)']>0):

        print(row)

    elif(row['STDs']==1 and row['STDs (number)']==0):

        print(row)

    if(row['STDs']==1 and row['STDs (number)']!=(row['STDs:condylomatosis']+

                                                 row['STDs:cervical condylomatosis']+

                                                 row['STDs:vaginal condylomatosis']+

                                                 row['STDs:vulvo-perineal condylomatosis']+

                                                 row['STDs:syphilis']+

                                                 row['STDs:pelvic inflammatory disease']+

                                                 row['STDs:genital herpes']+

                                                 row['STDs:molluscum contagiosum']+

                                                 row['STDs:AIDS']+

                                                 row['STDs:HIV']+

                                                 row['STDs:Hepatitis B']+

                                                 row['STDs:HPV'])):

        print(row)

    if(row['STDs: Number of diagnosis']==1 and row['STDs (number)']==1):

        print('Data Failure: not number of diagnosis:',index) 

        

#dropping columns with unknown meaning of data

data = data.drop(columns=['STDs: Number of diagnosis', 'Dx:Cancer','Dx:CIN','Dx:HPV','Dx'])

print('Columns with null values:\n', data.isnull().sum())  

        

    

    

    
#check some statistic functions

Data_Fisher=['Smokes','Hormonal Contraceptives','IUD','STDs:condylomatosis',

          'STDs:cervical condylomatosis','STDs:vaginal condylomatosis',

          'STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',

          'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV',

          'STDs:Hepatitis B','STDs:HPV']

Data_spearman=['Number of sexual partners','STDs']


#calculate chi-square for categorial feature 

from scipy.stats import chi2_contingency

from scipy.stats import chi2



for x in data_cat:

    for y in Target:

        print('Biopsy Scattering by:', x)

        contingency_table=pd.crosstab(data[x],data[y])

        print(contingency_table)

        table=np.array(contingency_table)

        # chi-squared test with similar proportions

        stat, p, dof, expected = chi2_contingency(table)

        #print('dof=%d' % dof)

        print(expected)

        # interpret test-statistic

        prob = 0.95

        critical = chi2.ppf(prob, dof)

        print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

        if abs(stat) >= critical:

            print('Dependent (reject H0)')

        else:

            print('Independent (fail to reject H0)')

        # interpret p-value

        alpha = 1.0 - prob

        print('significance=%.3f, p=%.3f' % (alpha, p))

        if p <= alpha:

            print('Dependent (reject H0)')

        else:

            print('Independent (fail to reject H0)')

        #print('array format\n',table)

        #print('Biopsy Correlation by:', x)

        #print(data[[x, y].groupby(x, as_index=False).mean())

        print('-'*10, '\n')
#checking feature correlation for numerical features

data_num=data[data_numeric]

corr = data_num.corr()

corr.head()





Data_spearman

data_spear=data[Data_spearman + Target]

corr_spear = data_spear.corr(method='spearman')

corr_spear
#Generating the correlation heatmap

sns.heatmap(corr)
#check fpt cotellation bigger then 0.5

for index,row in corr.iterrows():

    for col in corr:

        if col!=index:

            if row[col] >= 0.5:

                print(row[col])

                print(col,index)

                print(col==index)
#setting Y arrays

hinselmann = data['Hinselmann']

schiller = data['Schiller']

citology = data['Citology']

biopsy = data['Biopsy']

data = data.drop(['Hinselmann','Schiller','Citology','Biopsy'], axis=1)
data.head()
data[data_numeric].info()
"""Prepare data for modeling"""

#Normalization

from sklearn import preprocessing

np_data = data[data_numeric].values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(np_data)

data[data_numeric] = pd.DataFrame(x_scaled)



x=data

y=biopsy

print(type(y))

print("data balance:")

print (y.value_counts())

print(y)

#Spliting the data to train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33,random_state=42)

#Balancing the data

from collections import Counter

from imblearn.combine import SMOTETomek # doctest: +NORMALIZE_WHITESPACE

smt_test = SMOTETomek(random_state=42)

X_res, y_res = smt_test.fit_resample(x_test, y_test)

print('Resampled dataset tesr shape %s' % Counter(y_res))

x_test = pd.DataFrame(X_res)

y_test = pd.DataFrame(y_res)

print (y_test[0].value_counts())

smt_train = SMOTETomek(random_state=42)

Xt_res, yt_res = smt_train.fit_resample(x_train, y_train)

print('Resampled dataset tesr shape %s' % Counter(yt_res))

x_train = pd.DataFrame(Xt_res)

y_train = pd.DataFrame(yt_res)

print (y_train[0].value_counts())







class_data = {'train': {'X': x_train,

                        'y': y_train},

              'test': {'X': x_test,

                       'y': y_test},

              'n_classes': len(np.unique(y_train))}

print("Got %i training samples and %i test samples." %

     (len(class_data['train']['X']), len(class_data['test']['X'])))
data.head()
#"Train and check multiple classifiers for prediction"

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#from sklearn.neural_network import BernoulliRBM

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection

#import skflow

#import time



def main():



    # Get classifiers

    classifiers = [

        ('Logistic Regression (C=1)', LogisticRegression(C=4,penalty='l2')),

        ('Logistic Regression (C=1)', LogisticRegression(C=4,

                                                         solver='liblinear',

                                                         penalty='l1')),

        ('SVM, rbf', SVC(kernel="rbf",

                          C=100,

                          gamma=15)),

        ('SVM, rbf2', SVC(kernel="rbf",

                          gamma=15,

                          C=11)),

        ('knn', KNeighborsClassifier(3)),

        ('knn2', KNeighborsClassifier(3,

                                     weights='distance')),

        ('knn3', KNeighborsClassifier(2)),

        ('Decision Tree', DecisionTreeClassifier(max_depth=5, random_state=42)),

        ('Random Forest', RandomForestClassifier(n_estimators=50, n_jobs=10)),

        ('Random Forest 2', RandomForestClassifier(max_depth=5,

                                                   n_estimators=10,

                                                   max_features=1,

                                                   n_jobs=10)),

        ('AdaBoost', AdaBoostClassifier()),

        ('Naive Bayes', GaussianNB()),

        ('Gradient Boosting', GradientBoostingClassifier())#,

        #('LDA', LinearDiscriminantAnalysis()),

        #('QDA', QuadraticDiscriminantAnalysis())

    ]

    

    # Fit  all classifiers

    classifier_data = {}

    for clf_name, clf in classifiers:

        print("#" * 80)

        print("Start fitting '%s' classifier." % clf_name)

        clf.fit(class_data['train']['X'], class_data['train']['y'].values.ravel())

        an_data = analyze(clf, class_data, clf_name=clf_name)

        

   
from sklearn.model_selection import GridSearchCV



#Exploring Hyper-parameters

logistic = LogisticRegression(random_state=34)

# Create a list of all of the different penalty values 

penalty = ['l1', 'l2']

# Create a list of all of the different C values 

#C = [0.0001, 0.001, 0.01, 1,10, 100,250]

C = [5,6,7,8,9,10,11,12,13,14]

#solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

solver_list = ['liblinear']

hyperparameters = dict(C=C, penalty=penalty, solver=solver_list)

#hyperparameters = dict(solver=solver_list)

"""

#Fit your model using gridsearch

clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

best_model = clf.fit(class_data['train']['X'], class_data['train']['y'].values.ravel())

#Print all the Parameters that gave the best results:

print('Best Parameters',clf.best_params_)

print('Best parameters:', best_model.best_estimator_.get_params())

print(clf.cv_results_['mean_test_score'])

#print('Best solver:', best_model.best_estimator_.get_params()['solver'])

#scores = clf.cv_results_['mean_test_score']

#for score, solver, in zip(scores, solver_list):

#    print(f"{solver}: {score:.3f}")



svc = SVC()

kernel = ['linear', 'rbf'] 

gammas = [0.001, 0.01, 0.1, 1,8,10,15,16,17,18]

svc_parm  = dict(C=C,kernel=kernel, gamma= gammas)

clf_svc = GridSearchCV(svc, svc_parm, cv=2, verbose=0)

best_svc_model = clf_svc.fit(class_data['train']['X'], class_data['train']['y'].values.ravel())

print('Best Svc parameters:', best_svc_model.best_estimator_.get_params())

print(clf.cv_results_['mean_test_score'])



knn=KNeighborsClassifier()

n_neighbors = [1,2,3,4,5,11,19]

knnweights = ['uniform','distance']

algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

knn_parm  = dict(n_neighbors=n_neighbors,weights=knnweights,algorithm=algorithm )

clf_knn = GridSearchCV(knn, knn_parm, cv=2, verbose=0)

best_knn_model = clf_knn.fit(class_data['train']['X'], class_data['train']['y'].values.ravel())

print('Best knn parameters:', best_knn_model.best_estimator_.get_params())

print(clf.cv_results_['mean_test_score'])

"""

dt=DecisionTreeClassifier()

#criterion = ['gini', 'entropy']

#splitter = ['best', 'random'] 

max_depth = [1,2,3,4,5,6,7,8]

min_samples_split= np.linspace(0.1, 1.0, 10, endpoint=True)

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

num_leafs = [1, 5, 10, 20, 50, 100]

parameters = dict(max_depth=max_depth,

                  min_samples_split=min_samples_split,

                  min_samples_leafs=min_samples_leafs,

                  num_leafs=num_leafs)



def analyze(clf, data, clf_name=''):

    """

    Analyze how well a classifier performs on data.

    Parameters

    ----------

    clf : classifier object

    data : dict

    clf_name : str

    Returns

    -------

    dict

        accuracy 

    """

    results = {}



    target_names = ['healthy', 'sick']

    # Get confusion matrix

    from sklearn import metrics

    from sklearn.metrics import recall_score

    predicted = np.array([])

    predicted=clf.predict(data['test']['X'])

    

    print("Classifier: %s" % clf_name)

    print("Confusion matrix:\n%s" %

          metrics.confusion_matrix(data['test']['y'],

                                   predicted))

    results['accuracy'] = metrics.accuracy_score(data['test']['y'], predicted)

    print("Accuracy: %0.3f" % results['accuracy'])

    print("Sensitivity: %0.3f" % metrics.recall_score(data['test']['y'],predicted))

    print("Precision: %0.3f" % metrics.precision_score(data['test']['y'],predicted))

    print("F-measure: %0.3f" % metrics.f1_score(data['test']['y'],predicted))

    

    return results

if __name__ == '__main__':

    main()
#Permutation importance
