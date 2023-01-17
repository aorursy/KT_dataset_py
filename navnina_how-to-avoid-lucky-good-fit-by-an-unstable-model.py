%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

import math as ma



from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics

from sklearn.metrics import roc_curve, auc



from plotly import plotly

import plotly.offline as offline

import plotly.graph_objs as go

offline.init_notebook_mode()

from collections import Counter



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from collections import Counter

from sklearn.metrics import accuracy_score

from sklearn import model_selection
data = pd.read_csv('../input/heart.csv')
data.head(5)
data.shape
data.groupby("target")['age'].count().plot.bar()
y_value_counts = data['target'].value_counts()



print("Number of people getting heart attack ", y_value_counts[1], ", (", (y_value_counts[1]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")

print("Number of not getting heart attack ", y_value_counts[0], ", (", (y_value_counts[0]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")

data.columns.values

data.columns = ['age','sex','chestpain',

                     'blood_pressure','cholestoral','blood_sugar','ECG','max_heart_rate','angima','oldpeak','slope','coloured_vessels','thal','target']
data.head(5)
data.info()
data['sex'][data['sex'] == 0] = 'Female'

data['sex'][data['sex'] == 1] = 'Male'



data['chestpain'][data['chestpain'] == 0] = 'level_zero'

data['chestpain'][data['chestpain'] == 1] = 'level_one'

data['chestpain'][data['chestpain'] == 2] = 'level_two'

data['chestpain'][data['chestpain'] == 3] = 'level_three'



data['blood_sugar'][data['blood_sugar'] == 0] = 'low'

data['blood_sugar'][data['blood_sugar'] == 1] = 'high'



data['ECG'][data['ECG'] == 0] = 'normal'

data['ECG'][data['ECG'] == 1] = 'wave_abnormality'

data['ECG'][data['ECG'] == 2] = 'Estes_criteria'



data['angima'][data['angima'] == 0] = 'No'

data['angima'][data['angima'] == 1] = 'Yes'



data['slope'][data['slope'] == 0] = 'Upsloping'

data['slope'][data['slope'] == 1] = 'Flat'

data['slope'][data['slope'] == 2] = 'Downsloping'



data['thal'][data['thal'] == 0] = 'level_zero'

data['thal'][data['thal'] == 1] = 'level_one'

data['thal'][data['thal'] == 2] = 'level_two'

data['thal'][data['thal'] == 3] = 'level_three'



data['coloured_vessels'][data['coloured_vessels'] == 0] = 'zero'

data['coloured_vessels'][data['coloured_vessels'] == 1] = 'one'

data['coloured_vessels'][data['coloured_vessels'] == 2] = 'two'

data['coloured_vessels'][data['coloured_vessels'] == 3] = 'three'

data['coloured_vessels'][data['coloured_vessels'] == 4] = 'four'



data.head(5)
plt.figure(figsize=(12, 8))



plt.subplot(1,2,1)

sns.boxplot(x = 'target', y = 'age', data = data[0:])



plt.subplot(1,2,2)

sns.distplot(data[data['target'] == 0.0]['age'][0:] , label = "0", color = 'red')

sns.distplot(data[data['target'] == 1.0]['age'][0:] , label = "1" , color = 'blue' )

plt.legend()

plt.show()
plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)

sns.boxplot(x = 'target', y = 'blood_pressure', data = data[0:])



plt.subplot(1,2,2)

sns.distplot(data[data['target'] == 0.0]['blood_pressure'][0:] , label = "0", color = 'red')

sns.distplot(data[data['target'] == 1.0]['blood_pressure'][0:] , label = "1" , color = 'blue' )

plt.legend()

plt.show()
plt.figure(figsize=(12, 8))



plt.subplot(1,2,1)

sns.boxplot(x = 'target', y = 'cholestoral', data = data[0:])



plt.subplot(1,2,2)

sns.distplot(data[data['target'] == 0.0]['cholestoral'][0:] , label = "0", color = 'red')

sns.distplot(data[data['target'] == 1.0]['cholestoral'][0:] , label = "1" , color = 'blue' )

plt.legend()

plt.show()
plt.figure(figsize=(12, 8))



plt.subplot(1,2,1)

sns.boxplot(x = 'target', y = 'max_heart_rate', data = data[0:])



plt.subplot(1,2,2)

sns.distplot(data[data['target'] == 0.0]['max_heart_rate'][0:] , label = "0", color = 'red')

sns.distplot(data[data['target'] == 1.0]['max_heart_rate'][0:] , label = "1" , color = 'blue' )

plt.legend()

plt.show()
plt.figure(figsize=(12, 8))



plt.subplot(1,2,1)

sns.boxplot(x = 'target', y = 'oldpeak', data = data[0:])



plt.subplot(1,2,2)

sns.distplot(data[data['target'] == 0.0]['oldpeak'][0:] , label = "0", color = 'red')

sns.distplot(data[data['target'] == 1.0]['oldpeak'][0:] , label = "1" , color = 'blue' )

plt.legend()

plt.show()
heart_attack = data[data['target']==1]['oldpeak'].values

no_heart_attack = data[data['target']==0]['oldpeak'].values





from prettytable import PrettyTable



x = PrettyTable()

x.field_names = ["Percentile", "Heart Attack", "No Heart attack"]



for i in range(0,101,5):

    x.add_row([i,np.round(np.percentile(heart_attack,i), 3), np.round(np.percentile(no_heart_attack,i), 3)])

print(x)
def stack_plot(data, xtick, col2='heart attack cases', col3='total cases'):

    ind = np.arange(data.shape[0])

    

    plt.figure(figsize=(10,5))

    p1 = plt.bar(ind, data[col3].values)

    p2 = plt.bar(ind, data[col2].values)



    plt.ylabel('Number of people')

    plt.title('Heart Attack Vs No Heart Attack')

    plt.xticks(ind, list(data[xtick].values))

    plt.legend((p1[0], p2[0]), ('total cases', 'heart attack cases'))

    plt.show()
def univariate_barplots(data, col1, col2='target', top=False):

    # Count number of zeros in dataframe python: https://stackoverflow.com/a/51540521/4084039

    temp = pd.DataFrame(data.groupby(col1)[col2].agg(lambda x: x.eq(1).sum())).reset_index()

   

    # Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039

    

    temp['total'] = pd.DataFrame(data.groupby(col1)[col2].agg({'total':'count'})).reset_index()['total']



    temp['Avg'] = pd.DataFrame(data.groupby(col1)[col2].agg({'Avg':'mean'})).reset_index()['Avg']

    



    temp.sort_values(by=['total'],inplace=True, ascending=False)

    

    if top:

        temp = temp[0:top]

    

    stack_plot(temp, xtick=col1, col2=col2, col3='total')

    print(temp.head(5))

univariate_barplots(data, 'sex', 'target' , top=False)
univariate_barplots(data, 'chestpain', 'target' , top=False)
univariate_barplots(data, 'blood_sugar', 'target' , top=False)
univariate_barplots(data, 'ECG', 'target' , top=False)
univariate_barplots(data, 'angima', 'target' , top=False)
univariate_barplots(data, 'slope', 'target' , top=False)
univariate_barplots(data, 'coloured_vessels', 'target' , top=False)
univariate_barplots(data, 'thal', 'target' , top=False)
y = data['target']

data.drop(['target'], axis=1, inplace=True)

X=data

print(X.shape)

print(y.shape)
#Split the data into train and test



X_tr, X_test, y_tr, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
print(X_tr.shape)

print(y_tr.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.preprocessing import StandardScaler

age_scalar = StandardScaler(with_mean=False)

age_scalar.fit(X_tr['age'].values.reshape(-1,1)) # finding the mean and standard deviation of this data

print(f"Mean : {age_scalar.mean_[0]}, Standard deviation : {np.sqrt(age_scalar.var_[0])}")



# Now standardize the data with above maen and variance.

Tr_age_standardized = age_scalar.transform(X_tr['age'].values.reshape(-1, 1))

Test_age_standardized = age_scalar.transform(X_test['age'].values.reshape(-1, 1))



print("Shape of matrix after standarsation")

print(Tr_age_standardized.shape)

print(Test_age_standardized.shape)
blood_pressure_scalar = StandardScaler(with_mean=False)

blood_pressure_scalar.fit(X_tr['age'].values.reshape(-1,1)) # finding the mean and standard deviation of this data

print(f"Mean : {blood_pressure_scalar.mean_[0]}, Standard deviation : {np.sqrt(blood_pressure_scalar.var_[0])}")



# Now standardize the data with above maen and variance.

Tr_blood_pressure_standardized = blood_pressure_scalar.transform(X_tr['blood_pressure'].values.reshape(-1, 1))

Test_blood_pressure_standardized = blood_pressure_scalar.transform(X_test['blood_pressure'].values.reshape(-1, 1))



print("Shape of matrix after standarsation")

print(Tr_blood_pressure_standardized.shape)

print(Test_blood_pressure_standardized.shape)
cholestoral_scalar = StandardScaler(with_mean=False)

cholestoral_scalar.fit(X_tr['age'].values.reshape(-1,1)) # finding the mean and standard deviation of this data

print(f"Mean : {cholestoral_scalar.mean_[0]}, Standard deviation : {np.sqrt(cholestoral_scalar.var_[0])}")



# Now standardize the data with above maen and variance.

Tr_cholestoral_standardized = cholestoral_scalar.transform(X_tr['cholestoral'].values.reshape(-1, 1))

Test_cholestoral_standardized = cholestoral_scalar.transform(X_test['cholestoral'].values.reshape(-1, 1))



print("Shape of matrix after standarsation")

print(Tr_cholestoral_standardized.shape)

print(Test_cholestoral_standardized.shape)
max_heart_rate_scalar = StandardScaler(with_mean=False)

max_heart_rate_scalar.fit(X_tr['age'].values.reshape(-1,1)) # finding the mean and standard deviation of this data

print(f"Mean : {max_heart_rate_scalar.mean_[0]}, Standard deviation : {np.sqrt(max_heart_rate_scalar.var_[0])}")



# Now standardize the data with above maen and variance.

Tr_max_heart_rate_standardized = cholestoral_scalar.transform(X_tr['max_heart_rate'].values.reshape(-1, 1))

Test_max_heart_rate_standardized = cholestoral_scalar.transform(X_test['max_heart_rate'].values.reshape(-1, 1))



print("Shape of matrix after standarsation")

print(Tr_max_heart_rate_standardized.shape)

print(Test_max_heart_rate_standardized.shape)
oldpeak_scalar = StandardScaler(with_mean=False)

oldpeak_scalar.fit(X_tr['oldpeak'].values.reshape(-1,1)) # finding the mean and standard deviation of this data

print(f"Mean : {oldpeak_scalar.mean_[0]}, Standard deviation : {np.sqrt(oldpeak_scalar.var_[0])}")



# Now standardize the data with above maen and variance.

Tr_oldpeak_standardized = cholestoral_scalar.transform(X_tr['oldpeak'].values.reshape(-1, 1))

Test_oldpeak_standardized = cholestoral_scalar.transform(X_test['oldpeak'].values.reshape(-1, 1))



print("Shape of matrix after standarsation")

print(Tr_oldpeak_standardized.shape)

print(Test_oldpeak_standardized.shape)
def One_hot_encoding_tr(col):

    my_counter = Counter()

    for word in col.values:

        my_counter.update(word.split())



    col_dict = dict(my_counter)

    sorted_col_dict = dict(sorted(col_dict.items(), key=lambda kv: kv[1])) # sort categories in desc order as a dictionary



    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(vocabulary=list(sorted_col_dict.keys()), lowercase=False, binary=True)

    vectorizer.fit(col.values)

    print(vectorizer.get_feature_names())

    

    Tr_col_one_hot = vectorizer.transform(col.values)

    return Tr_col_one_hot



def One_hot_encoding_test(col,test_col):

    my_counter = Counter()

    for word in col.values:

        my_counter.update(word.split())



    col_dict = dict(my_counter)

    sorted_col_dict = dict(sorted(col_dict.items(), key=lambda kv: kv[1])) # sort categories in desc order as a dictionary



    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(vocabulary=list(sorted_col_dict.keys()), lowercase=False, binary=True)

    vectorizer.fit(col.values)

    print(vectorizer.get_feature_names())

    

    Tr_col_one_hot = vectorizer.transform(col.values)

    Test_col_one_hot = vectorizer.transform(test_col.values)

    return Test_col_one_hot





Tr_sex_one_hot=One_hot_encoding_tr(X_tr['sex'])

Test_sex_one_hot=One_hot_encoding_test(X_tr['sex'],X_test['sex'])

print("Shape of matrix after one hot encoding")

print(Tr_sex_one_hot.shape)

print(Test_sex_one_hot.shape)
Tr_chestpain_one_hot=One_hot_encoding_tr(X_tr['chestpain'])

Test_chestpain_one_hot=One_hot_encoding_test(X_tr['chestpain'],X_test['chestpain'])

print("Shape of matrix after one hot encoding")

print(Tr_chestpain_one_hot.shape)

print(Test_chestpain_one_hot.shape)
Tr_blood_sugar_one_hot=One_hot_encoding_tr(X_tr['blood_sugar'])

Test_blood_sugar_one_hot=One_hot_encoding_test(X_tr['blood_sugar'],X_test['blood_sugar'])

print("Shape of matrix after one hot encoding")

print(Tr_blood_sugar_one_hot.shape)

print(Test_blood_sugar_one_hot.shape)
Tr_ECG_one_hot=One_hot_encoding_tr(X_tr['ECG'])

Test_ECG_one_hot=One_hot_encoding_test(X_tr['ECG'],X_test['ECG'])

print("Shape of matrix after one hot encoding")

print(Tr_ECG_one_hot.shape)

print(Test_ECG_one_hot.shape)
Tr_angima_one_hot=One_hot_encoding_tr(X_tr['angima'])

Test_angima_one_hot=One_hot_encoding_test(X_tr['angima'],X_test['angima'])

print("Shape of matrix after one hot encoding")

print(Tr_angima_one_hot.shape)

print(Test_angima_one_hot.shape)
Tr_slope_one_hot=One_hot_encoding_tr(X_tr['slope'])

Test_slope_one_hot=One_hot_encoding_test(X_tr['slope'],X_test['slope'])

print("Shape of matrix after one hot encoding")

print(Tr_slope_one_hot.shape)

print(Test_slope_one_hot.shape)
Tr_coloured_vessels_one_hot=One_hot_encoding_tr(X_tr['coloured_vessels'])

Test_coloured_vessels_one_hot=One_hot_encoding_test(X_tr['coloured_vessels'],X_test['coloured_vessels'])

print("Shape of matrix after one hot encoding")

print(Tr_coloured_vessels_one_hot.shape)

print(Test_coloured_vessels_one_hot.shape)
Tr_thal_one_hot=One_hot_encoding_tr(X_tr['thal'])

Test_thal_one_hot=One_hot_encoding_test(X_tr['thal'],X_test['thal'])

print("Shape of matrix after one hot encoding")

print(Tr_thal_one_hot.shape)

print(Test_thal_one_hot.shape)
from scipy.sparse import hstack

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn import linear_model

import pylab

from matplotlib.pyplot import figure

##https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot

font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 12}



plt.rc('font', **font)
train = hstack((Tr_age_standardized,Tr_blood_pressure_standardized,Tr_cholestoral_standardized,Tr_chestpain_one_hot,Tr_blood_sugar_one_hot,Tr_ECG_one_hot,Tr_angima_one_hot,

            Tr_slope_one_hot,Tr_coloured_vessels_one_hot,Tr_thal_one_hot,Tr_max_heart_rate_standardized,Tr_oldpeak_standardized,Tr_sex_one_hot

             

           )) 



test = hstack((Test_age_standardized,Test_blood_pressure_standardized,Test_cholestoral_standardized,Test_chestpain_one_hot,Test_blood_sugar_one_hot,Test_ECG_one_hot,Test_angima_one_hot,

            Test_slope_one_hot,Test_coloured_vessels_one_hot,Test_thal_one_hot,Test_max_heart_rate_standardized,Test_oldpeak_standardized,Test_sex_one_hot

             

           )) 

print(train.shape)

print(test.shape)
import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')



lamda=list()

test_auc=list()

training_auc=list()

cv_auc=list()



iteration=np.arange(1,10)

for i in range(1,10):



    k=[ 0.0001,0.001,0.01,0.1,1,10,100,1000,10000]



    param_grid = {'alpha':[ 0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}



    #Hyper parameter Tuning for lamda with 3 fold cross validation

    clf = linear_model.SGDClassifier(loss='log')



    clf_cv= GridSearchCV(clf,param_grid,cv=3,scoring='roc_auc')



    clf_cv.fit(train.toarray(),y_tr)

    cv_auc.append(clf_cv.best_score_)

    

    #Instantiate Classifier

    clf = linear_model.SGDClassifier(loss='log',alpha=clf_cv.best_params_.get('alpha'))

    clf.fit(train.toarray(),y_tr)

    y_pred_proba_test = clf.predict_proba(test.toarray())[:,1]

    y_pred_proba_tr = clf.predict_proba(train.toarray())[:,1]

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_tr, y_pred_proba_tr)

   

    test_auc.append(roc_auc_score(y_test,y_pred_proba_test))

    training_auc.append(roc_auc_score(y_tr,y_pred_proba_tr))

    lamda.append(clf_cv.best_params_.get('alpha'))



figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

pylab.subplot(2,2,1)    

pylab.plot(iteration, training_auc, '-b', label='Training AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('Training AUC')

pylab.grid()





pylab.subplot(2,2,2)

pylab.plot(iteration, test_auc, '-b', label='Test AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('test AUC')

pylab.grid()





pylab.subplot(2,2,3)

pylab.plot(iteration, cv_auc, '-b', label='CV AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('CV AUC')

pylab.grid()



pylab.subplot(2,2,4)

pylab.plot(iteration, lamda, '-b', label='Lambda',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('Lambda')

pylab.yscale("log")

pylab.grid()

pylab.show()



import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from sklearn.neighbors import KNeighborsClassifier

k_nn=list()

test_auc=list()

training_auc=list()

cv_auc=list()



iteration=np.arange(1,10)



k=np.arange(1,70,5)

param_grid = {'n_neighbors':np.arange(1,70,5)}



for i in range(1,10):





    #Hyper parameter Tuning for lamda with 3 fold cross validation

    clf = KNeighborsClassifier(algorithm='brute')



    clf_cv= GridSearchCV(clf,param_grid,cv=3,scoring='roc_auc')



    clf_cv.fit(train.toarray(),y_tr)

    cv_auc.append(clf_cv.best_score_)

    

    #Instantiate Classifier

    clf = KNeighborsClassifier(n_neighbors=clf_cv.best_params_.get('n_neighbors'),algorithm='brute')

    clf.fit(train.toarray(),y_tr)

    y_pred_proba_test = clf.predict_proba(test.toarray())[:,1]

    y_pred_proba_tr = clf.predict_proba(train.toarray())[:,1]

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_tr, y_pred_proba_tr)

   

    test_auc.append(roc_auc_score(y_test,y_pred_proba_test))

    training_auc.append(roc_auc_score(y_tr,y_pred_proba_tr))

    k_nn.append(clf_cv.best_params_.get('n_neighbors'))

    

figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

pylab.subplot(2,2,1)    

pylab.plot(iteration, training_auc, '-b', label='Training AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('Training AUC')

pylab.grid()





pylab.subplot(2,2,2)

pylab.plot(iteration, test_auc, '-b', label='Test AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('test AUC')

pylab.grid()





pylab.subplot(2,2,3)

pylab.plot(iteration, cv_auc, '-b', label='CV AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('CV AUC')

pylab.grid()



pylab.subplot(2,2,4)

pylab.plot(iteration, k_nn, '-b', label='Number of nearest neighbours',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('Number of nearest neighbours')

pylab.grid()

pylab.show()
import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from sklearn.naive_bayes import MultinomialNB



alpha_list=list()

test_auc=list()

training_auc=list()

cv_auc=list()



iteration=np.arange(1,10)



param_grid = {'alpha':[ 0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}



for i in range(1,10):





    #Hyper parameter Tuning for lamda with 3 fold cross validation

    clf = MultinomialNB()



    clf_cv= GridSearchCV(clf,param_grid,cv=3,scoring='roc_auc')



    clf_cv.fit(train.toarray(),y_tr)

    cv_auc.append(clf_cv.best_score_)

    

    #Instantiate Classifier

    clf = MultinomialNB(alpha=clf_cv.best_params_.get('alpha'))

    clf.fit(train.toarray(),y_tr)

    y_pred_proba_test = clf.predict_proba(test.toarray())[:,1]

    y_pred_proba_tr = clf.predict_proba(train.toarray())[:,1]

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_tr, y_pred_proba_tr)

   

    test_auc.append(roc_auc_score(y_test,y_pred_proba_test))

    training_auc.append(roc_auc_score(y_tr,y_pred_proba_tr))

    alpha_list.append(clf_cv.best_params_.get('alpha'))

    

figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

pylab.subplot(2,2,1)    

pylab.plot(iteration, training_auc, '-b', label='Training AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('Training AUC')

pylab.grid()





pylab.subplot(2,2,2)

pylab.plot(iteration, test_auc, '-b', label='Test AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('test AUC')

pylab.grid()





pylab.subplot(2,2,3)

pylab.plot(iteration, cv_auc, '-b', label='CV AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('CV AUC')

pylab.grid()



pylab.subplot(2,2,4)

pylab.plot(iteration, alpha_list, '-b', label='Alpha',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('alpha')

pylab.grid()

pylab.show()

    

import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')



from sklearn.calibration import CalibratedClassifierCV



alpha_list=list()

test_auc=list()

training_auc=list()

cv_auc=list()



iteration=np.arange(1,10)



param_grid = {'alpha':[ 0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}



for i in range(1,10):





    #Hyper parameter Tuning for lamda with 3 fold cross validation

    clf = linear_model.SGDClassifier(loss='hinge')



    clf_cv= GridSearchCV(clf,param_grid,cv=3,scoring='roc_auc')



    clf_cv.fit(train.toarray(),y_tr)

    

    cv_auc.append(clf_cv.best_score_)

    

    #Instantiate Classifier

    clf = linear_model.SGDClassifier(loss='hinge',alpha=clf_cv.best_params_.get('alpha'))

    calibrated = CalibratedClassifierCV(clf, method='sigmoid', cv=5)

    calibrated.fit(train.toarray(),y_tr)



    y_pred_proba_test = calibrated.predict_proba(test.toarray())[:, 1]

    y_pred_proba_tr = calibrated.predict_proba(train.toarray())[:, 1]

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_tr, y_pred_proba_tr)

   

    test_auc.append(roc_auc_score(y_test,y_pred_proba_test))

    training_auc.append(roc_auc_score(y_tr,y_pred_proba_tr))

    alpha_list.append(clf_cv.best_params_.get('alpha'))

    



figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

pylab.subplot(2,2,1)

pylab.plot(iteration, training_auc, '-b', label='Training AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('Training AUC')

pylab.grid()







pylab.subplot(2,2,2)

pylab.plot(iteration, test_auc, '-b', label='Test AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('test AUC')

pylab.grid()





pylab.subplot(2,2,3)

pylab.plot(iteration, cv_auc, '-b', label='CV AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('CV AUC')

pylab.grid()





pylab.subplot(2,2,4)

pylab.plot(iteration, alpha_list, '-b', label='Alpha',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('alpha')

pylab.yscale("log")

pylab.grid()

pylab.show()



import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')



from sklearn.svm import SVC

c_list=list()

gamma_list=list()

test_auc=list()

training_auc=list()

cv_auc=list()



iteration=np.arange(1,10)



param_grid = {'C':[ 0.0001,0.001,0.01,0.1,1,10,100,1000,10000],

              'gamma':[ 0.0001,0.001,0.01,0.1,1,10,100,1000,10000]}



for i in range(1,10):





    #Hyper parameter Tuning for lamda with 3 fold cross validation

    clf = SVC(kernel='rbf',probability=True)



    clf_cv= GridSearchCV(clf,param_grid,cv=3,scoring='roc_auc')



    clf_cv.fit(train.toarray(),y_tr)

    cv_auc.append(clf_cv.best_score_)

    

    #Instantiate Classifier

    clf = SVC(kernel='rbf', gamma=clf_cv.best_params_.get('gamma'), C=clf_cv.best_params_.get('C'),probability=True)

    clf.fit(train.toarray(),y_tr)

    y_pred_proba_test = clf.predict_proba(test.toarray())[:,1]

    y_pred_proba_tr = clf.predict_proba(train.toarray())[:,1]

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)

    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_tr, y_pred_proba_tr)

   

    test_auc.append(roc_auc_score(y_test,y_pred_proba_test))

    training_auc.append(roc_auc_score(y_tr,y_pred_proba_tr))

    c_list.append(clf_cv.best_params_.get('C'))

    gamma_list.append(clf_cv.best_params_.get('gamma'))

    

figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

pylab.subplot(3,2,1)    

pylab.plot(iteration, training_auc, '-b', label='Training AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('Training AUC')

pylab.grid()





pylab.subplot(3,2,2)

pylab.plot(iteration, test_auc, '-b', label='Test AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('test AUC')

pylab.grid()





pylab.subplot(3,2,3)

pylab.plot(iteration, cv_auc, '-b', label='CV AUC',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('CV AUC')

pylab.grid()



pylab.subplot(3,2,4)

pylab.plot(iteration, c_list, '-b', label='C',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('c')

pylab.grid()





pylab.subplot(3,2,5)

pylab.plot(iteration, gamma_list, '-b', label='gamma',linewidth=2.0,marker='o')

pylab.legend(loc='upper right')

pylab.xlabel('Execution number')

pylab.ylabel('gamma')

pylab.grid()

pylab.show()

    


