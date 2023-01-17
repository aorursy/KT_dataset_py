import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../input/utiles/')

from process_data import pre_process_dataset

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import random
import time
from tqdm import tqdm
import IPython.display as ipd
import seaborn as sns
import itertools
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV , RandomizedSearchCV
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neighbors import KNeighborsClassifier as KNC

%matplotlib inline
def plot_confusion_matrix(y_true, y_pred, class_names,title="Confusion matrix",normalize=False,onehot = False, size=4):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    if onehot :
        cm = confusion_matrix([y_i.argmax() for y_i in y_true], [y_ip.argmax() for y_ip in y_pred])
    else:
        cm = confusion_matrix(y_true, y_pred)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2) if normalize else cm

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "red" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #return figure
DBtrain = pd.read_csv('../input/titanic/train.csv')
DBtest = pd.read_csv('../input/titanic/test.csv')
DBtrain.info(verbose=True)
DBtrain.head()
# Numerical Data
DBtrain.describe()
DBtrain.describe(include=['object'])
DBtrain_num = ['Age', 'SibSp', 'Parch', 'Fare']
DBtrain_cat = ['Sex', 'Ticket', 'Cabin', 'Embarked', 'Survived','Pclass']
rcParams['figure.figsize'] = 20, 10
fig, axes = plt.subplots(nrows=2, ncols=2)
colors = ['b', 'g', 'r', 'k']
n_bins = None
for n, zippack in enumerate(zip(axes.flatten(), DBtrain_num)):
    axhist , i_label = zippack
    axhist.hist(DBtrain[i_label],bins=n_bins, histtype='bar',color=colors[n] ,density =None)
    axhist.set_title(i_label)
rcParams['figure.figsize'] = 8, 4
n_bins = 5
DBtrain['AgeBand'] = pd.cut(DBtrain['Age'], [min(0,DBtrain['Age'].min()),16,32,48,64,max(100,DBtrain['Age'].max())])
_ = DBtrain[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False ).mean().sort_values(by='AgeBand', ascending=True)
_.index = _['AgeBand']
_ = _.plot(kind='barh',rot =30)
_.set_xlabel('Probability')
_.set_title('P-survive VS Age')
DBtrain[['AgeBand', 'PassengerId']].groupby(['AgeBand'], as_index=False ).count().sort_values(by='AgeBand', ascending=True)
rcParams['figure.figsize'] = 8, 4
n_bins = 3
DBtrain['FareBand'] = pd.cut(DBtrain['Fare'], [min(-1,DBtrain['Fare'].min()),85,170,256,426,max(600,DBtrain['Fare'].max())])
#DBtrain['FareBand'] = pd.cut(DBtrain['Fare'], n_bins)
_ = DBtrain[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False ).mean().sort_values(by='FareBand', ascending=True)
_.index = _['FareBand']
_ = _.plot(kind='barh',rot =30)
_.set_xlabel('Probability')
_.set_title('P-survive VS Fare')
DBtrain[['FareBand', 'PassengerId']].groupby(['FareBand'], as_index=False ).count().sort_values(by='FareBand', ascending=True)
rcParams['figure.figsize'] = 6, 4
coor_matrix = DBtrain[DBtrain_num].corr()
print(coor_matrix)
_ = sns.heatmap(coor_matrix)
pd.pivot_table(DBtrain, index= 'Survived',values= DBtrain_num,aggfunc=['mean' ])
rcParams['figure.figsize'] = 20, 10
fig, axes = plt.subplots(nrows=2, ncols=3)
colors = ['b', 'g', 'r', 'k','m','b']
for n, zippack in enumerate(zip(axes.flatten(), DBtrain_cat)):
    axhist , i_label = zippack
    axhist.bar(DBtrain[i_label].value_counts().index.astype('object'), DBtrain[i_label].value_counts(),color=colors[n],width=0.5)
    axhist.set_title(i_label)

rcParams['figure.figsize'] = 10, 2
not_pivot = ['Survived','Ticket', 'Cabin' ]
_ = pd.pivot_table(DBtrain, index= 'Survived',values= 'Ticket',aggfunc=['count'])
print(_)
for i_label in DBtrain_cat:
    if i_label not in not_pivot:
        _ = pd.pivot_table(DBtrain, index= i_label,values= 'Survived',aggfunc=['mean','count'])
        print(_)
        _ =_['mean'].plot(kind='barh')
        _.set_xlabel('P-survive')
DBtrain['n_parents'] = DBtrain['SibSp'] + DBtrain['Parch']
_ = pd.pivot_table(DBtrain, columns= 'n_parents', index= 'Survived',values= 'PassengerId',aggfunc='count')
print(_)
DBtrain['accompanied'] = DBtrain['n_parents'].apply(lambda x: 1 if x >0 else 0)
_ = pd.pivot_table(DBtrain, columns= 'accompanied', index= 'Survived',values= 'PassengerId',aggfunc='count')
print(_)
_=DBtrain['Cabin'].value_counts().sort_values(ascending=False )
print(_.shape)
print(_)
# Count how many cabins a passenger bought
DBtrain['Count_Cabin'] = DBtrain['Cabin'].apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
#DBtrain['Count_Cabin'].value_counts()

# Categorizing which type of cabin a passenger bought
DBtrain['w_Cabin'] = DBtrain['Cabin'].apply(lambda x: 'nn' if pd.isna(x) else x[0])
#DBtrain['w_Cabin'].value_counts() 
# We will see the relationship between survivors and the number of cabin that bought
_ = pd.pivot_table(DBtrain, columns= 'Count_Cabin', index= 'Survived',values= 'Ticket',aggfunc='count')
print(_)
print("______________________")
# we will see the relationship between survivors and the type of cabin that they bought
_ = pd.pivot_table(DBtrain, columns= 'w_Cabin', index= 'Survived',values= 'Ticket',aggfunc='count')
print(_)
# The passenger bought a cabin yes or not
DBtrain['b_Cabin'] = DBtrain['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
_ = pd.pivot_table(DBtrain, columns= 'b_Cabin', index= 'Survived',values= 'Ticket',aggfunc='count')
print(_)
DBtrain['title_Name'] = DBtrain['Name'].apply(lambda x: x.split(',')[1:][0].split('.')[0].strip())
DBtrain['title_Name'].value_counts() 
# Grouping title names using only the predominants
title_pred = ['Mr', 'Miss', 'Mrs']
DBtrain['title_Name'] = DBtrain['title_Name'].apply(lambda x: x if x in title_pred else 'Others')
print(DBtrain['title_Name'].value_counts() )
_ = pd.pivot_table(DBtrain, columns= 'title_Name', index= 'Survived',values= 'PassengerId',aggfunc='count')
print(_)
# We goint to pretend that there is a relatión between the ticket ID and the survivors
#   We take two new features
#      1. If a ticket is a number, take the logarithm10 round to whole
#      2. If a ticket start with a letter, take that string of letters. This technique is so risky, because in our test set could be a new string and will be in trouble
countbyTicket = DBtrain['Ticket'].value_counts()#.sort_values(ascending = False)
DBtrain['num_Ticket'] = DBtrain['Ticket'].apply(lambda x: 1 if x.isnumeric() else 0)
DBtrain['word_Ticket'] = DBtrain['Ticket'].apply(lambda x: x.split(' ')[0].replace('.', '').replace('/','').lower() if not x.split(' ')[0].isnumeric() else 0)
print(DBtrain['num_Ticket'].value_counts())
print(DBtrain['word_Ticket'].value_counts())
_ = pd.pivot_table(DBtrain, columns= 'num_Ticket', index= 'Survived',values= 'Ticket',aggfunc='count')
print(_)
_ = pd.pivot_table(DBtrain, columns= 'word_Ticket', index= 'Survived',values= 'Ticket',aggfunc='count')
print(_)
DBtrain=DBtrain.drop(columns=['Ticket','num_Ticket','word_Ticket'])
DBtrain.describe(include=['object','category'])
DBtrain.describe()
DBtrain.info()
DBtrain['Embarked'].mode()[0]
media_age = DBtrain.dropna(subset=['Age'])[['Age','Pclass','Sex']].groupby(['Pclass', 'Sex'], as_index=False ).mean().sort_values(by='Age', ascending=True)
def impute_years(x):
    if x['Age'] == x['Age'] :
        return x['Age']
    else:
        return media_age.loc[media_age['Pclass']==x['Pclass'] , ['Age', 'Sex']].loc[media_age['Sex']==x['Sex'] , ['Age']]['Age'].tolist()[0]

DBtrain['Embarked'] = DBtrain['Embarked'].fillna(DBtrain['Embarked'].mode()[0])
DBtrain['Age'] = DBtrain.apply(impute_years, axis=1)# DBtrain['Age'].fillna( DBtrain['Age'].mean())
DBtrain['AgeBand'] = pd.cut(DBtrain['Age'], [min(0,DBtrain['Fare'].min()),16,32,48,64,max(100,DBtrain['Age'].max())])
DBtrain.info()
# On "Histogram for numerical Data" we saw that "Fare" is not normalize, the distance between his values are significance
#     We try to normalize that data, taken the logarithm10 to those values
rcParams['figure.figsize'] = 10, 5
DBtrain['norm_Fare']= DBtrain['Fare'].apply(lambda x : np.log10(x+1))
#DBtrain['norm_Fare']= DBtrain['norm_Fare']
DBtrain['norm_Fare'].hist().set_title('Fare normalized')
# we can see that the histogram look like a normal distributions more accurate
DBtrain.head(2)
# Second Test
DBtrain_f = DBtrain.copy()
DBtrain_f.index  = DBtrain_f.PassengerId
DBtrain_f = DBtrain_f.drop(columns=[ 'Cabin','Fare','Name','PassengerId','w_Cabin', 'Count_Cabin','n_parents','SibSp','Parch','norm_Fare', 'Age'])
DBtrain_f['Pclass'] = DBtrain_f['Pclass'].astype(str) # categorical feature
DBtrain_f['accompanied'] = DBtrain_f['accompanied'].astype(str) # categorical feature
DBtrain_f['b_Cabin'] = DBtrain_f['b_Cabin'].astype(str) # categorical feature
DBtrain_f['AgeBand'] = DBtrain_f['AgeBand'].astype(str) # categorical feature
DBtrain_f['AgeBand'] = DBtrain_f['AgeBand'].apply(lambda x : x.replace('(','').replace(']','').replace(',','_'))
DBtrain_f['FareBand'] = DBtrain_f['FareBand'].astype(str) # categorical feature
DBtrain_f['FareBand'] = DBtrain_f['FareBand'].apply(lambda x : x.replace('(','').replace(']','').replace(',','_'))
DBtrain_f.info()
DBtrain_f.head()
# First test
DBtrain_f = DBtrain.copy()
DBtrain_f.index  = DBtrain_f.PassengerId
DBtrain_f = DBtrain_f.drop(columns=[ 'Cabin','Fare','Name','PassengerId','w_Cabin', 'Count_Cabin','n_parents','SibSp','Parch','AgeBand','FareBand' ])
DBtrain_f['Pclass'] = DBtrain_f['Pclass'].astype(str) # categorical feature
DBtrain_f['accompanied'] = DBtrain_f['accompanied'].astype(str) # categorical feature
DBtrain_f['b_Cabin'] = DBtrain_f['b_Cabin'].astype(str) # categorical feature
DBtrain_f.info()
DBtrain_f.head()
DBtrain_f = pd.get_dummies(DBtrain_f)
X_train = DBtrain_f.drop(columns='Survived')
Y_train = DBtrain_f['Survived']
DBtrain_f.head()
X_train, Y_train = pre_process_dataset(all_categorical=False, Test=False, fillna_age = None,path_test='../input/titanic/test.csv',
                                                                                           path_train='../input/titanic/train.csv', )
#Modelo with SVC
clf = svm.SVC(verbose= True,random_state=5,C=10, kernel='rbf',degree=3, gamma='auto',probability=True)
clf = BalancedBaggingClassifier(base_estimator=clf,
                                sampling_strategy='auto',
                                replacement=False,random_state=42)
fit_model = clf.fit(X_train, Y_train)
y_pred = clf.predict(X_train)
print("SVM: acc:"+ str(accuracy_score(Y_train, y_pred)))
plot_confusion_matrix(y_true=Y_train, y_pred=y_pred, class_names=['No', 'Yes'],title="SVC",normalize=True, size=4)
xgb_model_ = XGBClassifier(learning_rate=0.01,
                    n_estimators=150,
                    max_depth=100,
                    min_child_weight=.05,
                    gamma=0,
                    subsample=.5,
                    colsample_bytree=0.5,
                    objective='multi:softmax',
                    num_class=10,
                    num_parallel_tree = 8,
                    seed=27,verbosity= 1,n_jobs=8 )
xgb_model = BalancedBaggingClassifier(base_estimator=xgb_model_,
                                sampling_strategy='auto',
                                replacement=False,random_state=42)
xgb_model.fit(X_train, Y_train)
#xgb_model.save_model('models/xgbmodel')
y_pred = xgb_model.predict(X_train)
print("XGB: acc:"+ str(accuracy_score(Y_train, y_pred)))
plot_confusion_matrix(y_true=Y_train, y_pred=y_pred, class_names=['No', 'Yes'],title="XGB",normalize=True, size=4)
clf_KNC_ = KNC(n_neighbors=2,n_jobs=-1,algorithm='brute',p=2 )
clf_KNC = BalancedBaggingClassifier(base_estimator=clf_KNC_,
                                sampling_strategy='auto',
                                replacement=False,random_state=42)
clf_KNC.fit(X_train, Y_train)
y_pred = clf_KNC.predict(X_train)
print("KNC: acc:"+ str(accuracy_score(Y_train, y_pred)))
plot_confusion_matrix(y_true=Y_train, y_pred=y_pred, class_names=['No', 'Yes'],title="KNC",normalize=True,size=4)
clf_RFC_ = RandomForestClassifier( n_estimators=100, n_jobs=-1)
clf_RFC = BalancedBaggingClassifier(base_estimator=clf_RFC_,
                                sampling_strategy='auto',
                                replacement=False,random_state=42)
clf_RFC.fit(X_train, Y_train)
y_pred = clf_RFC.predict(X_train)
print("RFC: acc:"+ str(accuracy_score(Y_train, y_pred)))
plot_confusion_matrix(y_true=Y_train, y_pred=y_pred, class_names=['No', 'Yes'],title="RFC",normalize=True,size=4)
model_VC = VotingClassifier (estimators=[ ('xgb', xgb_model), ('knc', clf_KNC),('rfc',clf_RFC)], voting='hard', weights=[1,1,1],n_jobs=-1)
model_VC.fit(X_train, Y_train)
y_pred = model_VC.predict(X_train)
print("VotingClassifier: acc:"+ str(accuracy_score(Y_train, y_pred)))
plot_confusion_matrix(y_true=Y_train, y_pred=y_pred, class_names=['No', 'Yes'],title="VotingClassifier",normalize=True,size=4)
media = DBtrain[['Age','Pclass','Sex','Embarked']].groupby(['Pclass', 'Sex','Embarked'], as_index=False ).mean().sort_values(by='Pclass', ascending=True)
media
def impute_years(x):
    if x['Age'] == x['Age']:
        return x['Age']
    else:
        return media.loc[media['Pclass']==x['Pclass'] , ['Age', 'Sex','Embarked']].loc[media['Sex']==x['Sex'] , ['Age','Embarked']].loc[media['Embarked']==x['Embarked'] , ['Age']]['Age'].tolist()[0]
X_age = DBtrain.apply(impute_years, axis=1)                                                                                                                          
#from process_data import pre_process_dataset
X_test, X_test_origin = pre_process_dataset(all_categorical=False, Test=True, fillna_age = None,path_test='../input/titanic/test.csv',
                                                                                           path_train='../input/titanic/train.csv', )
estmt = model_VC.estimators
y_predict_svc = clf.predict(X_test)
y_predict_xgb = estmt[0][1].predict(X_test)
y_predict_knc = estmt[1][1].predict(X_test)
y_predict_rfc = estmt[2][1].predict(X_test)
y_predict_vc = model_VC.predict(X_test)
print("CLASSIF test: equal:"+ str(accuracy_score(y_predict_vc, y_predict_rfc)))
submission_file = pd.DataFrame({ 'PassengerId':np.stack(X_test.index.tolist()),'Survived': y_predict_vc})
submission_file.to_csv('submission/submm_vc2.csv',index=False, )
submission_file = pd.DataFrame({ 'PassengerId':np.stack(X_test.index.tolist()),'Survived': y_predict_xgb})
submission_file.to_csv('submission/submm_xgb2.csv',index=False, )
submission_file = pd.DataFrame({ 'PassengerId':np.stack(X_test.index.tolist()),'Survived': y_predict_knc})
submission_file.to_csv('submission/submm_knc2.csv',index=False, )
submission_file = pd.DataFrame({ 'PassengerId':np.stack(X_test.index.tolist()),'Survived': y_predict_rfc})
submission_file.to_csv('submission/submm_rfc2.csv',index=False, )

ls
a =1
a
a*2