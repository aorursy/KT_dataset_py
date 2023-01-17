%matplotlib inline

import graphviz

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier, VotingClassifier

from sklearn.naive_bayes import GaussianNB
train = pd.read_csv('../input/carInsurance_train.csv')

test = pd.read_csv('../input/carInsurance_test.csv')
print('The train dataset has %d observations and %d features' % (train.shape[0], train.shape[1]))

print('The test dataset has %d observations and %d features' % (test.shape[0], test.shape[1]))
train.head()
# Take a peak at the non-categorical

train.describe()
# Take a peak at the categorical

train.describe(include=['O'])
# merge train and test data here in order to impute missing values all at once

all=pd.concat([train,test],keys=('train','test'))
all.head()
# First check out correlations among numeric features

# Heatmap is a useful tool to get a quick understanding of which variables are important

cor = all.corr()

cor = cor.drop(['Id'],axis=1).drop(['Id'],axis=0)

plt.figure(figsize=(12,12))

sns.heatmap(cor,annot=True)
imp_feats = ['CarInsurance','Age','Balance','HHInsurance','NoOfContacts','DaysPassed','PrevAttempts']

sns.pairplot(all[imp_feats],hue='CarInsurance',size=2.5)

plt.show()
all.drop(['CarInsurance','Id'],axis=1,inplace=True)

print(all.shape)
total = all.isnull().sum()

pct = total/all.isnull().count()

NAs = pd.concat([total,pct],axis=1,keys=('Total','Pct'))

NAs[NAs.Total>0].sort_values(by='Total',ascending=False)
all_df = all.copy()



# Fill missing outcome as not in previous campaign

all_df[all_df['DaysPassed']==-1].count()

all_df.loc[all_df['DaysPassed']==-1,'Outcome']='NoPrev'



# Fill missing communication with none 

all_df['Communication'].value_counts()

all_df['Communication'].fillna('None',inplace=True)



# Fill missing education with the most common education level by job type

all_df['Education'].value_counts()



# Create job-education level mode mapping

edu_mode=[]

job_types = all_df.Job.value_counts().index

for job in job_types:

    mode = all_df[all_df.Job==job]['Education'].value_counts().nlargest(1).index

    edu_mode = np.append(edu_mode,mode)

edu_map=pd.Series(edu_mode,index=all_df.Job.value_counts().index)



# Apply the mapping to missing eductaion obs

for j in job_types:

    all_df.loc[(all_df['Education'].isnull()) & (all_df['Job']==j),'Education'] = edu_map.loc[edu_map.index==j][0]

all_df['Education'].fillna('None',inplace=True)



# Fill missing job with none

all_df['Job'].fillna('None',inplace=True)

# Double check if there is still any missing value

print("Remaining missing values: %d"%(all_df.isnull().sum().sum()))

all_df.head()
# Get call length

all_df['CallEnd'] = pd.to_datetime(all_df['CallEnd'])

all_df['CallStart'] = pd.to_datetime(all_df['CallStart'])

all_df['CallStartHour'] = all_df['CallStart'].dt.hour

all_df['CallLength'] = ((all_df['CallEnd'] - all_df['CallStart'])/np.timedelta64(1,'m')).astype(float)
all_df['CallLengthPercent'] = all_df['CallLength']/all_df['CallLength'].max()
all_df['AgePercent'] = all_df['Age']/all_df['Age'].max()
all_df['BalancePercent'] = all_df['Balance']/all_df['Balance'].max()
all_df['Education'] = all_df['Education'].replace({'None':0,'primary':1,'secondary':2,'tertiary':3})
all_df = all_df.drop(['Age','Balance', 'CallLength', 'CallStart', 'CallEnd'],axis=1)
all_df.head()
# Spilt numeric and categorical features

cat_feats = all_df.select_dtypes(include=['object']).columns

num_feats = all_df.select_dtypes(include=['float64','int64']).columns

num_df = all_df[num_feats]

print('There are %d numeric features and %d categorical features\n' %(len(num_feats),len(cat_feats)))

print('Numeric features:\n',num_feats.values)

print('Categorical features:\n',cat_feats.values)
# One hot encoding

exlude = ['CallStart' 'CallEnd']

cat_feats = [val for val in cat_feats if val not in exlude]

cat_df = all_df[cat_feats]

cat_df = pd.get_dummies(cat_df)

cat_feats
cat_df.head()
# Recombine data

all_data = pd.concat([num_df,cat_df],axis=1)

all_data.head()
# Split train and test

idx=pd.IndexSlice

train_df=all_data.loc[idx[['train',],:]]

test_df=all_data.loc[idx[['test',],:]]

train_label=train['CarInsurance']

print(train_df.shape)

print(len(train_label))

print(test_df.shape)

# Train test split

x_train, x_test, y_train, y_test = train_test_split(train_df,train_label,test_size = 0.3,random_state=3)
x_test.head()
train_with_label = x_train.copy()

train_with_label['CarInsurance'] = y_train.values

train_with_label.head()
col_names = ['CarInsurance', 'AgePercent', 'BalancePercent', 'CallLengthPercent', 'Education']

sns.pairplot(train_with_label[col_names],hue='CarInsurance',size=2.5)

plt.show()
# The confusion matrix plotting function is from the sklearn documentation below:

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



class_names = ['Success','Failure']
def model_fit(model, x_train=x_train, y_train=y_train, x_test=x_test):

    clf = model.fit(x_train,y_train)

    y_pred = clf.predict(x_test)

    return {

        'clf': clf,

        'x_train':x_train, 

        'y_train':y_train, 

        'x_test':x_test,

        'y_pred':y_pred

    }
# Create a cross validation function 

def get_best_model(estimator, params_grid={}):

    

    model = GridSearchCV(estimator = estimator,param_grid = params_grid,cv=3, scoring="accuracy", n_jobs= -1)

    model.fit(x_train,y_train)

    print('\n--- Best Parameters -----------------------------')

    print(model.best_params_)

    print('\n--- Best Model -----------------------------')

    best_model = model.best_estimator_

    print(best_model)

    return best_model
# Based off of: https://www.kaggle.com/emmaren/cold-calls-data-mining-and-model-selection

def model_report(clf, y_pred, y_test=y_test, class_names=['Success','Failure'], cv=5, feature_imp=True):

    # model report     

    cm = confusion_matrix(y_test,y_pred)

    plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')



    print('\n--- Train Set -----------------------------')

    print('Accuracy: %.5f +/- %.4f' % (np.mean(cross_val_score(clf,x_train,y_train,cv=cv)),np.std(cross_val_score(clf,x_train,y_train,cv=cv))))

    print('AUC: %.5f +/- %.4f' % (np.mean(cross_val_score(clf,x_train,y_train,cv=cv,scoring='roc_auc')),np.std(cross_val_score(clf,x_train,y_train,cv=cv,scoring='roc_auc'))))

    print('\n--- Validation Set -----------------------------')    

    print('Accuracy: %.5f +/- %.4f' % (np.mean(cross_val_score(clf,x_test,y_test,cv=cv)),np.std(cross_val_score(clf,x_test,y_test,cv=cv))))

    print('AUC: %.5f +/- %.4f' % (np.mean(cross_val_score(clf,x_test,y_test,cv=cv,scoring='roc_auc')),np.std(cross_val_score(clf,x_test,y_test,cv=cv,scoring='roc_auc'))))

    print('-----------------------------------------------') 



    # feature importance 

    if feature_imp:

        feat_imp = pd.Series(clf.feature_importances_,index=all_data.columns)

        feat_imp = feat_imp.nlargest(15).sort_values()

        plt.figure()

        feat_imp.plot(kind="barh",figsize=(6,8),title="Most Important Features")

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)
x_test.head()
dt.predict(x_train)
model_report(dt, dt.predict(x_test), y_test)
parameters = {"criterion": ["gini", "entropy"],

              "min_samples_split": [2, 10, 20],

              "max_depth": [None, 2, 5, 10],

              "min_samples_leaf": [1, 5, 10],

              "max_leaf_nodes": [None, 5, 10, 20],

              }

dt_best = get_best_model(dt, parameters)
model_report(dt_best, dt_best.predict(x_test), y_test)
dot_data = export_graphviz(dt_best, out_file=None, feature_names=list(x_test.columns), filled=True, rounded=True,) 

graph = graphviz.Source(dot_data)
graph
knn = KNeighborsClassifier()

parameters = {'n_neighbors':[5,6,7], 

              'p':[1,2],

              'weights':['uniform','distance']}

clf_knn = get_best_model(knn,parameters)
model_report(clf_knn, clf_knn.predict(x_test), y_test, feature_imp=False)
clf_nb = GaussianNB()

model_fit(model=clf_nb)
model_report(clf_nb, clf_nb.predict(x_test), y_test, feature_imp=False)
lg = LogisticRegression(random_state=3)

parameters = {'C':[0.8,0.9,1], 

              'penalty':['l1','l2']}

clf_lg = get_best_model(lg,parameters)
model_report(clf_lg, clf_lg.predict(x_test), y_test, feature_imp=False)
rf = RandomForestClassifier(random_state=3)

parameters={'n_estimators':[100],

            'max_depth':[10],

            'max_features':[13,14],

            'min_samples_split':[11]}

clf_rf= get_best_model(rf,parameters)
model_report(clf_rf, clf_rf.predict(x_test), y_test, feature_imp=False)
svc = svm.SVC(kernel='rbf', probability=True, random_state=3)

parameters = {'gamma': [0.005,0.01,0.02],

              'C': [0.5,1,5]}

clf_svc = get_best_model(svc, parameters)
model_report(clf_svc, clf_svc.predict(x_test), y_test, feature_imp=False)
clf_vc = VotingClassifier(estimators=[('rf', clf_rf),

                                      ('lg', clf_lg), 

                                      ('svc', clf_svc)], 

                          voting='hard',

                          weights=[4,1,1])

clf_vc = clf_vc.fit(x_train, y_train)
accuracy_score(y_test, clf_vc.predict(x_test))
dt.score(x_train)