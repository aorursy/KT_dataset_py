import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas_profiling

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



sns.set(style='ticks', context='talk')

plt.style.use('dark_background')
DATADIR = '../input/titanic/'



train  = pd.read_csv('{0}train.csv'.format(DATADIR))

test   = pd.read_csv('{0}test.csv'.format(DATADIR))



train_len = len(train)

IDtest = test['PassengerId'] 
train.head()
test.head()
dataset = pd.concat( objs=[train, test], axis=0 ).reset_index(drop=True)
fig, ax = plt.subplots(1,1,figsize=(10,6))



sns.distplot(dataset['Age'][(dataset['Age'].notnull())&(dataset['Survived']==0)] ,color='r', ax=ax )

sns.distplot(dataset['Age'][(dataset['Age'].notnull())&(dataset['Survived']==1)] ,color='b', ax=ax )



plt.legend(['Not Survived', 'Survived'])
def age_classify(x):

    return int(x//10)
dataset['Age'].isnull().sum()
dataset['n_Age'] = dataset['Age'][dataset['Age'].notnull()].apply(age_classify)
dataset['n_Age'].value_counts()
m = len(dataset['n_Age'].value_counts())

total = dataset['n_Age'].value_counts().sum()

total_null = dataset['n_Age'].isnull().sum()



age_percent = [0]*m

age_dist    = [0]*m



for i in range(m):

    age_percent[i]=round((dataset['n_Age'].value_counts()[i]/total),3)

    age_dist[i] = int(round(age_percent[i]*total_null))



age_dist[3]-=1

print('percentage[%] : ',np.multiply(age_percent,100))

print('counts for n_Age :',age_dist)



total_null = dataset['n_Age'].isnull().sum()



for i in dataset['n_Age'].index[dataset['n_Age'].isnull()]:

    if age_dist[0] != 0: 

        dataset['n_Age'][i]=0.0

        age_dist[0]-=1

    elif age_dist[1] != 0: 

        dataset['n_Age'][i]=1.0

        age_dist[1]-=1

    elif age_dist[2] != 0:

        dataset['n_Age'][i]=2.0

        age_dist[2]-=1

    elif age_dist[3] != 0: 

        dataset['n_Age'][i]=3.0

        age_dist[3]-=1

    elif age_dist[4] != 0: 

        dataset['n_Age'][i]=4.0

        age_dist[4]-=1

    elif age_dist[5] != 0:

        dataset['n_Age'][i]=5.0

        age_dist[5]-=1

    elif age_dist[6] != 0: 

        dataset['n_Age'][i]=6.0

        age_dist[6]-=1            

    elif age_dist[7] != 0: 

        dataset['n_Age'][i]=7.0

        age_dist[7]-=1

    elif age_dist[8] != 0: 

        dataset['n_Age'][i]=8.0

        age_dist[8]-=1
dataset['n_Age'].isnull().sum()
fig, ax = plt.subplots(1,2, figsize=(14, 6))



g1 = sns.countplot( x='n_Age', hue='Survived',data=dataset, palette='YlGnBu_r', ax=ax[0])

g2 = sns.factorplot(x='n_Age', y='Survived',  data=dataset, palette='YlGnBu_r', ax=ax[1], kind = 'bar')



plt.close(g2.fig)

plt.subplots_adjust(wspace = 0.3)
fig , ax= plt.subplots(1,2,figsize=(18,6))



g1 = sns.factorplot(x='n_Age',hue='Survived', data=dataset, kind='count', palette='YlGnBu_r', ax=ax[0])

g2 = sns.factorplot(x='n_Age',  y='Survived', data=dataset, kind='bar',   palette='YlGnBu_r', ax=ax[1])



ax[0].legend(['Not Survived', 'Survived'],loc='best')

ax[1].set_ylabel('Survived Probability')



plt.subplots_adjust(wspace=1)

plt.close(g1.fig)

plt.close(g2.fig)
dataset.drop(labels=['Age'], axis=1, inplace=True)
dataset['Sex'].value_counts()
fig, ax = plt.subplots(1,2, figsize=(12,6))



g = sns.factorplot( x='Sex', y='Survived', data=dataset, palette='Blues_r', kind='bar' ,ax=ax[0] )

plt.close(g.fig)



g2 = sns.countplot( x='Sex', data=dataset, palette='Blues_r', ax=ax[1] )

plt.subplots_adjust(wspace=0.5)



ax[0].set_xticklabels(['Female', 'Male']);

ax[1].set_xticklabels(['Female', 'Male']);
dataset['Sex'].head()
fig, ax = plt.subplots(2,2, figsize=(15,15))



g1 = sns.countplot(x='n_Age',                  hue='Sex', data=dataset, palette='Blues_r',     ax=ax[0,0])

g2 = sns.barplot(  x='n_Age',    y='Survived', hue='Sex', data=dataset, palette='Greens_r',    ax=ax[0,1])

g3 = sns.barplot(  x='Pclass',   y='Survived', hue='Sex', data=dataset, palette='Oranges_r',   ax=ax[1,0])

g4 = sns.barplot(  x='Embarked', y='Survived', hue='Sex', data=dataset, palette='Reds_r',      ax=ax[1,1])



g1.set_ylabel('Counts')

g2.set_ylabel('survived Probability')

g3.set_ylabel('survived Probability')

g4.set_ylabel('survived Probability')



ax[0,0].legend(['Female','Male'])



plt.subplots_adjust(wspace=0.3)
dataset['SibSp'].value_counts()
dataset['Parch'].value_counts()
dataset['FamSize']=dataset['SibSp']+dataset['Parch']+1
dataset['FamSize'].isnull().sum()
dataset['FamSize'].value_counts()
fig, ax = plt.subplots(2,1, figsize=(12,10))



g1 = sns.countplot(x='FamSize', data=dataset, palette='RdBu', ax=ax[0])

g2 = sns.factorplot(x='FamSize', y='Survived',kind='bar', data=dataset, palette='RdBu', ax=ax[1])

ax[1].set(ylabel='survived probability')



plt.close(g2.fig)

plt.subplots_adjust(hspace=0.4)
dataset.drop(labels=['SibSp','Parch'], axis=1, inplace=True)
dataset['Name'].unique()
dataset['Name'].isnull().sum()
dataset['Title'] = [0]*dataset['Name'].shape[0]
num=0

for x in dataset['Name']:

    title = x.split(',')[1].split(',')[0].split('.')[0]

    title = title.split(' ')[1]

    dataset['Title'][num]=title

    num+=1
dataset['Title'].value_counts()
dataset['Title'].replace(['Rev','Dr','Col','Major','Jonkheer','Ms','Mlle','Mme','Lady','Dona','the Countess','Sir','Capt','Don'],'Rare', inplace=True)

dataset['Title'] = dataset['Title'].replace(['Ms','Mlle','Mme','Lady','Dona','the'],'Miss')

dataset['Title'] = dataset['Title'].replace(['Sir','Capt','Don'],'Mr')



dataset['Title'].unique()
fig, ax = plt.subplots(1,2, figsize=(15, 6))



g1 = sns.countplot(x='Title', data=dataset, palette='YlGnBu_r', ax=ax[0] )

g2 = sns.factorplot(x='Title', y='Survived', data=dataset, palette='YlGnBu_r',kind='bar', ax=ax[1])

plt.close(g2.fig)
dataset.drop(labels=['Name'], axis=1, inplace=True)

dataset.head()
dataset['Embarked'].value_counts()
dataset['Embarked'].fillna(method='ffill', inplace = True)
dataset['Embarked'].isnull().sum()
fig, ax = plt.subplots(1,2, figsize=(18,6))



g = sns.factorplot(x='Embarked', y='Survived',data=dataset, kind='bar', palette='binary', ax=ax[0])

ax[0].set(ylabel='survived probability')



g2 = sns.countplot(x='Embarked', hue='Survived',data=dataset, palette='binary',ax=ax[1])

g2.set(ylabel='survived counts')

g2.legend(['Not Survived', 'Survived'])



plt.close(g.fig)

plt.subplots_adjust(wspace=0.3)
dataset['Fare'].fillna(method='ffill',inplace=True )
dataset['Fare'].isnull().sum()
dataset['n_Fare'] = (dataset['Fare']//1.0)
dataset['n_Fare'].isnull().sum()
dataset['n_Fare'] = pd.qcut(dataset['n_Fare'], 7)
def mapping(x):

    x= str(x)

    left= (x.split(', ')[0].split('(')[1])

    if   left =='-0.001': return 1

    elif left =='7.0'   : return 2

    elif left =='8.0'   : return 3

    elif left =='12.0'  : return 4

    elif left =='19.0'  : return 5

    elif left =='27.0'  : return 6

    elif left =='59.0'  : return 7    
dataset['n_Fare'] = dataset['n_Fare'].map(mapping)

dataset['n_Fare'].astype(int)
dataset['n_Fare'].value_counts()
fig, ax = plt.subplots(1,1, figsize=(15, 6))

g=sns.factorplot(x='n_Fare', y='Survived', data=dataset, kind='bar', palette='YlGnBu_r', ax=ax)

plt.close(g.fig)
fig, ax = plt.subplots(1,2,figsize=(17, 6))



g  = sns.countplot( x='n_Fare', hue='Survived', data=dataset, palette='Greens_r', ax=ax[0])

g2 = sns.factorplot(x='n_Fare', y='Survived',   data=dataset, palette='Blues_r',  ax=ax[1], kind='bar')

ax[0].set(ylabel='Survived counts')

ax[1].set(ylabel='Survived probability')



plt.close(g2.fig)

plt.subplots_adjust(wspace=0.3)
dataset.drop(labels=['Fare', 'PassengerId'],axis=1, inplace=True)
dataset['Pclass'].value_counts()
dataset['Pclass'].isnull().sum()
fig, ax = plt.subplots(1,2, figsize=(18, 6))



g = sns.factorplot( x='Pclass', y='Survived', data=dataset, palette='Greens_r', kind='bar', ax=ax[0] )

plt.close(g.fig)



g = sns.countplot( x='Pclass', hue='Survived',data=dataset, palette='bone_r', ax=ax[1] )

g.legend(['Not Survived', 'Survived'])
dataset['Cabin'].value_counts()
def refine(x):

    if type(x)==float: return 'X'

    else:

        if len(x)>4: return x.split(' ')[0][0]

        else: return x[0]
dataset['n_Cabin'] = dataset['Cabin'].apply(refine)
dataset['n_Cabin'].unique()
fig, ax = plt.subplots(1,2, figsize=(15, 6))

g1 = sns.countplot( x='n_Cabin', data=dataset, palette='rainbow' ,ax=ax[0] )

g2 = sns.factorplot( x='n_Cabin', y='Survived', data=dataset, palette='rainbow', kind='bar', ax=ax[1] )



plt.subplots_adjust(wspace=1.0)

plt.close(g2.fig)
dataset['n_Cabin']=dataset['n_Cabin'].map({'A':2,'B':1,'C':3,'D':1,'E':1,'F':3,'G':2,'T':1,'X':4})
sns.factorplot(x='n_Cabin',y='Survived', data=dataset, kind='bar', palette='YlGnBu_r')
dataset.drop(labels=['Cabin'], axis=1, inplace=True)
dataset['Ticket'].value_counts()
dataset['Ticket_Frequency'] = dataset.groupby('Ticket')['Ticket'].transform('count')
fig, ax = plt.subplots(1,2, figsize=(15, 6))



g1 = sns.countplot(x='Ticket_Frequency',              hue='Sex', data=dataset, palette='Oranges_r', ax=ax[0]  )

g2 = sns.barplot(  x='Ticket_Frequency', y='Survived',hue='Sex', data=dataset, palette='Oranges_r', ax=ax[1]  )

g1.legend(['Female', 'Male'])

plt.subplots_adjust(wspace=0.3)
dataset = dataset.drop(labels=['Ticket'],axis=1)
dataset.head()
#dataset.profile_report()
print('Emabarked unique() :' , dataset['Embarked'].unique())

print('Title.    unique() :' , dataset['Title'   ].unique())

print('Sex       unique() :' , dataset['Sex'     ].unique())

print('Pclass    unique() :' , dataset['Pclass'  ].unique())

print('n_Age     unique() :' , dataset['n_Age'   ].unique())

print('FamSize   unique() :' , dataset['FamSize' ].unique())

print('n_Cabin   unique() :' , dataset['n_Cabin' ].unique())

print('n_Fare    unique() :' , dataset['n_Fare'  ].unique())
# One-hot encoding for non integer values

dataset = pd.get_dummies(dataset, columns=[ 'Sex'     ], prefix='Sx')

dataset = pd.get_dummies(dataset, columns=[ 'Embarked'], prefix='Em')

dataset = pd.get_dummies(dataset, columns=[ 'Title'   ], prefix='Tt')

dataset = pd.get_dummies(dataset, columns=[ 'n_Fare'  ], prefix='nF')



# dataset = pd.get_dummies(dataset, columns=[ 'n_Age'   ], prefix='nA')

# dataset = pd.get_dummies(dataset, columns=[ 'Pclass'  ], prefix='Pc')

# dataset = pd.get_dummies(dataset, columns=[ 'FamSize' ], prefix='FS')

# dataset = pd.get_dummies(dataset, columns=[ 'Ticket_Frequency'], prefix='nT')
# This part is for removing low feature_importance_values.

# However that doesn't mean that we don't need these values.



# So to check this, I erased values which has low importance values, But the result doesn't changed.



# dataset.drop(['nA_1','nA_4','nA_5','nA_6','nA_7','nA_8'],axis=1, inplace=True)

# dataset.drop(['Tt_1','Tt_2','Tt_3','Tt_4'],axis=1,inplace=True)

# dataset.drop(['Em_1','Em_2'],              axis=1,inplace=True)

# dataset.drop(['Pc_1','Pc_2'],              axis=1,inplace=True)

# dataset.drop(['nT_5','nT_8','nT_7','nT_11','nF_2','nF_3','nF_7'],       axis=1,inplace=True)

# dataset.drop(['FS_1','FS_2','FS_4','FS_5','FS_6','FS_7','FS_8','FS_11'],axis=1,inplace=True)
dataset.head()
train = dataset[:train_len]

test  = dataset[train_len:]

test.drop(labels=['Survived'], axis=1, inplace=True)
train['Survived'] = train['Survived'].astype(int)

Y_train = train['Survived']

X_train = train.drop(labels=['Survived'], axis=1)
kfold = StratifiedKFold(n_splits=10)
from xgboost import XGBClassifier



random_state=3

classifiers = []



classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state=random_state))

classifiers.append(LinearDiscriminantAnalysis())

classifiers.append(XGBClassifier(random_state=random_state))
cv_results = []



for classifier in classifiers:

    cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring='accuracy', cv=kfold, n_jobs=4))



cv_means = []

cv_std   = []

for cv_result in cv_results:

    cv_means.append( cv_result.mean() )

    cv_std.append(   cv_result.std()  )

    

cv_res = pd.DataFrame({'CrossValMeans':cv_means, 'CorssValerrors':cv_std,

                       'Algorithm':['SVC','DecisionTree','AdaBoost','RandomForest','ExtraTrees',

                                    'GradientBoosting','MultipleLayerPerceptron','KNeighbors','LogisticRegression','LinearDiscriminantAnalysis','XGBoost']})



g = sns.barplot('CrossValMeans','Algorithm', data=cv_res, palette='YlGnBu', orient='h', **{'xerr':cv_std})

g = g.set(title='Cross Validation scores',xlabel='Mean Accuracy')
XGB = XGBClassifier(random_state=0)



xgb_param_grid={'colsample_bylevel':[0.9],

                'colsample_bytree' :[0.8],

                'gamma'            :[0.99],

                'max_depth'        :[4],

                'min_child_weight' :[1],

                'n_estimators'     :[10],

                'nthread'          :[4],

                'silent'           :[True]}



gsXGB = GridSearchCV(XGB, param_grid=xgb_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)

gsXGB.fit(X_train, Y_train)

XGBC_best = gsXGB.best_estimator_



gsXGB.best_score_
DTC = DecisionTreeClassifier(max_depth=1)



adaDTC = AdaBoostClassifier(DTC, random_state=0)



ada_param_grid = {'base_estimator__criterion':['gini', 'entropy'],

                  'base_estimator__splitter' :['best', 'random'],

                  'algorithm'                :['SAMME', 'SAMME.R'],

                  'n_estimators'             :[100],

                  'learning_rate'            :[0.0001, 0.001, 0.01, 0.1, 1, 5, 10]}



gsadaDTC = GridSearchCV( adaDTC, param_grid=ada_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1 )

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_



gsadaDTC.best_score_
ExtC = ExtraTreesClassifier(random_state=0)



ex_param_grid = {'max_depth'        :[4],

                 'max_features'     :[7],

                 'min_samples_split':[8],

                 'min_samples_leaf' :[4],

                 'bootstrap'        :[False],

                 'n_estimators'     :[100],

                 'criterion'        :['gini']}



gsExtC = GridSearchCV(ExtC, param_grid=ex_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)

gsExtC.fit(X_train, Y_train)



ExtC_best = gsExtC.best_estimator_



gsExtC.best_score_
RFC = RandomForestClassifier(random_state=0)



rf_param_grid = {'max_depth'        :[6],

                 'max_features'     :[10],

                 'min_samples_split':[10],

                 'bootstrap'        :[False],

                 'n_estimators'     :[100],

                 'criterion'        :['gini']}



gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)

gsRFC.fit(X_train, Y_train)

RFC_best = gsRFC.best_estimator_



gsRFC.best_score_
GBC = GradientBoostingClassifier(random_state=0)



gb_param_grid = {'loss'            :['deviance'],

                 'n_estimators'    :[10],

                 'learning_rate'   :[0.001, 0.01, 0.1, 1, 10, 100],

                 'max_depth'       :[4],

                 'min_samples_leaf':[8],

                 'max_features'    :[0.3, 0.1]

                }



gsGBC = GridSearchCV( GBC, param_grid=gb_param_grid, cv=kfold, scoring='accuracy',n_jobs=4, verbose=1 )



gsGBC.fit(X_train, Y_train)

GBC_best = gsGBC.best_estimator_



gsGBC.best_score_
SVMC = SVC(probability=True)

svc_param_grid = {'kernel':['rbf'],

                  'gamma' :[0.01,0.1,1,10,100],

                  'C'     :[0.01,0.1,1,10,100]}



gsSVMC = GridSearchCV(SVMC, param_grid=svc_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)



gsSVMC.fit(X_train, Y_train)

SVMC_best = gsSVMC.best_estimator_



gsSVMC.best_score_
parameters = {'solver':['adam'], 'max_iter':[800], 'alpha':10.0**-np.arange(1,10), 'hidden_layer_sizes':[40,20,10], 'random_state':[0,1] }

gsMLPC=GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)



gsMLPC.fit(X_train, Y_train)

MLPC_best = gsMLPC.best_estimator_



gsMLPC.best_score_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel('Training examples')

    plt.ylabel('Score')

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean( train_scores, axis=1 )

    train_scores_std  = np.std(  train_scores, axis=1 )

    test_scores_mean  = np.mean( test_scores, axis=1  )

    test_scores_std   = np.std(  test_scores, axis=1  )

    

    plt.fill_between( train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha=0.1, color='r' )

    plt.fill_between( train_sizes, test_scores_mean-train_scores_std,  test_scores_mean+train_scores_std,  alpha=0.1, color='g' )

    

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')

    plt.plot(train_sizes, test_scores_mean,  'o-', color='g', label='Cross-validation score')

    plt.legend(loc='best')

    return plt
g1 = plot_learning_curve( gsGBC.best_estimator_,    'Gradient Boosting Learning Curve', X_train, Y_train, cv=kfold )

g2 = plot_learning_curve( gsExtC.best_estimator_,   'ExtraTrees learning curves',       X_train, Y_train, cv=kfold )

g3 = plot_learning_curve( gsSVMC.best_estimator_,   'SVC learning curves',              X_train, Y_train, cv=kfold )

g4 = plot_learning_curve( gsadaDTC.best_estimator_, 'AdaBoost learning curves',         X_train, Y_train, cv=kfold )

g5 = plot_learning_curve( gsRFC.best_estimator_,    'RF learning curves',               X_train, Y_train, cv=kfold )

g6 = plot_learning_curve( gsMLPC.best_estimator_,   'MLP learning curves',              X_train, Y_train, cv=kfold )

g6 = plot_learning_curve( gsXGB.best_estimator_,    'XGB learning curves',              X_train, Y_train, cv=kfold )
nrows = 2

ncols = 2

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', figsize=(15,15))



names_classifiers = [('AdaBoosting', ada_best),('ExtraTrees',ExtC_best),('RandomForest',RFC_best),

                     ('GradientBoosting',GBC_best)]



nclassifier = 0

for row in range(nrows):

    for col in range(ncols):

        name = names_classifiers[nclassifier][0]

        classifier = names_classifiers[nclassifier][1]

        indices = np.argsort(classifier.feature_importances_)[::-1][:40]

        g = sns.barplot(y=X_train.columns[indices][:40],x=classifier.feature_importances_[indices][:40], orient='h', ax=axes[row][col])

        g.set_xlabel('Relative importance', fontsize=12)

        g.set_ylabel('Features', fontsize=12)

        g.tick_params(labelsize=9)

        g.set_title(name+' feature importance')

        nclassifier +=1

        

plt.subplots_adjust(wspace=0.4, hspace=0.2)
test_Survived_RFC  = pd.Series( RFC_best.predict(test), name='RFC' )

test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name='ExtC')

test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name='SVC' )

test_Survived_AdaC = pd.Series( ada_best.predict(test), name='Ada' )

test_Survived_GBC  = pd.Series( GBC_best.predict(test), name='GBC' )

test_Survived_MLP  = pd.Series(MLPC_best.predict(test), name='MLP' )

test_Survived_XGB  = pd.Series(XGBC_best.predict(test), name='XGB' )



ensemble_results = pd.concat( [test_Survived_RFC, test_Survived_ExtC, test_Survived_SVMC, 

                               test_Survived_AdaC, test_Survived_GBC, test_Survived_MLP, test_Survived_XGB], axis=1 )

g = sns.heatmap(ensemble_results.corr(), annot=True, annot_kws={'size':12})
# votingC = VotingClassifier(estimators = [('rfc',RFC_best),('extc',ExtC_best),('svc',SVMC_best),('adac',ada_best),('gbc',GBC_best),('mlp',MLPC_best)],voting='soft',n_jobs=4)

# print(votingC)

# votingC = votingC.fit(X_train, Y_train)
# test_Survived = pd.Series(votingC.predict(test), name='Survived')



# results = pd.concat([IDtest, test_Survived],axis=1)



# results.to_csv('51th_submission.csv', index=False)
test_Survived = gsExtC.predict(test)



submission = pd.DataFrame({

    'PassengerId' : IDtest,

    'Survived' : test_Survived

})



submission.to_csv('ExtC_55th.csv', index=False)
test_Survived = gsXGB.predict(test)



submission = pd.DataFrame({

    'PassengerId' : IDtest,

    'Survived' : test_Survived

})



submission.to_csv('XGB_55th.csv', index=False)
test_Survived = gsGBC.predict(test)



submission = pd.DataFrame({

    'PassengerId' : IDtest,

    'Survived' : test_Survived

})



submission.to_csv('GBC_55th.csv', index=False)
test_Survived = gsSVMC.predict(test)



submission = pd.DataFrame({

    'PassengerId' : IDtest,

    'Survived' : test_Survived

})



submission.to_csv('SVMC_55th.csv', index=False)
# test_Survived = mlp.predict(test)



# submission = pd.DataFrame({

#     'PassengerId' : IDtest,

#     'Survived' : test_Survived

# })



# submission.to_csv('my_NN_submission.csv', index=False)