%%capture 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pathlib import Path #flexible path files

import matplotlib.pyplot as plt #plotting

import seaborn as sns

import missingno as msno #library for missing values visualization

import warnings #ignoring warnings

warnings.filterwarnings('ignore')

import os



%matplotlib inline
# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = Path('/kaggle/input/titanic')

trpath = path/'train.csv'

cvpath = path/'test.csv'



df_train_raw = pd.read_csv(trpath)

df_test_raw = pd.read_csv(cvpath)



df_train = df_train_raw.copy(deep = True)

df_test  = df_test_raw.copy(deep = True)



data_cleaner = [df_train_raw, df_test_raw] #to clean both simultaneously
df_train.head(n=10)
df_train.info()
varnames = list(df_train.columns)

for name in varnames:

    print(name+": ",type(df_train.loc[1,name]))
print("Training Set")

print(df_train.isnull().sum(axis=0))

print("Test Set")

print(df_test.isnull().sum(axis=0))
msno.matrix(df_train)
msno.bar(df_test)
print('Overall survival quota:')

df_train['Survived'].value_counts(normalize = True)
plt.style.use('seaborn')
fig = df_train.groupby('Survived')['Age'].plot.hist(histtype= 'bar',

                                                    alpha = 0.7,

                                                    fontsize = 14,

                                                    figsize = [10,10])

plt.legend(('Died','Survived'), fontsize = 13)

plt.xlabel('Age', fontsize = 18)

plt.ylabel('Count', fontsize = 18)

plt.suptitle('Histogram of the ages of survivors and decased ones',fontsize =22)

plt.show()
df_train['Family onboard'] = df_train['Parch'] + df_train['SibSp']

plt.rcParams['figure.figsize'] = [20, 7]



fig, axes = plt.subplots(nrows=1, ncols=3)

axes[0].set_ylabel('Survival rate',fontsize = 18)

axes = iter(axes.flatten())



titles = iter(['Family onboard','parents / children aboard','siblings / spouses aboard'])

family_vars = ['Family onboard','Parch','SibSp']





for var in family_vars:

    ax = next(axes)

    df_train.groupby(var)['Survived'].value_counts(normalize = True).unstack().plot.bar(ax = ax, width = 0.85, fontsize = 14)

    ax.set_xlabel(next(titles),fontsize = 18)

    ax.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

    plt.sca(ax)

    plt.xticks(rotation=0)



plt.suptitle('Survival rates over Number of relatives onboard',fontsize =22)

plt.show()
fig = df_train.groupby(['Sex'])['Survived'].value_counts(normalize=True).unstack().plot.bar(figsize = [8,5], width = 0.5,fontsize = 14)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Gender',fontsize =18)

plt.xticks(rotation=0)

plt.ylabel('Survival rate',fontsize = 18)



plt.suptitle('Survival rates over Gender',fontsize =22)

plt.show()
fig = df_train.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack().plot.bar(figsize = [8,5], width = 0.5,fontsize = 14)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Ticket Class',fontsize =18)

plt.ylabel('Survival rate',fontsize = 18)

plt.suptitle('Survival rate over Ticket class', fontsize = 22)

plt.xticks(rotation=0)

plt.show()
df_train['Title'] = df_train['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()

varnames = list(df_train.columns)

    

print("Training set: " ,list(df_train['Title'].unique()))    

df_test['Title'] = df_test['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()

print("Test set: " ,list(df_test['Title'].unique()))    

def new_titles(df):

    new_titles = dict()

    assert 'Title' in df.columns

    for key in df['Title'].unique():

        females = ['Mrs','Miss','Ms','Mlle','Mme','Dona']

        males = ['Mr','Don']

        notable = ['Jonkheer','the Countess','Lady','Sir','Major','Col','Capt','Dr','Rev','Notable']

        titles = [females,males,notable,'Master']

        newtitles = ['Mrs','Mr','Notable','Master']

        idx = [key in sublist for sublist in titles]

        idx = np.where(idx)[0] 

        new_titles[key] = newtitles[idx[0]]

    return new_titles





new_titles_dict = new_titles(df_train)

df_train['Title'] = df_train['Title'].replace(new_titles_dict)
fig = df_train.groupby(['Title'])['Survived'].value_counts(normalize=True).unstack().plot.bar(figsize = [12,5], width = 0.7,fontsize = 14)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Title',fontsize =16)

plt.xticks(rotation=0)



plt.suptitle('Survival rates over Title',fontsize =22)

plt.show()
df_train['Cabin'].fillna('Missing',inplace = True)

df_train['Cabin'] = df_train['Cabin'].str.split(r'(^[A-Z])',expand = True)[1]
fig = df_train.groupby(['Cabin'])['Survived'].value_counts(normalize=True).unstack().plot.bar(figsize = [12,5], width = 0.9)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Cabin Deck',fontsize =18)

plt.suptitle('Survival rates over Cabin Deck',fontsize =22)

plt.xticks(rotation=0)

plt.show()
fig = df_train.groupby(['Embarked'])['Survived'].value_counts(normalize=True).unstack().plot.bar(figsize = [10,5], width = 0.7)

plt.legend(('Died','Survived'),fontsize = 13, loc = 'upper left')

plt.xlabel('Embarking Port',fontsize =18)

plt.suptitle('Survival rates over embarking port',fontsize =22)

plt.xticks(rotation=0)

plt.show()
df_train.groupby(['Embarked'])['Pclass'].value_counts(normalize=True).unstack()
df_train.corr(method='pearson')['Age'].abs()
def df_fill(datasets, mode):

    assert mode =='median' or mode =='sampling'

    datasets_cp =[]

    np.random.seed(2)

    varnames = ['Age','Fare']

    for d in datasets:

        df = d.copy(deep = True)

        for var in varnames:

            idx = df[var].isnull()

            if idx.sum()>0:

                if mode =='median':

                    medians = df.groupby('Pclass')[var].median()

                    for i,v in enumerate(idx):

                        if v:

                            df[var][i] = medians[df['Pclass'][i]]

                else:

                    g = df[idx==False].groupby('Pclass')[var]

                    for i,v in enumerate(idx):

                        if v:

                            df[var][i] = np.random.choice((g.get_group(df['Pclass'][i])).values.flatten())

    #Embarked                 

        idx = df['Embarked'].isnull()

        g = df[idx==False].groupby('Pclass')['Embarked']

        for i,v in enumerate(idx):

            if v:

                df['Embarked'][i] = np.random.choice((g.get_group(df['Pclass'][i])).values.flatten())                   

    #Cabin

        df['Cabin'][df['Cabin'].isnull()]='Missing'

        df['Cabin'] = df['Cabin'].str.split(r'(^[A-Z])',expand = True)[1]

        datasets_cp.append(df)

    return datasets_cp
def prepare_data(datasets):

        datasets_cp = []

        for d in datasets:

            df = d.copy(deep = True)

            df['Family onboard'] = df['Parch'] + df['SibSp']

            df['Title'] = df['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0].str.strip()

            new_titles_dict = new_titles(df)

            df['Title'] = df['Title'].replace(new_titles_dict)

            df.drop(columns = ['PassengerId','Name','Ticket'],axis = 1, inplace = True)

            

            datasets_cp.append(df)

        return datasets_cp      
train,test =prepare_data(df_fill(data_cleaner,mode = 'sampling'))  

print("Training data:")

print(train.isnull().sum())

print("Test data:")

print(test.isnull().sum())
ytrain = train['Survived']

xtrain = train.drop('Survived',axis = 1)

xtest = test
data = pd.concat([xtrain,xtest],copy =True)

sex_mapping = {'male'  : 0,

               'female': 1

              }



data = pd.get_dummies(data,columns = ['Title', 'Cabin', 'Embarked'],drop_first = True)

data['Sex'] = data['Sex'].map(sex_mapping)



m = xtrain.shape[0]

x_train = data[:m].astype('float64')

x_test = data[m:].astype('float64')

y_train = ytrain.astype('int64')
def normalize(df,cols,mu,sigma):

    df[cols] = (df[cols]-mu)/sigma

    return df
x_train_mean, x_train_std = x_train[['Age','Fare']].mean(), x_train[['Age','Fare']].std()

x_train = normalize(x_train,['Age','Fare'],x_train_mean, x_train_std)

x_test = normalize(x_test,['Age','Fare'],x_train_mean, x_train_std)
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import RandomizedSearchCV

from sklearn.base import BaseEstimator, clone

from sklearn.pipeline import make_pipeline

import xgboost as xgb
n_folds = 5

def acc_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(xtrain)

    acc= cross_val_score(model, x_train, y_train, scoring="accuracy", cv = kf)

    return(acc)
Adaboost = make_pipeline(RobustScaler(),

                         AdaBoostClassifier(base_estimator=None,

                                            n_estimators = 56,

                                            learning_rate= 0.18,

                                            algorithm='SAMME.R',

                                            random_state = 1)

                        )
GBoosting = make_pipeline(RobustScaler(), 

                          GradientBoostingClassifier(loss='deviance',

                                                     learning_rate = 0.05,

                                                     n_estimators = 56,

                                                     min_samples_split = 9,

                                                     min_samples_leaf = 2,

                                                     max_depth = 4,

                                                     random_state = 1,

                                                     max_features = 9)

                         )
SVC =  make_pipeline(RobustScaler(), 

                     SVC(decision_function_shape = 'ovr',

                         random_state = 1,

                         max_iter = 14888,

                         kernel = 'poly',

                         degree = 2,

                         coef0 = 0.49, 

                         C =  9.6)

                     )
RF = make_pipeline(RobustScaler(), 

                   RandomForestClassifier(criterion='gini', 

                                          n_estimators=364,

                                          max_depth = 11,                    

                                          min_samples_split=6,

                                          min_samples_leaf=1,

                                          max_features='auto',

                                          oob_score=True,

                                          random_state=1,

                                          )

                  )
xgbc = make_pipeline(RobustScaler(), 

                     xgb.XGBClassifier(n_estimators=121,

                                       reg_lambda = 0.9,

                                       reg_alpha = 0.5,

                                       max_depth = 9,

                                       learning_rate = 0.55,

                                       gamma = 0.5,

                                       colsample_bytree = 0.4,

                                       coldsample_bynode = 0.15,

                                       colsample_bylevel = 0.5)

                    )
score = acc_cv(Adaboost)

print("Adaboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = acc_cv(GBoosting)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = acc_cv(SVC)

print("SVC  score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = acc_cv(RF)

print("Random Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = acc_cv(xgbc)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class AveragingModels(BaseEstimator):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
%%capture

averaged_models = AveragingModels(models = (Adaboost,SVC, GBoosting, RF,xgbc))

averaged_models.fit(x_train, y_train)
train_pred = averaged_models.predict(x_train)

test_pred = averaged_models.predict(x_test)
train_pred = np.round(train_pred)

test_pred = np.round(test_pred)



acc_averaged = np.round((train_pred==y_train).sum()/train_pred.shape[0],5)

print(f"Averaged models accuracy: {acc_averaged}")
submission = pd.DataFrame()

submission['PassengerId'] = df_test['PassengerId'].astype('int32')

submission['Survived'] = test_pred

submission['Survived'] = submission['Survived'].astype('int32')

submission.to_csv('submission.csv',index=False)