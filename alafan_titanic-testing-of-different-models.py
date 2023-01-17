import pandas as pd

import numpy as np



from sklearn.model_selection import cross_val_score, KFold

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
from sklearn import svm

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB,MultinomialNB, ComplementNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier, VotingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
def scoring(clf, X, y):

    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(X)

    cvs = cross_val_score(clf, X, y, cv=kf, scoring="accuracy")

    return cvs
# Parameters fitting

def param_fit(model, known_params, param_name, values_list, X, y, silence = False):

    current_max = 0

    max_value = None

    for value in values_list:

        if known_params!='':

            model_string = model+'('+known_params+','+param_name + '=' + str(value) +')'

        else:

            model_string = model+'('+param_name + '=' + str(value) +')'

        clf = eval(model_string)

        #print('Scoring calculation starded')

        scr = scoring(

            clf,

            X,

            y)

        if silence == False:

            print('Value: {0}, scoring: {1}'.format(value,scr.mean()))

        if current_max<scr.mean():

            current_max = scr.mean()

            max_value = value

    #print(param_name,max_value)

    return max_value

def fit_different_models(models, X, y, print_threshold = 0):

    best_scoring = 0

    best_model = 0

    for model in models.keys():

        param_string = ''

        for param in models[model].keys():

            param_value = param_fit(model,param_string[:-1], param, models[model][param], X, y, silence = True)

            #print('param = {0}, param_value = {1}'.format(param, param_value))

            param_string+=param+'='+str(param_value)+','

            #print('param_string = {}'.format(param_string))

        model_string = model+'('+param_string+')'

        #print(model_string)

        clf = eval(model_string)

        scr = scoring(clf,X,y).min()

        if scr > print_threshold:

            print('Model: {}, accuracy = {:0.4f}'.format(clf,scr))

        if scr>best_scoring:

            best_scoring = scr

            best_model = clf

    print('\n\nBest scoring: {0:0.4f}\nBest model:\n{1}\n\n'.format(best_scoring, best_model))

    return best_model
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('Shape of train set: {0}\nShape of test set: {1}'.format(train.shape,test.shape))
train.head()
train.Survived.value_counts()
ax = sns.barplot(x = [0,1],y = train.groupby('Survived').Survived.count())
surv_share = len(train[train.Survived==1])*1.0/len(train)

print('Share of survived is {:4f}, share of lost is {:4f}'.format(surv_share, 1 - surv_share))
sns.pairplot(train[['Age','Fare','Pclass','SibSp','Parch']].dropna())
train[train.Fare>500]
train = train.drop(train[train.Fare>500].index)
train['Sex'] = train.Sex.apply(lambda x: 1 if x == 'male' else 0)

test['Sex'] = test.Sex.apply(lambda x: 1 if x == 'male' else 0)
sns.pairplot(train.dropna(),

            hue = 'Survived',

            x_vars=['Sex','Age','Fare','Pclass','Parch','SibSp'],

            y_vars=['Sex','Age','Fare','Pclass','Parch','SibSp'])
# Moving PassengerId in separate arrays

train_pass_id = train.PassengerId.values

test_pass_id = test.PassengerId.values
ntrain = train.shape[0]

ntest = test.shape[0]



y_train = train.Survived.values



all_data = pd.concat((train, test),sort=False).reset_index(drop=True)

all_data.drop(['Survived'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing share' :all_data_na})

missing_data
f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
#Removing feature

all_data = all_data.drop('Cabin',axis = 1)
sns.distplot(all_data[(pd.notna(all_data.Age))].Age)
#filling the NaN with median

all_data['Age'] = all_data['Age'].fillna(all_data.Age.median())
#let's see again at distribution

sns.distplot(all_data.Age)
embarked = all_data.groupby('Embarked')['PassengerId'].count()

sns.barplot(x = embarked.index, y = embarked)
all_data['Embarked'] = all_data['Embarked'].fillna(all_data.Embarked.mode()[0])
sns.distplot(all_data[pd.notna(all_data.Fare)].Fare)
all_data['Fare'] = all_data['Fare'].fillna(all_data.Fare.median())
all_data[['Name']].head(5)
all_data['LenName'] = all_data['Name'].apply(lambda x:len(x))
all_data['Status'] = all_data.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())
f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation='90')

sns.barplot(x = all_data.Status.unique(), y = all_data.Status.value_counts())
def status_update(x):

    if x['Status'] in all_data.Status.value_counts()[all_data.Status.value_counts()<10].index:

        if x['Sex']==1:

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return x['Status']

all_data['Status'] = all_data.apply(status_update, axis = 1)
f, ax = plt.subplots(figsize=(10, 6))

plt.xticks(rotation='90')

sns.barplot(x = all_data.Status.unique(), y = all_data.Status.value_counts())
all_data = all_data.drop('Name',axis = 1)
all_data.groupby('Ticket')['Fare'].agg(

    ['count','mean','min','max']).sort_values(

    by = 'count',ascending = False).head(10)
all_data = all_data.drop('Ticket',axis = 1)
all_data = pd.get_dummies(all_data,columns = ['Status', 'Embarked'])
all_data = all_data.drop('PassengerId', axis = 1)
all_data.head()
# New train and test:

train = all_data[:ntrain]

test = all_data[ntrain:]
corr = pd.concat([train, pd.Series(y_train)], axis = 1).rename(columns = {0:'Survived'}).corr()
# Generate a mask for the upper triangle

mask = np.zeros_like(np.abs(corr), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))





# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(np.abs(corr), mask=mask,  vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
np.round(np.abs(corr),2)
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

train_var = sel.fit_transform(all_data)[:ntrain]

test_var = sel.fit_transform(all_data)[ntrain:]
train.shape, train_var.shape
simple_models = {

    'svm.SVC':{'gamma':['\'auto\'','\'scale\''],

               'C':range(10,100,10),

               'kernel':['\'rbf\'','\'linear\''],

               

              },

    'GaussianNB':{},

    'BernoulliNB':{},

    'MultinomialNB':{},

    'ComplementNB':{},

    'DecisionTreeClassifier':{'random_state':[42],

                              'criterion':['\'gini\'','\'entropy\''],

                              'max_depth':[1,2,3,4,5,10,20],

                              'min_samples_leaf':[1,2,5,10]

                             },

    'KNeighborsClassifier':{'n_neighbors':range(1,100,2),

                            'metric':['\'manhattan\'','\'minkowski\'']},

                         

         }
%%time

clf = fit_different_models(simple_models, train_var, y_train, print_threshold = 0.7)
base_model = svm.SVC(C=5, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)



def predict(model, train, y_train, test,test_pass_id, fname):

    model.fit(train, y_train)

    pred = model.predict(test)

    res = pd.DataFrame()

    res['PassengerId'] = test_pass_id

    res['Survived'] = pred

    res['Survived'] = res.Survived.astype(int)

    res.to_csv(fname, index = False)

    
predict(clf, train, y_train, test, test_pass_id, 'submission.csv')