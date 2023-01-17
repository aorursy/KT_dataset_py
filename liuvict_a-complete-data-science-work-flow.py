# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pylab import rcParams



# Configure visualisations

get_ipython().magic(u'matplotlib inline')

mpl.style.use( 'ggplot' )

sns.set_style( 'white' )

pylab.rcParams[ 'figure.figsize' ] = 8 , 6

from pylab import rcParams



# Data Science Models

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn import tree

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
train = pd.read_csv('../input/train.csv', header=0, dtype={'Age': np.float64})

test = pd.read_csv('../input/test.csv', header=0, dtype={'Age': np.float64})

fulldata = [train, test]
class MultiColumnLabelEncoder:

    def __init__(self,columns = None):

        self.columns = columns # array of column names to encode



    def fit(self,X,y=None):

        return self # not relevant here



    def transform(self,X):

        '''

        Transforms columns of X specified in self.columns using

        LabelEncoder(). If no columns specified, transforms all

        columns in X.

        '''

        output = X.copy()

        if self.columns is not None:

            for col in self.columns:

                output[col] = LabelEncoder().fit_transform(output[col])

        else:

            for colname,col in output.iteritems():

                output[colname] = LabelEncoder().fit_transform(col)

        return output



    def fit_transform(self,X,y=None):

        return self.fit(X,y).transform(X)



def plt_pearson_corr_feature(data):

    colormap = plt.cm.viridis

    plt.figure(figsize=(12,12))

    plt.title('Pearson Correlation of Features', y=1.05, size=15)

    sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

    plt.yticks(rotation=0)

    plt.xticks(rotation=90)

    plt.show()



def factorplot(x, y, hue, data, axi):

    ax = fig.add_subplot(axi)

    g=sns.factorplot(x=x, y=y, hue=hue, data=data, kind="bar", size=4, aspect=2, ax=ax)

    plt.close(g.fig) 



def surv_rate_size_chart(data, col):

    dum_ind = data.sort_values([col])[col].unique()

    s = pd.Series(0, index=dum_ind)

    groupby_Mean = data.groupby([col]).mean()[['Survived']]

    groupby_Sum = (data.groupby(['Survived',col]).sum()).loc[1]['nCount']

    x = dum_ind

    y = groupby_Mean.values

    s = (groupby_Sum+s)*6

    plt.scatter(x, y, s, alpha=0.4, edgecolors='black', linewidths=2)

    plt.xlabel(col)

    plt.ylabel('Survived(Mean)')



def feat_corr_target(data, feature, target, ncolumn):

    

    dum_ind = range(len(df[feature].unique()))

    s = pd.Series(0, index=dum_ind)

    groupby = data.groupby([target, feature]).sum()[ncolumn].dropna()

    not_survived = pd.Series(groupby.loc[0].values) + s

    survived = pd.Series(groupby.loc[1].values) + s

    

    # positions of the left bar-boundaries

    bar_l = [i+1 for i in range(len(data[feature].dropna().unique()))]

    bar_width = 0.3

    # positions of the x-axis ticks (center of the bars as bar labels)

    tick_pos = [i for i in bar_l]



    p1 = plt.bar(bar_l, survived, color='g', width=bar_width, label='Survived')

    p2 = plt.bar(pd.Series(bar_l).map(lambda x: x+bar_width), not_survived, color='r', width=bar_width, label='Not Survived')



    plt.xticks(tick_pos, sorted(data[feature].dropna().unique()))

    plt.ylabel('Counts')

    plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width*2])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
print ('Columns: ')

print (train.columns.values)

print ('-'*20)

print ('shape: ', train.shape)
print (train.info())

print ('*'*20)

print (test.info())
#Categorical Features

train.describe(include=['O'])
train.describe()
# #### Observation (1 - Age):

# Age has the minimum value of 0.42, which indicates some data errors in this column

train['Age'].unique()
# #### Observation (1 - Age):

# There are some unusual values: 0.83, 0.67, 0.42, 0.92 and nan. These values could be 

# miss-typing or missing value

train[train['Age'] < 1]
# #### Obervation (2 - Ticket/Fare)

# Tickets are the same for all members of a family. And the Fare is also the total amount 

# of all the tickets.

train[train['Ticket'] == '2651']
# Or a group of people who bought the tickets together

train[train['Ticket'] == '1601']
# #### Observation (3 - Cabin)

# Some values of the Cabins show all the Cabins accommodated the family or group. And the 

#first letter of Cabin indicates the Deck Level.

train[train['Ticket'] == '113760']
# #### Obervation (4 - Unique Values)

fields = pd.Series(['Sex', 'SibSp', 'Parch', 'Embarked'])

def show_unique_value(column):

    return column, train[column].unique()

print (fields.apply(show_unique_value))
# ### 3.1 Age

# 1. If Age is less than 1, convert it to 1

# 2. If Age is NaN, fill with mean value of column 'Age'

# 3. Create a new column 'Age Bin', which categorizes the Age into Age Group

for dataset in fulldata:

#If Age is less than 1, convert to 1

    tmp_index = dataset[dataset['Age'] < 1].index

    dataset.loc[tmp_index, 'Age'] = 1



    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

    #if Age is NaN, fill with mean value of column 'Age'

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

    #create Age Bin Column

    def age_bin(age):

        if age >0 and age <= 10:

            return '0-10'

        elif age >10 and age <=20:

            return '11-20'

        elif age >20 and age <=30:

            return '21-30'

        elif age >30 and age <=40:

            return '31-40'

        elif age >40 and age <=50:

            return '41-50'

        elif age >50 and age <=60:

            return '51-60'

        elif age >60:

            return '>60'

    dataset['Age_bin'] = dataset['Age'].apply(age_bin)
# ### 3.2 Count

# 1. Create a new numerical value column 'nCount' which will be used for summation calculation

for dataset in fulldata:

    dataset['nCount'] = 1
# ### 3.3 Ticket Fare

# Tickets can be bought together as a group (family). The fare column is the total price the 

# group paid, rather than the individual ticket fare. The single ticket fare needs to be evenly 

# divided by the purchase group. Then the single fare will be categorized into 4 pricing 

# groups

train['Fare'] = train['Fare'].fillna(0)

Fare_gp = train.groupby(['Ticket']).sum()[['Fare', 'nCount']]

Fare_gp['Single_Fare'] = Fare_gp.Fare / (Fare_gp.nCount**2)

Fare_gp = Fare_gp.reset_index()[['Ticket', 'Single_Fare']]

train = train.merge(Fare_gp, how='left', on='Ticket')

train['CategoricalFare'] =  pd.qcut(train['Single_Fare'], 4, labels=[0,1,2,3]).astype(int)
test['Fare'] = test['Fare'].fillna(0)

Fare_gp = test.groupby(['Ticket']).sum()[['Fare', 'nCount']]

Fare_gp['Single_Fare'] = Fare_gp.Fare / (Fare_gp.nCount**2)

Fare_gp = Fare_gp.reset_index()[['Ticket', 'Single_Fare']]

test = test.merge(Fare_gp, how='left', on='Ticket')

test['CategoricalFare'] =  pd.qcut(test['Single_Fare'], 4, labels=[0,1,2,3]).astype(int)
# ### 3.3 Cabin

# ** There are many missing values in Cabin column. The observation from this field won't 

# represent the whole dataset.

# Extract the first letter of Cabin and form a new column 'Deck', and fill the missing 

# values with letter 'U'.

train['Deck'] = train['Cabin'].str.get(0)

train['Deck'] = train['Deck'].fillna('U')

test['Deck'] = test['Cabin'].str.get(0)

test['Deck'] = test['Deck'].fillna('U')
# ### 3.4 Embarked

# Fill the two missing values with the most occurred value 'S'

train["Embarked"] = train["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].fillna("S")
# ### 3.5 SibSp and Parch

# SibSp' and 'Parch' don't provide enough interesting insight to the dataset, 

# but together it can represent the size of the family. If the total of these 

# two columns is zero, it means the passenger was alone.

# 1. Add SibSp and Parch together and create a new Column 'Family_Size'

# 2. Create a new column 'Family'

# 3. If Family_Size > 0, assign 1 to 'Family' and If total = 0, assign 0 to 'Family' 

# 0: Alone; 1: Family

train['Family_Size'] =  train['Parch'] + train['SibSp'] + 1

train['IsAlone'] = 0

train['IsAlone'].loc[train['Family_Size'] > 1] = 1

train['IsAlone'].loc[train['Family_Size'] == 1] = 0

test['Family_Size'] =  test['Parch'] + test['SibSp'] + 1

test['IsAlone'] = 0

test['IsAlone'].loc[test['Family_Size'] > 1] = 1

test['IsAlone'].loc[test['Family_Size'] == 1] = 0
# ### 3.6 Remove Not Used Columns

train = train.drop(['Name','Ticket', 'Fare','Cabin', 'PassengerId', 'Age', 'Parch', 'SibSp', 'Single_Fare'], axis=1)

test = test.drop(['Name','Ticket', 'Fare','Cabin', 'PassengerId', 'Age', 'Parch', 'SibSp', 'Single_Fare'], axis=1)
# ### 3.7 Convert Categorical Features to Numeric Values

train = MultiColumnLabelEncoder(columns = ['Sex', 'Embarked', 'Age_bin', 

                                   'Deck']).fit_transform(train)

test = MultiColumnLabelEncoder(columns = ['Sex', 'Embarked', 'Age_bin', 

                                   'Deck']).fit_transform(test)
# ### 3.8 Data After Wrangling

train.info()
test.info()
rcParams['figure.figsize'] = 13,8

train_1 = train.drop(['nCount'], axis=1)

train_1.plot(kind='density', subplots=True, layout=(4,3), sharex=False)

plt.show()
rcParams['figure.figsize'] = 10,10

fig = plt.figure()



i = [421, 422, 423, 424, 425, 426, 427, 428]

ind = 0

for feature in train.columns.values[1:]:

    if feature in ('nCount', 'Survived', 'Pclass', 'Single_Fare'):

        continue

    else:

        factorplot(feature, 'Survived', 'Pclass', train, i[ind])

        plt.title('Survived Rate and Size by ' + feature, size=12)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8, title='Pclass')

        ind +=1  

plt.tight_layout()

plt.subplots_adjust(wspace = 0.35) 

plt.show()
rcParams['figure.figsize'] = 10,10

fig = plt.figure()



i = [421, 422, 423, 424, 425, 426, 427, 428]

ind = 0

for feature in train.columns.values[1:]:

    if feature in ('nCount', 'Survived', 'Sex'):

        continue

    else:

        factorplot(feature, 'Survived', 'Sex', train, i[ind])

        plt.title('Survived Rate and Size by ' + feature, size=12)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=8, title='Sex')

        ind +=1   

plt.tight_layout()

plt.subplots_adjust(wspace = 0.35) 

plt.show()
train_1=train.drop(['nCount'], axis=1)

plt_pearson_corr_feature(train_1)
train = train.drop(['Deck', 'nCount', 'CategoricalFare'], axis=1)

test = test.drop(['Deck', 'nCount', 'CategoricalFare'], axis=1)

feature_train = train.iloc[:,1:]

label_train = train['Survived']

feature_test = test
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier



X = train.iloc[:, 1:]

y = train.iloc[:,:1]

names = X.columns.values
# Class to extend the Sklearn classifier

SEED = 0 # for reproducibility

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

#        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        return self.clf.fit(x,y).feature_importances_

    

    def score(self, x, y):

        return self.clf.score(x, y)



# Put in our parameters for said classifiers # Random Forest parameters

rf_params = {

              'n_jobs': -1,

              'n_estimators': 500,

              'warm_start': True,

               #'max_features': 0.2,

              'max_depth': 6,

              'min_samples_leaf': 2,

              'max_features' : 'sqrt',

              'verbose': 0

}

et_params = {

              'n_jobs': -1,

              'n_estimators':500,

              #'max_features': 0.5,

              'max_depth': 8,

              'min_samples_leaf': 2,

              'verbose': 0

}

ada_params = {

              'n_estimators': 500,

              'learning_rate' : 0.75

}

gb_params = {

              'n_estimators': 500,

               #'max_features': 0.2,

              'max_depth': 5,

              'min_samples_leaf': 2,

              'verbose': 0

}

dtc_params = {

   

}
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)

ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)

gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)

dtc = SklearnHelper(clf=DecisionTreeClassifier, seed=SEED, params=dtc_params)



rf_feature = rf.fit(feature_train, label_train).feature_importances_

et_feature = et.fit(feature_train, label_train).feature_importances_

ada_feature = ada.fit(feature_train, label_train).feature_importances_

gb_feature = gb.fit(feature_train, label_train).feature_importances_

dtc_feature = dtc.fit(feature_train, label_train).feature_importances_



columns = pd.Series(feature_train.columns.values)



rf_feature_df = pd.concat([columns, pd.Series(rf_feature)], axis=1)

et_feature_df = pd.concat([columns, pd.Series(et_feature)], axis=1)

ada_feature_df = pd.concat([columns, pd.Series(ada_feature)], axis=1)

gb_feature_df = pd.concat([columns, pd.Series(gb_feature)], axis=1)

dtc_feature_df = pd.concat([columns, pd.Series(dtc_feature)], axis=1)



rcParams['figure.figsize'] = 10, 10

def feature_importance_chart(feature_importance, model):

    plt.title("Feature importances - " + model)

    plt.bar(range(len(feature_importance[0])), feature_importance[1], color="g", align="center")

    plt.xticks(range(len(feature_importance[0])), feature_importance[0])

plt.figure()

plt.subplot(321)

feature_importance_chart(rf_feature_df, "Random Forest")

plt.subplot(322)

feature_importance_chart(et_feature_df, "Extrac Trees") 

plt.subplot(323)

feature_importance_chart(ada_feature_df, "AdaBoosting")

plt.subplot(324)

feature_importance_chart(gb_feature_df, "GradientBoosting")

plt.subplot(325)

feature_importance_chart(dtc_feature_df, "Decision Trees")

plt.tight_layout()

plt.show()
classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel='linear', C=0.025),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=500, warm_start='True', max_depth=6, min_samples_leaf=2, max_features='sqrt'),

    AdaBoostClassifier(n_estimators=500, learning_rate=0.75),

    GradientBoostingClassifier(n_estimators=500, max_depth=5, min_samples_leaf=2, verbose=0),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()

]

log_cols = ["Classifier", "Accuracy"]

log  = pd.DataFrame(columns=log_cols)



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train.iloc[:,1:]

y = train['Survived']



acc_dict = {}

for train_index, test_index in sss.split(X, y):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y.loc[train_index], y.loc[test_index]



    for clf in classifiers:

        name = clf.__class__.__name__

        clf.fit(X_train, y_train)

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



rcParams['figure.figsize'] = 10,5

plt.xlabel('Accuracy', size=12)

plt.ylabel('Classifier', size=12)

plt.title('Classifier Accuracy')

plt.tick_params(axis='both', which='major', labelsize=12)

sns.set_color_codes("muted")



log = log.sort_values(['Accuracy'], ascending=False).reset_index(drop=True)

ax = sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

for p in ax.patches:

    ax.annotate(

        s='{0:.1f}%'.format(round(p.get_width(), 3)*100),

        xy=(p.get_x()+p.get_width(),p.get_y()),

        ha='center',va='center',

        xytext= (20,-10),

        textcoords='offset points',

        size = 14)
feature_test = feature_test.drop('PassengerId', axis=1)
final_clf = RandomForestClassifier(n_estimators=500, warm_start='True', max_depth=6, min_samples_leaf=2, max_features='sqrt')

final_clf.fit(feature_train, label_train)

y_pred = final_clf.predict(feature_test)

org_test = pd.read_csv('../input/test.csv', header=0, dtype={'Age': np.float64})['PassengerId']

test['PassengerId'] = org_test

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Predicted Survived": y_pred

    })

#submission.to_csv('../output/submission.csv', index=False)