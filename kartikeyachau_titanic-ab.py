# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline



import re



import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)



import pandas as pd

pd.options.display.max_columns = 100



from matplotlib import pyplot as plt

import numpy as np



import seaborn as sns



import pylab as plot

params = { 

    'axes.labelsize': "large",

    'xtick.labelsize': 'x-large',

    'legend.fontsize': 20,

    'figure.dpi': 150,

    'figure.figsize': [25, 7]

}

plot.rcParams.update(params)
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from pydoc import help

from scipy.stats.stats import pearsonr
df = pd.read_csv('../input/train.csv')

df.head(5)
df.describe()
df.Age = df.Age.fillna(df.Age.median())

df.describe()
df['Died'] = 1 - df.Survived

df.describe()
'''Survival proportion for each gender'''

print(df.groupby('Sex').agg('sum')[['Survived', "Died"]])

print(df.groupby('Sex').agg('sum')[['Survived', "Died"]].plot(title = 'Deaths vs sex'

                                                              ,kind = 'bar', stacked = True, colors = ['turquoise', 'coral']))

print("We observe than a larger portion of males have died as compared to females, depicting that women were given priority during the rescue")
'''Analysing if the age affects the survival for males and females'''

fig = plt.figure(figsize= (30,7))

sns.set_style("darkgrid")

sns.set_context("poster")

#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.violinplot('Sex', 'Age', 'Survived', df, split = True, palette= {0: 'r', 1: 'g'}, title = 'The differing effect of age on survival')



print("WOMEN AND CHILDREN FIRST!! \n1: Males in the age of 20-40 are more susceptible to dying whereas teenagers have better chances of survival \n2: Females do not see a corelation between age and deaths")
'''The relation between fare and survival '''

fig = plt. figure(figsize= (25,7))

plt.style.use(['ggplot'])

plt.hist([df[df['Survived'] == 1]['Fare'], df[df['Survived'] == 0]['Fare']], stacked = True, color = ['g','r'], 

         bins = 50, label = ['Survived','Dead'])

plt.xlabel("Fare")

plt.ylabel('#')

plt.legend()



print("We notice that most of the passengers with fares")
'''Analysing the combined role of Age and fare on survival '''

plt.figure(figsize=(25, 7))

ax = plt.subplot()

plt.xlabel("Age")

plt.ylabel('Fare#')

plt.legend()



ax.scatter(df[df['Survived'] == 1]['Age'], df[df['Survived'] == 1]['Fare'], c = 'orange', s = df[df['Survived'] == 1]['Fare'])

ax.scatter(df[df['Survived'] == 0]['Age'], df[df['Survived'] == 0]['Fare'], c = 'black', s = df[df['Survived'] == 0]['Fare'])



print("1-Most of the black dots are between 0 to 100 dollars indicating that the rich passengers were prioritized while rescue")

print("2-There is a small bunch of green dots in the lower left corner indicating that the babies were given priority regardless of the fare")

print("3-None of the passengers from the $500 fare has died")

print("4-Elderly passengers with low fares also share the similar distributon of survival as others stipulating that elderly were not given special treatment while rescuing.Passengers near 80 years of age however were all rescued")
df.groupby(["Pclass"]).agg('mean')['Fare'].plot(title = "Mean fare of each Pclass")

print("The fares of Pclass 2 and 3 were much lower than Pclass 1")

print("This could mean that a larger proprtion of Pclass 1 has survived.")
'''Were Pclass 1 passengers a priority??'''

fig = plt. figure(figsize= (25,7))

plt.style.use(['ggplot'])

plt.hist([df[df['Survived'] == 1]['Pclass'], df[df['Survived'] == 0]['Pclass']], stacked = True, color = ['g','r'], 

         bins = 50, label = ['Survived','Dead'])

plt.xlabel("Pclass")

plt.ylabel('#')

plt.legend()

print("It seems that our hypothesis about Pclass 1 passengers being a priority was true...")
fig, axs = plt.subplots(nrows=2)

sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=df, split=True, palette={0: "r", 1: "g"}, ax=axs[0])



sns.violinplot(x='Embarked', y='Age', hue='Survived', data=df, split=True, palette={0: "r", 1: "g"}, ax=axs[1])
def status(feature):

    print("Processing", feature, ":ok")
def get_combined_data():

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    

    targets = train['Survived']

    #train.drop('Survived', 1, inplace = True)

    

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop(['PassengerId', 'index'], 1, inplace = True)

    

    return combined



combined = get_combined_data()
'''Extracting titles'''

titles = set()

for name in df['Name']:

    titles.add(name.split(',')[1].split('.')[0].strip())

    

print(titles)



Title_Dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty"

}

    
def get_titles():

    combined['Title'] = combined['Name'].map(lambda name : name.split(',')[1].split('.')[0].strip())

    combined['Title'] = combined['Title'].map(Title_Dictionary)

    

    status('Titles')

    return combined



combined = get_titles()

combined.head()

print(combined['Title'].unique())



print(combined[combined['Title'].isnull()])
print("the number of missing age values : ", combined.Age[:891].isnull().sum(),"\n")

print("Median ages for each class : \n\n")

gtrain = combined[:891].groupby(['Sex', 'Pclass', 'Title'])['Age'].median().reset_index()

print(gtrain.head())

print(type(gtrain))
def select_cat(row):

    cond = ((gtrain['Sex'] == row['Sex']) & (gtrain['Pclass'] == row['Pclass']) & (gtrain['Title'] == row['Title']))

    return gtrain[cond]['Age'].values[0]

temp= combined



def impute_age():

    global temp

    temp['Age'] = temp.apply(lambda row: select_cat(row) if np.isnan(row['Age']) else row['Age'], axis = 1)

    status('age')

    return temp



cond = ((gtrain['Sex'] == 'male' ) & (gtrain['Pclass'] == 3) & (gtrain['Title'] == 'Mr'))

cond

gtrain[cond]['Age'].values[0]
temp = impute_age()



print("null values in age now: ",temp['Age'].isna().sum())

temp.head()

combined = temp
combined.head()
combined.drop('Name', inplace=True, axis = 1)
combined.head()
temp = combined

def dum_title():

    global temp

    temp = pd.concat([temp, pd.get_dummies(temp['Title'], prefix = "Title")], axis = 1)

    temp.drop('Title', inplace = True, axis = 1)

    return temp



temp = dum_title()

temp.head()    
combined = temp

combined.columns
temp = combined



def impute_fare():

    global temp

    temp['Fare'].fillna(temp.iloc[:891]['Fare'].mean(), inplace = True)

    status("Fare")

    return temp



temp = impute_fare()



print(temp["Fare"].isna().sum())

temp.head()



combined = temp

combined.head()
temp = combined

def impute_embarked():

    global temp

    temp['Embarked'].fillna('S', inplace = True)

    status('Embarked')

    return temp



temp = impute_embarked()



def dum_embarked():

    global temp

    temp = pd.concat([temp, pd.get_dummies(temp['Embarked'], prefix = 'Emb')], axis = 1)

    status('dummies embarked')

    return temp



temp = dum_embarked()



temp.columns

combined = temp
combined.head()
temp = combined

def dum_cabin():

    global temp

    temp['Cabin'].fillna("U", inplace = True)

    temp['Cabin'] = temp['Cabin'].map(lambda x: x[0])

    status("imputed cabins")

    

    temp = pd.concat([temp, pd.get_dummies(temp['Cabin'], prefix = 'Cabin')], axis = 1)

    status('dummies for cabins')

    return temp



temp = dum_cabin()

print(temp.columns)

combined = temp

combined.head()
temp =combined

def dum_sex():

    global temp

    temp['Sex'] = temp['Sex'].map({'male':1, 'female':0})

    status('Sex dummies')

    return temp



temp = dum_sex()

combined = temp
def process_pclass():

    global combined

    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    combined = pd.concat([combined, pclass_dummies],axis=1)

    combined.drop('Pclass',axis=1,inplace=True)

    status('Pclass')

    return combined

combined = process_pclass()
combined.head()
combined.drop(['Ticket', 'Cabin', 'Embarked'], inplace = True, axis = 1)
pearsonr(train.Survived, train.Fare)
a,b,c = [],[],[]

p_cols = ['Feature', 'pear_cff', 'p_value']



def pr():

    for i in train.columns:

        if i == 'Survived':

            continue

        else:

            a.append(i)

            b.append(pearsonr(train.Survived, train[i])[0])

            c.append(pearsonr(train.Survived, train[i])[1])

    return a,b,c

pr()

pear_cf = pd.DataFrame({'Feature' : a, 'pear_cff' : b, 'p_value' : c }, columns = p_cols)

pear_cf

plt.plot(pear_cf.Feature, pear_cf.pear_cff, color = 'blue')

plt.plot(pear_cf.Feature, pear_cf.p_value, color = 'red')
def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)

    return np.mean(xval)
def recover_test_train():

    target = pd.read_csv('../input/train.csv', usecols = ['Survived'])['Survived'].values

    train = combined.iloc[:891]

    test = combined.iloc[891:]

    

    return test, train, target



test, train, targets = recover_test_train()
clf = RandomForestClassifier(n_estimators=50, max_features = 'sqrt')

clf = clf.fit(train, targets)
feature = pd.DataFrame()

feature['feature'] = train.columns

feature['importance'] = clf.feature_importances_

feature.sort_values(by = ['importance'], ascending = True, inplace = True)

feature.plot('feature', 'importance', kind = 'barh', figsize = (25,25), style  = 'seaborn-dark')
model = SelectFromModel(clf, prefit = True)

train_red = model.transform(train)

test_red = model.transform(test)

print(train_red.shape, test_red.shape, '\n', train_red)
logreg = LogisticRegression()

logreg_cv = LogisticRegressionCV()

rf = RandomForestClassifier()

gboost = GradientBoostingClassifier()



models = [logreg, logreg_cv, rf, gboost]
for model in models:

    print ('Cross-validation of : {0}'.format(model.__class__))

    score = compute_score(clf=model, X=train_red, y=targets, scoring='accuracy')

    print ('CV score = {0}'.format(score))

    print ('****')