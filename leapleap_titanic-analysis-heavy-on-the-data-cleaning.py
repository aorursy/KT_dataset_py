import pandas as pd

from os import path

import matplotlib.pyplot as plt

import numpy as np

from random import random, randint



import seaborn as sea
# make pandas window less tiny

pd.set_option('display.max_columns', 20)

pd.set_option('expand_frame_repr', True)



# for my own sanity temporarily:

pd.options.mode.chained_assignment = None # None|'warn'|'raise'
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
def bar_graph_by(label, metric,  df, pal= sea.color_palette()):

    fig, axis1 = plt.subplots(1,1)

    dat = df[[label, metric]].groupby([label],as_index=False).mean()

    sea.barplot(x=label, y=metric, data=dat, palette = pal)

    axis1.set_title(metric + ' by ' + label)

    plt.show()

    

    

def multi_bar_count_by(x_val, y_val, hue_val, data):

    dat = data[[x_val, y_val, hue_val]].groupby((hue_val, x_val), as_index=False).count()

    g =sea.factorplot(x= x_val, y= y_val, hue= hue_val, data=dat,

                     kind="bar", palette="muted")

    g.fig.suptitle(x_val + ' counts by ' + hue_val)

    plt.show()



def multi_bar_sum_by(x_val, y_val, hue_val, data):

    dat = data[[x_val, y_val, hue_val]].groupby((hue_val, x_val), as_index=False).sum()

    g = sea.factorplot(x= x_val, y= y_val, hue= hue_val, data=dat,

                   kind="bar", palette="muted", size = 6)

    g.fig.suptitle(x_val + ' sum by ' + hue_val)

    plt.show()

    

def facet_by(label, metric, df, plt_type,  asp = 1, r_lab = None, c_lab= None, pal= sea.color_palette(), **kwargs):

    g = sea.FacetGrid(data = df, row= r_lab, col= c_lab, hue= metric, palette = pal)

    g.map(plt_type, label,  **kwargs)

    g.set(xlim=(0, df[label].max()))

    g.add_legend()

    plt.show()
train.describe()
test.describe()
train[train['Survived'] ==1].describe()
train.head()
train.drop('Ticket', axis=1, inplace=True)

test.drop('Ticket', axis=1, inplace=True)
train.isnull().sum()
train[train['Age'].isnull()][:20]
train[65:66]
males = train.loc[train['Sex'] == 'male']

t_males = test.loc[test['Sex'] == 'male']



masters = males.loc[males['Name'].str.contains("Master")]

t_masters = t_males.loc[t_males['Name'].str.contains("Master")]



allmasters= pd.concat((masters, t_masters), ignore_index =True)



print(allmasters['Age'].min(), allmasters['Age'].max(), allmasters['Age'].median())

allmasters.describe()
def get_age_range(df):

    m = df['Age'].median()

    st = df['Age'].std()

    return int(max((m - st), df['Age'].min())), int(min((m + st), df['Age'].max()))

    



def get_rand_age(x, age_range):

    if np.isnan(x):

        return float(randint(*age_range))

    else:

        return x

    
masters_range = get_age_range(allmasters)

null_masters = masters[masters['Age'].isnull()]

null_masters['Age'] = null_masters['Age'].apply(lambda x : get_rand_age(x, masters_range))

males.loc[null_masters.index] = null_masters



# and the test data

null_t_masters = t_masters[t_masters['Age'].isnull()]

null_t_masters['Age'] = null_t_masters['Age'].apply( lambda x : get_rand_age(x, masters_range))

t_males.loc[null_t_masters.index] = null_t_masters
train[65:66]
men = males[males['Age'] > 14.5].dropna()

t_men = t_males[t_males['Age'] > 14.5].dropna()

allmen = pd.concat((men, t_men), ignore_index=True )

allmen.describe()
m_range = get_age_range(allmen)



null_men = males[males['Age'].isnull()]

null_men['Age'] = null_men['Age'].apply(lambda x : get_rand_age(x, m_range))

males.loc[null_men.index] = null_men



#and test data

null_t_men = t_males[t_males['Age'].isnull()]

null_t_men['Age'] = null_t_men['Age'].apply(lambda x : get_rand_age(x, m_range))

t_males.loc[null_t_men.index] = null_t_men
males['Age'].isnull().sum() + t_males['Age'].isnull().sum()
train.loc[males.index] = males

test.loc[t_males.index] = t_males
women = train.loc[train['Sex'] == 'female']

t_women = test.loc[test['Sex'] == 'female']
unmarried_women = women.loc[women['Name'].str.contains("Miss")]

t_unmarried_women = t_women.loc[t_women['Name'].str.contains("Miss")]

all_unmarried_women =  pd.concat((unmarried_women, t_unmarried_women), ignore_index =True)

all_unmarried_women.describe()
bar_graph_by('Parch', 'Age',  all_unmarried_women)
all_unmarried_women['withparent'] = all_unmarried_women['Parch'] >= 1

without_parent = all_unmarried_women[~all_unmarried_women['withparent']]

without_parent_range = get_age_range(without_parent)

without_parent_range
withparent = all_unmarried_women[all_unmarried_women['withparent']]

with_parent_range = get_age_range(withparent)

with_parent_range
unmarried_women = unmarried_women[unmarried_women['Age'].isnull()]

wp_unmarried = unmarried_women[(unmarried_women['Parch'] >= 1)]

wp_unmarried['Age'] = wp_unmarried['Age'].apply(lambda x : get_rand_age(x, with_parent_range))

women.loc[wp_unmarried.index] = wp_unmarried



np_unmarried = unmarried_women[(unmarried_women['Parch'] < 1)]

np_unmarried['Age'] = np_unmarried['Age'].apply(lambda x : get_rand_age(x, without_parent_range))

women.loc[np_unmarried.index] = np_unmarried



t_unmarried_women = t_unmarried_women[t_unmarried_women['Age'].isnull()]

twp_unmarried = t_unmarried_women[(t_unmarried_women['Parch'] >= 1)]

twp_unmarried['Age'] = twp_unmarried['Age'].apply(lambda x : get_rand_age(x, with_parent_range))

t_women.loc[twp_unmarried.index] = twp_unmarried



tnp_unmarried = t_unmarried_women[~(t_unmarried_women['Parch'] >= 1)]

tnp_unmarried['Age'] = tnp_unmarried['Age'].apply(lambda x : get_rand_age(x, without_parent_range))

t_women.loc[tnp_unmarried.index] = tnp_unmarried
married_women = women.loc[women['Name'].str.contains("Mrs")]

t_married_women = t_women.loc[t_women['Name'].str.contains("Mrs")]

all_married_women = pd.concat((married_women, t_married_women), ignore_index =True)

all_married_women.describe()
all_married_women[all_married_women['Age'] < 18.]
mar_age = all_married_women[all_married_women['Age'] >= 18.]

mw_age_range = get_age_range(mar_age)

mw_age_range
mw_train = women[women['Age'].isnull()]

mw_train['Age'] = mw_train['Age'].apply(lambda x : get_rand_age(x, mw_age_range))

women.loc[mw_train.index] = mw_train



mw_test =  t_women[t_women['Age'].isnull()]

mw_test['Age'] = mw_test['Age'].apply(lambda x : get_rand_age(x, mw_age_range))

t_women.loc[mw_test.index] = mw_test
train.loc[women.index] = women

test.loc[t_women.index] = t_women

train['Age'].isnull().sum() + test['Age'].isnull().sum()
train.isnull().sum()
common_embarked = train['Embarked'].value_counts().index[0]

train['Embarked'] = train['Embarked'].fillna(common_embarked)

test['Embarked'] = test['Embarked'].fillna(common_embarked)
train['HasCabin'] = train['Cabin'].notnull()

test['HasCabin'] = test['Cabin'].notnull()



bar_graph_by('HasCabin', 'Survived',  train)
bar_graph_by('Pclass', 'HasCabin',  train)
withcabins = train.loc[train['HasCabin'] ==True]

withcabins_t = test.loc[test['HasCabin'] ==True]



withcabins['Cabin'] = withcabins['Cabin'].apply(lambda cab : cab[0])

withcabins_t['Cabin'] = withcabins_t['Cabin'].apply(lambda cab : cab[0])



train.loc[withcabins.index] = withcabins

test.loc[withcabins_t.index] = withcabins_t



test_train_withcabins = pd.concat([withcabins, withcabins_t], ignore_index=True)[['Cabin', 'Pclass', 'HasCabin']]



# have a look at counts, in this case we're just using 'HasCabin'

# on y axis as it's a nice column full of 1's.

multi_bar_count_by('Cabin', 'HasCabin', 'Pclass', test_train_withcabins)
def weighted_random_cabin(cls):

    c_decks = test_train_withcabins[test_train_withcabins['Pclass'] == cls]['Cabin'].value_counts()

    return np.random.choice(list(c_decks.index), p=list(c_decks /c_decks.sum()))
null_cab = train.loc[train['Cabin'].isnull()]#['Pclass']#.apply(lambda c : weighted_random_cabin(c))

null_cab['Cabin'] = null_cab['Pclass'].apply(lambda c : weighted_random_cabin(c))

train.loc[null_cab.index] = null_cab



null_cabx = test.loc[test['Cabin'].isnull()]

null_cabx['Cabin'] = null_cabx['Pclass'].apply(lambda c : weighted_random_cabin(c))

test.loc[null_cabx.index] = null_cabx
train['HasCabin'] = True

multi_bar_count_by('Cabin', 'HasCabin', 'Pclass', train)
bar_graph_by('Cabin', 'Survived', train)
deck_vals = dict((d, float(i)) for i, d in enumerate(sorted(list(train['Cabin'].unique()))))

train['Deck'] = train['Cabin'].apply(lambda v : deck_vals[v])

test['Deck'] = test['Cabin'].apply(lambda v : deck_vals[v])
train = train.drop('HasCabin', axis =1)

test = test.drop('HasCabin', axis =1)
train['Fare'].describe()
facet_by('Fare', 'Survived', train,  sea.distplot, asp=3)
fare_med = pd.concat((train['Fare'], test['Fare']), ignore_index = True).median()

train['Fare'] = train['Fare'].fillna(fare_med)

test['Fare'] = test['Fare'].fillna(fare_med)
gender_colors = sea.xkcd_palette(['cool blue', "coral pink"])

sea.palplot(gender_colors)

plt.show()
bar_graph_by('Sex', 'Survived',  train, pal = gender_colors)
bar_graph_by('Pclass', 'Survived',  train)
bar_graph_by('Embarked', 'Survived',  train)
bar_graph_by('Pclass', 'Survived',  train)

bar_graph_by('Embarked', 'Survived',  train)
bar_graph_by('Parch', 'Survived',  train)

bar_graph_by('SibSp', 'Survived',  train)
train['Family'] = train['SibSp'] + train['Parch']



# do the same thing to test before i forget...

test['Family'] = test['SibSp'] + test['Parch']



bar_graph_by('Family', 'Survived',  train, )

facet_by('Family', 'Survived', train,  sea.distplot, asp=3)

facet_by('Age', 'Survived', train,  sea.kdeplot , asp = 3, shade= True)

facet_by('Age', 'Sex', train,  sea.kdeplot, asp=3, c_lab = 'Survived', pal = gender_colors[::-1],  shade= True)
facet_by('Age', 'Survived', train, sea.kdeplot, c_lab = 'Sex', asp = 2, shade= True)
facet_by('Age', 'Survived', train, sea.distplot, c_lab = 'Pclass', r_lab = 'Sex' )
train[['Survived', 'Pclass', 'Sex']].groupby(['Sex', 'Pclass']).mean()
train.dtypes
train['Gender'] =  train['Sex'] == 'female'

train['Gender'] = train['Gender'].astype(float)



test['Gender'] =  test['Sex'] == 'female'

test['Gender'] = test['Gender'].astype(float)
embarked_map = dict([(e, float(i)) for i, e in enumerate(list(test['Embarked'].unique()))])

train['Embarked'] = train['Embarked'].apply(lambda v : embarked_map[v])

test['Embarked'] = test['Embarked'].apply(lambda v : embarked_map[v])
nf_vals = ['Pclass', 'Family', 'Parch', 'SibSp']

train[nf_vals] = train[nf_vals].astype(float)

test[nf_vals] = test[nf_vals].astype(float)
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier

from sklearn.model_selection import GridSearchCV, ParameterGrid,  cross_val_score, train_test_split
def test_features(features):

    rf = RandomForestClassifier()

    dat, labels =  train[features], train[ 'Survived']

    importances = pd.DataFrame()

    importances['features'] = features

    importances['score'] = 0.

    rf.fit(dat, labels)

    importances['score'] += rf.feature_importances_

    return importances
feats = ['Gender',  'Pclass', 'Fare', 'Age', 'Family', 'Deck', 'SibSp', 'Parch', 'Embarked']

importances = test_features(feats)

importances
samp_size = print(test.shape[0]/ train.shape[0])

samp_size
train_dat, test_dat, train_lab, test_lab = train_test_split(train[feats], train['Survived'], test_size= samp_size, random_state=42)
rfc_param_grid = [{ 

                    'criterion' : ['gini', 'entropy'],

                   'min_samples_split' : [10, 20, 30, 40, 50],

                   'n_estimators': [100, 500, 1000],

                   'max_features' : ['auto', 'sqrt']

                  }]

                   

rfc = RandomForestClassifier(bootstrap = False, warm_start = True )

rfc_gs = GridSearchCV(rfc, rfc_param_grid)

rfc_gs.fit(train_dat, train_lab)

print(rfc_gs.best_score_)

rfc_gs.best_params_
best_rfc = RandomForestClassifier(bootstrap = False, 

                             warm_start = True, 

                             criterion = 'entropy' ,

                             max_features = 'auto',

                             min_samples_split = 30,

                             n_estimators = 1000

                            )



best_rfc = best_rfc.fit(train_dat, train_lab)

best_rfc_predictions = best_rfc.predict(test_dat)

best_rfc.score(test_dat, test_lab)
ada_param_grid = [{ 'n_estimators': [100, 300, 500], 

                 'learning_rate': [.1, .2, .5]}]



ada = AdaBoostClassifier()

ada_gs = GridSearchCV(ada, ada_param_grid)

ada_gs = ada_gs.fit(train_dat, train_lab)

print(ada_gs.best_score_)

ada_gs.best_params_
ada2 = AdaBoostClassifier(learning_rate = .5, n_estimators = 100)



ada2 = ada2.fit( train_dat, train_lab)

ada_predictions = ada2.predict(test_dat)

ada2.score(test_dat, test_lab)
def make_submission(mdl, features, filename):

    mdl = mdl.fit(train[features], train['Survived'])

    predictions = mdl.predict(test[features])

    results = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': predictions})

    results.to_csv(filename)
submission_rfc =   RandomForestClassifier(   bootstrap = False, 

                                             warm_start = True, 

                                             criterion = 'entropy' ,

                                             max_features = 'auto',

                                             min_samples_split = 30,

                                             n_estimators = 1000)



make_submission(submission_rfc, feats, 'rfcmodel.csv')