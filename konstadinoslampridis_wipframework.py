import os



from itertools import product

from functools import reduce



import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# helpers

from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection

#OneHotEncoder
print("FILES provided:\n{}".format('\n'.join([os.path.join(dirname, filename) for dirname, _, filenames in os.walk('/kaggle/input') for filename in filenames])))

        

def load_data():

    global train_data, test_data

    train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

    test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

    

ld = load_data

ld()
def split_attributes(dataframe):

    """Return the categorical and numerical columns/attributes of the given dataframe"""

    _ = dataframe._get_numeric_data().columns.values

    return list(set(dataframe.columns) - set(_)), _



def missing_values(dataframe):

    return {k:v for k,v in dataframe.isnull().sum().to_dict().items() if v != 0}
print('Train #datapoints {}, #attributes {}: [{}]'.format(len(train_data), len(train_data.columns), ', '.join(train_data.columns)))

print('Test #datapoints {}, #attributes {}: [{}]'.format(len(test_data), len(test_data.columns), ', '.join(test_data.columns)))



print()

train_data.info()

print()

test_data.info()
def print_missing():

    train_missing = missing_values(train_data)

    test_missing = missing_values(test_data)

    print("Train data missing values:\n{}".format('\n'.join(['{}: {} ({:.2f}%)'.format(x, y, 100*float(y)/len(train_data)) for x,y in sorted(train_missing.items(), key=lambda z: z[0])])))

    print()

    print("Test data missing values:\n{}".format('\n'.join(['{}: {} ({:.2f}%)'.format(x, y, 100*float(y)/len(test_data)) for x,y in sorted(test_missing.items(), key=lambda z: z[0])])))

print_missing()
def variable_types():

    train_categorical_attrs, train_numerical_attrs = split_attributes(train_data)

    test_categorical_attrs, test_numerical_attrs = split_attributes(test_data)



    print('"Train" data categorical attributes: [{}]'.format(', '.join(train_categorical_attrs)))

    print('"Test" data categorical attributes: [{}]\n'.format(', '.join(test_categorical_attrs)))

    print('"Train" data numerical attributes: [{}]'.format(', '.join(train_numerical_attrs)))

    print('"Test" data numerical attributes: [{}]'.format(', '.join(test_numerical_attrs)))



variable_types()
train_data.sample(10)
# train_data.describe('all')  # for distribution of all attributes
train_data.describe()
train_data.describe(include=['O'])
g = sns.FacetGrid(train_data, col='Pclass')

g.map(plt.hist, 'Survived')
print (100 * train_data.groupby(['Pclass','Survived']).size() / len(train_data.index))
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Pclass', 'Survived']].groupby(['Survived'], as_index=False).mean().sort_values(by='Pclass', ascending=False)
train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
sns.distplot(train_data['Age'][train_data['Age'].notna()])
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
def drop_columns(dataframe, *columns):

    """Call this method to remove given columns for the given dataframe and get a new dataframe reference"""

    return dataframe.drop([*columns], axis=1)



def add_column(dataframe, name, values):

    """Call this method to add a new column with the given values and get a new dataframe reference"""

    return dataframe.assign(**{name: values})

########

def bin_column(column_ref, nb_bins):

    return pd.cut(column_ref, nb_bins)



def qbin_column(column_ref, nb_bins):

    return pd.qcut(column_ref, nb_bins)



def string_map(column_ref, strings, target_form):

    """Call this method to replace values found in 'strings' list with the target form, given a column (ie Series) reference and return the reference"""

    return column_ref.replace(strings, target_form)

####

def add_bin_for_continuous(dataframe, column, new_column, nb_bins):

    return add_column(dataframe, new_column, list(bin_column(dataframe[column], nb_bins)))



def add_reg(dataframe, name, regex, target):

    """Call this method to add a new column by applying a regex extractor to an existing column and get a new dataframe reference"""

    return add_column(dataframe, name, list(dataframe[target].str.extract(regex, expand=False)))



def map_replace_string(dataframe, column, strings_data, target_forms):

    for strings, target in zip(strings_data, target_forms):

        dataframe[column] = string_map(dataframe[column], strings, target)

    return dataframe



###############################

def df_map(a_callable, dataframes, *args, **kwargs):

    return [a_callable(x, *args, **kwargs) for x in dataframes]

#########################
# Starting of data altering operations

print(train_data.columns)
# DROP 'Ticket' and 'Cabin' columns

train_data, test_data = df_map(drop_columns, [train_data, test_data], 'Ticket', 'Cabin')

print('Train data columns: [{}]'.format(', '.join(train_data.columns.values)))
print_missing()
def complete_categorical_with_most_freq(dataframe, column):

    return dataframe.assign(**{column: dataframe[column].fillna(dataframe[column].dropna().mode()[0])})
# COMPLETE the 'Embarked' categorical attribute with the most common/frequent occurance

train_data, test_data = df_map(complete_categorical_with_most_freq, [train_data, test_data], 'Embarked')



train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print_missing()
# COMPLETE 'Fare' data

def complete_numerical_with_most_freq(dataframe, column):

    return dataframe.assign(**{column: dataframe[column].fillna(dataframe[column].dropna().median())})



test_data = complete_numerical_with_most_freq(test_data, 'Fare')
print_missing()
class MedianFiller:

    def __call__(self, dataframe, column, columns):

        """

        Call this method to fill missing values in a dataframe's column according to the medians computed on correlated columns\n

        :param str dataframe:

        :param str column: column with missing values

        :param list columns: correlated columns

        :return: a dataframe reference with the column completed

        """

        for vector in product(*[list(dataframe[c].unique()) for c in columns]):

            self._set_value(dataframe, column, self._condition(dataframe, columns, vector)) 

        return dataframe.assign(**{column: dataframe[column].astype(int)})

    

    def _set_value(self, dataframe, column, condition):

        dataframe.loc[(dataframe[column].isnull()) & condition, column] = self._convert(dataframe[condition][column].dropna().median())



    def _condition(self, dataframe, columns, values_vector):

        return reduce(lambda i,j: i & j, [dataframe[c] == values_vector[e] for e, c in enumerate(columns)])



    def _convert(self, age):

        """Call this method to convert the input float number to its nearest 0.5"""

        return int(age / 0.5 + 0.5) * 0.5



median_filler = MedianFiller()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
# FILL in missing values in 'Age' column with medians computed on correlated 'Sex' and 'Pclass' columns

train_data, test_data = df_map(median_filler, [train_data, test_data], 'Age', ['Sex', 'Pclass'])
print_missing()
# DROP 'PassengetId' from train_data only

# train_data = drop_columns(train_data, 'PassengerId')
def create_column(dataframe, name, a_callable):

    return dataframe.assign(**{name: a_callable(dataframe)})



# CREATE 'FamilySize' column/feature out of 'SibSp' and 'Parch'

train_data, test_data = df_map(create_column, [train_data, test_data], 'FamilySize', lambda x: x['SibSp'] + x['Parch'] + 1)



# CREATE 'IsAlone' FEATURE by checking the family size (if familySize=1 then IsAlone=1, else IsAlone=0)  # where replaces values where condition is false

train_data, test_data = df_map(lambda x,y: x.assign(**{y:pd.Series([1]*len(x)).where(x['FamilySize'] == 1, 0)}), [train_data, test_data], 'IsAlone')



# CREATE a 'Age*Class' feature as 'Age' * 'Class'

train_data, test_data = df_map(create_column, [train_data, test_data], 'Age*Class', lambda x: x['Age'] * x['Pclass'])

train_data.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)



# Check potential correlation

print(train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
# # DROP 'Parch', 'SibSp', FEATURES (in favor of 'FamilySize')

# train_data, test_data = df_map(drop_columns, [train_data, test_data], 'Parch', 'SibSp')
# ADD Title attribute

train_data, test_data = df_map(add_reg, [train_data, test_data], 'Title', ' ([A-Za-z]+)\.', 'Name')



print('Train #datapoints {}, #attributes {}: [{}]'.format(len(train_data), len(train_data.columns), ', '.join(train_data.columns)))

print('Test #datapoints {}, #attributes {}: [{}]'.format(len(test_data), len(test_data.columns), ', '.join(test_data.columns)))
tc = train_data['Title'].value_counts()

tec = test_data['Title'].value_counts()



print("Train data 'Title' #distinct values: {}, frequencies:\n{}".format(len(tc), '\n'.join(['{}: {}'.format(z[0], z[1]) for z in sorted(tc.items(), key=lambda x: x[1], reverse=True)])))

print()

print("Test data 'Title' #distinct values: {}, frequencies:\n{}".format(len(tec), '\n'.join(['{}: {}'.format(z[0], z[1]) for z in sorted(tec.items(), key=lambda x: x[1], reverse=True)])))



pd.crosstab(train_data['Title'], train_data['Sex'])
# DROP 'Name' column

train_data, test_data = df_map(drop_columns, [train_data, test_data], 'Name')
# Define mappings to perform 'normalization'

norm = {

    'Rare': ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],

    'Miss': ['Mlle', 'Ms'],

    'Mrs': ['Mme']

}
# NORMALIZATION of string terms

# Map values to a 'normal' form, looking at the 'Title' column. Apply the replacements both for train_data and test_data

train_data, test_data = df_map(map_replace_string, [train_data, test_data], 'Title', *list(reversed([list(_) for _ in zip(*list(norm.items()))])))



tc = train_data['Title'].value_counts()

tec = test_data['Title'].value_counts()



print("Train data 'Title' #distinct values: {}, frequencies:\n{}".format(len(tc), '\n'.join(['{}: {}'.format(z[0], z[1]) for z in sorted(tc.items(), key=lambda x: x[1], reverse=True)])))

print()

print("Test data 'Title' #distinct values: {}, frequencies:\n{}".format(len(tec), '\n'.join(['{}: {}'.format(z[0], z[1]) for z in sorted(tec.items(), key=lambda x: x[1], reverse=True)])))
def test():

    d1 = train_data.copy()

    d1['Title'] = d1['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    d1['Title'] = d1['Title'].replace(['Mlle' 'Ms'], 'Miss')

    d1['Title'] = d1['Title'].replace('Mme', 'Mrs')

    assert list(d1['Title']) == list(train_data['Title'])

test()
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
variable_types()
sorted(pd.cut(train_data.Age, 5).unique())
sorted(pd.cut(test_data.Age, 5).unique())
# CREATE 5 BINS for 'Age' column (discreetize) and add a column in 'train' data

train_data = train_data.assign(**{'AgeBand': pd.cut(train_data.Age.astype(int), 5)})

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_data.sample(3)
def add_qbin(dataframe, target, nb_bins, destination):

    """Call this function to create a column (with 'destination' name) of quantisized bins of the continuous variable given in the target column"""

    return dataframe.assign(**{destination: pd.qcut(dataframe[target], nb_bins)})
# CREATE 4 quantisized bins for the 'Fare' attribute and add a column in the 'train' data

train_data = add_qbin(train_data, 'Fare', 4, 'FareBand')
train_data.sample(3)
def binned_indices(values, left_boundaries):

    """Call this function to get an array of indices the given values belong based on the input boundaries.\n

        If values in `x` are beyond the bounds of `left_boundaries`, 0 or ``len(left_boundaries)`` is returned as appropriate."""

    return np.digitize(values, left_boundaries)



def _map(intervals_list):

    """Call this function to get a dictionary mapping Interval objects to numerical codes (0, 1, ..).

        Assumes that the input Intervals list is sorted"""

    return {(interval_obj): index for index, interval_obj in enumerate(intervals_list)}



## operations for dfs with constructed bins/bands

def encode_bands(dataframe, target_column, intervals_list, destination_column):

    """Call this function to get a dictionary mapping Interval objects to numerical codes (0, 1, ..).

        Assumes that the input Intervals list is sorted"""

    return dataframe.assign(**{destination_column: dataframe[target_column].map(_map(intervals_list)).astype(int)})



def encode_bands_many(dataframe, targets, intervals_lists, destinations):

    return dataframe.assign(**{dest_c: dataframe[target_c].map(_map(intervals_list)).astype(int) 

                               for target_c, intervals_list, dest_c in zip(targets, intervals_lists, destinations)})

    

## operations for dfs without constructed bins/bands

def encode_continuous(dataframe, target_column, intervals_list, destination_column):

    return dataframe.assign(**{destination_column: binned_indices(dataframe[target_column], iter(x.left for x in intervals_list)) - 1})



def encode_continuous_many(dataframe, targets, intervals_lists, destinations):

    return dataframe.assign(**{dest_c: binned_indices(dataframe[target_c], [x.left for x in intervals_list]) - 1 

                               for target_c, intervals_list, dest_c in zip(targets, intervals_lists, destinations)})
def _op_gen(dataframe, columns, band_str='Band', post_str='_Code'):

#     interval_lists = [sorted(dataframe[c+band_str].unique()) for c in columns]

#     coded = ['{}{}'.format(c, post_str) for c in columns]

    _ = [list(_) for _ in zip(*list([(sorted(dataframe[c+band_str].unique()), '{}{}'.format(c, post_str)) for c in columns]))]

    yield lambda x: encode_bands_many(x, [c+band_str for c in columns], _[0], _[1])

    while 1:

        yield lambda x: encode_continuous_many(x, columns, _[0], _[1])
#### CONSTANTS #####

POST_STR = '_Code'  # postfix string for encoded features

BAND_STR = 'Band'



#### SETTINGS ###

# PICK columns with categorical variables to encode with sklearn LabelEncoder

TO_ENCODE_WITH_SKLEARN = ['Embarked', 'Sex', 'Title']



TO_ENCODE_WITH_INTERVALS = ['Age', 'Fare']
# Get categorical features except for custom "bands" (columns with data binned from continuous variables into categories)

categ = set(split_attributes(train_data)[0]).intersection(set(split_attributes(test_data)[0]))



# Sanity check

assert categ == set(TO_ENCODE_WITH_SKLEARN)

assert all(set(train_data[x].unique()) == set(test_data[x].unique()) for x in TO_ENCODE_WITH_SKLEARN)



def label_encode(dataframe, columns, encode_callback, code_str='_Code'):

    return dataframe.assign(**{c+code_str: encode_callback(dataframe[c]) for c in columns})
# Add 'Sex_Code', 'Embarked_Code' and 'Title_Code' columns to 'train_data' and to 'test_data'

train_data, test_data = df_map(label_encode,

                               [train_data, test_data],

                               ['Embarked', 'Sex', 'Title'],

                               LabelEncoder().fit_transform)  # encodes categorical objects into indices starting from 0

train_data.sample(3)
print("'Train' data Fare bins (FareBand): [{}]\n'Test' data Fare bins (FareBand): [{}]".format(

    ', '.join(str(_) for _ in sorted(train_data.FareBand.unique())),

    ', '.join(str(_) for _ in sorted(add_qbin(test_data, 'Fare', 4, 'FareBand').FareBand.unique()))))
print("'Train' data Age bins (FareBand): [{}]\n'Test' data Age bins (FareBand): [{}]".format(

    ', '.join(str(_) for _ in sorted(train_data.AgeBand.unique())),

    ', '.join(str(_) for _ in sorted(pd.cut(test_data.Age, 5).unique()))))
# ENCODE 'Age' and 'Fare' by creating the 'Age_code' and 'Fare_Code' coluns in 'train_data' and 'test_data'

# train_data, test_data = encode_bands([train_data, test_data], TO_ENCODE_WITH_INTERVALS, bin_str=BAND_STR, post=POST_STR)



op_gen = _op_gen(train_data, ['Age', 'Fare'], band_str='Band', post_str='_Code')

train_data, test_data = [next(op_gen)(df) for df in [train_data, test_data]]



train_data.head(4)
test_data.head(4)
train_data.columns
train_data = train_data.drop(['PassengerId'], axis=1)
#define y variable aka target/outcome

Target = ['Survived']



# FEATURE SLECTION

# define features (original and encoded)

feature_titles = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts

feature_names = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation

# data1_xy =  Target + data1_x

# print('Original X Y: ', data1_xy, '\n')





#define x variables for original w/bin features to remove continuous variables

data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'Age_Code', 'Fare_Code']

# data1_xy_bin = Target + data1_x_bin

# print('Bin X Y: ', data1_xy_bin, '\n')





#define x and y variables for dummy features original

data1_dummy = pd.get_dummies(train_data[feature_titles])

data1_x_dummy = data1_dummy.columns.tolist()

# data1_xy_dummy = Target + data1_x_dummy

# print('Dummy X Y: ', data1_xy_dummy, '\n')
train_data.head()
train_dataframe = train_data



# SELECT features

numerical_feats = ['Pclass', 'Fare_Code', 'Age_Code', 'FamilySize']  # ordering makes sence: eg: class_1 < class_2, 

binary_feats = ['Sex_Code', 'IsAlone']

categorical_feats = ['Embarked']



assert all(all(x in train_dataframe.columns for x in y) for y in [numerical_feats, binary_feats, categorical_feats])
pd.get_dummies(train_data[categorical_feats]).head()
X_train = pd.concat([train_data[numerical_feats + binary_feats], pd.get_dummies(train_data[categorical_feats])], axis=1)

X_test = pd.concat([test_data[numerical_feats + binary_feats], pd.get_dummies(test_data[categorical_feats])], axis=1)

X_train.head()
X_test.head()
data1_dummy.head()
#split train and test data with function defaults

#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(train_data[data1_x_calc], train_data[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(train_data[data1_x_bin], train_data[Target] , random_state = 0)

train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], train_data[Target], random_state = 0)





print("Data1 Shape: {}".format(train_data.shape))

print("Train1 Shape: {}".format(train1_x.shape))

print("Test1 Shape: {}".format(test1_x.shape))
train1_x_bin.head()
data1 = train_data.drop(['PassengerId'], axis=1)
for x in data1_x:

    if data1[x].dtype != 'float64' :

        print('Survival Correlation by:', x)

        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())

        print('-'*10, '\n')
# print(pd.crosstab(data1['Title'],data1[Target[0]]))
#graph distribution of quantitative data

plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(data1['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
# multi-variable comparison

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])



sns.pointplot(x = 'FareBand', y = 'Survived',  data=data1, ax = saxis[1,0])

sns.pointplot(x = 'AgeBand', y = 'Survived',  data=data1, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')
# pair plots of entire dataset

pp = sns.pairplot(data1, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )

pp.set(xticklabels=[])
# correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
correlation_heatmap(data1)
param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini

              #'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best

              'max_depth': [2,4,6,8,10,None], #max depth tree can grow; default is none

              #'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2

              #'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1

              #'max_features': [None, 'auto'], #max features to consider when performing split; default none or all

              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation

             }

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0 )



#decision tree w/full dataset modeling submission score: defaults= 0.76555, tuned= 0.77990



submit_dt = model_selection.GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', cv=cv_split)

submit_dt.fit(X_train, train_data[Target])

# submit_dt.fit(train_data[data1_x_bin], train_data[Target])

print('Best Parameters: ', submit_dt.best_params_) #Best Parameters:  {'criterion': 'gini', 'max_depth': 4, 'random_state': 0}



predictions = submit_dt.predict(X_test)



predictions_df = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

predictions_df.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
# X_train = train_data.drop(["Survived", 'PassengerId'], axis=1)

# Y_train = train_data["Survived"]

# X_test  = test_data.drop("PassengerId", axis=1).copy()

# X_train.shape, Y_train.shape, X_test.shape
# dtc = DecisionTreeClassifier()

# dtc.fit(data1[data1_x_bin], Y_train)

# Y_pred = dtc.predict(data1[data1_x_bin])

# acc_dtc = round(dtc.score(data1[data1_x_bin], Y_train) * 100, 2)

# print(acc_dtc)
# random_forest = RandomForestClassifier(n_estimators=100)

# random_forest.fit(X_train, Y_train)

# Y_pred = random_forest.predict(X_test)

# random_forest.score(X_train, Y_train)

# acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# acc_random_forest
# def targets(dataframe):

#     return dataframe['Survived']



# def feature_vectors(dataframe, features):

#     return pd.get_dummies(dataframe[features])



# def accuracy(y_pred, y_true):

#     return sum(np.array(y_pred) == np.array(y_true)) / float(len(y_true))

    

# def submit(model, x_test, test_dataframe):

#     predictions = model.predict(x_test)

#     output = pd.DataFrame({'PassengerId': test_dataframe.PassengerId, 'Survived': predictions})

#     output.to_csv('my_submission.csv', index=False)

#     print("Your submission was successfully saved!")

#     return predictions



# def build_model(estimator, *args, **kwargs):

#     return {

#         'random-forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1),

#         'adaboost': AdaBoostClassifier(n_estimators=100),

#         'linear-svc': LinearSVC(random_state=0, tol=1e-5, max_iter=1000),

#         'bagging': BaggingClassifier(base_estimator=SVC(), n_estimators=100, random_state=0),

#         'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4, solver='lbfgs', verbose=0, random_state=1, learning_rate_init=.01),

#         'gaussian': GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=0)

#     }[estimator]

#     return pd.get_dummies(dataframe[features])



# def train(model, X, y, kfold=None, random_state=100):

#     res = {'model': model}

#     if not kfold:

#         model.fit(X, y)

#         res['accuracy'] = accuracy(model.predict(X), y)

#     else:

#         try:

#             kfold = int(kfold)

#             assert 1 < kfold

#         except (TypeError, AssertionError):

#             raise("The 'kfold' parameter should evaluate either to None or an integer >= 2")

#         kfold_results = model_selection.cross_val_score(model, X, y, cv=model_selection.KFold(n_splits=kfold, random_state=random_state))

#         res.update(accuracy=kfold_results.mean(), kfold_results=kfold_results)

#     return res
# # FEATURE SELECTION

# drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Fare', 'Parch', 'Age', 'Sex']

# train = train_data.drop(drop_elements, axis = 1)

# print(dataframe_stats(train))

# #train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

# test  = test_data.drop(drop_elements, axis = 1)
# train = pd.get_dummies(train)

# y = targets(train_data)

# X = feature_vectors(train_data, features)



# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

# imp_mean.fit(X)

# X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]

# print(imp_mean.transform(X))



# X_test = feature_vectors(test_data, features)



# colormap = plt.cm.RdBu

# plt.figure(figsize=(7,6))

# plt.title('Pearson Correlation of Features', y=1.05, size=15)

# sns.heatmap(train.astype(float).corr(), linewidths=0.1,vmax=1.0, 

#             square=True, cmap=colormap, linecolor='white', annot=True)
# y = targets(train_data)

# X = feature_vectors(train_data, features)



# # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

# # imp_mean.fit(X)

# # X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]

# # print(imp_mean.transform(X))



# X_test = feature_vectors(test_data, features)



# X_test = imp_mean.transform(X_test)
# m = [

#     'random-forest',

#     'adaboost',

#     'linear-svc',

#     'bagging',

#     'mlp',

#     'gaussian'][5]



# kfold = None



# model = build_model(m)

# res = train(model, X, y, kfold=kfold, random_state=100)

# print("Train accuracy: {:.2f}".format(res['accuracy']))
# Get predictions on blind 'test' set

# y_test_pred = submit(res['model'], X_test, test_data)