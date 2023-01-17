# data analysis and data handling

import pandas as pd

import numpy as np



# data visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#train_df = pd.read_csv('data/train.csv')

#test_df = pd.read_csv('data/test.csv')

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
print('Shape of training set = {}'.format(train_df.shape))

print('_'*40)

print('Shape of testing set  = {}'.format(test_df.shape))
train_df.head(5)
test_df.head(5)
print(train_df.columns)

print('_'*40)

print(test_df.columns)
print(train_df.dtypes)

print('_'*40)

print(test_df.dtypes)
train_df.isnull().sum()
test_df.isnull().sum()
print(train_df.info())

print('_'*40)

print(test_df.info())
train_df.groupby('Survived').size()
train_df.describe()
train_df.describe(include=['O'])
test_df.describe()
test_df.describe(include=['O'])
train_df.hist(figsize=(20,10), layout=(3,3), bins=20)
train_df.plot(

    figsize=(20,10), 

    kind='box', 

    sharex=False, 

    subplots=True, 

    layout=(3,3)

)
#from pandas.plotting import scatter_matrix

temp_filtered_df = train_df[train_df['Age'].notnull()]

pd.plotting.scatter_matrix(

    temp_filtered_df, 

    figsize=(20,10), 

    c=temp_filtered_df['Survived'], 

    alpha=0.3

)

#plt.legend(loc = 'best')
names = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

corr_train_df = train_df[names].corr()



fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

corr_plot = ax.matshow(corr_train_df, vmin=-1, vmax=1, cmap='Spectral')

fig.colorbar(corr_plot)

ticks = np.arange(0,5,1) # total 6 items

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

plt.show()
(

    train_df[['Pclass', 'Survived']]

    .groupby(['Pclass'], as_index=False)

    .mean()

    .sort_values(by='Survived', ascending=False)

)
(

    train_df[['Sex', 'Survived']]

    .groupby(['Sex'], as_index=False)

    .mean().sort_values(by='Survived', ascending=False)

)
(

    train_df[['SibSp', 'Survived']]

    .groupby(['SibSp'], as_index=False)

    .mean()

)
(

    train_df[['Parch', 'Survived']]

    .groupby(['Parch'], as_index=False)

    .mean()

)
(

    train_df[['Embarked', 'Survived']]

    .groupby(['Embarked'], as_index=False)

    .mean()

    .sort_values(by='Survived', ascending=False)

)
(

    train_df[['Pclass', 'SibSp', 'Survived']]

    .groupby(['Pclass', 'SibSp'], as_index=False)

    .mean()

    .sort_values(by='Survived', ascending=False)

)
(

    train_df[['Pclass', 'Parch', 'Survived']]

    .groupby(['Pclass', 'Parch'], as_index=False)

    .mean()

    .sort_values(by='Survived', ascending=False)

)
sns.FacetGrid(train_df, col='Survived', height=4, aspect=1).map(plt.hist, 'Age', bins=20)



print("Mean age relative to survival:")

print(train_df[['Age', 'Survived']].groupby(['Survived'], as_index=False).mean())

print("Median age relative to survival:")

print(train_df[['Age', 'Survived']].groupby(['Survived'], as_index=False).median())
sns.FacetGrid(train_df, col='Survived', height=4, aspect=1).map(plt.hist, 'Fare', bins=20)



print("Mean fare relative to survival:")

print(train_df[['Fare', 'Survived']].groupby(['Survived'], as_index=False).mean())

print("Median fare relative to survival:")

print(train_df[['Fare', 'Survived']].groupby(['Survived'], as_index=False).median())
grid = sns.FacetGrid(train_df, height=5, aspect=1)

grid.map(sns.pointplot, 'SibSp', 'Survived', ci=95)

grid.add_legend()
grid = sns.FacetGrid(train_df, height=5, aspect=1)

grid.map(sns.pointplot, 'Parch', 'Survived', ci=95)

grid.add_legend()
sns.FacetGrid(

    train_df, 

    col='Survived', 

    row='Pclass', 

    height=3, 

    aspect=1

).map(

    plt.hist, 

    'Age', 

    alpha=.5, 

    bins=20

).add_legend();
sns.FacetGrid(

    train_df, 

    col='Survived', 

    row='Pclass', 

    height=3, 

    aspect=1

).map(

    plt.hist, 

    'Fare', 

    alpha=.5, 

    bins=20

).add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=3, aspect=1)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=95, order=['male', 'female'])

grid.add_legend()
grid = sns.FacetGrid(train_df, col='Embarked', height=3, aspect=1)

grid.map(sns.pointplot, 

         'Pclass', 

         'Survived', 

         'Sex', 

         order = [1,2,3], 

         hue_order=['male', 'female'], 

         palette='deep', 

         ci=95

        )

grid.add_legend()



(

    train_df[['Embarked', 'Pclass', 'Sex']]

    .groupby(['Embarked', 'Pclass', 'Sex'], as_index=False)

    .size()

)
(

    train_df[['Embarked']]

    .groupby(['Embarked'], as_index=False)

    .size()

)
test_feature_df = train_df.copy()



test_feature_df['FamilySize'] = test_feature_df['SibSp'] + test_feature_df['Parch'] + 1
print(

    test_feature_df[['FamilySize', 'Survived']]

    .groupby(['FamilySize'], as_index=False)

    .mean()

)

print(

    test_feature_df[['FamilySize']]

    .groupby(['FamilySize'], as_index=False)

    .size()

)

print (

    test_feature_df[['FamilySize', 'Pclass']]

    .groupby(['Pclass', 'FamilySize'], as_index=False)

    .size()

)
grid = sns.FacetGrid(test_feature_df, height=5, aspect=1)

grid.map(sns.pointplot, 'FamilySize', 'Survived', ci=95)

grid.add_legend()
print(

    test_feature_df[['FamilySize', 'Sex', 'Survived']]

    .groupby(['FamilySize', 'Sex'], as_index=False)

    .mean()

)

print(

    test_feature_df[['FamilySize', 'Sex', 'Survived']]

    .groupby(['FamilySize', 'Sex'], as_index=False)

    .size()

)
grid = sns.FacetGrid(test_feature_df, height=5, aspect=1)

grid.map(

    sns.pointplot, 

    'FamilySize', 'Survived', 'Sex', 

    order = [1,2,3,4,5,6,7,8,9,10,11], 

    hue_order=['male', 'female'], 

    palette='deep', 

    ci=95

)

grid.add_legend()
grid = sns.FacetGrid(test_feature_df, col='Pclass', height=5, aspect=1)

grid.map(sns.pointplot, 'FamilySize', 'Survived', ci=95)

grid.add_legend()
test_feature_df = train_df.copy()



test_feature_df['FamilySize'] = test_feature_df['SibSp'] + test_feature_df['Parch'] + 1



test_feature_df['IsAlone'] = (test_feature_df['FamilySize'] == 1).astype(int)
grid = sns.FacetGrid(test_feature_df, height=5, aspect=1)

grid.map(sns.pointplot, 'IsAlone', 'Survived', ci=95)

grid.add_legend()
grid = sns.FacetGrid(test_feature_df, col='Pclass', height=5, aspect=1)

grid.map(sns.pointplot, 'IsAlone', 'Survived', ci=95)

grid.add_legend()
test_feature_df = train_df.copy()

num_bins = 10

test_feature_df['age_cat'] = pd.cut(test_feature_df['Age'], 

                                    num_bins, 

                                    #labels=[i for i in range(0,num_bins)]

                                   )



#test_feature_df['age_cat'] = (test_feature_df['age_cat']).astype(int)



(

    test_feature_df[['age_cat', 'Survived']]

    .groupby(['age_cat'], as_index=False)

    .mean()

    .sort_values(by='age_cat', ascending=True)

)
(

    test_feature_df[['age_cat', 'Survived']]

    .groupby(['age_cat'], as_index=False)

    .size()

)
test_feature_df = train_df.copy()

num_bins = 10

test_feature_df['fare_cat'] = pd.cut(test_feature_df['Fare'], 

                                     num_bins, 

                                     #labels=[i for i in range(0,num_bins)]

                                    )



#test_feature_df['fare_cat'] = (test_feature_df['fare_cat']).astype(int)



(

    test_feature_df[['fare_cat', 'Survived']]

    .groupby(['fare_cat'], as_index=False)

    .mean()

    .sort_values(by='fare_cat', ascending=True)

)
(

    test_feature_df[['fare_cat', 'Survived']]

    .groupby(['fare_cat'], as_index=False)

    .size()

)
MEDIAN_AGE = None

MEDIAN_FARE = None

MEDIAN_EMBARKED = 'S'


def create_family_size_feature(df):

    df['num_family_travelling_with'] = df['SibSp'] + df['Parch'] + 1

    return df



def create_is_alone_feature(df):

    df['is_alone'] = (df['num_family_travelling_with'] == 1).astype(int)

    return df



def discretize_continuous_feature(df, new_column, column_to_bin, num_bins):

    df[new_column] = pd.cut(df[column_to_bin],

                            num_bins,

                            labels=[i for i in range(0,num_bins)]

                           )

    df[new_column] = (df[new_column]).astype(int)

    return df

    

def categorize_feature(df, new_column, old_column, old_column_value):

    def get_new_column_value(row, old_column, old_column_value):

        if row[old_column] == old_column_value:

            new_value = 1

        else:

            new_value = 0

        return new_value

    

    df[new_column] = df.apply(

        lambda row: get_new_column_value(row, old_column, old_column_value),

        axis=1

    )

    return df





# data transform pipeline

def transform_pipeline(df, data_set_type):

    # global is not great to use. Need to refactor pipeline into a class :( 

    global MEDIAN_AGE, MEDIAN_FARE, MEDIAN_EMBARKED

    

    # engineer family size feature

    df = create_family_size_feature(df)

    

    # engineer IsAlone feature

    df = create_is_alone_feature(df)

    

    # calculate median values of missing data columns if data_set_type = 'train'

    if data_set_type == 'train':

        MEDIAN_AGE = df['Age'].median()

        print('Median age of training data is {}'.format(MEDIAN_AGE))

        MEDIAN_FARE = df['Fare'].median()

        print('Median Fare of training data is {}'.format(MEDIAN_FARE))

        print('Median embarked is {}'.format(MEDIAN_EMBARKED))

        

    # Fill missing values with medians

    fill_na_dict = {

        'Age': MEDIAN_AGE,

        'Fare': MEDIAN_FARE,

        'Embarked': MEDIAN_EMBARKED

    }

    df = df.fillna(fill_na_dict)

    

    # discretize certian continuous variable features

    num_bins = 10

    # Age

    df = discretize_continuous_feature(df, 'age_cat', 'Age', num_bins)

    # Fare

    df = discretize_continuous_feature(df, 'fare_cat', 'Fare', num_bins)

    

    # Engineer categorical classification features into binary features

    new_column_list = ['is_first_class', 'is_second_class', 'is_third_class',

                       'embarked_S', 'embarked_C', 'embarked_Q',

                       'is_male']

    old_column_list = ['Pclass', 'Pclass', 'Pclass',

                       'Embarked', 'Embarked', 'Embarked',

                       'Sex']

    old_column_value_list = [1, 2, 3,

                             'S', 'C', 'Q',

                             'male']

    for new_column, old_column, old_column_value in zip(new_column_list,

                                                       old_column_list,

                                                       old_column_value_list):

        df = categorize_feature(df, new_column, old_column, old_column_value)

    

    

    

    # drop columns

    features_to_drop = ['Name', 

                        'Parch', 

                        'Ticket', 

                        'Cabin', 

                        'Pclass', 

                        'Embarked', 

                        'Sex', 

                        'Age', 

                        'Fare',

                        'PassengerId']

    df.drop(features_to_drop, axis=1, inplace=True)

    

    # rename columns

    df = df.rename(

        columns = {

            'SibSp': 'num_siblings_spouse',

        }

    )

    

    # assert no null values in dataframe

    assert pd.notnull(df).all().all()

    

    return df.values
train_labels = train_df['Survived'].copy()

train_prepared = transform_pipeline(train_df.drop(['Survived'], axis=1).copy(), 'train')
train_prepared
train_prepared.shape
train_labels.shape
# import sklearn stuff

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform

import numpy as np
log_reg = LogisticRegression()



hyper_param_distribs = {

    'penalty': ['l1', 'l2'],

    'dual': [False],

    'tol': [1e-4],

    'C': uniform(loc=0, scale=4),

    'fit_intercept': [True, False],

    'intercept_scaling': [1.0],

    'class_weight': [None],

    'random_state': [None],

    'solver': ['liblinear'],

    'max_iter': [100],

    'multi_class': ['ovr'],

    'verbose': [0],

    'warm_start': [False],

    'n_jobs': [None]

}



rnd_search = RandomizedSearchCV(

    log_reg, 

    hyper_param_distribs, 

    n_iter=1000, 

    scoring='accuracy', 

    cv=10, 

    verbose=1, 

    random_state=78, 

    n_jobs=1, 

    error_score=np.nan 

)
rnd_search.fit(train_prepared, train_labels)
rnd_search.best_params_
rnd_search.best_estimator_
rnd_search.best_score_
#cv_results = rnd_search.cv_results_

#for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):

#    print(mean_score, params)
# define best model

final_model = rnd_search.best_estimator_

# prepare (transform) test data

test_IDs = test_df['PassengerId'].copy()

test_prepared = transform_pipeline(test_df.copy(), 'test')



test_prepared.shape
# make predictions for test data

test_predictions = final_model.predict(test_prepared)
# print submission

submission = pd.DataFrame({

    'PassengerId': test_IDs,

    'Survived': test_predictions

})

submission.to_csv('submission_logistic_regression.csv', index=False)
