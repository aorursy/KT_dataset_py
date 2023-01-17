# imports



# data analysis

import pandas as pd

import numpy as np

from scipy import stats, integrate



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns



# machine learning

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error
# configuration



# seaborn

sns.set(color_codes=True)
# load datasets as DataFrames

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# combine datasets to make it easier to run operations on both sets together

test_df['Survived'] = np.NaN

combined_df = pd.concat([train_df, test_df], ignore_index=True)
# take a look at the data

combined_df.head()
# get information on the datasets

combined_df.info()

print("------------------------")

combined_df.info()
# first create a new feature to track passengers whose age is known (those passengers are probably more likely to have survived)

# might be useful later on

combined_df['age_known'] = combined_df.Age.notnull()



# create 12"x 12" figure

fig = plt.figure(figsize=(12,12))



# create "5-years large" bins (0-5, 5-10, 10-15...)

age_bins = np.arange(0, 90, 5)



# plot age distribution histogram

#-------------------------------------

# filter out missing ages

age_not_null_series = combined_df['Age'][combined_df.Age.notnull()]



plt.subplot(211)

ax1 = sns.distplot(age_not_null_series, bins=age_bins, kde=False,

                  hist_kws=dict(edgecolor="k", linewidth=1))



plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Age distribution')

plt.grid(False)



# plot Survived per Age group

#----------------------------------

# filter out missing ages from training set

age_not_null_series_train = train_df['Age'][combined_df.Age.notnull()]



# create a new feature that stores passengers age group (0 to 5 years old, 5 to 10, etc.)

train_df['age_group'] = pd.cut(age_not_null_series_train, bins=age_bins, include_lowest=True, right=False)

# compute mean for each age group

age_group_surv_df = train_df[['age_group', 'Survived']].groupby(by='age_group', as_index=False).mean()

age_group_surv_df.columns = ['age_group', 'surv_pct']



plt.subplot(212)

ax2 = sns.barplot(x='age_group', y='surv_pct', color='salmon', data=age_group_surv_df, linewidth=1, edgecolor='black')



plt.xlabel('Age group')

plt.ylabel('Survived')

plt.title('Survived per Age group')



sns.plt.show()
# get details of Survived percentage per age group

age_group_surv_df
# take a closer look at passengers over 65 years old

train_df[['Age', 'Survived']][train_df['Age'] > 65]
# get median age

combined_df['Age'].median()
# get median age per sex

combined_df[['Sex','Age']].groupby(by='Sex').median()
# compute median age, number of observations and Survived % per class

combined_df[['Age', 'Pclass', 'Survived']].groupby(by='Pclass').agg({'Age':['median','count'], 'Survived' : 'mean'})
# extract the title ('Mr', 'Mrs', 'Miss', 'Master', ...) from the Name feature

# use a regular expression that gets the first word '(\w+)' finishing by a period '\.

combined_df['title_extracted'] = combined_df['Name'].str.extract('(\w+)\.', expand=False)



# Replacing 'Mme' (Mrs. in french) by 'Mrs'

combined_df['title'] = combined_df['title_extracted'].str.lower()



# Replacing 'Mme' (Mrs. in french) by 'Mrs'

combined_df['title'] = combined_df['title'].replace('mme', 'mrs')



# Replacing 'Mlle' - french for Miss - and 'Ms' (only one in the dataset and she's 28) by 'Miss' 

combined_df['title'] = combined_df['title'].replace(['ms','mlle'], 'miss')



# All other titles are military/religion/honorific titles, we group them under a 'Notable' title

# Note: we do not try to separate male and female notable titles here.

combined_df['title'] = combined_df['title'].replace(['master','rev','dr','col','major','lady','countess','jonkheer','don','capt','dona','sir'], 'notable')



# compute median age, number of observations and Survived % per title

combined_df[['Age', 'title', 'Survived']].groupby(by='title').agg({'Age':['median','count'], 'Survived' : 'mean'})
# taking a closer look at our 'Notable' passengers, their median age is only 9 years old ??!

combined_df[combined_df['title'] == 'notable']
# set title='Master' (instead of 'Notable') for passengers for which we have extracted the title 'Master' previously

combined_df.loc[combined_df['title_extracted'] == 'Master', 'title'] = 'master'



# drop the 'title_extracted' column, we don't need anymore

combined_df.drop('title_extracted', axis=1, inplace=True)



# compute median age, number of observations and Survived % per title

combined_df[['Age', 'title', 'Survived']].groupby(by='title').agg({'Age':['median','count'], 'Survived' : 'mean'})
combined_df.groupby(by=['Pclass', 'title']).agg({'Age':['median','count'], 'Survived' : 'mean'})
# create a mapping table with 15 rows (3 classes x 5 titles)

map_table_age_df = pd.DataFrame(np.nan, index=range(0,15),columns=['Pclass', 'title', 'age_pred'])



titles = ['master', 'miss', 'mrs', 'mr', 'notable']



# 1st class passengers

map_table_age_df.iloc[0:5, 0] = 1

map_table_age_df.iloc[0:5, 1] = titles

# input our guesses

map_table_age_df.iloc[0:5, 2] = [4, 30, 45, 41, 48]



# 2nd class passengers

map_table_age_df.iloc[5:10, 0] = 2

map_table_age_df.iloc[5:10, 1] = titles

map_table_age_df.iloc[5:10, 2] = [4, 20, 31, 30, 41]



# 3rd class passengers

map_table_age_df.iloc[10:15, 0] = 3

map_table_age_df.iloc[10:15, 1] = titles

map_table_age_df.iloc[10:15, 2] = [4, 18, 31, 26, 41]



map_table_age_df
# join mapping table with the dataset, this adds a new 'age_pred' column to the set

combined_df = combined_df.merge(map_table_age_df, on=['Pclass','title'], how='left')



# evaluate mean error for non-null ages

age_not_null_slice = combined_df['Age'].notnull()

mae = mean_absolute_error(combined_df['age_pred'][age_not_null_slice], combined_df['Age'][age_not_null_slice])

print('mean absolute error for non-null ages =', mae)



# what would have been the mean error if we had set missing ages to the overall median value ?

age_pred_median = [combined_df['Age'][age_not_null_slice].median()]*1046

mae_static_median = mean_absolute_error(age_pred_median, combined_df['Age'][age_not_null_slice])

print('mean absolute error for non-null ages with simple overall median =', mae_static_median)
# fill in missing ages

combined_df['Age'] = combined_df['Age'].fillna(combined_df['age_pred'])



# drop the 'age_pred' column, we don't need anymore

combined_df.drop('age_pred', axis=1, inplace=True)



combined_df.head()
combined_df[combined_df['Embarked'].isnull()]
# counting the number of distinct values of Embarked per ticket #

combined_df[['Ticket','Embarked']][combined_df['Embarked'].notnull()].groupby(by='Ticket').Embarked.nunique().value_counts()
combined_df[combined_df['Ticket'] == '113572']
# filter 1st class passengers, group them by Embarked values 

combined_df.loc[combined_df['Pclass'] == 1][['Embarked','PassengerId']].groupby(by='Embarked').count()
# get number of tickets starting by '113'

print('Number of tickets starting by 113 =', combined_df.loc[combined_df['Ticket'].str.match('^113'), 'Ticket'].count(), '\n')



# get class of tickets starting by '113'

print('Class of tickets starting by 113: \n', combined_df.loc[combined_df['Ticket'].str.match('^113'), ['Pclass', 'PassengerId']].groupby(by='Pclass').count(), '\n')



# get Embarked value of passengers which tiket starts by '113'

print('Embarked values of tickets starting by 113: \n', combined_df.loc[combined_df['Ticket'].str.match('^113') , 'Embarked'].value_counts() / combined_df['Ticket'].str.match('^113').sum(), '\n')
# replace missing Embarked value with 'S'

combined_df['Embarked'].fillna('S', inplace=True)
combined_df[combined_df['Fare'].isnull()]
# get passengers with ticket # = '3701'

combined_df[combined_df['Ticket'] == '3701']
# get the median fare for a 3rd class passenger embarking at S (= 8.5)

median_fare_3rd_S = combined_df.loc[(combined_df['Embarked'] == 'S') & (combined_df['Pclass'] == 3) , 'Fare'].median()



# impute fare price

combined_df.loc[combined_df['Fare'].isnull(), 'Fare'] = median_fare_3rd_S
# set Sex to 0 for female and 1 for male

combined_df['Sex'] = combined_df['Sex'].astype("category")

combined_df['Sex'].cat.categories = [0,1]

combined_df['Sex'] = combined_df['Sex'].astype("int")
print('Under 14 y.o. survivded pct:', combined_df[combined_df.Age < 14].Survived.mean())

print('Under 15 y.o. survivded pct:', combined_df[combined_df.Age < 15].Survived.mean())

print('Under 16 y.o. survivded pct:', combined_df[combined_df.Age < 16].Survived.mean())

print('Under 17 y.o. survivded pct:', combined_df[combined_df.Age < 17].Survived.mean())

print('Under 18 y.o. survivded pct:', combined_df[combined_df.Age < 18].Survived.mean())
combined_df['age_group'] = np.nan # create new empty column

combined_df.loc[combined_df['Age'] < 16, 'age_group'] = 0

combined_df.loc[(combined_df['Age'] >= 16) & (combined_df['Age'] < 32), 'age_group'] = 0.25

combined_df.loc[(combined_df['Age'] >= 32) & (combined_df['Age'] < 48), 'age_group'] = 0.50

combined_df.loc[(combined_df['Age'] >= 48) & (combined_df['Age'] < 64), 'age_group'] = 0.75

combined_df.loc[combined_df['Age'] >= 64, 'age_group'] = 1
combined_df['Pclass'] = combined_df['Pclass'].astype("int")
# create new feature that contains deck number, by extracting the first character of the Cabin feature

# if the string contains multiple cabin #, we will catch only the first one, assuming the other cabin # of the string are part of the same deck

combined_df['deck'] = combined_df['Cabin'].str.extract('([a-zA-Z])', expand=False)



combined_df[combined_df['deck'].notnull()].head(5)
# get survival rate, sex, age and class per deck

deck_notnull_slice = combined_df.loc[combined_df['deck'].notnull(), :]

deck_analysis_df = deck_notnull_slice.groupby(by='deck').agg({'Survived':'mean', \

                                                              'PassengerId':'count', \

                                                              'Sex':'mean', \

                                                              'Age':'mean', \

                                                               'Pclass':'mean'})



# rename columns

deck_analysis_df.columns = ['pct_surv', 'count', 'pct_male', 'age_mean', 'class_mean']



# moves 'deck' from the index to a column

deck_analysis_df.reset_index(inplace=True)



# compute survival rate, median age, pct male

test_df=combined_df.iloc[0:890, :]

surv_rate_1st = test_df[test_df.Pclass == 1].Survived.mean()

surv_rate_2nd = test_df[test_df.Pclass == 2].Survived.mean()

surv_rate_3rd = test_df[test_df.Pclass == 3].Survived.mean()

o_median_age = combined_df.Age.median()

o_pct_male = combined_df.Sex.mean()

o_class_mean = combined_df.Pclass.mean()



# plot

fig = plt.figure(figsize=(14,14))



plt.subplot(221)

sns.barplot(x='deck', y='pct_surv', data=deck_analysis_df)

surv_rate_1st_line = plt.axhline(y=surv_rate_1st, xmin=0, xmax=1, color='g', label='survival rate 1st class')

surv_rate_2nd_line = plt.axhline(y=surv_rate_2nd, xmin=0, xmax=1, color='b', label='survival rate 2nd class')

surv_rate_3rd_line = plt.axhline(y=surv_rate_3rd, xmin=0, xmax=1, color='r',  label='survival rate 3rd class')

plt.legend(handles=[surv_rate_1st_line, surv_rate_2nd_line, surv_rate_3rd_line])



plt.subplot(222)

sns.barplot(x='deck', y='count', data=deck_analysis_df)



plt.subplot(234)

sns.barplot(x='deck', y='pct_male', data=deck_analysis_df)

pct_male_line = plt.axhline(y=o_pct_male, xmin=0, xmax=1, label='overall pct of males')

plt.legend(handles=[pct_male_line])



plt.subplot(235)

sns.barplot(x='deck', y='age_mean', data=deck_analysis_df)

median_age_line = plt.axhline(y=o_median_age, xmin=0, xmax=1, label='overall median age')

plt.legend(handles=[median_age_line])



plt.subplot(236)

sns.barplot(x='deck', y='class_mean', data=deck_analysis_df)

class_mean_line = plt.axhline(y=o_class_mean, xmin=0, xmax=1, label='overall class mean')

plt.legend(handles=[class_mean_line])



sns.plt.show()



deck_analysis_df
# create a new feature 'deck_down' which value is True when 'deck' is not null

combined_df['deck_known'] = combined_df['deck'].notnull()
# create feature for Age scaled down to [0, 1]

combined_df['age_scaled'] = combined_df.Age - combined_df.Age.min()

combined_df['age_scaled'] = combined_df.age_scaled / combined_df.Age.max()



# create feature for Fare scaled down to [0, 1]

# use logarithm because we have extreme Fare prices

combined_df['fare_scaled'] = np.log10(combined_df['Fare'] + 1)

combined_df['fare_scaled'] = (combined_df.fare_scaled - combined_df.fare_scaled.min()) / combined_df.fare_scaled.max()



# create boolean feature to track passengers whose cabin # is known (those passengers are probably more likely to have survived)

combined_df['cabin_known'] = combined_df.Cabin.notnull()



# create feature for the total number of family members

combined_df['family_size'] = combined_df.SibSp + combined_df.Parch



# create boolean feature for passengers that travelled alone

combined_df['is_alone'] = combined_df['family_size'] == 0



combined_df.head(5)
title_mapping = {"mr": 0, "notable": 0.2, "master": 0.3, "miss": 0.4, "mrs": 0.5}

combined_df['title_encoded'] = combined_df['title'].map(title_mapping)

combined_df['title_encoded'] = combined_df['title_encoded'].fillna(0)
combined_df.head (5)
# create dummy features for the Embarked feature

embarked_dummies_df = pd.get_dummies(combined_df.Embarked, prefix='embarked')

combined_df = pd.concat([combined_df, embarked_dummies_df], axis=1)



# drop the 'Embarked_Q' dummy feature, because if a passenger did not embarked at C or S, he has necessarily embarked at Q.

combined_df.drop('embarked_Q', axis=1, inplace=True)
cols = ['Pclass', 'Sex', 'age_scaled', 'fare_scaled', 'title_encoded', \

         'is_alone', 'embarked_C', 'embarked_S', 'deck_known', 'Survived']



train_df = combined_df.loc[:890, cols]

test_df = combined_df.loc[891:, cols]

test_df.drop('Survived', axis=1, inplace=True)



ax = plt.subplots(figsize =(12, 12))

heat = sns.heatmap(train_df.corr(), vmax=1.0, square=True, annot=True)



sns.plt.show()
y_train = train_df['Survived']

X_train = train_df.drop('Survived', axis=1)

X_test = test_df
# instantiate KNN classifier

knn = KNeighborsClassifier()



# define the parameters values that should be searched

k_range = range(1, 40)



# create a parameter grid: map the parameter names to the values that should be searched

param_grid = {'n_neighbors' : k_range}



# instantiate the grid

grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
# fit the grid

grid.fit(X_train,y_train)
# get mean score for each value of params (n=1, n=2, ...)

mean_test_score = grid.cv_results_['mean_test_score']



# plot the results

plt.plot(k_range, mean_test_score)

plt.xlabel('Value of k for KNN')

plt.ylabel('Cross-validated accuracy')

sns.plt.show()



# display best param and corresponding score

result_str = "Best params: {}; score: {}"

print(result_str.format(grid.best_params_, grid.best_score_))
# create an instance of a KNN classifier to make our prediction

# testing with different values of hyper parameter n_neighbors (the [15-30] look like a good range)

knn_sub = KNeighborsClassifier(n_neighbors=25)



# fit our model all the WHOLE training set

knn_sub.fit(X_train, y_train)



# make our prediction

y_pred = knn_sub.predict(X_test)



# insert our prediction in a DataFrame with the PassengerId

submit = pd.DataFrame({'PassengerId' : combined_df.loc[891:,'PassengerId'],

                         'Survived': y_pred.astype(int)})
submit.info()
submit.head(10)
submit.to_csv("prediction-20170627-Q.csv", index=False)