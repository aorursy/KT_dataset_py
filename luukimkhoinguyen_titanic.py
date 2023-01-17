import pandas as pd

import numpy as np

from functools import reduce 

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# for regression problems

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



# for classification

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



# to split and standarize the datasets

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# to evaluate regression models

from sklearn.metrics import mean_squared_error



# to evaluate classification models

from sklearn.metrics import roc_auc_score



import warnings

warnings.filterwarnings('ignore')
texts = [['i', 'have', 'a', 'cat'], 

        ['he', 'have', 'a', 'dog'], 

        ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]



dictionary = list(enumerate(set(list(reduce(lambda x, y: x + y, texts)))))



print (dictionary)



def vectorize(text): 

    vector = np.zeros(len(dictionary)) 

    for i, word in dictionary: 

        num = 0 

        for w in text: 

            if w == word: 

                num += 1 

        if num: 

            vector[i] = num 

    return vector



for t in texts: 

    print(vectorize(t))
print(os.listdir("../input"))
data = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

data.head()
data = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

data.head()
# you can determine the total number of missing values using

# the isnull method plus the sum method on the dataframe

data.isnull().sum()
# alternatively, you can call the mean method after isnull

# to visualise the percentage of the dataset that 

# contains missing values for each variable



data.isnull().mean()
# we create a dummy variable that indicates whether the value

# of the variable cabin is missing



data['AMT_REQ_CREDIT_BUREAU_WEEK_null'] = np.where(data.AMT_REQ_CREDIT_BUREAU_WEEK.isnull(), 1, 0)



# find percentage of null values

data.AMT_REQ_CREDIT_BUREAU_WEEK.mean()
# and then we evaluate the mean of the missing values in

# cabin for the people who survived vs the non-survivors.



# group data by Survived vs Non-Survived

# and find nulls for cabin

data.groupby(['TARGET'])['AMT_REQ_CREDIT_BUREAU_WEEK_null'].mean()
# we repeat the exercise for the variable age:

# First we create a dummy variable that indicates

# whether the value of the variable Age is missing



data['AMT_REQ_CREDIT_BUREAU_WEEK_null'] = np.where(data.AMT_REQ_CREDIT_BUREAU_WEEK.isnull(), 1, 0)



# and then look at the mean in the different survival groups:

# there are more NaN for the people who did not survive

data.groupby(['TARGET'])['AMT_REQ_CREDIT_BUREAU_WEEK_null'].mean()
# slice the dataframe to show only those observations

# with missing value for Embarked



data[data.OWN_CAR_AGE.isnull()]
# let's look at the percentage of NA



data.isnull().mean()
# let's separate into training and testing set



X_train, X_test, y_train, y_test = train_test_split(data, data.TARGET, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
X_train.AMT_REQ_CREDIT_BUREAU_YEAR.median()
# let's make a function to create 2 variables from Age:

# one filling NA with median, and another one filling NA with zeroes



def impute_na(df, variable, median):

    df[variable+'_median'] = df[variable].fillna(median)

    df[variable+'_zero'] = df[variable].fillna(0) 
impute_na(X_train, 'AMT_REQ_CREDIT_BUREAU_YEAR', X_train.AMT_REQ_CREDIT_BUREAU_YEAR.median())

X_train.head(15)
impute_na(X_test, 'AMT_REQ_CREDIT_BUREAU_YEAR', X_train.AMT_REQ_CREDIT_BUREAU_YEAR.median())
# we can see that the distribution has changed slightly with now more values accumulating towards the median

fig = plt.figure()

ax = fig.add_subplot(111)

X_train['AMT_REQ_CREDIT_BUREAU_YEAR'].plot(kind='kde', ax=ax)

X_train.AMT_REQ_CREDIT_BUREAU_YEAR_median.plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()

ax.legend(lines, labels, loc='best')
# filling NA with zeroes creates a peak of population around 0, as expected

fig = plt.figure()

ax = fig.add_subplot(111)

X_train['AMT_REQ_CREDIT_BUREAU_YEAR'].plot(kind='kde', ax=ax)

X_train.AMT_REQ_CREDIT_BUREAU_YEAR_zero.plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()

ax.legend(lines, labels, loc='best')
# Let's compare the performance of Logistic Regression using Age filled with zeros or alternatively the median



# model on NA imputed with zeroes

logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_zero']], y_train)

print('Train set zero imputation')

pred = logit.predict_proba(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_zero']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set zero imputation')

pred = logit.predict_proba(X_test[['AMT_REQ_CREDIT_BUREAU_YEAR_zero']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

print()



# model on NA imputed with median

logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_median']], y_train)

print('Train set median imputation')

pred = logit.predict_proba(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_median']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set median imputation')

pred = logit.predict_proba(X_test[['AMT_REQ_CREDIT_BUREAU_YEAR_median']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
# load the Titanic Dataset with a few variables for demonstration



data = pd.read_csv('titanic.csv', usecols = ['Age', 'Fare', 'Survived'])

data.head()
# let's separate into training and testing set



X_train, X_test, y_train, y_test = train_test_split(data, data.TARGET, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
def impute_na(df, variable, median):

    df[variable+'_median'] = df[variable].fillna(median)

    df[variable+'_zero'] = df[variable].fillna(0)

    

    # random sampling

    df[variable+'_random'] = df[variable]

    # extract the random sample to fill the na

    random_sample = X_train[variable].dropna().sample(df[variable].isnull().sum(), random_state=0)

    # pandas needs to have the same index in order to merge datasets

    random_sample.index = df[df[variable].isnull()].index

    df.loc[df[variable].isnull(), variable+'_random'] = random_sample
impute_na(X_train, 'AMT_REQ_CREDIT_BUREAU_YEAR', X_train.AMT_REQ_CREDIT_BUREAU_YEAR.median())

X_train.head(20)
impute_na(X_test, 'AMT_REQ_CREDIT_BUREAU_YEAR', X_train.AMT_REQ_CREDIT_BUREAU_YEAR.median())
# we can see that the distribution of the variable after filling NA is exactly the same as that one before filling NA

fig = plt.figure()

ax = fig.add_subplot(111)

X_train['AMT_REQ_CREDIT_BUREAU_YEAR'].plot(kind='kde', ax=ax)

X_train.AMT_REQ_CREDIT_BUREAU_YEAR_random.plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()

ax.legend(lines, labels, loc='best')
# let's compare the performance of logistic regression on Age NA imputed by zeroes, or median or random sampling



logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_zero']], y_train)

print('Train set zero imputation')

pred = logit.predict_proba(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_zero']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set zero imputation')

pred = logit.predict_proba(X_test[['AMT_REQ_CREDIT_BUREAU_YEAR_zero']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

print()

logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_median']], y_train)

print('Train set median imputation')

pred = logit.predict_proba(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_median']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set median imputation')

pred = logit.predict_proba(X_test[['AMT_REQ_CREDIT_BUREAU_YEAR_median']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

print()

logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_random']], y_train)

print('Train set random sample imputation')

pred = logit.predict_proba(X_train[['AMT_REQ_CREDIT_BUREAU_YEAR_random']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set random sample imputation')

pred = logit.predict_proba(X_test[['AMT_REQ_CREDIT_BUREAU_YEAR_random']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
# load the Titanic Dataset with a few variables for demonstration



data = pd.read_csv('../input/home-credit-default-risk/application_train.csv', usecols = ['OWN_CAR_AGE','TARGET'])

data.head()
# let's separate into training and testing set



X_train, X_test, y_train, y_test = train_test_split(data, data.TARGET, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
# create variable indicating missingness



X_train['OWN_CAR_AGE_NA'] = np.where(X_train['OWN_CAR_AGE'].isnull(), 1, 0)

X_test['OWN_CAR_AGE_NA'] = np.where(X_test['OWN_CAR_AGE'].isnull(), 1, 0)



X_train.head()
# let's replace the NA with the median value in the training set

X_train['OWN_CAR_AGE'].fillna(X_train.OWN_CAR_AGE.median(), inplace=True)

X_test['OWN_CAR_AGE'].fillna(X_train.OWN_CAR_AGE.median(), inplace=True)



X_train.head(20)
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train))

X_test = pd.DataFrame(scaler.transform(X_test))



X_train.columns = ['TARGET','OWN_CAR_AGE','OWN_CAR_AGE_NA']

X_test.columns = ['TARGET','OWN_CAR_AGE','OWN_CAR_AGE_NA']
# we compare the models built using Age filled with median, vs Age filled with median + additional

# variable indicating missingness



logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['OWN_CAR_AGE']], y_train)

print('Train set')

pred = logit.predict_proba(X_train[['OWN_CAR_AGE']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set')

pred = logit.predict_proba(X_test[['OWN_CAR_AGE']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))



logit = LogisticRegression(random_state=44, C=1000) # c big to avoid regularization

logit.fit(X_train[['OWN_CAR_AGE','OWN_CAR_AGE_NA']], y_train)

print('Train set')

pred = logit.predict_proba(X_train[['OWN_CAR_AGE','OWN_CAR_AGE_NA']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set')

pred = logit.predict_proba(X_test[['OWN_CAR_AGE','OWN_CAR_AGE_NA']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
# load the Titanic Dataset with a few variables for demonstration



data = pd.read_csv('titanic.csv', usecols = ['Age', 'Fare','Survived'])

data.head()
# let's separate into training and testing set



X_train, X_test, y_train, y_test = train_test_split(data, data.TARGET, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
X_train.OWN_CAR_AGE.hist(bins=50)
# far end of the distribution

X_train.OWN_CAR_AGE.mean()+3*X_train.OWN_CAR_AGE.std()
# we see that there are a few outliers for Age, according to its distribution

# these outliers will be masked when we replace NA by values at the far end 

# see below



sns.boxplot('OWN_CAR_AGE', data=data)
def impute_na(df, variable, median, extreme):

    df[variable+'_far_end'] = df[variable].fillna(extreme)

    df[variable].fillna(median, inplace=True)
# let's replace the NA with the median value in the training and testing sets

impute_na(X_train, 'OWN_CAR_AGE', X_train.OWN_CAR_AGE.median(), X_train.OWN_CAR_AGE.mean()+3*X_train.OWN_CAR_AGE.std())

impute_na(X_test, 'OWN_CAR_AGE', X_train.OWN_CAR_AGE.median(), X_train.OWN_CAR_AGE.mean()+3*X_train.OWN_CAR_AGE.std())



X_train.head(20)
# we see an accumulation of values around the median for the median imputation

X_train.OWN_CAR_AGE.hist(bins=50)
# we see an accumulation of values at the far end for the far end imputation



X_train.OWN_CAR_AGE_far_end.hist(bins=50)
# indeed, far end imputation now indicates that there are no outliers in the variable

sns.boxplot('OWN_CAR_AGE_far_end', data=X_train)
# on the other hand, replacing values by the median, now generates the impression of a higher

# amount of outliers



sns.boxplot('OWN_CAR_AGE', data=X_train)
# we compare the models built using Age filled with median, vs Age filled with values at the far end of the distribution

# variable indicating missingness



logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['OWN_CAR_AGE']], y_train)

print('Train set')

pred = logit.predict_proba(X_train[['OWN_CAR_AGE']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set')

pred = logit.predict_proba(X_test[['OWN_CAR_AGE']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))



logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['OWN_CAR_AGE_far_end']], y_train)

print('Train set')

pred = logit.predict_proba(X_train[['OWN_CAR_AGE_far_end']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set')

pred = logit.predict_proba(X_test[['OWN_CAR_AGE_far_end']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
# load the Titanic Dataset with a few variables for demonstration



data = pd.read_csv('titanic.csv', usecols = ['Age', 'Fare','Survived'])

data.head()
# let's separate into training and testing set



X_train, X_test, y_train, y_test = train_test_split(data, data.TARGET, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
def impute_na(df, variable):

    df[variable+'_zero'] = df[variable].fillna(0)

    df[variable+'_hundred']= df[variable].fillna(100)
# let's replace the NA with the median value in the training set

impute_na(X_train, 'OWN_CAR_AGE')

impute_na(X_test, 'OWN_CAR_AGE')



X_train.head(20)
# we compare the models built using Age filled with zero, vs Age filled with 100



logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['OWN_CAR_AGE_zero']], y_train)

print('Train set')

pred = logit.predict_proba(X_train[['OWN_CAR_AGE_zero']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set')

pred = logit.predict_proba(X_test[['OWN_CAR_AGE_zero']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))



logit = RandomForestClassifier(random_state=44) # c big to avoid regularization

logit.fit(X_train[['OWN_CAR_AGE_hundred']], y_train)

print('Train set')

pred = logit.predict_proba(X_train[['OWN_CAR_AGE_hundred']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set')

pred = logit.predict_proba(X_test[['OWN_CAR_AGE_hundred']])

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
import pandas as pd



data = pd.read_csv('../input/home-credit-default-risk/application_train.csv', usecols=['CODE_GENDER'])

data.head()
# one hot encoding



pd.get_dummies(data).head()
# for better visualisation

pd.concat([data, pd.get_dummies(data)], axis=1).head()
data = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

data.head()
# let's make a copy of the dataset, in which we encode the categorical variables using OHE



data_OHE = pd.concat([data[['TARGET', 'OWN_CAR_AGE', 'CNT_CHILDREN']], # numerical variables 

                      pd.get_dummies(data.CODE_GENDER),   # binary categorical variable

                      pd.get_dummies(data.FLAG_OWN_REALTY)],  # k categories in categorical

                    axis=1)



data_OHE.head()
# and now let's separate into train and test set



X_train, X_test, y_train, y_test = train_test_split(data_OHE[['OWN_CAR_AGE', 'CNT_CHILDREN', 'F', 'M', 'XNA', 'N', 'Y']].fillna(0),

                                                    data_OHE.TARGET,

                                                    test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
def impute_na(df, variable, extreme):

    df[variable].fillna(extreme, inplace=True)

    

impute_na(X_train, 'OWN_CAR_AGE', X_train.OWN_CAR_AGE.mean()+3*X_train.OWN_CAR_AGE.std())

impute_na(X_test, 'OWN_CAR_AGE', X_train.OWN_CAR_AGE.mean()+3*X_train.OWN_CAR_AGE.std())
# and finally a logistic regression



logit = RandomForestClassifier(random_state=44)

logit.fit(X_train, y_train)

print('Train set')

pred = logit.predict_proba(X_train)

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set')

pred = logit.predict_proba(X_test)

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
data = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

data.head()
X_train, X_test, y_train, y_test = train_test_split(data[['OWN_CAR_AGE', 'CNT_CHILDREN']].fillna(0),

                                                    data.TARGET,

                                                    test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
# And now let's replace each label in X2 by its count



# first we make a dictionary that maps each label to the counts

X_frequency_map = X_train.OWN_CAR_AGE.value_counts().to_dict()



# and now we replace X2 labels both in train and test set with the same map

X_train.Sex = X_train.OWN_CAR_AGE.map(X_frequency_map)

X_test.Sex = X_test.OWN_CAR_AGE.map(X_frequency_map)



X_train.head()
# And now let's replace each label in X2 by its count



# first we make a dictionary that maps each label to the counts

X_frequency_map = X_train.CNT_CHILDREN.value_counts().to_dict()



# and now we replace X2 labels both in train and test set with the same map

X_train.Embarked = X_train.CNT_CHILDREN.map(X_frequency_map)

X_test.Embarked = X_test.CNT_CHILDREN.map(X_frequency_map)



X_train.head()
# and finally a logistic regression



logit = RandomForestClassifier(random_state=44)

logit.fit(X_train, y_train)

print('Train set')

pred = logit.predict_proba(X_train)

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

print('Test set')

pred = logit.predict_proba(X_test)

print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
# let's load again the titanic dataset



data = pd.read_csv('../input/home-credit-default-risk/application_train.csv', usecols=['OWN_CAR_AGE', 'TARGET'])

data.head()
# let's first fill NA values with an additional label



data.OWN_CAR_AGE.fillna('OWN_CAR_AGE', inplace=True)

data['OWN_CAR_AGE'] = data['OWN_CAR_AGE'].astype(str).str[0]

data.OWN_CAR_AGE.unique()
data.head()
# Let's separate into train and test set



X_train, X_test, y_train, y_test = train_test_split(

    data[['OWN_CAR_AGE', 'TARGET']], data.TARGET, test_size=0.3, random_state=0)



X_train.shape, X_test.shape
# now we order the labels according to the mean target value



X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean().sort_values()
# and now we create a dictionary that maps each label to the number

ordered_labels = X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean().sort_values().index

ordinal_label = {k:i for i, k in enumerate(ordered_labels, 0)} 

ordinal_label
# replace the labels with the ordered numbers

# both in train and test set (note that we created the dictionary only using the training set)



X_train['OWN_CAR_AGE_ordered'] = X_train.OWN_CAR_AGE.map(ordinal_label)

X_test['OWN_CAR_AGE_ordered'] = X_test.OWN_CAR_AGE.map(ordinal_label)
# check the results



X_train.head()
# let's inspect the newly created monotonic relationship with the target



#first we plot the original variable for comparison, there is no monotonic relationship



fig = plt.figure()

fig = X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean().plot()

fig.set_title('Normal relationship between variable and target')

fig.set_ylabel('TARGET')
# plot the transformed result: the monotonic variable



fig = plt.figure()

fig = X_train.groupby(['OWN_CAR_AGE_ordered'])['TARGET'].mean().plot()

fig.set_title('Monotonic relationship between variable and target')

fig.set_ylabel('Survived')
# let's load again the titanic dataset



data = pd.read_csv('../input/home-credit-default-risk/application_train.csv', usecols=['OWN_CAR_AGE', 'TARGET'])

data.head()
# let's first fill NA values with an additional label



data.OWN_CAR_AGE.fillna('Missing', inplace=True)

data['OWN_CAR_AGE'] = data['OWN_CAR_AGE'].astype(str).str[0]
# Let's separate into training and testing set



X_train, X_test, y_train, y_test = train_test_split(data[['OWN_CAR_AGE', 'TARGET']], data.TARGET, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
# let's calculate the target frequency for each label



X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean()

ordered_labels = X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean().to_dict()

X_train['OWN_CAR_AGE_ordered'] = X_train.OWN_CAR_AGE.map(ordered_labels)

X_test['OWN_CAR_AGE_ordered'] = X_test.OWN_CAR_AGE.map(ordered_labels)

X_train.head()
# plot the original variable



fig = plt.figure()

fig = X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean().plot()

fig.set_title('Normal relationship between variable and target')

fig.set_ylabel('Survived')
# plot the transformed result: the monotonic variable



fig = plt.figure()

fig = X_train.groupby(['OWN_CAR_AGE_ordered'])['TARGET'].mean().plot()

fig.set_title('Monotonic relationship between variable and target')

fig.set_ylabel('Survived')
# let's load again the titanic dataset



data = pd.read_csv('../input/home-credit-default-risk/application_train.csv', usecols=['OWN_CAR_AGE', 'TARGET'])

data.head()
# let's first fill NA values with an additional label



data.OWN_CAR_AGE.fillna('Missing', inplace=True)

data['OWN_CAR_AGE'] = data['OWN_CAR_AGE'].astype(str).str[0]
# Let's separate into training and testing set



X_train, X_test, y_train, y_test = train_test_split(data[['OWN_CAR_AGE', 'TARGET']],

                                                    data.TARGET, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
# now let's  calculate the probability of target = 0 (people who did not survive)

prob_df = X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean()

prob_df = pd.DataFrame(prob_df)

prob_df['Difficult'] = 1-prob_df.TARGET

prob_df['ratio'] = prob_df.TARGET/prob_df.Difficult

ordered_labels = prob_df['ratio'].to_dict()

X_train['OWN_CAR_AGE_ordered'] = X_train.OWN_CAR_AGE.map(ordered_labels)

X_test['OWN_CAR_AGE_ordered'] = X_test.OWN_CAR_AGE.map(ordered_labels)

X_train.head()
# plot the original variable



fig = plt.figure()

fig = X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean().plot()

fig.set_title('Normal relationship between variable and target')

fig.set_ylabel('TARGET')
# plot the transformed result: the monotonic variable



fig = plt.figure()

fig = X_train.groupby(['OWN_CAR_AGE_ordered'])['TARGET'].mean().plot()

fig.set_title('Monotonic relationship between variable and target')

fig.set_ylabel('TARGET')
# let's load again the titanic dataset



data = pd.read_csv('../input/home-credit-default-risk/application_train.csv', usecols=['OWN_CAR_AGE', 'TARGET'])

data.head()
# let's first fill NA values with an additional label



data.OWN_CAR_AGE.fillna('Missing', inplace=True)

data['OWN_CAR_AGE'] = data['OWN_CAR_AGE'].astype(str).str[0]
# Let's divide into train and test set



X_train, X_test, y_train, y_test = train_test_split(data[['OWN_CAR_AGE', 'TARGET']], data.TARGET, test_size=0.3,

                                                    random_state=0)

X_train.shape, X_test.shape
# and now the probability of target = 0 

# and we add it to the dataframe



prob_df = X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean()

prob_df = pd.DataFrame(prob_df)

prob_df['Difficult'] = 1-prob_df.TARGET

# since the log of zero is not defined, let's set this number to something small and non-zero



prob_df.loc[prob_df.TARGET == 0, 'TARGET'] = 0.00001

prob_df['WoE'] = np.log(prob_df.TARGET/prob_df.Difficult)

ordered_labels = prob_df['WoE'].to_dict()



X_train['OWN_CAR_AGE_ordered'] = X_train.OWN_CAR_AGE.map(ordered_labels)

X_test['OWN_CAR_AGE_ordered'] = X_test.OWN_CAR_AGE.map(ordered_labels)



X_train.head()
# plot the original variable



fig = plt.figure()

fig = X_train.groupby(['OWN_CAR_AGE'])['TARGET'].mean().plot()

fig.set_title('Normal relationship between variable and target')

fig.set_ylabel('Survived')
# plot the transformed result: the monotonic variable



fig = plt.figure()

fig = X_train.groupby(['OWN_CAR_AGE_ordered'])['TARGET'].mean().plot()

fig.set_title('Monotonic relationship between variable and target')

fig.set_ylabel('Survived')