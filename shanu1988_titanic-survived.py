# to handle datasets

import pandas as pd

import numpy as np



# for text / string processing

import re



# for ploting

import matplotlib.pyplot as plt

%matplotlib inline



# To divide train and test set

#from sklearn.model_selection import train_test_split



# feature scaling

from sklearn.preprocessing import MinMaxScaler



# for tree binarisation

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



# To build the models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



# to evaluate the models

from sklearn.metrics import roc_auc_score

from sklearn import metrics



pd.pandas.set_option('Display.max_columns', None)



import warnings

warnings.filterwarnings('ignore')
# Load data

##### Load train and Test set



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.head()
train.shape, test.shape
train.dtypes, test.dtypes
# Let's get more information about data

print('Number of passengersID labels in train data:', len(train.PassengerId.unique()))

print('Number of passengersID labels in test data:', len(test.PassengerId.unique()))



print('Number of passengers on Titanic:', len(train), len(test))
# find categorical variables



categorical = [var for var in train.columns if train[var].dtypes=='O']

print('There are {} categorical variables in train set'. format(len(categorical)))



categorical = [var for var in test.columns if test[var].dtypes=='O']

print('There are {} categorical variables in test set'. format(len(categorical)))
# find the numarical variables



numerical = [var for var in train.columns if train[var].dtypes!='O']

print('there are {} numerical variables in train set'.format(len(numerical)))



numerical = [var for var in test.columns if test[var].dtypes!='O']

print('there are {} numerical variables in test set'.format(len(numerical)))
# view of categorical variables

train[categorical].head() 
test[categorical].head()
# view of numerical variables

train[numerical].head()
test[numerical].head()
# let's visualise the values of the discrete variables

for var in ['Pclass','SibSp','Parch']:

    print(var, 'values:', train[var].unique())
# let's visualise the percentage of missing values



train.isnull().mean(), test.isnull().mean()
numerical = [var for var in numerical if var not in['Survived','PassengerId']]

numerical
#### let's make boxplots to visualise outliers in the continuous variables in test data 

##### Age and Fare

#Age

plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = train.boxplot(column='Age')

fig.set_title("")

fig.set_ylabel('Age as in train set')



# Fare

plt.subplot(1, 2, 2)

fig = train.boxplot(column='Fare')

fig.set_title("")

fig.set_ylabel('Fare as in train set')
#### let's make boxplots to visualise outliers in the continuous variables in test data 

##### Age and Fare

plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = test.boxplot(column='Age')

fig.set_title("")

fig.set_ylabel('Age as in Test set')



# Fare

plt.subplot(1, 2, 2)

fig = train.boxplot(column='Fare')

fig.set_title("")

fig.set_ylabel('Fare as in test set')
# Plot the distributions to find out if they are Gaussian or skewed in train data.

# Depending on the distribution, we will use the normal assumption or the interquantile

# range to find outliers



plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = train.Age.hist(bins=20)

fig.set_title('Number of passenger')

fig.set_ylabel('Age')





plt.subplot(1, 2, 2)

fig = train.Fare.hist(bins=20)

fig.set_title('Number of passenger')

fig.set_ylabel('Fare')
# Plot the distributions to find out if they are Gaussian or skewed in test data.



plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = test.Age.hist(bins=20)

fig.set_title('Number of passenger')

fig.set_ylabel('Age')





plt.subplot(1, 2, 2)

fig = test.Fare.hist(bins=20)

fig.set_title('Number of passenger')

fig.set_ylabel('Fare')
# find outliers in train data

#Age

upper_boundary = train.Age.mean() + 3* train.Age.std()

lower_boundary = train.Age.mean() - 3* train.Age.std()

print('Age outliers are values in train set < {lowerboundary} or > {upperboundary}'.format(lowerboundary = lower_boundary, upperboundary = upper_boundary))



# Fare

IQR = train.Fare.quantile(0.75) - train.Fare.quantile(0.25)

lower_fence = train.Fare.quantile(0.25) - (IQR * 3)

upper_fence = train.Fare.quantile(0.75) + (IQR * 3)

print('Fare outliers are values in train set < {lowerboundary} or > {upperboundary}'.format(lowerboundary = lower_fence, upperboundary=upper_fence))





# find outliers in test data

#Age

upper_boundary = test.Age.mean() + 3* test.Age.std()

lower_boundary = test.Age.mean() - 3* test.Age.std()

print('Age outliers are values in test set < {lowerboundary} or > {upperboundary}'.format(lowerboundary = lower_boundary, upperboundary = upper_boundary))



# Fare

IQR = train.Fare.quantile(0.75) - test.Fare.quantile(0.25)

lower_fence = test.Fare.quantile(0.25) - (IQR * 3)

upper_fence = test.Fare.quantile(0.75) + (IQR * 3)

print('Fare outliers are values in test set < {lowerboundary} or > {upperboundary}'.format(lowerboundary = lower_fence, upperboundary=upper_fence))
# outlies in discrete variables in train set



for var in ["Pclass", "SibSp","Parch"]:

  print(train[var].value_counts()/ np.float(len(train)))
# outlies in discrete variables in train set  

for var in ["Pclass", "SibSp","Parch"]:

  print(test[var].value_counts()/ np.float(len(test)))  
for var in categorical:

  print(var, 'contains', len(train[var].unique()),'labels in train data')  
for var in categorical:

  print(var, 'contains', len(test[var].unique()),'labels in test data') 
# Cabin

train['Cabin_numerical'] = train.Cabin.str.extract('(\d+)') # extract number from string

train['Cabin_numerical'] = train['Cabin_numerical'].astype('float') # passes the abouv variable to float type



train['Cabin_categorical'] = train['Cabin'].str[0] # captures first letter of string (the letter for cabin)



# same for test data

# Cabin

test['Cabin_numerical'] = test.Cabin.str.extract('(\d+)') # extract number from string

test['Cabin_numerical'] = test['Cabin_numerical'].astype('float') # passes the abouv variable to float type



test['Cabin_categorical'] = test['Cabin'].str[0] # captures first letter of string (the letter for cabin)



train[['Cabin','Cabin_numerical','Cabin_categorical']].head(),test[['Cabin','Cabin_numerical','Cabin_categorical']].head(20)
# drop the original variable



train.drop(labels='Cabin', inplace=True, axis=1)

test.drop(labels='Cabin', inplace=True, axis=1)
# Ticket for train set

# extract the last bit of ticket as number

train['Ticket_numerical'] = train.Ticket.apply(lambda s: s.split()[-1])

train['Ticket_numerical'] = np.where(train.Ticket_numerical.str.isdigit(), train.Ticket_numerical, np.nan)

train['Ticket_numerical'] = train['Ticket_numerical'].astype('float')



# Ticket for test set

test['Ticket_numerical'] = test.Ticket.apply(lambda s: s.split()[-1])

test['Ticket_numerical'] = np.where(test.Ticket_numerical.str.isdigit(), test.Ticket_numerical, np.nan)

test['Ticket_numerical'] = test['Ticket_numerical'].astype('float')
# Extract the first part of ticket as category from train

train['Ticket_categorical'] = train.Ticket.apply(lambda s: s.split()[0])

train['Ticket_categorical'] = np.where(train.Ticket_categorical.str.isdigit(), np.nan, train.Ticket_categorical)





# Extract the first part of ticket as category from test

test['Ticket_categorical'] = test.Ticket.apply(lambda s: s.split()[0])

test['Ticket_categorical'] = np.where(test.Ticket_categorical.str.isdigit(), np.nan, test.Ticket_categorical)



train[['Ticket_numerical', 'Ticket_categorical']].head()
test[['Ticket_numerical', 'Ticket_categorical']].head(20)
# let's explore the ticket categorical part a bit further

train.Ticket_categorical.unique(), test.Ticket_categorical.unique()
# it contains several labels, some of them seem very similar apart from the punctuation in both dataset

# I will try to reduce this number of labels a bit further



# remove non letter characters from string

text = train.Ticket_categorical.apply(lambda x: re.sub("[^a-zA-Z]",'', str(x)))



# To visualise the output and compare with inputs this is based on train data

pd.concat([text, train.Ticket_categorical], axis=1).head()
# To visualise the output and compare with inputs this is based on test data

text = test.Ticket_categorical.apply(lambda x: re.sub("[^a-zA-Z]",'', str(x)))



pd.concat([text, test.Ticket_categorical], axis=1).head(20)
# set to upper case: we reduce the number of labels quite a bit

text = text.str.upper()

text.unique()
# drop the original variable

train.drop(labels='Ticket', inplace=True, axis=1)

test.drop(labels='Ticket', inplace=True, axis=1)
def get_title(passenger):

  line = passenger

  if re.search('Mr', line):

    return 'Mr'

  elif re.search('Mrs', line):

    return 'Mrs'

  elif re.search('Miss', line):

    return 'Miss'

  elif re.search('Master', line):

    return 'Master'

  else:

    return 'Mr'
train['Title'] = train['Name'].apply(get_title)

test['Title'] = test['Name'].apply(get_title)



train[['Name','Title']].head(), test[['Name','Title']].head()
# drop the original variable form both dataset

train.drop(labels='Name', inplace=True, axis=1)

test.drop(labels='Name', inplace=True, axis=1)
# create a variable indicating family size (including the passenger) on both dataset

# sums siblings and parents



train['Family_size'] = train['SibSp'] + train['Parch']+1

print(train.Family_size.value_counts() / np.float(len(train)))



(train.Family_size.value_counts() / np.float(len(train))).plot.line()
test['Family_size'] = test['SibSp'] + test['Parch']+1

print(test.Family_size.value_counts() / np.float(len(test)))



# let's understand better

(test.Family_size.value_counts() / np.float(len(test))).plot.line()
# variable indicating if passenger was a mother

train['is_mother'] = np.where((train.Sex == 'female')&(train.Parch>=1)&(train.Age>18)&(train.Age<60),1,0)

test['is_mother'] = np.where((test.Sex == 'female')&(test.Parch>=1)&(test.Age>18)&(test.Age<60),1,0)

                              

                              

train.loc[train.is_mother==1,['Sex','Parch','Age','is_mother']].head()
print('Mothers were {} in the titanic as per train set '. format(train.is_mother.sum()))

print('Mothers were {} in the titanic as per test set '. format(test.is_mother.sum()))
# variable indicating if passenger was a father

train['is_father'] = np.where((train.Sex == 'male')&(train.Parch>=1)&(train.Age>20)&(train.Age<65),1,0)

test['is_father'] = np.where((test.Sex == 'male')&(test.Parch>=1)&(test.Age>20)&(test.Age<65),1,0)

                              

                              

train.loc[train.is_father==1,['Sex','Parch','Age','is_father']].head()
print('Fathers were {} in the titanic as per train set '. format(train.is_father.sum()))

print('Fathers were {} in the titanic as per test set '. format(test.is_father.sum()))
train[['Cabin_numerical','Ticket_numerical','is_mother','is_father']].isnull().mean()
test[['Cabin_numerical','Ticket_numerical','is_mother','is_father']].isnull().mean()
# Train Set

# first we plot the distributions to find out if they are Gaussian or skewed.

# Depending on the distribution, we will use the normal assumption or the interquantile

# range to find outliers





plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = train.Cabin_numerical.hist(bins=50)

fig.set_xlabel('Cabin Number')

fig.set_ylabel('Number of passengers')



plt.subplot(1, 2, 2)

fig = train.Ticket_numerical.hist(bins=50)

fig.set_xlabel('Cabin Number')

fig.set_ylabel('Number of passengers')
#Test Set

plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = test.Cabin_numerical.hist(bins=50)

fig.set_xlabel('Cabin Number')

fig.set_ylabel('Number of passengers')



plt.subplot(1, 2, 2)

fig = test.Ticket_numerical.hist(bins=50)

fig.set_xlabel('Cabin Number')

fig.set_ylabel('Number of passengers')
# Train Set

# let's visualise outliers with the boxplot and whiskers

plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = train.boxplot(column='Cabin_numerical')

fig.set_title('')

fig.set_ylabel('Cabin Number')



plt.subplot(1, 2, 2)

fig = train.boxplot(column='Ticket_numerical')

fig.set_title('')

fig.set_ylabel('Cabin Number')
#Test Set

# let's visualise outliers with the boxplot and whiskers

plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)

fig = test.boxplot(column='Cabin_numerical')

fig.set_title('')

fig.set_ylabel('Cabin Number')



plt.subplot(1, 2, 2)

fig = test.boxplot(column='Ticket_numerical')

fig.set_title('')

fig.set_ylabel('Cabin Number')
#Train Set

# Ticket Numerical

IQR = train.Ticket_numerical.quantile(0.75) - train.Ticket_numerical.quantile(0.25)

lower_fence = train.Ticket_numerical.quantile(0.25) - (IQR * 3)

upper_fence = train.Ticket_numerical.quantile(0.75) + (IQR * 3)



print('Ticket Number outliers are values < {lowerboundary} or {upperboundary}'.

     format(lowerboundary=lower_fence, upperboundary=upper_fence))

passengers = len(train[train.Ticket_numerical > upper_fence]) / np.float(len(train))



print('Number of passengers with ticket values higher than {upperboundary}: {passengers}'.format(upperboundary=upper_fence,\

                                                                                                 passengers=passengers))
#Test Set

# Cabin Numerical

IQR = test.Cabin_numerical.quantile(0.75) - test.Cabin_numerical.quantile(0.25)

lower_fence = test.Cabin_numerical.quantile(0.25) - (IQR * 3)

upper_fence = test.Cabin_numerical.quantile(0.75) + (IQR * 3)



print('Cabin Number outliers are values < {lowerboundary} or {upperboundary}'.

     format(lowerboundary=lower_fence, upperboundary=upper_fence))

passengers = len(test[test.Cabin_numerical > upper_fence]) / np.float(len(test))



print('Number of passengers in Cabin more than {upperboundary}: {passengers}'.format(upperboundary=upper_fence,\

                                                                                                 passengers=passengers))





# Ticket Numerical

IQR = test.Ticket_numerical.quantile(0.75) - test.Ticket_numerical.quantile(0.25)

lower_fence = test.Ticket_numerical.quantile(0.25) - (IQR * 3)

upper_fence = test.Ticket_numerical.quantile(0.75) + (IQR * 3)



print('Ticket Number outliers are values < {lowerboundary} or {upperboundary}'.

     format(lowerboundary=lower_fence, upperboundary=upper_fence))

passengers = len(test[test.Ticket_numerical > upper_fence]) / np.float(len(test))



print('Number of passengers with ticket values higher {upperboundary}: {passengers}'.format(upperboundary=upper_fence,\

                                                                                                 passengers=passengers))
train[['Cabin_categorical','Ticket_categorical','Title']].isnull().mean()
test[['Cabin_categorical','Ticket_categorical','Title']].isnull().mean()
# rare / infrequnet labels (less than 1% of passengers)



for var in ['Cabin_categorical','Ticket_categorical','Title']:

  print(var,' contains',len(train[var].unique()), 'labels')
#Test

# rare / infrequnet labels (less than 1% of passengers)



for var in ['Cabin_categorical','Ticket_categorical','Title']:

  print(var,' contains',len(test[var].unique()), 'labels')
# Train

# rare / infrequent labels



for var in ['Cabin_categorical','Ticket_categorical','Title']:

    print(train[var].value_counts() / np.float(len(train)))
# Let's check both dataset shape

train.shape, test.shape
# let's group again the variables into categorical or numerical

# now considering the newly created variables



def find_categorical_and_numerical_variables(datafram):

    cat_vars = [col for col in train.columns if train[col].dtypes == 'O']

    num_vars = [col for col in train.columns if train[col].dtypes != 'O']

    return cat_vars, num_vars

     

        

categorical, numerical = find_categorical_and_numerical_variables(train)        
# let's check results

categorical
#test

def find_categorical_and_numerical_variables(datafram):

    cat_vars = [col for col in test.columns if test[col].dtypes == 'O']

    num_vars = [col for col in test.columns if test[col].dtypes != 'O']

    return cat_vars, num_vars

       

        

categorical, numerical = find_categorical_and_numerical_variables(test)     
# let's check results

categorical
numerical = [var for var in numerical if var not in ['Survived','PassengerId']]

numerical
# print variables with missing data in train set

for col in numerical:

    if train[col].isnull().mean()>0:

        print(col, train[col].isnull().mean())
# print variables with missing data in test set

for col in numerical:

    if test[col].isnull().mean()>0:

        print(col, test[col].isnull().mean())
def impute_na(train, df, variable):

    # Make the temporary df copy

    temp = df.copy()

    

    # extract random from train set to fill the NA's

    random_sample = train[variable].dropna().sample(temp[variable].isnull().sum(), random_state=0)

        

    # pandas needs to have the same index in order to merge dataset

    random_sample.index = temp[temp[variable].isnull()].index

    temp.loc[temp[variable].isnull(), variable] = random_sample

    return temp[variable]
# Age and tickect

# Add variable indicating missingness

for df in [train, test]:

    for var in ['Age', 'Ticket_numerical']:

        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)

        

# replacing by random sampling

for df in [train, test]:

    for var in ['Age', 'Ticket_numerical']:

        df[var] = impute_na(train, df, var)



#Cabin numerical

extrem = train.Cabin_numerical.mean() + train.Cabin_numerical.std()*3

for df in [train, test]:

    df.Cabin_numerical.fillna(extrem, inplace=True)



#Fare

extrem = train.Fare.mean() + train.Fare.std()*3

for df in [train, test]:

    df.Fare.fillna(extrem, inplace=True)    
### Engineering Missing Data in categorical variables

# print variables with missing data

for col in categorical:

    if train[col].isnull().mean()>0:

        print(col, train[col].isnull().mean())
# print variables with missing data in test set

for col in categorical:

    if test[col].isnull().mean()>0:

        print(col, test[col].isnull().mean())
# add label indicating 'Missing' to Cabin categorical & Ticket_categorical

# or replace by most frequent label in Embarked



for df in [train, test]:

    df['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

    df['Cabin_categorical'].fillna('Missing', inplace=True)

    df['Ticket_categorical'].fillna('Missing', inplace=True)
# check absence of null values

train.isnull().mean()
# check absence of null values

test.isnull().mean()
def top_code(df, variable, top):

    return np.where(df[variable]>top, top, df[variable])



for df in [train, test]:

    df['Age'] = top_code(df,'Age', 73)

    df['SibSp'] = top_code(df, 'SibSp',4)

    df['Parch'] = top_code(df, 'Parch',2)

    df['Family_size'] = top_code(df, 'Family_size',7)
#let's check that it worked

for var in ['Age','SibSp','Parch','Family_size']:

    print(var, 'max_value:', train[var].max())
# find quantiles and discretise train & test set

train['Fare'], bins = pd.qcut(x=train['Fare'], q=8, retbins=True, precision=3, duplicates='raise')

test['Fare'] = pd.cut(x = test['Fare'], bins=bins, include_lowest=True)
test.Fare.isnull().sum(), train.Fare.isnull().sum()
t1 = train.groupby(['Fare'])['Fare'].count() / np.float(len(train))

t2 = test.groupby(['Fare'])['Fare'].count() / np.float(len(test))



temp = pd.concat([t1,t2], axis=1)

temp.columns = ['train','test']

temp.plot.bar(figsize=(12,6))
# find quantiles and discretise train & test set



train['Ticket_numerical'], bins = pd.qcut(x = train['Ticket_numerical'], q=8, retbins=True, precision=3, duplicates='raise')

test['Ticket_numerical'] = pd.cut(x = test['Ticket_numerical'], bins=bins, include_lowest=True)
test.Ticket_numerical.isnull().sum(),train.Ticket_numerical.isnull().sum()
# inspect the ticket bins in training set

train.Ticket_numerical.sort_values().unique()
test.loc[test.Ticket_numerical.isnull(), 'Ticket_numerical'] = train.Ticket_numerical.unique()[0]

test.Ticket_numerical.isnull().sum()
# find unfrequent labels in categorical variables

for var in categorical:

    print(var, train[var].value_counts()/np.float(len(train)))

    print()
def rare_imputation(variable, which='rare'):    

    # find frequent labels

    temp = train.groupby([variable])[variable].count()/np.float(len(train))

    frequent_cat = [x for x in temp.loc[temp>0.01].index.values]

    

    # create new variables, with Rare labels imputed

    if which=='frequent':

        # find the most frequent category

        mode_label = train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]

        train[variable] = np.where(train[variable].isin(frequent_cat), train[variable], mode_label)

        test[variable] = np.where(test[variable].isin(frequent_cat), test[variable], mode_label)

        

    else:

        train[variable] = np.where(train[variable].isin(frequent_cat), train[variable], 'Rare')

        test[variable] = np.where(test[variable].isin(frequent_cat), test[variable], 'Rare')
rare_imputation('Cabin_categorical', 'frequent')

rare_imputation('Ticket_categorical', 'rare')
#let's check that it worked

for var in categorical:

    print(var, train[var].value_counts()/np.float(len(train)))

    print()
# let's check that it worked for test dataset

for var in categorical:

    print(var, test[var].value_counts()/np.float(len(test)))

    print()
categorical
for df in [train, test]:

    df['Sex']  = pd.get_dummies(df.Sex, drop_first=True)
train.Sex.unique()
test.Sex.unique()
def encode_categorical_variables(var, target):

        # make label to risk dictionary

        ordered_labels = train.groupby([var])[target].mean().to_dict()

        

        # encode variables

        train[var] = train[var].map(ordered_labels)

        test[var] = test[var].map(ordered_labels)

        



# enccode labels in categorical vars

for var in categorical:

    encode_categorical_variables(var, 'Survived')
# parse discretised variables to object before encoding

for df in [train, test]: 

    df.Fare = df.Fare.astype('O')

    df.Ticket_numerical = df.Ticket_numerical.astype('O')
# encode labels

for var in ['Fare', 'Ticket_numerical']:

    print(var)

    encode_categorical_variables(var, 'Survived')
#let's inspect the dataset

train.head()
#let's inspect the test dataset

test.head()
train.describe()
variables_that_need_scaling = ['Pclass', 'Age', 'Sibsp', 'Parch', 'Cabin_numerical', 'Family_size']
# Let's create the predictors

training_vars = [var for var in train.columns if var not in ['PassengerId', 'Survived']]

training_vars
# fit scaler

scaler = MinMaxScaler() # create an instance

scaler.fit(train[training_vars]) #  fit  the scaler to the train set and then transform it
# Let make it y lable for train dataset 

# We will copy the Survived variable from train data

y_train = train.Survived

y_train.head()
xgb_model = xgb.XGBClassifier()



xgb_model.fit(train[training_vars], y_train, verbose=False)



pred = xgb_model.predict_proba(train[training_vars])

print('xgb train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = xgb_model.predict_proba(test[training_vars])

rf_model = RandomForestClassifier()

rf_model.fit(train[training_vars], y_train)



pred = rf_model.predict_proba(train[training_vars])

print('RF train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = rf_model.predict_proba(test[training_vars])
ada_model = AdaBoostClassifier()

ada_model.fit(train[training_vars], y_train)



pred = ada_model.predict_proba(train[training_vars])

print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = ada_model.predict_proba(test[training_vars])
logit_model = LogisticRegression()

logit_model.fit(scaler.transform(train[training_vars]), y_train)



pred = logit_model.predict_proba(scaler.transform(train[training_vars]))

print('Logit train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = ada_model.predict_proba(scaler.transform(test[training_vars]))

pred_ls = []

for model in [xgb_model, rf_model, ada_model, logit_model]:

    pred_ls.append(pd.Series(model.predict_proba(test[training_vars])[:,1]))



final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)
final_pred = pd.Series(np.where(final_pred>0.40,1,0))

final_pred.head()
temp = pd.concat([test.PassengerId, final_pred], axis=1)

temp.columns = ['PassengerId', 'Survived']

temp.head()
temp.to_csv('submit_titanic.csv', index=False)
# XGB

importance = pd.Series(xgb_model.feature_importances_)

importance.index = training_vars

importance.sort_values(inplace=True, ascending=False)

importance.plot.bar(figsize=(12,6))
#Logistic_Regression

importance = pd.Series(np.abs(logit_model.coef_.ravel()))

importance.index = training_vars

importance.sort_values(inplace=True, ascending=False)

importance.plot.bar(figsize=(12,6))