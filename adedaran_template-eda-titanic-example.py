# imports
import pandas as pd
import numpy as np
import os
# read the data with all default parameters
train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')
# get the type
type(train_df)
# use .info() to get brief information about the dataframe 
train_df.info()
test_df.info()
test_df['Survived'] = -888 # Adding Survived with a default value
df = pd.concat((train_df, test_df),axis=0)
df.info()
# use .head() to get top 5 rows
df.head()
# use .tail() to get last 5 rows
df.tail()
# use .head(n) to get top-n rows
df.head(10)
# column selection using dot
df.Name
# selection using column name as string
df['Name']
# selecting multiple columns using a list of column name strings
df[['Name','Age']]
# indexing : use loc for label based indexing 
# all columns
df.loc[5:10,]
# selecting column range
df.loc[5:10, 'Age' : 'Pclass']
# selecting discrete columns
df.loc[5:10, ['Survived', 'Fare','Embarked']]
# indexing : use iloc for position based indexing 
df.iloc[5:10, 3:8]
# filter rows based on the condition 
male_passengers = df.loc[df.Sex == 'male',:]
print('Number of male passengers : {0}'.format(len(male_passengers)))
# use & or | operators to build complex logic
male_passengers_first_class = df.loc[((df.Sex == 'male') & (df.Pclass == 1)),:]
print('Number of male passengers in first class: {0}'.format(len(male_passengers_first_class)))
# use .describe() to get statistics for all numeric columns
df.describe()
# numerical feature
# centrality measures
print('Mean fare : {0}'.format(df.Fare.mean())) # mean
print('Median fare : {0}'.format(df.Fare.median())) # median
# dispersion measures
print('Min fare : {0}'.format(df.Fare.min())) # minimum
print('Max fare : {0}'.format(df.Fare.max())) # maximum
print('Fare range : {0}'.format(df.Fare.max()  - df.Fare.min())) # range
print('25 percentile : {0}'.format(df.Fare.quantile(.25))) # 25 percentile
print('50 percentile : {0}'.format(df.Fare.quantile(.5))) # 50 percentile
print('75 percentile : {0}'.format(df.Fare.quantile(.75))) # 75 percentile
print('Variance fare : {0}'.format(df.Fare.var())) # variance
print('Standard deviation fare : {0}'.format(df.Fare.std())) # standard deviation
%matplotlib inline
# box-whisker plot
df.Fare.plot(kind='box')
# use .describe(include='all') to get statistics for all  columns including non-numeric ones
df.describe(include='all')
# categorical column : Counts
df.Sex.value_counts()
# categorical column : Proprotions
df.Sex.value_counts(normalize=True)
# apply on other columns
df[df.Survived != -888].Survived.value_counts() 
# count : Passenger class
df.Pclass.value_counts() 
# visualize counts
df.Pclass.value_counts().plot(kind='bar')
# title : to set title, color : to set color,  rot : to rotate labels 
df.Pclass.value_counts().plot(kind='bar',rot = 0, title='Class wise passenger count', color='c');
# use hist to create histogram
df.Age.plot(kind='hist', title='histogram for Age', color='c');
# use bins to add or remove bins
df.Age.plot(kind='hist', title='histogram for Age', color='c', bins=20);
# use kde for density plot
df.Age.plot(kind='kde', title='Density plot for Age', color='c');
# histogram for fare
df.Fare.plot(kind='hist', title='histogram for Fare', color='c', bins=20);
print('skewness for age : {0:.2f}'.format(df.Age.skew()))
print('skewness for fare : {0:.2f}'.format(df.Fare.skew()))
# use scatter plot for bi-variate distribution
df.plot.scatter(x='Age', y='Fare', color='c', title='scatter plot : Age vs Fare');
# use alpha to set the transparency
df.plot.scatter(x='Age', y='Fare', color='c', title='scatter plot : Age vs Fare', alpha=0.1);
df.plot.scatter(x='Pclass', y='Fare', color='c', title='Scatter plot : Passenger class vs Fare', alpha=0.15);
# group by 
df.groupby('Sex').Age.median()
# group by 
df.groupby(['Pclass']).Fare.median()
df.groupby(['Pclass']).Age.median()
df.groupby(['Pclass'])['Fare','Age'].median()
df.groupby(['Pclass']).agg({'Fare' : 'mean', 'Age' : 'median'})
# more complicated aggregations 
aggregations = {
    'Fare': { # work on the "Fare" column
        'mean_Fare': 'mean',  # get the mean fare
        'median_Fare': 'median', # get median fare
        'max_Fare': max,
        'min_Fare': np.min
    },
    'Age': {     # work on the "Age" column
        'median_Age': 'median',   # Find the max, call the result "max_date"
        'min_Age': min,
        'max_Age': max,
        'range_Age': lambda x: max(x) - min(x)  # Calculate the age range per group
    }
}
df.groupby(['Pclass']).agg(aggregations)
df.groupby(['Pclass', 'Embarked']).Fare.median()
# crosstab on Sex and Pclass
pd.crosstab(df.Sex, df.Pclass)
pd.crosstab(df.Sex, df.Pclass).plot(kind='bar');
# pivot table
df.pivot_table(index='Sex',columns = 'Pclass',values='Age', aggfunc='mean')
df.groupby(['Sex','Pclass']).Age.mean()
df.groupby(['Sex','Pclass']).Age.mean().unstack()
# use .info() to detect missing values (if any)
df.info()
# extract rows with Embarked as Null
df[df.Embarked.isnull()]
# how many people embarked at different points
df.Embarked.value_counts()
# which embarked point has higher survival count
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Embarked)
# impute the missing values with 'S'
# df.loc[df.Embarked.isnull(), 'Embarked'] = 'S'
# df.Embarked.fillna('S', inplace=True)
# Option 2 : explore the fare of each class for each embarkment point
df.groupby(['Pclass', 'Embarked']).Fare.median()
# replace the missing values with 'C'
df.Embarked.fillna('C', inplace=True)
# check if any null value remaining
df[df.Embarked.isnull()]
# check info again
df.info()
df[df.Fare.isnull()]
median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S'),'Fare'].median()
print(median_fare)
df.Fare.fillna(median_fare, inplace=True)
# check info again
df.info()
# set maximum number of rows to be displayed
pd.options.display.max_rows = 15
# return null rows
df[df.Age.isnull()]
df.Age.plot(kind='hist', bins=20, color='c');
# get mean
df.Age.mean()
# replace the missing values
# df.Age.fillna(df.Age.mean(), inplace=True)
# median values
df.groupby('Sex').Age.median()
# visualize using boxplot
df[df.Age.notnull()].boxplot('Age','Sex');
# replace : 
# age_sex_median = df.groupby('Sex').Age.transform('median')
# df.Age.fillna(age_sex_median, inplace=True)
df[df.Age.notnull()].boxplot('Age','Pclass');
# replace : 
# pclass_age_median = df.groupby('Pclass').Age.transform('median')
# df.Age.fillna(pclass_age_median , inplace=True)
df.Name
# Function to extract the title from the name 
def GetTitle(name):
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title
# use map function to apply the function on each Name value row i
df.Name.map(lambda x : GetTitle(x)) # alternatively you can use : df.Name.map(GetTitle)
df.Name.map(lambda x : GetTitle(x)).unique()
# Function to extract the title from the name 
def GetTitle(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]


# create Title feature
df['Title'] =  df.Name.map(lambda x : GetTitle(x))
# head 
df.head()
# Box plot of Age with title
df[df.Age.notnull()].boxplot('Age','Title');
# replace missing values
title_age_median = df.groupby('Title').Age.transform('median')
df.Age.fillna(title_age_median , inplace=True)
# check info again
df.info()
# use histogram to get understand the distribution
df.Age.plot(kind='hist', bins=20, color='c');
df.loc[df.Age > 70]
# histogram for fare
df.Fare.plot(kind='hist', title='histogram for Fare', bins=20, color='c');
# box plot to indentify outliers 
df.Fare.plot(kind='box');
# look into the outliers
df.loc[df.Fare == df.Fare.max()]
# Try some transformations to reduce the skewness
LogFare = np.log(df.Fare + 1.0) # Adding 1 to accomodate zero fares : log(0) is not defined
# Histogram of LogFare
LogFare.plot(kind='hist', color='c', bins=20);
# binning
pd.qcut(df.Fare, 4)
pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high']) # discretization
pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high']).value_counts().plot(kind='bar', color='c', rot=0);
# create fare bin feature
df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels=['very_low','low','high','very_high'])
# AgeState based on Age
df['AgeState'] = np.where(df['Age'] >= 18, 'Adult','Child')
# AgeState Counts
df['AgeState'].value_counts()
# crosstab
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].AgeState)
# Family : Adding Parents with Siblings
df['FamilySize'] = df.Parch + df.SibSp + 1 # 1 for self
# explore the family feature
df['FamilySize'].plot(kind='hist', color='c');
# further explore this family with max family members
df.loc[df.FamilySize == df.FamilySize.max(),['Name','Survived','FamilySize','Ticket']]
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].FamilySize)
# a lady aged more thana 18 who has Parch >0 and is married (not Miss)
df['IsMother'] = np.where(((df.Sex == 'female') & (df.Parch > 0) & (df.Age > 18) & (df.Title != 'Miss')), 1, 0)
# Crosstab with IsMother
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].IsMother)
# explore Cabin values
df.Cabin
# use unique to get unique values for Cabin feature
df.Cabin.unique()
# look at the Cabin = T
df.loc[df.Cabin == 'T']
# set the value to NaN
df.loc[df.Cabin == 'T', 'Cabin'] = np.NaN
# look at the unique values of Cabin again
df.Cabin.unique()
# extract first character of Cabin string to the deck
def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')
df['Deck'] = df['Cabin'].map(lambda x : get_deck(x))
# check counts
df.Deck.value_counts()
# use crosstab to look into survived feature cabin wise
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].Deck)
# info command 
df.info()
# sex
df['IsMale'] = np.where(df.Sex == 'male', 1, 0)
# columns Deck, Pclass, Title, AgeState
df = pd.get_dummies(df,columns=['Deck', 'Pclass','Title', 'Fare_Bin', 'Embarked','AgeState'])
print(df.info())
# drop columns
df.drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis=1, inplace=True)
# reorder columns
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived'] + columns
df = df[columns]
# check info again
df.info()
import matplotlib.pyplot as plt
%matplotlib inline
plt.hist(df.Age)
plt.hist(df.Age, bins=20, color='c')
plt.show()
plt.hist(df.Age, bins=20, color='c')
plt.title('Histogram : Age')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.show()
f , ax = plt.subplots()
ax.hist(df.Age, bins=20, color='c')
ax.set_title('Histogram : Age')
ax.set_xlabel('Bins')
ax.set_ylabel('Counts')
plt.show()
# Add subplots
f , (ax1, ax2) = plt.subplots(1, 2 , figsize=(14,3))

ax1.hist(df.Fare, bins=20, color='c')
ax1.set_title('Histogram : Fare')
ax1.set_xlabel('Bins')
ax1.set_ylabel('Counts')

ax2.hist(df.Age, bins=20, color='tomato')
ax2.set_title('Histogram : Age')
ax2.set_xlabel('Bins')
ax2.set_ylabel('Counts')

plt.show()
# Adding subplots
f , ax_arr = plt.subplots(3 , 2 , figsize=(14,7))

# Plot 1
ax_arr[0,0].hist(df.Fare, bins=20, color='c')
ax_arr[0,0].set_title('Histogram : Fare')
ax_arr[0,0].set_xlabel('Bins')
ax_arr[0,0].set_ylabel('Counts')

# Plot 2
ax_arr[0,1].hist(df.Age, bins=20, color='c')
ax_arr[0,1].set_title('Histogram : Age')
ax_arr[0,1].set_xlabel('Bins')
ax_arr[0,1].set_ylabel('Counts')

# Plot 3
ax_arr[1,0].boxplot(df.Fare.values)
ax_arr[1,0].set_title('Boxplot : Age')
ax_arr[1,0].set_xlabel('Fare')
ax_arr[1,0].set_ylabel('Fare')

# Plot 4
ax_arr[1,1].boxplot(df.Age.values)
ax_arr[1,1].set_title('Boxplot : Age')
ax_arr[1,1].set_xlabel('Age')
ax_arr[1,1].set_ylabel('Age')

# Plot 5
ax_arr[2,0].scatter(df.Age, df.Fare, color='c', alpha=0.15)
ax_arr[2,0].set_title('Scatter Plot : Age vs Fare')
ax_arr[2,0].set_xlabel('Age')
ax_arr[2,0].set_ylabel('Fare')

ax_arr[2,1].axis('off')
plt.tight_layout()


plt.show()
# family size 
family_survived = pd.crosstab(df[df.Survived != -888].FamilySize, df[df.Survived != -888].Survived)
print(family_survived)
# impact of family size on survival rate
family_survived =  df[df.Survived != -888].groupby(['FamilySize','Survived']).size().unstack()
print(family_survived)
family_survived.columns = ['Not Survived', 'Survived']
# Mix and Match
f, ax = plt.subplots(figsize=(10,3))
ax.set_title('Impact of family size on survival rate')
family_survived.plot(kind='bar', stacked=True, color=['tomato','c'], ax=ax, rot=0)
plt.legend(bbox_to_anchor=(1.3,1.0))
plt.show()
family_survived.sum(axis = 1)
scaled_family_survived = family_survived.div(family_survived.sum(axis=1), axis=0)
scaled_family_survived.columns = ['Not Survived', 'Survived']
# Mix and Match
f, ax = plt.subplots(figsize=(10,3))
ax.set_title('Impact of family size on survival rate')
scaled_family_survived.plot(kind='bar', stacked=True, color=['tomato','c'], ax=ax, rot=0)
plt.legend(bbox_to_anchor=(1.3,1.0))
plt.show()
df.info()
def read_data():
    # set the path of the raw data
    raw_data_path = os.path.join(os.path.pardir,'data','raw')
    train_file_path = os.path.join(raw_data_path, 'train.csv')
    test_file_path = os.path.join(raw_data_path, 'test.csv')
    # read the data with all default parameters
    train_df = pd.read_csv(train_file_path, index_col='PassengerId')
    test_df = pd.read_csv(test_file_path, index_col='PassengerId')
    test_df['Survived'] = -888
    df = pd.concat((train_df, test_df), axis=0)
    return df

def process_data(df):
    # using the method chaining concept
    return (df
         # create title attribute - then add this 
         .assign(Title = lambda x: x.Name.map(get_title))
         # working missing values - start with this
         .pipe(fill_missing_values)
         # create fare bin feature
         .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels=['very_low','low','high','very_high']))
         # create age state
         .assign(AgeState = lambda x : np.where(x.Age >= 18, 'Adult','Child'))
         .assign(FamilySize = lambda x : x.Parch + x.SibSp + 1)
         .assign(IsMother = lambda x : np.where(((x.Sex == 'female') & (x.Parch > 0) & (x.Age > 18) & (x.Title != 'Miss')), 1, 0))
          # create deck feature
         .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.nan, x.Cabin)) 
         .assign(Deck = lambda x : x.Cabin.map(get_deck))
         # feature encoding 
         .assign(IsMale = lambda x : np.where(x.Sex == 'male', 1,0))
         .pipe(pd.get_dummies, columns=['Deck', 'Pclass','Title', 'Fare_Bin', 'Embarked','AgeState'])
         # add code to drop unnecessary columns
         .drop(['Cabin','Name','Ticket','Parch','SibSp','Sex'], axis=1)
         )

def get_title(name):
    title_group = {'mr' : 'Mr', 
               'mrs' : 'Mrs', 
               'miss' : 'Miss', 
               'master' : 'Master',
               'don' : 'Sir',
               'rev' : 'Sir',
               'dr' : 'Officer',
               'mme' : 'Mrs',
               'ms' : 'Mrs',
               'major' : 'Officer',
               'lady' : 'Lady',
               'sir' : 'Sir',
               'mlle' : 'Miss',
               'col' : 'Officer',
               'capt' : 'Officer',
               'the countess' : 'Lady',
               'jonkheer' : 'Sir',
               'dona' : 'Lady'
                 }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')

def fill_missing_values(df):
    # embarked
    df.Embarked.fillna('C', inplace=True)
    # fare
    median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')]['Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)
    # age
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median , inplace=True)
    return df
train_df = process_data(train_df)
test_df = process_data(test_df)
test_df.info()
X = train_df.loc[:,'Age':].values.astype('float')
y = train_df['Survived'].ravel()
print(X.shape, y.shape)
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# average survival in train and test
print('mean survival in train : {0:.3f}'.format(np.mean(y_train)))
print('mean survival in test : {0:.3f}'.format(np.mean(y_test)))
import sklearn

# import function
from sklearn.dummy import DummyClassifier
# create model
model_dummy = DummyClassifier(strategy='most_frequent', random_state=0)
# train model
model_dummy.fit(X_train, y_train)
print('score for baseline model : {0:.2f}'.format(model_dummy.score(X_test, y_test)))
# peformance metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
# accuracy score
print('accuracy for baseline model : {0:.2f}'.format(accuracy_score(y_test, model_dummy.predict(X_test))))
# confusion matrix
print('confusion matrix for baseline model: \n {0}'.format(confusion_matrix(y_test, model_dummy.predict(X_test))))
# precision and recall scores
print('precision for baseline model : {0:.2f}'.format(precision_score(y_test, model_dummy.predict(X_test))))
print('recall for baseline model : {0:.2f}'.format(recall_score(y_test, model_dummy.predict(X_test))))
# import function
from sklearn.linear_model import LogisticRegression
# create model
model_lr_1 = LogisticRegression(random_state=0)
# train model
model_lr_1.fit(X_train,y_train)
# evaluate model
print('score for logistic regression - version 1 : {0:.2f}'.format(model_lr_1.score(X_test, y_test)))
# performance metrics
# accuracy
print('accuracy for logistic regression - version 1 : {0:.2f}'.format(accuracy_score(y_test, model_lr_1.predict(X_test))))
# confusion matrix
print('confusion matrix for logistic regression - version 1: \n {0}'.format(confusion_matrix(y_test, model_lr_1.predict(X_test))))
# precision 
print('precision for logistic regression - version 1 : {0:.2f}'.format(precision_score(y_test, model_lr_1.predict(X_test))))
# precision 
print('recall for logistic regression - version 1 : {0:.2f}'.format(recall_score(y_test, model_lr_1.predict(X_test))))
# model coefficients
model_lr_1.coef_
# base model 
model_lr = LogisticRegression(random_state=0)
from sklearn.model_selection import GridSearchCV
parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}
clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)
clf.fit(X_train, y_train)
clf.best_params_
print('best score : {0:.2f}'.format(clf.best_score_))
# evaluate model
print('score for logistic regression - version 2 : {0:.2f}'.format(clf.score(X_test, y_test)))
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# feature normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled[:,0].min(),X_train_scaled[:,0].max()
# normalize test data
X_test_scaled = scaler.transform(X_test)
# feature standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# base model 
model_lr = LogisticRegression(random_state=0)
parameters = {'C':[1.0, 10.0, 50.0, 100.0, 1000.0], 'penalty' : ['l1','l2']}
clf = GridSearchCV(model_lr, param_grid=parameters, cv=3)
clf.fit(X_train_scaled, y_train)
clf.best_score_
# evaluate model
print('score for logistic regression - version 2 : {0:.2f}'.format(clf.score(X_test_scaled, y_test)))
from sklearn.ensemble import RandomForestClassifier
model_rf_1 = RandomForestClassifier(random_state=0)
model_rf_1.fit(X_train_scaled, y_train)
# evaluate model
print('score for random forest - version 1 : {0:.2f}'.format(model_rf_1.score(X_test_scaled, y_test)))
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[10, 100, 200], 
              'min_samples_leaf':[1, 5,10,50],
              'max_features' : ('auto','sqrt','log2'),
               }
rf = RandomForestClassifier(random_state=0, oob_score=True)
clf = GridSearchCV(rf, parameters)
clf.fit(X_train, y_train)
clf.best_estimator_
# best score
print('best score for random forest : {0:.2f}'.format(clf.best_score_))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn import metrics
model.score(X_test, y_test)
pred = model.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
metrics.auc(fpr, tpr)
# Predict on Final Test data
test_X = test_df.as_matrix().astype('float')
test_X = scaler.transform(test_X)
predictions = model.predict_proba(test_X)
print(predictions.shape)
