import numpy as np
import pandas as pd
import os

train_file_path ='../input/train.csv'
test_file_path = '../input/test.csv'
#### import dataset as pandas DataFrames
train_df = pd.read_csv(train_file_path, index_col='PassengerId')
test_df = pd.read_csv(test_file_path, index_col='PassengerId')
#get data frame information
train_df.info()
test_df.info()
test_df['Survived'] = -222 #enter any values
test_df.info()
df = pd.concat((train_df,test_df), axis=0)
df.info()
#head function returns top n row. default no. of rows is 5.
df.head()
#tail returns last n rows. default no of rows is 5.
df.tail(6)
#select column from DataFrame
df['Name']
#selecting colums as list of columns

df[['Name', 'Age']]
#select specified rows using slicing
df.loc[1:10, 'Age': 'Name']
#select discrete columns 
df.loc[1:5, ['Name', 'Age', 'Sex']]
#use iloc for location based indexing
df.iloc[1:5, 2:6]
#filter rows based on some condition on columns
female_passengers = df.loc[(df.Sex=='female') & (df.Pclass==1)]
len(female_passengers)
df.describe()
#we can also get mean and other statistics diretly
fare_mean = df.Fare.mean()
df.Fare.min()

df.Fare.max()

df.Fare.quantile(.25)
%matplotlib inline

df.Fare.plot(kind='box')
#with include parameter we can filter out 
df.describe(include='all')
#counts the values on basis of unique categories
df.Sex.value_counts()
#categorical column: #proportion
df.Sex.value_counts(normalize='True')
df[df.Survived != -888].Survived.value_counts()
df.Pclass.value_counts()
df.Pclass.value_counts().plot(kind='bar')
df.Pclass.value_counts().plot(kind='bar', title= "Passenger counts class wise", rot=0)
#plot a histogram for Age 
df.Age.plot(kind='hist', title="histogram for Age")
#histogram with specified bins
df.Age.plot(kind='hist', title="histogram for Age", bins=20)
#KDE plot for age
df.Age.plot(kind='kde', title="Kernel density plot for Age")
df.Fare.plot(kind='hist', title='histogram for fare', bins=20)
#skewness for Age column
df.Age.skew()
#skewness for Fare column
df.Fare.skew()
#scatter plot for age and fare
df.plot.scatter(x='Age', y='Fare', title="scatter plot: Age vs Fare")
#we can also use alpha for opacity
df.plot.scatter(x='Age', y='Fare', title="scatter plot: Age vs Fare", alpha=0.1)
#scatter plot between Pclass and Fare: pclass is categorical feature
df.plot.scatter(x='Pclass', y='Fare', title="scatter plot: Pclass vs Fare", alpha=0.12)
#groupby
df.groupby('Sex').Fare.mean()
df.groupby(['Pclass']).Fare.mean()
df.groupby(['Pclass']).Age.mean()
df.groupby(['Pclass'])['Age', 'Fare'].mean()
#using agg
df.groupby(['Pclass']).agg({'Fare': 'mean', 'Age': 'median'})
#aggregate dictionary
aggregate = {
    'Fare': {
        'fare_mean': 'mean',
        'fare_median': 'median',
        'fare_max': max,
        'fare_min': np.min
    },
    'Age': {
        'age_mean': 'mean',
        'age_median': 'median',
        'age_max': max,
        'age_min': min,
        'age_range': lambda x : max(x)-min(x)
    }
}
df.groupby(['Pclass']).agg(aggregate)
#group based on two and more variables
df.groupby(['Pclass', 'Sex']).Fare.mean()
df.groupby(['Pclass', 'Sex', 'Embarked']).Fare.mean()
#crosstabs
pd.crosstab(df.Sex, df.Pclass)
#crosstabs using bars
pd.crosstab(df.Sex, df.Pclass).plot(kind='bar',title='class vs sex', rot=0)
#pivot table
df.pivot_table(index='Sex', columns='Pclass', values='Age', aggfunc='mean')
df.groupby(['Pclass', 'Sex']).Age.mean()
#same result we can get from groupby 
df.groupby(['Pclass', 'Sex']).Age.mean().unstack()
#information about data
df.info()
#find rows for null values
df[df.Embarked.isnull()]
#find how many type of Embarked, or categorical feature
df.Embarked.value_counts()
#which embarked point has highest survived counts
pd.crosstab(df[df.Survived != -888].Embarked, df[df.Survived != -888].Survived).plot(kind='bar')
#set Embarked value on basis of survived count
#df.loc[df.Embarked.isnull(), 'Embarked']='S'
#df.Embarked.fillna('S', inplace=True)
#options: categories on basis of Pclass and fare
df.groupby(['Pclass', 'Embarked']).Fare.median().plot(kind='bar', rot=0)
#fill value of Embarked with 'C'
df.Embarked.fillna('C', inplace=True)
len(df.Embarked.isnull().values)
df[df.Embarked.isnull()]
df.info()
df[df.Fare.isnull()]
df.groupby(['Pclass', 'Embarked']).Fare.median()
mean_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'C'), 'Fare'].median()
print("mean fare value where class is 3 and Embarked value is C: {0}".format(mean_fare))
#fill missing value of fare with median values
df.Fare.fillna(mean_fare, inplace=True)
df[df.Fare.isnull()]
df.info()
#set maximum number of raws to display in case of large rendered data
pd.options.display.max_rows = 15
df[df.Age.isnull()]
#histogram of age ranges
df.Age.plot(kind='hist', bins=20)
df.Age.plot(kind='kde')
#mean value of age
mean_age = df.Age.mean()
median_age = df.Age.median()
print("mean of Age: {0}".format(mean_age))
print("median of Age: {0}".format(median_age))

df.groupby(['Sex']).Age.median()
df[df.Age.notnull()].boxplot('Age', 'Sex');
#age_sex_median = df.groupby('Sex').Age.transform('median')
#df.Age.fillna(age_sex_median, inplace=True)
df[df.Age.notnull()].boxplot('Age', 'Pclass')
#age_pclass_median = df.groupby('Pclass').Age.transform('median)
#df.Age.fillna(age_sex_median, inplace=True)
df.head()
#get title of name
def getTitle(name):
    name_with_title = name.split(',')[1]
    title_of_name = name_with_title.split('.')[0]
    title = title_of_name.strip().lower()
    return title
#testing of getTitle function
name = "BLR, Mr. Pulkit Agrawal"
print(getTitle(name))
#we need unquie title for these data sets
df.Name.map(lambda x: getTitle(x)).unique()
#get specified category title for name
def getSpecifiedTitle(name):
    title_category ={
        'mr': 'Mr',
        'mrs': 'Mrs',
        'miss': 'Miss',
        'master': 'Master',
        'don': 'Sir',
        'rev': 'Sir',
        'dr': 'Officer',
        'mme': 'Mrs',
        'ms': 'Mrs',
        'major': 'Master',
        'lady': 'Lady', 
        'sir': 'Sir', 
        'mlle': 'Lady', 
        'col': 'Officer', 
        'capt': 'officer', 
        'the countess': 'Lady',
        'jonkheer': 'Sir',
        'dona': 'Lady'
    }
    name_with_title = name.split(',')[1]
    title_of_name = name_with_title.split('.')[0]
    title = title_of_name.strip().lower()
    return title_category[title]
#create a new Title column
df['Title']=df.Name.map(lambda x: getTitle(x))
df.info()
df[df.Age.notnull()].boxplot('Age', 'Title')
#replace missing Age values with median of title
age_title_median = df.groupby('Title').Age.transform('median')
df.Age.fillna(age_title_median, inplace=True)
df.info()
df.Age.plot(kind='hist',bins = 20)
df.loc[df.Age > 70]
df.Fare.plot(kind='hist', bins=20)
df.Fare.plot(kind='box')
logFare = np.log(df.Fare + 1)
logFare.plot(kind='hist', bins = 20)
logFare.plot(kind='box')
#binning
pd.qcut(df.Fare, 4)
#discritization
pd.qcut(df.Fare, 4, labels = ['very_low', 'low', 'high', 'very_high'])
pd.qcut(df.Fare, 4, labels = ['very_low', 'low', 'high', 'very_high']).value_counts().plot(kind='bar', rot=0)
#create new feature column
df['Fare_Bin'] = pd.qcut(df.Fare, 4, labels = ['very_low', 'low', 'high', 'very_high'])
df.info()
#create a AgeState feature
df['AgeState'] = np.where(df['Age'] >= 18, 'Adult', 'Child')
df.info()
df.AgeState.value_counts()
pd.crosstab(df.loc[df.Survived != -888].Survived, df.loc[df.Survived != -888].AgeState)
# create a familysize feature
df['FamilySize'] = df.Parch + df.SibSp +1
df.FamilySize.plot(kind='hist')
df.loc[df.FamilySize == df.FamilySize.max() , ['Age', 'Ticket', 'FamilySize', 'Survived']]
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived != -888].FamilySize)
df.Cabin.unique()
def getDeck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')
df['Deck'] = df['Cabin'].map(lambda x: getDeck(x))
df.Deck.value_counts()
pd.crosstab(df[df.Survived != -888].Survived, df[df.Survived !=-888].Deck)
df.info()
#change categorical feature into values
df['isMale'] = np.where(df.Sex=='Male', 1, 0)
#create one-hot encoding 
df = pd.get_dummies(df, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'AgeState'])
df.info()
#drop unused columns
df.drop(['Cabin', 'Name', 'Parch', 'SibSp', 'Ticket', 'Sex'], axis=1, inplace=True)
#reorder columns
columns = [column for column in df.columns if column != 'Survived']
columns = ['Survived'] + columns
df = df[columns]
df.info()
#write train data
df_train = df.loc[df.Survived != -888]

#write test data 
columns = [column for column in df.columns if column != 'Survived']
df_test = df.loc[df.Survived == -888, columns]
df_train.info()
#convert input and output features
X = df_train.loc[:,'Age':].as_matrix().astype('float')
y = df_train['Survived'].ravel()
print(X.shape)
print(y.shape)
#split data into 80/20 using train_test_split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("train: {0}, {1}".format(X_train.shape, y_train.shape))
print("test: {0}, {1}".format(X_test.shape, y_test.shape))
#import logistic regression
from sklearn.linear_model import LogisticRegression
logisticRg_model = LogisticRegression(random_state = 0)
logisticRg_model.fit(X_train, y_train)
print("score of the Logistic Regression model: {0:.3f}".format(logisticRg_model.score(X_test, y_test)))
logistic_predicted_model = logisticRg_model.predict(X_test)
#imports performance matrices
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, precision_recall_curve
#confusion metices
print("Confusion Metrices of Logistic Regression model : \n {0}".format(confusion_matrix(y_test, logistic_predicted_model)))

