#fundamental package for scientific computing with Python, arrays, matrices
import numpy as np 
# data analysis and wrangling
import pandas as pd
# regular expressions
import re

# machine learning


# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)
%matplotlib inline

# util functions
from subprocess import check_output # for ls

# Set jupyter's max row display
pd.set_option('display.max_row', 1000)
# Set jupyter's max column width to 50
pd.set_option('display.max_columns', 50)

# allow interactivity not only with last line in cell
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
# check available libraries (versions)
# !pip list
# !pip list | grep "pandas"
!cat __notebook_source__.ipynb
#!pwd
#!cat __notebook_source__.ipynb | grep -i "loading"
# check input directory
#print(check_output(["ls", "../input"]).decode())
# load train and test datasets into data frames
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df.head(5)
train_df.tail(10)
# 891 Zeilen x 12 Spalten
print('train shape', train_df.shape)
print('test shape', test_df.shape)

# get set difference on column level
train_df.columns.difference(test_df.columns) 
# check data types 
pd.DataFrame(train_df.dtypes)
train_df.info()
print("_"*40)
test_df.info()
data = [] # blank initial list
for f in train_df.columns:
    # Defining the data type 
    dtype = train_df[f].dtype
        
    # Defining the level
    if train_df[f].dtype == float:
        level = 'numerical' #continuous
    elif train_df[f].dtype == object:
        level = 'categorical' #nominal
    else:
        level = 'N.A.'
        
    # Initialize keep to True for all variables 
    keep = True
    role = 'input'
    
    # Creating a Dict that contains all the metadata for the current variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
    
summary_df = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
summary_df.set_index('varname', inplace=True)

# set special roles manually
summary_df.loc['PassengerId','role']='ID'
summary_df.loc['Survived','role']='target'

# set levels individually
summary_df.loc['PassengerId','level']='categorical' #nominal
summary_df.loc['Survived','level']='categorical' #nominal
# 1st class > 2nd class > 3rd class
summary_df.loc['Pclass','level']='categorical' #ordinal
summary_df.loc['SibSp','level']='numerical' #discrete
summary_df.loc['Parch','level']='numerical' #discrete

summary_df
summary_df.groupby(['role', 'level']).size().to_frame(name = 'count').reset_index()
#pd.DataFrame({'count' : summary_df.groupby( ['role', 'level'] ).size()}).reset_index()
# all stats
#train_df.describe(include='all')

# basic stats for numerical features
# train_df.describe(include=['number'])
#train_df[["Survived", "Age", "Fare", "SibSp", "Parch"]].describe()


# or use our newly designed metadata
#s = summary_df[(summary_df['level'].isin(['continuous', 'discrete'])) & (summary_df.keep)].index
s = summary_df[(summary_df['level'].isin(['numerical'])) & (summary_df.keep)].index
# use obtained index for describe and add survived as well
train_df[s.union(['Survived'])].describe()
# only features of type Object
#train_df.describe(include=['O'])
s = summary_df[(summary_df['level'].isin(['categorical'])) & (summary_df.keep)].index

# use obtained index for describe, all to ensure type object as well
train_df[s].describe(include='all')
def Print_Missing_Values_Overview(input_df):
    # put all variables with missing values in a list
    vars_with_missing = []

    for f in input_df.columns:
        missings = input_df[f].isnull().sum()
        if missings > 0:
            vars_with_missing.append(f)
            missings_perc = missings/input_df.shape[0]

            print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))

    print('==> In total, there are {} variables with missing values'.format(len(vars_with_missing)))
print("missings for train_df")
Print_Missing_Values_Overview(train_df)

print("\nmissings for test_df")
Print_Missing_Values_Overview(test_df)
def Get_Missing_Values(input_df):
    null_cnt = input_df.isnull().sum().sort_values(ascending=False)
    tot_cnt = input_df.isnull().count()
    pct_tmp = input_df.isnull().sum()/tot_cnt*100
    pct = (round(pct_tmp, 1)).sort_values(ascending=False)
    missing_data = pd.concat([null_cnt, tot_cnt, pct], axis=1, keys=['#null', '#Tot','%'])\
        .reindex(pct.index)
    return missing_data
missing_vals = Get_Missing_Values(train_df)
display(missing_vals.head(5))

summary_df['#null'] = missing_vals['#null']
summary_df['%null'] = missing_vals['%']
Get_Missing_Values(test_df).head(5)
train_df["Cabin"].value_counts().shape[0]
# get list of unique values
#pd.unique(train_df["Cabin"])
# get overview of uniques
train_df.nunique()
# get number of unique values
#train_df.apply(lambda x: len(x.unique()))

train_df.apply(pd.Series.nunique).sort_values(ascending=False)

summary_df['#unique'] = train_df.nunique()
# list all unique values
# pd.unique(train_df["Ticket"])
# check for duplicate ticket numbers
train_df.groupby('Ticket').size().sort_values(ascending=False).head()
train_df[(train_df['Ticket']=='CA. 2343')]
# is there a duplicate ticket number with differing prices?
train_df.groupby(['Ticket', 'Fare']).size().groupby('Ticket').size().sort_values(ascending=False).head()
train_df[(train_df['Ticket']=='7534')]
summary_df['#non_null'] = train_df.count()
# reorder columns
summary_df = summary_df[['role', 'level', 'keep', 'dtype', '#null', '%null', '#non_null', '#unique']]
summary_df
# group by each value and get counts
# train_df["Age"].value_counts()
# get value counts for all columns
#pieces = []
#for col in train_df.columns:
#    tmp_series = train_df[col].value_counts()
#    tmp_series.name = col
#    pieces.append(tmp_series)
#pd.DataFrame(pieces).T
# !pip list | grep profiling
import pandas_profiling

# double click left next to cell to collapse/expand output
profile = pandas_profiling.ProfileReport(train_df)
profile
# rejected_variables = profile.get_rejected_variables(threshold=0.9)
# profile.to_file(outputfile="/tmp/myoutputfile.html")
train_df.Survived.value_counts(normalize=True)
sns.countplot(x='Survived', data=train_df);
sns.distplot(train_df.Fare, kde=False);
print(train_df.Fare.mean())
# delete n.a./missing values beforehand
sns.distplot(train_df.Age.dropna())
plt.title('Age Distribution of Passengers', fontdict={'fontsize': 16})
plt.show()
# show box plots for numerical featues fare and age
fig, axes = plt.subplots(2, 1)
sns.boxplot(x="Fare", data=train_df, ax=axes[0])
sns.boxplot(x="Age", data=train_df, ax=axes[1])
train_df['Age'].hist(bins=50)
facet_grid = sns.FacetGrid(train_df, col='Sex', size=5, aspect=1)
# using histogram (distplot)
facet_grid.map(sns.distplot, "Age")
# move to ensure enough space for title
plt.subplots_adjust(top=0.9)
facet_grid.fig.suptitle('Age Distribution (Males vs Females)', fontsize=16)
#use FacetGrid to plot multiple kdeplots on one plot
fig = sns.FacetGrid(train_df,hue='Sex',aspect=4)
#call FacetGrid.map() to use sns.kdeplot() to show age distribution
fig.map(sns.kdeplot,'Age',shade=True)
#set the x max limit by the oldest passenger
oldest = train_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
fig = sns.FacetGrid(train_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade='True')
oldest = train_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
# use facet grid for a box plot of age distribution
fg = sns.FacetGrid(train_df, col="Pclass")
# using boxplots, order parameter has to be set to prevent wrong output !!
fg.map(sns.boxplot, "Sex", "Age", order=["male", "female"])
sns.swarmplot(x="Pclass", y="Age", hue="Sex", data=train_df)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.title("Age distribution vs Class", fontsize=15)

sns.factorplot('Pclass',data=train_df,hue='Sex',kind='count')
# 2 rows, 4 columns
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

sns.countplot('Survived',data=train_df,ax=axes[0,0])
sns.countplot('Pclass',data=train_df,ax=axes[0,1])
sns.countplot('Sex',data=train_df,ax=axes[0,2])
sns.countplot('SibSp',data=train_df,ax=axes[0,3])

sns.countplot('Parch',data=train_df,ax=axes[1,0])
sns.countplot('Embarked',data=train_df,ax=axes[1,1])

# numeric/continuous features
sns.distplot(train_df['Fare'], ax=axes[1,2])
# remove null values/rows beforehand
sns.distplot(train_df['Age'].dropna(),ax=axes[1,3])
plt.figure(figsize=(15,8))
sns.kdeplot(train_df["Age"][train_df.Survived == 1], color="green", shade=True)
sns.kdeplot(train_df["Age"][train_df.Survived == 0], color="red", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Age for Surviving Population and Deceased Population')
plt.show()
#s = summary_df[(summary_df['level'].isin(['categorical'])) & (summary_df['#unique'] <= 10)].index
#sl = s.tolist()
cat_vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

fig, axs = plt.subplots(nrows=len(cat_vars), figsize=(8,20), sharex=False)
for i in range(len(cat_vars)):
    sns.countplot(x=cat_vars[i], data=train_df, hue='Survived', ax=axs[i])
# count plot for exact numbers 
sns.countplot(x="Sex", hue="Survived", data=train_df)

train_df.groupby(["Sex", "Survived"]).size()
# figure out survival proportions
print(train_df[train_df.Sex == 'female'].Survived.sum()/train_df[train_df.Sex == 'female'].Survived.count())
print(train_df[train_df.Sex == 'male'].Survived.sum()/train_df[train_df.Sex == 'male'].Survived.count())
sns.factorplot(x='Survived', col='Pclass', kind='count', data=train_df);
# Passenger class wise distribution of counts of survival statistics for men and women
sns.factorplot("Sex", col="Pclass", data=train_df, kind="count", hue="Survived")
train_df.groupby(["Sex", "Pclass", "Survived"]).size()
sns.pointplot(x="Pclass", y="Survived", hue="Pclass", data=train_df)
# surviving ratio of different classes
sns.factorplot('Pclass','Survived',data=train_df)
sns.barplot(x='Sex', y='Survived', hue="Pclass", data=train_df)
sns.factorplot(x='Survived', col='Embarked', kind='count', data=train_df);
sns.factorplot("Pclass", col="Embarked", data=train_df, kind="count", hue="Survived")
# check relation to target attribute

# Set up the matplotlib figure
figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))

train_df.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
train_df.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
train_df.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
train_df.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
train_df.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])

sns.boxplot(x="Survived", y="Age", data=train_df,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=train_df,ax=axesbi[1,2])
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
fig, axs = plt.subplots(ncols=2, figsize=(15, 3))
sns.pointplot(x="Embarked", y="Survived", hue="Sex", data=train_df, ax=axs[0]);
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_df, ax=axs[1]);
train_df.groupby('Survived').Fare.hist(alpha=0.6);
plt.figure(figsize=(15,8))
sns.kdeplot(train_df["Fare"][train_df.Survived == 1], color="green", shade=True)
sns.kdeplot(train_df["Fare"][train_df.Survived == 0], color="red", shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density Plot of Fare for Surviving Population and Deceased Population')
# limit x axis to zoom on most information. there are a few outliers in fare. 
plt.xlim(-20,200)
plt.show()
# although there appears to be a small tendency upwards shown by the regression, 
# there appears to be almost no correlation between the variables “age” and “fare”, 
# as shown by the Pearson correlation coefficient. 
sns.jointplot(x="Age", y="Fare", data=train_df, kind='reg');
sns.lmplot(x='Age', y='Fare', hue='Survived', data=train_df, fit_reg=False, scatter_kws={'alpha':0.5});
sns.factorplot(x="Pclass", y="Age", hue="Survived", data=train_df, kind="box")
# display most of the information in a single grid of plots.
# drop nulls before
sns.pairplot(train_df.dropna(), hue='Survived');
# correlation matrix for int64 and float64 types
# use pearsons R, alternatively Spearman or Kendal-Tau could be used for categorical features

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 8))
corr = train_df.corr()
#display(corr)

sns.heatmap(corr,
            mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, linewidths=.5, annot=True)
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.graph_objs as go
pyo.init_notebook_mode(connected=False)

corr = train_df.corr().abs().Survived.sort_values(ascending=False)[1:]
data = [go.Bar(
            x=corr.index.values,
            y=corr.values
    )]

pyo.iplot(data, filename='basic-bar')
# check fare = 0
# every ticket should have a value greater than 0
print((train_df.Fare == 0).sum())
print((test_df.Fare == 0).sum())

for df in train_df, test_df:
    # mark zero values as missing or NaN
    df.Fare = df.Fare.replace(0, np.NaN)
    
    # impute the missing Fare values with the mean Fare value    
    df.Fare.fillna(df.Fare.mean(),inplace=True)

# we see that there are no Zero values
print((train_df.Age == 0).sum())
print((test_df.Age == 0).sum())

# impute the missing Age values with the mean Fare value
for df in train_df, test_df:
    df.Age.fillna(df.Age.mean(),inplace=True)
    
# use median, as it deals better with outliers ??
# train_df['Age'] = train_df.Age.fillna(train_df.Age.median())
# We see that a majority 77% of the Cabin variable has missing values.
# Hence will drop the column from training a machine learnign algorithem
train_df.Cabin.isnull().mean()
# fill with most frequent value
# train_df.Embarked.mode()
for df in train_df, test_df:
    df["Embarked"].fillna("S", inplace=True)
for df in train_df, test_df:
    df['Name_len'] = df.Name.str.len()
    df['Name_parts'] = df.Name.str.count(" ") + 1

train_df.shape, test_df.shape
sns.distplot(train_df.Name_len, kde=False);
print(train_df.Name_len.mean())
train_df[(train_df.Name_len > 50)]
# Regular expression to get the title of the Name
for df in train_df, test_df:
    # exlude point
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

print(train_df.shape, test_df.shape)
train_df.Title.value_counts().reset_index()
# Show title counts by sex
pd.crosstab(train_df.Sex, train_df.Title)
# clean up titles
for df in train_df, test_df:
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', \
                                       'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
# show new distribution
train_df['Title'].value_counts()
# Finally, grab surname from passenger name
# just for information/analysis purposes => no relevant predictor
def substrSurname(name):
    end = name.find(', ')
    return name[0:end].strip()

for df in train_df, test_df:
    df["Surname"] = df.Name.map(substrSurname)
# short version with lamda function
#train_df["Surname"] = train_df.Name.map(lambda x: x[0:x.find(', ')].strip())

print('We have', len(np.unique(train_df.Surname)), 'unique surnames.')

train_df.sort_values('Surname').head(20)
# show age values that are approximated
train_df[(train_df['Age'] % 1 == 0.5)]
#interaction variable: since age and class are both numbers we can just multiply them
for df in train_df, test_df:
    df['Age*Class'] = df['Age'] * train_df['Pclass']
# linear combination of features
for df in train_df, test_df:
    # including the passenger themselves
    df['Family_Size'] = df.SibSp + df.Parch + 1
print("Travelling with family: ", train_df[train_df['Family_Size'] > 1].PassengerId.count())
print("Travelling alone: ",train_df[train_df['Family_Size'] == 1].PassengerId.count())

sns.factorplot('Family_Size', hue='Survived', data=train_df, kind='count')
# further group familiy size into new column, if traveled alone or with family
for df in train_df, test_df:
    df['Travel_Alone'] = df['Family_Size'].map(lambda x: True if x == 1 else False)
sns.factorplot('Travel_Alone', hue='Survived', data=train_df, kind='count')
train_df[['Travel_Alone', 'Survived']].groupby(['Travel_Alone'], as_index=False).mean()
sns.barplot('Travel_Alone', 'Survived', data=train_df, color="mediumturquoise")
plt.show()
# tickets have high variance but there seems to be an indication for something
for df in train_df, test_df:
    df['Ticket_First'] = df.Ticket.str[0]

train_df['Ticket_First'].value_counts()
#Here we divide the fare by the number of family members traveling together
for df in train_df, test_df:
    df['Fare_Per_Person'] = df['Fare'] / (df['Family_Size'])
# first letter of the cabin denotes the cabin level (e.g. A,B,C,D,E,F,G).
# Create a Deck variable. Get passenger deck A - F:
def getCabinDeck(cabin):
    if not cabin or pd.isnull(cabin):
        # use string to make clear it is a category on its own
        return 'None'
    else:
        return cabin[0]

for df in train_df, test_df: 
    df["Deck"] = df.Cabin.map(getCabinDeck)
sns.factorplot('Deck',data=train_df,kind='count', hue='Survived')
print(train_df.shape, test_df.shape)
train_df.columns.difference(test_df.columns)
train_df.columns
train_df.head()
# Factorize the values 
labels,levels = pd.factorize(train_df.Sex)

train_df['Sex_Class'] = labels

# print(levels)
train_df.head()

#drop again
train_df.drop('Sex_Class', axis=1, inplace=True)
for df in train_df, test_df:
    df['Sex_Class'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
for df in train_df, test_df:
    df['Embarked_Class'] = df["Embarked"].map(dict(zip(("S", "C", "Q"), (0, 1, 2))))
# convert categorical to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for df in train_df, test_df:
    df['Title_Class'] = df['Title'].map(title_mapping)
# convert from bool to int (0,1)
for df in train_df, test_df:
    df['Travel_Alone'] = df['Travel_Alone'].astype(int)
dummy_df = pd.get_dummies(train_df, columns=["Pclass", "Embarked", "Sex"])
dummy_df.drop('Sex_female', axis=1, inplace=True)
dummy_df.head()
# define bins automatically
mybins = range(0, int(train_df.Age.max()+10), 10)

# Cut the data with the help of the bins
train_df['age_bucket'] = pd.cut(train_df.Age, bins=mybins)

# Count the number of values per bucket
train_df['age_bucket'].value_counts()

train_df.drop(['age_bucket'] , axis=1, inplace=True, errors='ignore')
## create bins for age
def get_age_group(age):
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 11:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 65:
        a = 'adult'
    else:
        a = 'senior'
    return a
for df in train_df, test_df:
    df['Age_Group'] = df['Age'].map(get_age_group)

    # Factorize the values 
    labels,levels = pd.factorize(df.Age_Group)
    df['Age_Group'] = labels

# levels are the same for train and test
print(levels)
train_df.head(20)
dummy_df2 = pd.get_dummies(train_df, columns=["Age_Group"])
dummy_df2.head()
train_df.drop(['Age_Group'] , axis=1, inplace=True, errors='ignore')
test_df.drop(['Age_Group'] , axis=1, inplace=True, errors='ignore')
def get_name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'normal'
    else:
        a = 'long'
    return a
train_df['Name_len_Class'] = train_df['Name_len'].map(get_name_length_group)
# this should then be factorized again
## cuts the column by given bins based on the range of name_length
group_names = ['short', 'medium', 'normal', 'long']
train_df['Name_len_Class2'] = pd.cut(train_df['Name_len'], bins = 4, labels=group_names)
train_df['Name_len_Class2'].value_counts()
train_df.drop(['Name_len_Class2','Name_len_Class'] , axis=1, inplace=True, errors='ignore')
def get_family_group(size):
    a = ''
    if (size <= 1):
        a = 'alone'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
train_df['Family_Size_Class'] = train_df['Family_Size'].map(get_family_group)
train_df.drop(['Family_Size_Class'] , axis=1, inplace=True, errors='ignore')
train_df.head()
# manually set candidates for deletion
# typically those features are being determined automatically

## # PassengerId has too high variance
## summary_df.loc['PassengerId','keep']=False
## # Ticket column has a lot of various values. It will have no significant impact
## summary_df.loc['Ticket','keep']=False
## # Cabin has too many missings
## summary_df.loc['Cabin','keep']=False
## # name as too many distinct values as well
## summary_df.loc['Name','keep']=False
## 
## s = summary_df[(summary_df.keep == False)].index
## 
## for df in train_df, test_df:
##     for i in s:
##         df.drop(i, axis=1, inplace=True)


# PassengerId has too high variance,  IDs are unnecessary for classification
# drop only in training set, as ID is needed for submission file
train_df.drop('PassengerId', axis=1, inplace=True, errors='ignore')

for df in train_df, test_df:
    # Ticket column has a lot of various values. It will have no significant impact
    df.drop('Ticket', axis=1, inplace=True, errors='ignore')
    # Cabin has too many missings
    df.drop('Cabin', axis=1, inplace=True, errors='ignore')
    # name has too many distinct values as well
    df.drop('Name', axis=1, inplace=True, errors='ignore')
    # drop due to factorization/encoding
    df.drop('Sex', axis=1, inplace=True, errors='ignore')
    df.drop('Embarked', axis=1, inplace=True, errors='ignore')
    df.drop('Title', axis=1, inplace=True, errors='ignore')
    df.drop('Surname', axis=1, inplace=True, errors='ignore')
    df.drop('Ticket_First', axis=1, inplace=True, errors='ignore')
    df.drop('Deck', axis=1, inplace=True, errors='ignore')
    

# drop a list of colu
# drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
# train_df = train_df.drop(drop_elements, axis = 1)
train_df.head()
train_df.columns
#train_df.columns[1:]
# obtain feature importances

# Import `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = list(train_df.drop(['Survived'], axis=1).columns.values)

X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['Survived'], axis=1), train_df["Survived"])

# Build the model
# randomly generates thousands of decision trees and takes turns leaving out each variable in fitting the model
rfc = RandomForestClassifier()

# Fit the model
rfc.fit(X_train, y_train)

# Print the results
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), features), reverse=True))

# Isolate feature importances 
#importance = rfc.feature_importances_

# Sort the feature importances 
#sorted_importances = np.argsort(importance)
# imports
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
X = train_df.drop(['Survived'], axis=1)
y = train_df["Survived"]
# splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

# default split is with 25%
X_train.shape, X_test.shape
# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
test_df['Survived'] = 0
test_df[['PassengerId', 'Survived']].to_csv('output_model1.csv', index=False)
test_df.head()
test_df.drop('Survived', axis=1, inplace=True, errors='ignore')
test_df['Survived'] = test_df.Sex_Class == 1
# convert bool to int (0,1)
test_df['Survived'] = test_df.Survived.apply(lambda x: int(x))
test_df[['PassengerId', 'Survived']].to_csv('output_model2.csv', index=False)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier()
dectree.fit(X_train, y_train)
y_pred = dectree.predict(X_test)
dectree_accy = round(accuracy_score(y_pred, y_test), 3)
print(dectree_accy)
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=100,max_depth=9,min_samples_split=6, min_samples_leaf=4)
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
random_accy = round(accuracy_score(y_pred, y_test), 3)
print(random_accy)
test_df.drop('Survived', axis=1, inplace=True, errors='ignore')
test_df.shape
# ignore PaxID
test_prediction = randomforest.predict(test_df.iloc[:,1:])
submission = pd.DataFrame({
        "PassengerId": test_df.PassengerId,
        "Survived": test_prediction
    })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv('output_model_RF.csv', index=False)
from xgboost import XGBClassifier
XGBClassifier = XGBClassifier()
XGBClassifier.fit(X_train, y_train)

y_pred = XGBClassifier.predict(X_test)

XGBClassifier_accy = round(accuracy_score(y_pred, y_test), 3)

print(XGBClassifier_accy)
