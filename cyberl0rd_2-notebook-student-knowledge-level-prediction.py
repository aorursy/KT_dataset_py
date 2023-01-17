import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

'''
Bunch of statements by Kaggle to get path to your data. 
Don't worry if you don't understand them they are not of much concern right now.
'''

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Loading Data in daaframe

info_df = pd.read_csv('/kaggle/input/predict-students-knowledge-level/Data_User_Modeling_Dataset - Information.csv')
test_df = pd.read_csv('/kaggle/input/predict-students-knowledge-level/Data_User_Modeling_Dataset - Test_Data.csv')
train_df = pd.read_csv('/kaggle/input/predict-students-knowledge-level/Data_User_Modeling_Dataset - Training_Data.csv')
# Information.csv
info_df.head() #to print the first 5 rows of the dataframe
'''
To remove the column width restriction by pandas,
An esy solution is to see the column by converting it to a list
'''
list(info_df['Attribute Information:']) 
train_df.head()
train_df.info()
train_df[' UNS']
test_df.head()
test_df.info()
# Let us start with training data

train_df.columns
'''
* drop() function drops all the columns passed into it, 
* axis=1 is for telling drop() to delete values in the column not in the row, 
* inplace=True is for telling drop to save all the changes to oue dataframe.
'''
train_df.drop(['Unnamed: 6', 'Unnamed: 7','Attribute Information:'], axis=1, inplace=True)
# Now lets fix the name of 6th column.
# Its a good practice to keep column names the same as provided in the data set but here for better understanding we will change UNS to Knowldege Level
train_df.rename(columns = {' UNS':'Knowledge Level'}, inplace = True) 
# And the magic has been done.
train_df.head()
def fix_data(temp_df):
    #remenber you can always optimize you work by defining a function or using a loop.
    
    temp_df.drop(['Unnamed: 6', 'Unnamed: 7','Attribute Information:'], axis=1, inplace=True)
    temp_df.rename(columns = {' UNS':'Knowledge Level'}, inplace = True)
    return temp_df
test_df = fix_data(test_df)
# And like this evrything is done.
test_df.head()
# Checking data for nan values
'''
As above we need to do all operation twice for both of the data frames so we will design a function.
'''
def clean_my_data(df):
    if df.isnull().values.any(): # isnull().values.any() returns a boolean value depending upon the presence of null value anywhere in data
        print(df.isnull()) # returns dataframe filled with boolean values for presence of null
        df = df.dropna()
    else: print('Your data do not contain any missing values.')
    return df
train_df = clean_my_data(train_df)
test_df = clean_my_data(test_df)
import seaborn as sns
'''
* Here pairplot does is it takes all the features of the dataframe and plot them pair wise as can be seen below.
* train_df.select_dtypes(include=[np.number]).columns this returns only those columns which contain numeric data.
* passing the above columns in train_df we can easily return dataframe with only numeric data so that we can see the distribution of data 
a swell as can observe the outliers presence.
'''
sns.pairplot(train_df[train_df.select_dtypes(include=[np.number]).columns])
f,ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data = train_df, x='Knowledge Level', y='PEG', ax=ax)
train_df.drop(train_df[train_df['PEG'] < 0 ].index, axis=1, inplace=True)
#fig = plt.figure(figsize=)
#axes = fig.subplots(2, 3)
train_df.boxplot(layout=(2,3), by='Knowledge Level', figsize=[15,10])

f,ax = plt.subplots(figsize=(10, 4))
sns.countplot(x = 'Knowledge Level', data = train_df, palette="Set2", ax=ax)
plt.grid(True)
'''
Pandas come with the best method to start an EDA. You can directly use pandas to analyze and describe your data
'''
train_df.describe()
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
f,ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(train_df['STG'], train_df['PEG'], hue=train_df['Knowledge Level'], ax=ax, palette="Set1")
ax.set_facecolor('#e0f2fc')
plt.grid(True)
f,ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(train_df['SCG'], train_df['PEG'], hue=train_df['Knowledge Level'], ax=ax, palette="Set1")
ax.set_facecolor('#e0f2fc')
plt.grid(True)
f,ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(train_df['PEG'], train_df['LPR'], hue=train_df['Knowledge Level'], ax=ax, palette="Set1")
ax.set_facecolor('#e0f2fc')
plt.grid(True)
train_df['Knowledge Level'].unique()
Knwoledge_levels = train_df['Knowledge Level'].unique()
df = train_df
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
df['Knowledge Level']= label_encoder.fit_transform(df['Knowledge Level']) 
test_df['Knowledge Level'] = label_encoder.fit_transform(test_df['Knowledge Level'])
df

print('Encoding Approach:')
for i,j in zip(Knwoledge_levels, df['Knowledge Level'].unique()):
    print('{}  ==>  {}'.format(i,j))
df = df.append(test_df, ignore_index = True) #using append to add two datasets.
X = df.drop(['Knowledge Level'],axis=1) # assigning X all the independent variable
y = df['Knowledge Level'] #assigning y the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #using train_test_split() to randomly split dataset to train and test.
# In actual we will be using Support Vector Classifier thats why we create an object of SVC.
clf = SVC()
clf.fit(X_train, y_train) #fitting data in the model.
clf.score(X_test, y_test)
clf_1 = SVC(C=50, gamma=1)
clf_1.fit(X_train, y_train)
clf.score(X_test, y_test)
clf_2 = SVC(C=100, gamma=0.1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)