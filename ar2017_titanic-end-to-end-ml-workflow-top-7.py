# Importing data analysis packages
import pandas as pd
import numpy as np

# Importing data visualization packages
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style="whitegrid")

# Importing feature selection packages
from sklearn.feature_selection import RFECV

# Importing model selection packages
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV, cross_val_score

# Importing machine learning packages
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC 
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Miscellaneous
import time
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_colwidth',80)
# Importing datasets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_combine = df_train.append(df_test, sort=False)
df_train.head()
df_train.tail()
df_train.shape
df_test.shape
feat_desc = pd.DataFrame({'Description': ['Passenger ID',
                                          'Whether the passenger was survived or not',
                                          'The ticket class that the passenger bought',
                                          'The passenger name',
                                          'The gender of the passenger',
                                          'The age of the passenger',
                                          'The number of siblings/spouses that the passenger has aboard the Titanic',
                                          'The number of parents/children that the passenger has aboard the Titanic',
                                          'The ticket number of the passenger',
                                          'The ticket fare that the passenger paid',
                                          'The cabin number that the passenger boarded',
                                          'The passenger port of embarkation'], 
                          'Values': [df_train[i].unique() for i in df_train.columns],
                          'Number of unique values': [len(df_train[i].unique()) for i in df_train.columns]}, 
                          index = df_train.columns)

feat_desc
# Setting 'PassengerId' as the index of training and test dataset
df_train.set_index('PassengerId', inplace=True)
df_test.set_index('PassengerId', inplace=True)
df_combine = df_train.append(df_test, sort=False)
plt.figure(figsize=(11,9))

# Creating univariate distribution of Survived feature
plt.subplot(221)
sns.countplot(df_combine['Survived'], color='sandybrown')
plt.ylabel('Number of passenger')
plt.xlabel('Survived')
plt.title('Number of passenger per survival (Survived)', size=13)

# Creating univariate distribution of Sex feature
plt.subplot(222)
sns.countplot(df_combine['Sex'], color='sandybrown')
plt.ylabel('Number of passenger')
plt.xlabel('Sex')
plt.title('Number of passenger per gender class (Sex)', size=13)

# Creating univariate distribution of Pclass feature
plt.subplot(223)
sns.countplot(df_combine['Pclass'], color='sandybrown')
plt.ylabel('Number of passenger')
plt.xlabel('Ticket class (Pclass)')
plt.title('Number of passenger per ticket class (Pclass)', size=13)

# Creating univariate distribution of Embarked feature
plt.subplot(224)
sns.countplot(df_combine['Embarked'], color='sandybrown')
plt.ylabel('Number of passenger')
plt.xlabel('Port of embarkation (Embarked)')
plt.title('Number of passenger per port of embarkation (Embarked)', size=13)

# Adjusting the spaces between graphs
plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

plt.show()
plt.figure(figsize=(12,10))

# Creating univariate distribution of 'SibSp' feature
plt.subplot(221)
sns.countplot(df_combine['SibSp'], color='lightcoral')
plt.ylabel('Number of Passengers')
plt.xlabel('Number of Siblings/Spouses (SibSp)')
plt.title('Number of Passengers per\nNumber of Siblings/Spouses (SibSp)', size=13)

# Creating univariate distribution of 'Parch' feature
plt.subplot(222)
sns.countplot(df_combine['Parch'], color='lightcoral')
plt.ylabel('Number of Passengers')
plt.xlabel('Number of Parents/Childrens (Parch)')
plt.title('Number of Passengers per\nNumber of Parents/Childrens (Parch)', size=13)

# Creating univariate distribution of 'Age' feature
plt.subplot(223)
sns.distplot(df_combine['Age'].dropna(), color='lightcoral', kde=False, norm_hist=False)
plt.ylabel('Number of Passengers')
plt.xlabel('Age')
plt.title('Distribution of Passengers\' Age', size=13)

# Creating univariate distribution of 'Fare' feature
plt.subplot(224)
sns.distplot(df_combine['Fare'].dropna(), color='lightcoral', kde=False, norm_hist=False)
plt.ylabel('Number of Passengers')
plt.xlabel('Fare')
plt.title('Distribution of Fare', size=13)

plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

plt.show()
plt.figure(figsize=(7,10))

# Creating boxplot of 'SibSp' feature
plt.subplot(221)
sns.boxplot(x='SibSp', data=df_train, color='powderblue', orient='v')
plt.ylabel('Number of siblings/spouses (SibSp)')

# Creating boxplot of 'Parch' feature
plt.subplot(222)
sns.boxplot(x='Parch', data=df_train, color='powderblue', orient='v')
plt.ylabel('Number of parents/children (Parch)')

# Creating boxplot of 'Age' feature
plt.subplot(223)
sns.boxplot(x='Age', data=df_train, color='powderblue', orient='v')

# Creating boxplot of 'Fare' feature
plt.subplot(224)
sns.boxplot(x='Fare', data=df_train, color='powderblue', orient='v')

plt.subplots_adjust(hspace = 0.3, wspace = 0.5)

plt.show()
# Identifying missing data in the training and test dataset
pd.DataFrame({'Number of Missing Values (Training)': df_train.isna().sum(), 
              '% of Missing Values (Training)': (df_train.isna().sum()/df_train.shape[0] * 100).round(2),
              'Number of Missing Values (Test)': df_test.isna().sum().round(0), 
              '% of Missing Values (Test)': (df_test.isna().sum()/df_test.shape[0] * 100).round(2)})
# Imputing 'Cabin' feature
df_train['Cabin'].fillna('Z', inplace=True)
df_test['Cabin'].fillna('Z', inplace=True)
df_combine['Cabin'] = df_train['Cabin'].str.get(0)
df_combine.groupby('Cabin')['Pclass'].value_counts().to_frame('Count')
# Viewing rows that contain missing values in 'Embarked' feature
df_train.loc[df_train['Embarked'].isna()]
# Imputing 'Embarked' feature
df_train.loc[df_train['Embarked'].isna(), 'Embarked'] = 'S'
# Viewing rows that contain missing values in 'Fare' feature
df_test.loc[df_test['Fare'].isna()]
# Imputing 'Fare' feature
df_test.loc[df_test['Fare'].isna()] = df_train['Fare'].mean()
plt.figure(figsize=(12,10))

# Creating correlation matrix
sns.heatmap(df_train.corr(), annot=True, cmap='coolwarm')

plt.show()
# Creating 'df_age', which contains instances with complete data 
# and will be the training data for the linear regression model
df_age = df_train.loc[~df_train['Age'].isna()]

# Initiating linear regression model
reg = LinearRegression()

# Training linear regression model
reg.fit(df_age[['SibSp', 'Pclass']], df_age['Age'])

# Predicting 'Age' feature by using linear regression model
pred_age_train = pd.Series(reg.predict(df_train[['SibSp', 'Pclass']]), index=df_train.index)
pred_age_test = pd.Series(reg.predict(df_test[['SibSp', 'Pclass']]), index=df_test.index)

# Filling missing values based on the predicted 'Age' values
df_train['Age'].fillna(pred_age_train, inplace=True)
df_test['Age'].fillna(pred_age_test, inplace=True)
plt.figure(figsize=(12,6))

# Creating histogram of 'Age' feature from the training set
plt.subplot(121)
sns.distplot(df_train['Age'])
plt.title('Age Distribution of Passengers in the Training Dataset After Imputation', size=13)

# Creating histogram of 'Age' feature from the test set
plt.subplot(122)
sns.distplot(df_test['Age'])
plt.title('Age Distribution of Passengers in the Test Dataset After Imputation', size=13)

plt.show()
# Checking instances that have negative value in the training dataset
df_train.loc[df_train['Age'] < 0]
# Checking instances that have negative value in the test dataset
df_test.loc[df_test['Age'] < 0]
# Replacing instances that have negative value with the mean of the 'Age' feature
df_train.loc[df_train['Age'] < 0, 'Age'] = df_train['Age'].mean()
df_test.loc[df_test['Age'] < 0, 'Age'] = df_train['Age'].mean()
# Rounding values in the 'Age' feature
df_train['Age'] = df_train['Age'].round().astype('int')
df_test['Age'] = df_test['Age'].round().astype('int')
# Creating 'Title' feature
df_train['Title'] = df_train['Name'].str.split(',', expand=True)[1].str.split('.').str.get(0)
df_test['Title'] = df_test['Name'].str.split(',', expand=True)[1].str.split('.').str.get(0)
# Viewing the distribution of 'Title' feature
df_train['Title'].value_counts().to_frame('Number of Passengers').T
# Creating 'SibSp+Parch' feature
df_train['SibSp+Parch'] = df_train['SibSp'] + df_train['Parch'] 
df_test['SibSp+Parch'] = df_test['SibSp'] + df_test['Parch'] 
# Creating 'IsAlone' feature
df_train['IsAlone'] = df_train['SibSp+Parch'].map(lambda x: 1 if x == 0 else 0)
df_test['IsAlone'] = df_test['SibSp+Parch'].map(lambda x: 1 if x == 0 else 0)
train_size = df_train.shape[0]
test_size = df_test.shape[0]
df_combine = df_train.append(df_test, sort=False)
df_combine['Last_Name'] = df_combine['Name'].str.split(',', expand=True)[0]
fare_df = df_combine.loc[df_combine['SibSp+Parch'] > 0, ['Last_Name', 'Fare', 'SibSp+Parch']]
fare_diff = (fare_df.groupby(['Last_Name', 'SibSp+Parch'])['Fare'].aggregate('max') - fare_df.groupby(['Last_Name', 'SibSp+Parch'])['Fare'].aggregate('min')).value_counts()
print('Percentage of families with the same fare: {:.2f}%'.format(fare_diff[0]/fare_diff.sum()*100))
train_temp_df = df_combine.iloc[:train_size]
family_group_df = train_temp_df.loc[train_temp_df['SibSp+Parch']>0, 
                                    ['Last_Name', 'Fare', 'SibSp+Parch', 'Survived']].groupby(['Last_Name', 'Fare'])
family_df = pd.DataFrame(data=family_group_df.size(), columns=['Size_in_training_dataset'])
family_df['Survived_Total'] = family_group_df['Survived'].sum().astype('int')
family_df['SibSp+Parch'] = family_group_df['SibSp+Parch'].mean().astype('int')
all_survived = (family_df['Size_in_training_dataset'] == family_df['Survived_Total']).sum()/len(family_df)*100
print('Families with the whole members survived: {:.1f}%'.format(all_survived))
all_not_survived = (family_df['Survived_Total']==0).sum()/len(family_df)*100
print('Families with the whole members not survived: {:.1f}%'.format(all_not_survived))
df_combine['FamilySurvival'] = 0.5

for _, grp_df in df_combine[['Survived', 'Last_Name', 'Fare']].groupby(['Last_Name', 'Fare']):
    if len(grp_df) > 1:
        for ind, row in grp_df.iterrows():
            ## Finding out if any family members survived or not
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            ## If any family members survived, put this feature as 1
            if smax == 1: 
                df_combine.loc[ind, 'FamilySurvival'] = 1
            ## Otherwise if any family members perished, put this feature as 0
            elif smin == 0: 
                df_combine.loc[ind, 'FamilySurvival'] = 0
train_temp_df = df_combine.iloc[:train_size]
ticket_group_df = train_temp_df.groupby('Ticket')
ticket_df = pd.DataFrame(data=ticket_group_df.size(), columns=['Size_in_training_dataset'])
ticket_df['Survived_Total'] = ticket_group_df['Survived'].sum().astype('int')
ticket_df['Not_Family'] = ticket_group_df['Last_Name'].unique().apply(len)
ticket_df = ticket_df.loc[(ticket_df['Size_in_training_dataset'] > 1) & (ticket_df['Not_Family'] > 1)]
print('Number of groups in training set that is not family: {}'.format(len(ticket_df)))
all_survived = (ticket_df['Size_in_training_dataset'] == ticket_df['Survived_Total']).sum()/len(ticket_df)*100
print('Families with the whole members survived: {:.1f}%'.format(all_survived))
all_not_survived = (ticket_df['Survived_Total'] == 0).sum()/len(ticket_df)*100
print('Families with the whole members not survived: {:.1f}%'.format(all_not_survived))
for grp, grp_df in df_combine.groupby('Ticket'):
    if len(grp_df) > 1:
        for ind, row in grp_df.iterrows():
            if (row['FamilySurvival']) == 0 or (row['FamilySurvival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                if smax == 1:
                    df_combine.loc[ind, 'FamilySurvival'] = 1
                elif smin == 0:
                    df_combine.loc[ind, 'FamilySurvival'] = 0

df_train['FamilySurvival'] = df_combine.iloc[:train_size]['FamilySurvival']
df_test['FamilySurvival'] = df_combine.iloc[train_size:]['FamilySurvival']
df_combine['RealFare'] = 0

for _, grp_df in df_combine.groupby(['Ticket']):
    grp_size = len(grp_df)
    for ind, row in grp_df.iterrows():
        real_fare = row['Fare']/grp_size
        df_combine.loc[ind, 'RealFare'] = real_fare

df_train['Fare'] = df_combine.iloc[:train_size]['RealFare']
df_test['Fare'] = df_combine.iloc[train_size:]['RealFare']
# Viewing data type of each feature in the dataset
df_train.dtypes.to_frame(name='Data type')
# Converting 'Sex' feature data type
df_train.replace({'male': 1, 'female': 0}, inplace=True)
df_test.replace({'male': 1, 'female': 0}, inplace=True)
# Converting 'Embarked' feature data type
df_train.replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)
df_test.replace({'S': 0, 'C': 1, 'Q': 2}, inplace=True)
# Dropping 'Name' and 'Ticket' feature
df_train.drop(columns=['Name', 'Ticket'], inplace=True)
df_test.drop(columns=['Name', 'Ticket'], inplace=True)
plt.figure(figsize=(13,10))

# Creating a bar chart of ticket class (Pclass) vs probability of survival (Survived)
ax1 = plt.subplot(221)
g1 = sns.barplot(x='Pclass', y='Survived', data=df_train, color='seagreen')
plt.ylabel('Probability of Survival')
plt.xlabel('Ticket Class (Pclass)')
ax1.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
plt.title('Ticket Class (Pclass) Survival Comparison', size=13)

# Creating a bar chart of ticket class (Pclass) and gender (Sex) vs probability of survival (Survived)
ax2 = plt.subplot(222)
g2 = sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df_train, palette='BuGn_r')
plt.ylabel('Probability of Survival')
plt.xlabel('Ticket Class (Pclass)')
ax2.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
handles, _ = g2.get_legend_handles_labels()
ax2.legend(handles, ['Female', 'Male'], title='Gender')
plt.title('Ticket Class (Pclass) | Gender (Sex) Survival Comparison', size=13)

# Creating a bar chart of ticket class (Pclass) and port of embarkation (Embarked) vs probability of survival (Survived)
ax3 = plt.subplot(223)
g3 = sns.barplot(x='Pclass', y='Survived', hue='Embarked', data=df_train, palette='BuGn_r')
plt.ylabel('Probability of Survival')
plt.xlabel('Ticket Class (Pclass)')
ax3.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
handles, _ = g3.get_legend_handles_labels()
ax3.legend(handles, ['Southampton', 'Cherbourg', 'Queenstown'], title='Port of Embarkation')
plt.title('Ticket Class (Pclass) | Port of Embarkation (Embarked) \n Survival Comparison', size=13)

# Creating a bar chart of ticket class (Pclass) and passenger is alone (IsAlone) vs probability of survival (Survived)
ax4 = plt.subplot(224)
g4 = sns.barplot(x='Pclass', y='Survived', hue='IsAlone', data=df_train, palette='BuGn_r')
plt.ylabel('Probability of Survival')
plt.xlabel('Ticket Class (Pclass)')
ax4.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
handles, _ = g4.get_legend_handles_labels()
ax4.legend(handles, ['No', 'Yes'], title='Is Alone?')
plt.title('Ticket Class (Pclass) | Passenger Is Alone (IsAlone) \n Survival Comparison', size=13)

plt.subplots_adjust(hspace = 0.4, wspace = 0.3)

plt.show()
plt.figure(figsize=(13,10))

# Creating a bar chart of gender (Sex) vs probability of survival (Survived)
ax1 = plt.subplot(221)
g1 = sns.barplot(x='Sex', y='Survived', data=df_train, color='dodgerblue')
plt.ylabel('Probability of Survival')
plt.xlabel('Gender (Sex)')
ax1.set_xticklabels(['Female', 'Male'])
plt.title('Gender (Sex) Survival Comparison', size=13)

# Creating a bar chart of gender (Sex) and ticket class (Pclass) vs probability of survival (Survived)
ax2 = plt.subplot(222)
g2 = sns.barplot(x='Sex', y='Survived', hue='Pclass', data=df_train, palette='GnBu_d')
plt.ylabel('Probability of Survival')
plt.xlabel('Gender (Sex)')
ax2.set_xticklabels(['Female', 'Male'])
handles, _ = g2.get_legend_handles_labels()
ax2.legend(handles, ['1st Class', '2nd Class', '3rd Class'], title='Ticket Class')
plt.title('Gender (Sex) | Ticket Class (Pclass) Survival Comparison', size=13)

# Creating a bar chart of gender (Sex) and port of embarkation (Embarked) vs probability of survival (Survived)
ax3 = plt.subplot(223)
g3 = sns.barplot(x='Sex', y='Survived', hue='Embarked', data=df_train, palette='GnBu_d')
plt.ylabel('Probability of Survival')
plt.xlabel('Gender (Sex)')
ax3.set_xticklabels(['Female', 'Male'])
handles, _ = g3.get_legend_handles_labels()
ax3.legend(handles, ['Southampton', 'Cherbourg', 'Queenstown'], title='Port of Embarkation')
plt.title('Gender (Sex) | Port of Embarkation (Embarked) \n Survival Comparison', size=13)

# Creating a bar chart of gender (Sex) and passenger is alone (IsAlone) vs probability of survival (Survived)
ax4 = plt.subplot(224)
g4 = sns.barplot(x='Sex', y='Survived', hue='IsAlone', data=df_train, palette='GnBu_d')
plt.ylabel('Probability of Survival')
plt.xlabel('Gender (Sex)')
ax4.set_xticklabels(['Female', 'Male'])
handles, _ = g4.get_legend_handles_labels()
ax4.legend(handles, ['No', 'Yes'], title='Is Alone?')
plt.title('Gender (Sex) | Passenger Is Alone (IsAlone) \n Survival Comparison', size=13)

plt.subplots_adjust(hspace = 0.4, wspace = 0.3)

plt.show()
plt.figure(figsize=(13,10))

# Creating a bar chart of port of embarkation (Embarked) vs probability of survival (Survived)
ax1 = plt.subplot(221)
g1 = sns.barplot(x='Embarked', y='Survived', data=df_train, color='steelblue')
plt.ylabel('Probability of Survival')
plt.xlabel('Port of Embarkation (Embarked)')
ax1.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'])
plt.title('Port of Embarkation (Embarked) \n Survival Comparison', size=13)

# Creating a bar chart of port of embarkation (Embarked) and ticket class (Pclass) 
# vs probability of survival (Survived)
ax2 = plt.subplot(222)
g2 = sns.barplot(x='Embarked', y='Survived', hue='Pclass', data=df_train, palette='ocean_r')
plt.ylabel('Probability of Survival')
plt.xlabel('Port of Embarkation (Embarked)')
ax2.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'])
handles, _ = g2.get_legend_handles_labels()
ax2.legend(handles, ['1st Class', '2nd Class', '3rd Class'], title='Ticket Class')
plt.title('Port of Embarkation (Embarked) | Ticket Class (Pclass) \n Survival Comparison', size=13)

# Creating a bar chart of port of embarkation (Embarked) and gender (Sex) vs probability of survival (Survived)
ax3 = plt.subplot(223)
g3 = sns.barplot(x='Embarked', y='Survived', hue='Sex', data=df_train, palette='ocean_r')
plt.ylabel('Probability of Survival')
plt.xlabel('Port of Embarkation (Embarked)')
ax3.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'])
handles, _ = g3.get_legend_handles_labels()
ax3.legend(handles, ['Female', 'Male'], title='Gender')
plt.title('Port of Embarkation (Embarked) | Gender (Sex) \n Survival Comparison', size=13)

# Creating a bar chart of port of embarkation (Embarked) and passenger is alone (IsAlone) 
# vs probability of survival (Survived)
ax4 = plt.subplot(224)
g4 = sns.barplot(x='Embarked', y='Survived', hue='IsAlone', data=df_train, palette='ocean_r')
plt.ylabel('Probability of Survival')
plt.xlabel('Port of Embarkation (Embarked)')
ax4.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'])
handles, _ = g4.get_legend_handles_labels()
ax4.legend(handles, ['No', 'Yes'], title='Is Alone?')
plt.title('Port of Embarkation (Embarked) | Passenger Is Alone (IsAlone) \n Survival Comparison', size=13)

plt.subplots_adjust(hspace = 0.4, wspace = 0.3)

plt.show()
plt.figure(figsize=(13,10))

# Creating a bar chart of passenger is alone (IsAlone) vs probability of survival (Survived)
ax1 = plt.subplot(221)
g1 = sns.barplot(x='IsAlone', y='Survived', data=df_train, color='steelblue', ci = None)
plt.ylabel('Probability of Survival')
plt.xlabel('Passenger Is Alone (IsAlone)')
ax1.set_xticklabels(['No', 'Yes'])
plt.title('Passenger Is Alone (IsAlone) \n Survival Comparison', size=13)

# Creating a bar chart of passenger is alone (IsAlone) and ticket class (Pclass) 
# vs probability of survival (Survived)
ax2 = plt.subplot(222)
g2 = sns.barplot(x='IsAlone', y='Survived', hue='Pclass', data=df_train, palette='ocean_r', ci = None)
plt.ylabel('Probability of Survival')
plt.xlabel('Passenger Is Alone (IsAlone)')
ax2.set_xticklabels(['No', 'Yes'])
handles, _ = g2.get_legend_handles_labels()
ax2.legend(handles, ['1st Class', '2nd Class', '3rd Class'], title='Ticket Class')
plt.title('Passenger Is Alone (IsAlone) | Ticket Class (Pclass) \n Survival Comparison', size=13)

# Creating a bar chart of passenger is alone (IsAlone) and gender (Sex) vs probability of survival (Survived)
ax3 = plt.subplot(223)
g3 = sns.barplot(x='IsAlone', y='Survived', hue='Sex', data=df_train, palette='ocean_r', ci = None)
plt.ylabel('Probability of Survival')
plt.xlabel('Passenger Is Alone (IsAlone)')
ax3.set_xticklabels(['No', 'Yes'])
handles, _ = g3.get_legend_handles_labels()
ax3.legend(handles, ['Female', 'Male'], title='Gender')
plt.title('Passenger Is Alone (IsAlone) | Gender (Sex) \n Survival Comparison', size=13)

# Creating a bar chart of passenger is alone (IsAlone) and port of embarkation (Embarked) 
# vs probability of survival (Survived)
ax4 = plt.subplot(224)
g4 = sns.barplot(x='IsAlone', y='Survived', hue='Embarked', data=df_train, palette='ocean_r', ci = None)
plt.ylabel('Probability of Survival')
plt.xlabel('Passenger Is Alone (IsAlone)')
ax4.set_xticklabels(['No', 'Yes'])
handles, _ = g4.get_legend_handles_labels()
ax4.legend(handles, ['Southampton', 'Cherbourg', 'Queenstown'], title='Port of Embarkation')
plt.title('Passenger Is Alone (IsAlone) | Port of Embarkation (Embarked) \n Survival Comparison', size=13)

plt.subplots_adjust(hspace = 0.4, wspace = 0.3)

plt.show()
plt.figure(figsize=(18,3))

# Creating a bar chart of passenger's title (Title) vs probability of survival (Survived)
sns.barplot(x='Title', y='Survived', data=df_train, color='cadetblue', ci = None)
plt.ylabel('Probability of Survival')
plt.xlabel('Passenger Title')
plt.title('Passenger Title Survival Comparison', size=13)

plt.show()
# Viewing the distribution of passenger's title ('Title') in the training dataset
df_train['Title'].value_counts().to_frame('Number of Passengers').T
# Viewing the distribution of passenger's title ('Title') in the test dataset
df_test['Title'].value_counts().to_frame('Number of Passengers').T
# Binning titles with count less than 10 into a new category named 'Other'
df_train['Title'] = df_train['Title'].str.strip().map(lambda x: x if x == 'Mr' or x == 'Miss' or x == 'Mrs' or x == 'Master' else 'Other')
df_test['Title'] = df_test['Title'].str.strip().map(lambda x: x if x == 'Mr' or x == 'Miss' or x == 'Mrs' or x == 'Master' else 'Other')
plt.figure(figsize=(15,10))

# Creating a bar chart of passenger's title (Title) vs probability of survival (Survived)
ax1 = plt.subplot(221)
g1 = sns.barplot(x='Title', y='Survived', data=df_train, color='cadetblue')
plt.ylabel('Probability of Survival')
plt.xlabel('Title')
plt.title('Passenger\'s Title Survival Comparison', size=13)

# Creating a bar chart of passenger's title (Title) and ticket class (Pclass) vs 
# probability of survival (Survived)
ax2 = plt.subplot(222)
g2 = sns.barplot(x='Title', y='Survived', hue='Pclass', data=df_train, palette='GnBu_d')
plt.ylabel('Probability of Survival')
plt.xlabel('Title')
handles, _ = g2.get_legend_handles_labels()
ax2.legend(handles, ['1st Class', '2nd Class', '3rd Class'], title='Ticket Class')
plt.title('Passenger\'s Title | Ticket Class (Pclass) Survival Comparison', size=13)

# Creating a bar chart of passenger's title (Title) and passenger is alone (IsAlone) vs 
# probability of survival (Survived)
ax3 = plt.subplot(223)
g3 = sns.barplot(x='Title', y='Survived', hue='IsAlone', data=df_train, palette='GnBu_d')
plt.ylabel('Probability of Survival')
plt.xlabel('Title')
handles, _ = g3.get_legend_handles_labels()
ax3.legend(handles, ['No', 'Yes'], title='Is Alone?')
plt.title('Passenger\'s Title | Passenger Is Alone (IsAlone) Survival Comparison', size=13)

# Creating a bar chart of passenger's title (Title) and port of embarkation (Embarked) vs 
# probability of survival (Survived)
ax4 = plt.subplot(224)
g4 = sns.barplot(x='Title', y='Survived', hue='Embarked', data=df_train, palette='GnBu_d')
plt.ylabel('Probability of Survival')
plt.xlabel('Title')
handles, _ = g4.get_legend_handles_labels()
ax4.legend(handles, ['Southampton', 'Cherbourg', 'Queenstown'], title='Port of Embarkation')
plt.title('Passenger\'s Title | Port of Embarkation (Embarked) Survival Comparison', size=13)

plt.subplots_adjust(hspace = 0.4, wspace = 0.3)

plt.show()
# Checking the data type of 'Title' feature
df_train['Title'].dtype
# Performing dummy coding scheme to 'Title' feature
df_train = df_train.join(pd.get_dummies(df_train['Title'], prefix='Title'), how='outer')
df_test = df_test.join(pd.get_dummies(df_test['Title'], prefix='Title'), how='outer')
# Dropping 'Title' feature
df_train.drop(columns=['Title'], inplace=True)
df_test.drop(columns=['Title'], inplace=True)
plt.figure(figsize=(20,5))

# Creating a bar chart of passenger's age (Age) vs probability of survival (Survived)
sns.barplot(x='Age', y='Survived', data=df_train, ci=True, color='cadetblue')
plt.ylabel('Probability of Survival')
plt.title('Passenger\'s Age Survival Comparison', size=15)

plt.show()
# Binning 'Age' feature
df_train['Age_binned'] = pd.cut(df_train['Age'], np.arange(0, 85, 5), include_lowest=True)
df_test['Age_binned'] = pd.cut(df_test['Age'], np.arange(0, 85, 5), include_lowest=True)
# Creating a bar chart of passenger's age group (Age_binned) vs probability of survival (Survived)
plt.figure(figsize=(20,5))
sns.barplot(x='Age_binned', y='Survived', data=df_train, ci=False, color='cadetblue')
plt.xlabel('Age Group')
plt.ylabel('Probability of Survival')
plt.title('Passenger\'s Age Survival Comparison', size=13)

plt.show()
# Binning 'Age' feature
df_train['Age_binned'] = pd.cut(df_train['Age'], [0, 5, 30, 60, 80], include_lowest=True)
df_test['Age_binned'] = pd.cut(df_test['Age'], [0, 5, 30, 60, 80], include_lowest=True)
#plt.figure(figsize=(8,6))

# Creating a bar chart of passenger's age group (Age_binned) vs probability of survival (Survived)
sns.barplot(x='Age_binned', y='Survived', data=df_train, color='cadetblue')
plt.xlabel('Age Group')
plt.ylabel('Probability of Survival')
plt.title('Passenger\'s Age Survival Comparison', size=13)

plt.show()
# Checking the data type of 'Age_binned' feature
df_train['Age_binned'].dtype
# Converting 'Age_binned' feature data type
df_train['Age_binned'] = pd.cut(df_train['Age'], [0, 5, 30, 60, 80], labels=[0, 1, 2, 3], retbins=False, include_lowest=True)
df_train['Age_binned'] = df_train['Age_binned'].astype('int')
df_test['Age_binned'] = pd.cut(df_test['Age'], [0, 5, 30, 60, 80], labels=[0, 1, 2, 3], retbins=False, include_lowest=True)
df_test['Age_binned'] = df_test['Age_binned'].astype('int')
# Dropping 'Age' feature
df_train.drop(columns='Age', inplace=True)
df_test.drop(columns='Age', inplace=True)
plt.figure(figsize=(8,5))

# Creating a bar chart of number of siblings/spouses (SibSp) vs probability of survival (Survived)
sns.barplot(x='SibSp', y='Survived', data=df_train, color='cadetblue')
plt.xlabel('Number of Siblings/Spouses (SibSp)')
plt.ylabel('Probability of Survival')
plt.title('Number of Siblings/Spouses (SibSp) Survival Comparison', size=13)

plt.show()
plt.figure(figsize=(14,5))

# Creating boxplot of 'SibSp' feature
plt.subplot(121)
sns.boxplot(x='SibSp', data=df_train, color='cadetblue', orient='v')
plt.ylabel('Number of Siblings/Spouses (SibSp)')

# Creating univariate distribution of 'SibSp' feature
plt.subplot(122)
sns.countplot(x='SibSp', data=df_train, color='cadetblue', orient='v')
plt.ylabel('Number of Passengers')
plt.xlabel('Number of Siblings/Spouses (SibSp)')
plt.title('Number of Passengers per Number of Siblings/Spouses (SibSp)', size=13)

plt.show()
# In SibSp feature, binning values 3 to 8 together into value 3 
df_train['SibSp'] = df_train['SibSp'].map(lambda x: 3 if x == 4 or x == 5 or x == 8 else x)
df_test['SibSp'] = df_test['SibSp'].map(lambda x: 3 if x == 4 or x == 5 or x == 8 else x)
plt.figure(figsize=(8,5))

# Creating a bar chart of number of siblings/spouses (SibSp) vs probability of survival (Survived)
sns.barplot(x='SibSp', y='Survived', data=df_train, color='cadetblue')
plt.xlabel('Number of Siblings/Spouses (SibSp)')
plt.ylabel('Probability of Survival')
plt.title('Number of Siblings/Spouses (SibSp) Survival Comparison', size=13)

plt.show()
plt.figure(figsize=(8,5))

# Creating a bar chart of number parents/children (Parch) vs probability of survival (Survived)
sns.barplot(x='Parch', y='Survived', data=df_train, color='cadetblue')
plt.xlabel('Number of Parents/Children (Parch)')
plt.ylabel('Probability of Survival')
plt.title('Number of Parents/Children (Parch) Survival Comparison', size=13)

plt.show()
plt.figure(figsize=(14,5))

# Creating boxplot of 'Parch' feature
plt.subplot(121)
sns.boxplot(x='Parch', data=df_train, color='cadetblue', orient='v')
plt.ylabel('Number of Parents/Children (Parch)')

# Creating univariate distribution of 'Parch' feature
plt.subplot(122)
sns.countplot(x='Parch', data=df_train, color='cadetblue', orient='v')
plt.ylabel('Number of Passengers')
plt.xlabel('Number of Parents/Children (Parch)')
plt.title('Number of Passengers per Parents/Children (Parch)', size=13)

plt.show()
# In Parch feature, binning value 1 to 6 together into value 1 
df_train['Parch'] = df_train['Parch'].map(lambda x: x if x == 0 else 1)
df_test['Parch'] = df_test['Parch'].map(lambda x: x if x == 0 else 1)
plt.figure(figsize=(6,5))

# Creating a bar chart of number of parents/children (Parch) vs probability of survival (Survived)
sns.barplot(x='Parch', y='Survived', data=df_train, color='cadetblue')
plt.xlabel('Number of Parents/Children (Parch)')
plt.ylabel('Probability of Survival')
plt.title('Number of Parents/Children (Parch) Survival Comparison', size=13)

plt.show()
plt.figure(figsize=(8,5))

# Creating a bar chart of number siblings/spouses/parents/children (SibSp+Parch) vs 
# probability of survival (Survived)
sns.barplot(x='SibSp+Parch', y='Survived', data=df_train, color='cadetblue')
plt.xlabel('Number Siblings/Spouses/Parents/Children (SibSp+Parch)')
plt.ylabel('Probability of Survival')
plt.title('Number Siblings/Spouses/Parents/Children (SibSp+Parch)\n Survival Comparison', size=13)

plt.show()
plt.figure(figsize=(14,5))

# Creating boxplot of 'SibSp+Parch' feature
plt.subplot(121)
sns.boxplot(x='SibSp+Parch', data=df_train, color='cadetblue', orient='v')
plt.ylabel('Number Siblings/Spouses/Parents/Children (SibSp+Parch)')

# Creating univariate distribution of 'SibSp+Parch' feature
plt.subplot(122)
sns.countplot(x='SibSp+Parch', data=df_train, color='cadetblue', orient='v')
plt.ylabel('Number of Passengers')
plt.xlabel('Number Siblings/Spouses/Parents/Children (SibSp+Parch)')
plt.title('Number of Passengers per\n Number Siblings/Spouses/Parents/Children (SibSp+Parch)', size=13)

plt.show()
# In SibSp+Parch feature, binning values 1, 2, and 3 together into value 1
df_train['SibSp+Parch'] = df_train['SibSp+Parch'].map(lambda x: 1 if x == 1 or x == 2 or x == 3 else x)
df_test['SibSp+Parch'] = df_test['SibSp+Parch'].map(lambda x: 1 if x == 1 or x == 2 or x == 3 else x)

# In SibSp+Parch feature, binning values 4 to 10 together into value 2
df_train['SibSp+Parch'] = df_train['SibSp+Parch'].map(lambda x: 2 if x == 4 or x == 5 or x == 6 or x == 7 or x == 10 else x)
df_test['SibSp+Parch'] = df_test['SibSp+Parch'].map(lambda x: 2 if x == 4 or x == 5 or x == 6 or x == 7 or x == 10 else x)
plt.figure(figsize=(8,5))

# Creating a bar chart of number siblings/spouses/parents/children (SibSp+Parch) vs 
# probability of survival (Survived)
sns.barplot(x='SibSp+Parch', y='Survived', data=df_train, color='cadetblue')
plt.xlabel('Number Siblings/Spouses/Parents/Children (SibSp+Parch)')
plt.ylabel('Probability of Survival')
plt.title('Number Siblings/Spouses/Parents/Children (SibSp+Parch)\n Survival Comparison', size=13)

plt.show()
plt.figure(figsize=(14,5))

plt.hist([df_train.loc[df_train['Survived']==0,'Fare'], 
          df_train.loc[df_train['Survived']==1,'Fare']], stacked=True, bins=20, label=['Not Survived', 'Survived'])
plt.ylabel('Number of Passengers')
plt.xlabel('Ticket Price (Fare)')
plt.title('Histogram of Ticket Price (Fare) with Survived/Not-Survived Stacked', size=13)
plt.legend()

plt.show()
# Binning 'Fare' feature
df_train['Fare_binned'] = pd.cut(df_train['Fare'], bins=[0,25,75,513], include_lowest=True)
df_test['Fare_binned'] = pd.cut(df_test['Fare'], bins=[0,25,75,513], include_lowest=True)
plt.figure(figsize=(8,5))

# Creating a bar chart of ticket price (Fare_binned) vs probability of survival (Survived)
sns.barplot(x='Fare_binned', y='Survived', data=df_train, color='cadetblue')
plt.ylabel('Probability of Survival')
plt.xlabel('Fare Group')

plt.show()
df_train['Fare_binned'].dtype
# Converting 'Fare' feature data type
df_train['Fare_binned'] = pd.cut(df_train['Fare'], bins=[0,25,75,513], labels=[0, 1, 2], retbins=False, include_lowest=True)
df_train['Fare_binned'] = df_train['Fare_binned'].astype('int')
df_test['Fare_binned'] = pd.cut(df_test['Fare'], bins=[0,25,75,513], labels=[0, 1, 2], retbins=False, include_lowest=True)
df_test['Fare_binned'] = df_test['Fare_binned'].astype('int')
# Dropping 'Fare' feature
df_train.drop(columns='Fare', inplace=True)
df_test.drop(columns='Fare', inplace=True)
# Getting the first letter of 'Cabin' feature
df_train['Cabin'] = df_train['Cabin'].str.get(0)
plt.figure(figsize=(8,5))

# Creating a bar chart of Cabin vs probability of survival (Survived)
sns.barplot(x='Cabin', y='Survived', data=df_train, color='cadetblue')
plt.ylabel('Probability of Survival')
plt.xlabel('Cabin')
plt.title('Cabin Survival Comparison', size=13)

plt.show()
# Creating 'HaveCabin' feature
#df_train['HaveCabin'] = df_train['Cabin'].str.get(0)
df_test['HaveCabin'] = df_test['Cabin'].str.get(0)
df_train['HaveCabin'] = df_train['Cabin'].map(lambda x: 0 if x == 'Z' else 1)
df_test['HaveCabin'] = df_test['HaveCabin'].map(lambda x: 0 if x == 'Z' else 1)
plt.figure(figsize=(8,5))

sns.barplot(x='HaveCabin', y='Survived', data=df_train, color='cadetblue')
plt.ylabel('Probability of Survival')
plt.xlabel('HaveCabin')
plt.title('Cabin Survival Comparison', size=13)

plt.show()
# Dropping 'Cabin' feature
df_train.drop(columns=['Cabin'], inplace=True)
df_test.drop(columns=['Cabin'], inplace=True)
plt.figure(figsize=(14,10))

# Creating a heatmap of correlation among features
sns.heatmap(df_train.corr(), cmap='RdYlGn', annot=True)
plt.title('Correlation Among Features', size=15)

plt.show()
# List of machine learning algorithms that will be used for predictions
estimator = [('Logistic Regression', LogisticRegression), ('Ridge Classifier', RidgeClassifier), 
             ('SGD Classifier', SGDClassifier), ('Passive Aggressive Classifier', PassiveAggressiveClassifier), 
             ('SVC', SVC), ('Linear SVC', LinearSVC), ('Nu SVC', NuSVC), 
             ('K-Neighbors Classifier', KNeighborsClassifier),
             ('Gaussian Naive Bayes', GaussianNB), ('Multinomial Naive Bayes', MultinomialNB), 
             ('Bernoulli Naive Bayes', BernoulliNB), ('Complement Naive Bayes', ComplementNB), 
             ('Decision Tree Classifier', DecisionTreeClassifier), 
             ('Random Forest Classifier', RandomForestClassifier), ('AdaBoost Classifier', AdaBoostClassifier), 
             ('Gradient Boosting Classifier', GradientBoostingClassifier), ('Bagging Classifier', BaggingClassifier), 
             ('Extra Trees Classifier', ExtraTreesClassifier), ('XGBoost', XGBClassifier)]

# Separating independent features and dependent feature from the dataset
X_train = df_train.drop(columns='Survived')
y_train = df_train['Survived']

# Creating a dataframe to compare the performance of the machine learning models
comparison_cols = ['Algorithm', 'Training Time (Avg)', 'Accuracy (Avg)', 'Accuracy (3xSTD)']
comparison_df = pd.DataFrame(columns=comparison_cols)

# Generating training/validation dataset splits for cross validation
cv_split = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

# Performing cross-validation to estimate the performance of the models
for idx, est in enumerate(estimator):
    
    cv_results = cross_validate(est[1](), X_train, y_train, cv=cv_split)
    
    comparison_df.loc[idx, 'Algorithm'] = est[0]
    comparison_df.loc[idx, 'Training Time (Avg)'] = cv_results['fit_time'].mean()
    comparison_df.loc[idx, 'Accuracy (Avg)'] = cv_results['test_score'].mean()
    comparison_df.loc[idx, 'Accuracy (3xSTD)'] = cv_results['test_score'].std() * 3

comparison_df.set_index(keys='Algorithm', inplace=True)
comparison_df.sort_values(by='Accuracy (Avg)', ascending=False, inplace=True)

#Visualizing the performance of the models
fig, ax = plt.subplots(figsize=(12,10))

y_pos = np.arange(len(comparison_df))
ax.barh(y_pos, comparison_df['Accuracy (Avg)'], xerr=comparison_df['Accuracy (3xSTD)'], color='skyblue')
ax.set_yticks(y_pos)
ax.set_yticklabels(comparison_df.index)
ax.set_xlabel('Accuracy Score (Average)')
ax.set_title('Performance Comparison After Simple Modelling', size=13)
ax.set_xlim(0, 1)

plt.show()
# A list of machine learning algorithms that will be optimized
estimator = [('Logistic Regression', LogisticRegression), ('Ridge Classifier', RidgeClassifier), ('SVC', SVC), 
             ('Linear SVC', LinearSVC), ('Nu SVC', NuSVC), ('Random Forest Classifier', RandomForestClassifier), 
             ('AdaBoost Classifier', AdaBoostClassifier), 
             ('Gradient Boosting Classifier', GradientBoostingClassifier), 
             ('Bagging Classifier', BaggingClassifier), ('XGBoost', XGBClassifier)
            ]

index = [est[0] for est in estimator]

# A dictionary containing hyperparameters that are to be optimized for each machine learning algorithm
grid_params = {'SVC': {'C': np.arange(1,21,1), 'gamma': [0.005, 0.01, 0.015, 0.02], 'random_state': [0]},
               'Ridge Classifier': {'alpha': [0.001, 0.0025, 0.005], 'random_state': [0]},
               'Nu SVC': {'nu': [0.5], 'gamma': [0.001, 0.01, 0.1, 1], 'random_state': [0]},
               'Gradient Boosting Classifier': {'learning_rate': [0.001, 0.005, 0.01, 0.015], 'random_state': [0],
                                                'max_depth': [1,2,3,4,5], 'n_estimators': [300, 350, 400, 450, 500]},
               'Linear SVC': {'C': [1, 5, 10], 'random_state': [0]},
               'Logistic Regression': {'C': np.arange(2,7.5,0.25), 
                                       'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
                                       'random_state': [0]},
               'AdaBoost Classifier': {'learning_rate': np.arange(0.05, 0.21, 0.01), 'n_estimators': [50, 75, 100, 125, 150], 
                                       'random_state': [0]},
               'Random Forest Classifier': {'n_estimators': [200, 250, 300, 350], 'max_depth': [1,2,3,4,5,6], 
                                            'criterion': ['gini', 'entropy'], 'random_state': [0]},
               'Bagging Classifier': {'n_estimators': np.arange(200, 300, 10), 'random_state': [0]},
               'XGBoost': {'learning_rate': [0.001, 0.005, 0.01, 0.015], 'random_state': [0],
                           'max_depth': [1,2,3,4,5], 'n_estimators': [300, 350, 400, 450, 500]}
              }

# Creating a dataframe to compare the performance of the machine learning models after hyperparameter optimization 
best_params_df = pd.DataFrame(columns=['Optimized Hyperparameters', 'Accuracy'], index=index)

# start_total = time.perf_counter()

# Performing grid-search cross-validation to optimize hyperparameters and estimate the performance of the models
for idx, est in enumerate(estimator):
    
    # start = time.perf_counter()
    
    best_clf = GridSearchCV(est[1](), param_grid=grid_params[est[0]], cv=cv_split, scoring='accuracy', n_jobs=12)
    best_clf.fit(X_train, y_train)
    
    # run = time.perf_counter() - start
    
    best_params_df.loc[est[0], 'Optimized Hyperparameters'] = [best_clf.best_params_]
    best_params_df.loc[est[0], 'Accuracy'] = best_clf.best_score_
    
    #print('The best parameter for {} is {} with a runtime of {:.2f} seconds.'.format(est[0], best_clf.best_params_, run))

    
#run_total = time.perf_counter() - start_total
#print('Total optimization time was {:.2f} minutes.'.format(run_total/60))
# Showing the results of grid-search cross-validation
best_params_df.sort_values('Accuracy')
# Performing feature selection using RFECV
xg = GradientBoostingClassifier(learning_rate=0.005, max_depth=2, n_estimators=450, random_state=0)
selector = RFECV(xg, step=1, cv=cv_split, scoring='accuracy', n_jobs=8)
selector = selector.fit(X_train, y_train)
# Showing the result of RFECV
pd.DataFrame([X_train.columns, selector.ranking_], index=['Features', 'Ranking']).T.sort_values(by='Ranking')
# Listing the selected features based on RFECV
selected_features = ['Pclass', 'Sex', 'SibSp+Parch', 'FamilySurvival', 'Title_Mr', 'Fare_binned']

# Training the gradient boosting classifier model
gb = GradientBoostingClassifier(learning_rate=0.005, max_depth=2, n_estimators=450, random_state=0)
gb.fit(X_train[selected_features], y_train)

# Estimating the performance of the model by using cross-validation
gb_acc_score = cross_val_score(gb, X_train[selected_features], y_train, cv=cv_split, scoring='accuracy')

print('The performance of the model using the selected features: {:.2f}%'.format(gb_acc_score.mean()*100))
# Training the model
gb = GradientBoostingClassifier(learning_rate=0.005, max_depth=2, n_estimators=450, random_state=0)
gb.fit(X_train[selected_features], y_train)

# Creating a submission file
test_Survived = pd.DataFrame(gb.predict(df_test[selected_features]), columns=['Survived'], index=np.arange(892,1310,1))
test_Survived = test_Survived.reset_index()
test_Survived.rename(columns={'index': 'PassengerID'}, inplace=True)
test_Survived.to_csv("gb.csv",index=False)