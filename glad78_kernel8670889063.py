import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

%matplotlib inline
sns.set(style="darkgrid", color_codes=True, palette='deep')
train_csv = "../datasets/train.csv"
test_csv = "../datasets/test.csv"
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
train_df.head()
test_df.head()
train_df.describe()
test_df.describe()
train_df.isnull().sum()
test_df.isnull().sum()
def get_honoric(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for df in [train_df, test_df]:
    df['Honoric'] = df['Name'].apply(get_honoric)
# lets us plot many diffrent shaped graphs together 

train_df['Honoric'].value_counts().plot(kind='bar', fontsize=12, figsize=(8, 3))
plt.title('train data Honoric histgram')
plt.show()

test_df['Honoric'].value_counts().plot(kind='bar', fontsize=12, figsize=(8, 3))
plt.title('test data Honoric histgram')
plt.show()
# Replace
for df in [train_df, test_df]:
    df['Honoric'] = df['Honoric'].replace({'Mlle' :'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    rare_honorics = set(df['Honoric']) - set(['Mr', 'Miss', 'Mrs', 'Master'])
    replace_dict = {rh: 'Rare' for rh in rare_honorics}
    df['Honoric'] = df['Honoric'].replace(replace_dict)
# lets us plot many diffrent shaped graphs together 
train_df['Honoric'].value_counts().plot(kind='bar', fontsize=12, figsize=(8, 3))
plt.title('train data Honoric histgram')
plt.show()

test_df['Honoric'].value_counts().plot(kind='bar', fontsize=12, figsize=(8, 3))
plt.title('test data Honoric histgram')
plt.show()
for df in [train_df, test_df]:
    df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Name_length'] = df['Name'].apply(len)
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
for df in [train_df, test_df]:
    is_estimated = df['Age'].apply(lambda x: True if x - np.floor(x) == 0.5 and x >1 else False)
    df['is_estimated_Age'] = is_estimated
train_df[train_df['is_estimated_Age'] == True]
# create bins
def make_bins(df, bins, col='Age'):
    bins_col_name = col + '_bins'
    bin_labels = []
    for i in range(len(bins) - 1):
        bins_string = str(bins[i]) + '~' + str(bins[i + 1] - 1)
        bin_labels.append(bins_string)
    df[bins_col_name] = pd.cut(df[col], bins, labels=bin_labels, right=False)
    
# bining Age
age_bins = np.arange(0, 90, 5)
make_bins(train_df, bins=age_bins, col='Age')
make_bins(test_df, bins=age_bins, col='Age')

# bining Fare
fare_bins = np.arange(0, 600, 50)
make_bins(train_df, bins=fare_bins, col='Fare')
make_bins(test_df, bins=fare_bins, col='Fare')
plt.figure(figsize=(10, 10))
sns.heatmap(train_df.corr(), square=True, annot=True, cmap='Greens')
plt.show()
# Fare > 100 Passengers are all 1 Pclass
print(train_df[train_df['Fare'] > 100]['Pclass'].unique())

plt.scatter(train_df['Fare'], train_df['Pclass'])
plt.title('scatter plot by Fare and Pclass')
plt.ylabel('Pclass')
plt.xlabel('Fare')
plt.show()
def survival_plot(df, ax, col='Sex', stacked=True):
    df.groupby([col, 'Survived']).size().unstack().plot(kind='bar', stacked=True, ax=ax)
    plt.title("Survival by {}, (1 = Survived)".format(col))
    
def age_kde_plot(df, ax):
    df.groupby('Pclass')['Age'].plot(kind='kde')
    plt.xlabel("Age")
    plt.title("Age Distribution within classes")
    # sets our legend for our graph.
    plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
    
    
def  original_plot(df, survival_plot=survival_plot, age_kde_plot=age_kde_plot):
    fig = plt.figure(figsize=(18,30)) 
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    rows = 3
    columns = 5

    ax = plt.subplot2grid((columns, rows),(0,0))
    survival_plot(df=df, ax=ax, col='Sex', stacked=True)

    ax  = plt.subplot2grid((columns, rows),(0,1))
    survival_plot(df=df, ax=ax, col='Age_bins', stacked=True)

    ax = plt.subplot2grid((columns, rows),(0,2))
    survival_plot(df=df, ax=ax, col='Pclass', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(1,0))
    survival_plot(df=df, ax=ax, col='Fare_bins', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(1,1))
    survival_plot(df=df, ax=ax, col='Has_Cabin', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(1,2))
    survival_plot(df=df, ax=ax, col='Embarked', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(2,0))
    survival_plot(df=df, ax=ax, col='SibSp', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(2,1))
    survival_plot(df=df, ax=ax, col='Parch', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(2,2))
    survival_plot(df=df, ax=ax, col='FamilySize', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(3,0))
    survival_plot(df=df, ax=ax, col='IsAlone', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(3,1), colspan=2)
    survival_plot(df=df, ax=ax, col='Name_length', stacked=True)
    
    ax = plt.subplot2grid((columns, rows),(4,0), colspan=3)
    age_kde_plot(df=df, ax=ax)
    
    plt.show()
original_plot(train_df)
def survival_plot_normed(df, ax, col='Sex', stacked=True):
    (df.groupby([col, 'Survived']).size().unstack().T / df.groupby([col, 'Survived']).size().unstack().sum(axis=1)).T.plot(kind='bar', stacked=stacked, ax=ax)
    plt.title("Survival by {}, (1 = Survived)".format(col))
original_plot(train_df, survival_plot_normed)
train_df['is_train'] = True
test_df['is_train'] = False
merged_df = pd.concat([train_df, test_df])
def survival_plot_merged(df, ax, col='Sex', stacked=False):
    (df.groupby([col, 'is_train']).size().unstack() / df.groupby([col, 'is_train']).size().unstack().sum(axis=0)).plot(kind='bar', ax=ax)
    plt.title("is_train by {}, (1 = Survived)".format(col))
    
def age_kde_plot_merged(df, ax):
    df.groupby('is_train')['Age'].plot(kind='kde')
    plt.xlabel("Age")
    plt.title("Age Distribution within merged_data")
    # sets our legend for our graph.
    plt.legend(('Train', 'Test'),loc='best') 
    
original_plot(merged_df, survival_plot=survival_plot_merged, age_kde_plot=age_kde_plot_merged)
train_df.to_csv('../datasets/new_train.csv')
test_df.to_csv('../datasets/new_test.csv')
merged_df.to_csv('../datasets/new_merged.csv')