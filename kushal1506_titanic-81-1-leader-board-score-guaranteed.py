# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization(for EDA)

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



#We will use the popular scikit-learn library to develop our machine learning algorithms



# Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc



# Models

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB



import string



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
# link --->https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/

df_test = pd.read_csv("../input/titanic/test.csv")

df_train = pd.read_csv("../input/titanic/train.csv")



# link---> w3resource.com/pandas/concat.php

def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_df(all_data):

    # Use DataFrame.loc attribute to access a particular cell in the given Dataframe using the index and column labels.

    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

    # Returns divided dfs of training and test set 



df_all = concat_df(df_train, df_test)



df_train.name = 'Training Set'

df_test.name = 'Test Set'

df_all.name = 'All Set' 



dfs = [df_train, df_test]  # List consisting of both Train and Test set



# Pls note:- df_all and dfs is not same (df_all is a Dataframe and dfs is a list)
# Pandas sample() is used to generate a sample random row or column from the function caller data frame.

df_all.sample(10)
#preview data

print (df_train.info()) # link ---> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
#df_train.head() # link --> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html

#df_train.tail() # link --> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html



df_train.sample(10) # link --> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html



#If u look at the 'cabin' Feature , u can see 'NAN' depicting missing values. 
df_test.info()

df_test.sample(10) #https://www.geeksforgeeks.org/python-pandas-dataframe-sample/
df_train.describe() #link --> https://www.geeksforgeeks.org/python-pandas-dataframe-describe-method/
# link --> https://www.geeksforgeeks.org/matplotlib-pyplot-subplots-in-python/

# link --> https://www.geeksforgeeks.org/plot-a-pie-chart-in-python-using-matplotlib/

# link --> https://www.geeksforgeeks.org/countplot-using-seaborn-in-python/



f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots 

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)



ax[0].set_title('Survived') 

ax[0].set_ylabel('')



sns.countplot('Survived',data=df_train,ax=ax[1])



ax[1].set_title('Survived') # ax[0] & ax[1] are different axis for different plots.



plt.show()
# Counting the total missing values in respective features

total_missing_train = df_train.isnull().sum().sort_values(ascending=False)



# Calculating the percent of missing values in respective features

percent_1 = df_train.isnull().sum()/df_train.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False) # Rounding the percent calculated in percent_1 to one decimal.



#w3resource.com/pandas/concat.php

train_missing_data = pd.concat([total_missing_train, percent_2], axis=1, keys=['Total', '%'])



print(total_missing_train)



print('_'*25)



train_missing_data.head(5) # prints/shows top 5 rows of dataframe
total_missing_test = df_test.isnull().sum().sort_values(ascending=False)



percent_3 = df_test.isnull().sum()/df_test.isnull().count()*100

percent_4 = (round(percent_3, 1)).sort_values(ascending=False) 



test_missing_data = pd.concat([total_missing_test, percent_4], axis=1, keys=['Total', '%']) #w3resource.com/pandas/concat.php



print(total_missing_test)



print('_'*25)



test_missing_data.head(5)
# link --> https://www.geeksforgeeks.org/matplotlib-pyplot-subplots-in-python/

f,ax=plt.subplots(figsize=(18,8))



# link --> https://seaborn.pydata.org/generated/seaborn.violinplot.html

sns.violinplot("Pclass","Age", hue="Survived", data=df_train,split=True,ax=ax)



ax.set_title('Pclass and Age vs Survived')



ax.set_yticks(range(0,110,10)) # set_yticks() function in axes module is used to Set the y ticks with list of ticks.



plt.show()
# link --> https://www.geeksforgeeks.org/python-pandas-dataframe-corr/

df_all_corr = df_all.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()



df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)



df_all_corr[df_all_corr['Feature 1'] == 'Pclass'] 
f,ax=plt.subplots(figsize=(18,8))



# link --> http://alanpryorjr.com/visualizations/seaborn/violinplot/violinplot/

sns.violinplot("Sex","Age", hue="Survived", data=df_train,split=True,ax=ax)



ax.set_title('Sex and Age vs Survived') # setting the title of plot



ax.set_yticks(range(0,110,10))



plt.show()
# link ---> https://www.geeksforgeeks.org/python-pandas-dataframe-groupby/

age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']



for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {} '.format(pclass, sex, age_by_pclass_sex[sex][pclass].astype(int)))



# Filling the missing values in Age with the medians of Sex and Pclass groups

df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

# link --> https://www.w3schools.com/python/python_lambda.asp
# link --> https://www.geeksforgeeks.org/python-seaborn-factorplot-method/

sns.factorplot('Embarked','Survived',data=df_train)

fig=plt.gcf() # pyplot. gcf() is primarily used to get the current figure. 

fig.set_size_inches(5,3)

plt.show()
df_all[df_all['Embarked'].isnull()]
# Filling the missing values in Embarked with S

df_all['Embarked'] = df_all['Embarked'].fillna('S')

# link --> https://www.geeksforgeeks.org/python-pandas-dataframe-fillna-to-replace-null-values-in-dataframe/
# link --> https://www.kaggle.com/residentmario/faceting-with-seaborn

FacetGrid = sns.FacetGrid(df_train, row='Embarked', size=4.5, aspect=1.6)



# link --> https://www.geeksforgeeks.org/python-seaborn-pointplot-method/

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')



FacetGrid.add_legend() # Draw a legend, maybe placing it outside axes and resizing the figure.
df_all[df_all['Fare'].isnull()]
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0]

# Median of a Fare satisying condition([3][0][0] -- 3=Pclass,0=Parch,SibSp) 



# Filling the missing value in Fare with the median Fare of 3rd class alone passenger

df_all['Fare'] = df_all['Fare'].fillna(med_fare)
# link --> https://www.geeksforgeeks.org/seaborn-barplot-method-in-python/

sns.barplot(x='Pclass', y='Survived',hue='Sex',data=df_train)
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)



grid.map(plt.hist, 'Age', alpha=.5, bins=20)



grid.add_legend();
data1=df_train.copy() # shallow copy

data1['Family_size'] = data1['SibSp'] + data1['Parch'] +1

# 1 is considered 'Alone'



data1['Family_size'].value_counts().sort_values(ascending=False)
axes = sns.factorplot('Family_size','Survived', data=data1, aspect = 2.5, )
# Creating Deck column by extracting the first letter of the Cabin(string s) column M stands for Missing

df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')





df_all_decks = df_all.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 

                                                                        'Fare', 'Embarked', 'Cabin', 'PassengerId', 

                                                                        'Ticket']).rename(columns={'Name': 'Count'})



df_all_decks
# Transpose is done for accessbility

df_all_decks=df_all_decks.transpose()
def get_pclass_dist(df):

    

    # Creating a dictionary for every passenger class count in every deck

    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}

    

    #Deck column is extracted from df_all_decks 

    decks = df.columns.levels[0]    

    

    # Creating a new dataframe just a copy of df_all_decks with 0 in respective Pclass if empty ... See Output below.

    # Start

    for deck in decks:

        for pclass in range(1, 4):

            try:

                count = df[deck][pclass][0]

                deck_counts[deck][pclass] = count 

            except KeyError:

                deck_counts[deck][pclass] = 0

                

    df_decks = pd.DataFrame(deck_counts) 

    # End

    

    deck_percentages = {}

   

    # Creating a dictionary for every passenger class percentage in every deck

    for col in df_decks.columns:

        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]

        

    return deck_counts, deck_percentages,df_decks





all_deck_count, all_deck_per,df_decks_return = get_pclass_dist(df_all_decks)



print(df_decks_return)



print("_"*25)



all_deck_per
def display_pclass_dist(percentages):

    

    #converting dictionary to dataframe and then transpose

    df_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')

    bar_count = np.arange(len(deck_names))  

    bar_width = 0.85

    

    pclass1 = df_percentages[0]

    pclass2 = df_percentages[1]

    pclass3 = df_percentages[2]

    

    plt.figure(figsize=(20, 10))

    

    # link --> https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm

    plt.bar(bar_count, pclass1,width=bar_width,edgecolor='white',label='Passenger Class 1')

    plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width, label='Passenger Class 2')

    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width, label='Passenger Class 3')



    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='best',bbox_to_anchor=(1, 1),prop={'size': 15})

    plt.title('Passenger Class Distribution in Decks',size=18, y=1.05)   

    

    plt.show()    

    

display_pclass_dist(all_deck_per)    
# Passenger in the T deck is changed to A

idx = df_all[df_all['Deck'] == 'T'].index

df_all.loc[idx, 'Deck'] = 'A'
# Same Method is applied as above just this time , deck is grouped with 'Survived' Feature



df_all_decks_survived = df_all.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 

                                                                                   'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()



def get_survived_dist(df):

    

    # Creating a dictionary for every survival count in every deck

    surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}

    decks = df.columns.levels[0]    



    for deck in decks:

        for survive in range(0, 2):

            surv_counts[deck][survive] = df[deck][survive][0]

            

    df_surv = pd.DataFrame(surv_counts)

    surv_percentages = {}



    for col in df_surv.columns:

        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]

        

    return surv_counts, surv_percentages



def display_surv_dist(percentages):

    

    df_survived_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')

    bar_count = np.arange(len(deck_names))  

    bar_width = 0.85    



    not_survived = df_survived_percentages[0]

    survived = df_survived_percentages[1]

    

    plt.figure(figsize=(20, 10))

    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")

    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")

 

    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Survival Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)    

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

    plt.title('Survival Percentage in Decks', size=18, y=1.05)

    

    plt.show()



all_surv_count, all_surv_per = get_survived_dist(df_all_decks_survived)

display_surv_dist(all_surv_per)
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')

df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')

df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')



df_all['Deck'].value_counts()
# Dropping the Cabin feature

df_all.drop(['Cabin'], inplace=True, axis=1)



df_train, df_test = divide_df(df_all)

dfs = [df_train, df_test]



for df in dfs:

    print(df_test.isnull().sum())

    print('-'*25)
cont_features = ['Age', 'Fare']

surv = df_train['Survived'] == 1



fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))

plt.subplots_adjust(right=1.5)



for i, feature in enumerate(cont_features): # link --> https://www.geeksforgeeks.org/enumerate-in-python/   

    # Distribution of survival in feature

    sns.distplot(df_train[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i]) 

    # [-surv] means "Not Survived"

    sns.distplot(df_train[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

    

    # Distribution of feature in dataset

    sns.distplot(df_train[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])

    sns.distplot(df_test[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

    

    axs[0][i].set_xlabel('')

    axs[1][i].set_xlabel('')

     

    # just providing the ticks for x & y axis in respective plots    

    for j in range(2):        

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

    

    axs[0][i].legend(loc='upper right', prop={'size': 20})

    axs[1][i].legend(loc='upper right', prop={'size': 20})

    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)



axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)

axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)

        

plt.show()
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']



fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(cat_features, 1):    

    plt.subplot(2, 3, i)

    sns.countplot(x=feature, hue='Survived', data=df_train)

    

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15)    

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})

    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)



plt.show()
df_all = concat_df(df_train, df_test)

df_all.head()
# link ---> https://likegeeks.com/seaborn-heatmap-tutorial/

sns.heatmap(df_all.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
df_all['Fare'] = pd.qcut(df_all['Fare'], 13) # visit the link above
fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='Fare', hue='Survived', data=df_all)



plt.xlabel('Fare', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)



plt.show()
df_all['Age'] = pd.qcut(df_all['Age'], 10)
fig, axs = plt.subplots(figsize=(22, 9))

sns.countplot(x='Age', hue='Survived', data=df_all)



plt.xlabel('Age', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Survival Counts in {} Feature'.format('Age'), size=15, y=1.05)



plt.show()
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1



fig, axs = plt.subplots(figsize=(20, 20), ncols=2, nrows=2)

plt.subplots_adjust(right=1.5)



sns.barplot(x=df_all['Family_Size'].value_counts().index, y=df_all['Family_Size'].value_counts().values, ax=axs[0][0])

sns.countplot(x='Family_Size', hue='Survived', data=df_all, ax=axs[0][1])



axs[0][0].set_title('Family Size Feature Value Counts', size=20, y=1.05)

axs[0][1].set_title('Survival Counts in Family Size ', size=20, y=1.05)



# Mapping Family Size

family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)



sns.barplot(x=df_all['Family_Size_Grouped'].value_counts().index, y=df_all['Family_Size_Grouped'].value_counts().values, ax=axs[1][0])

sns.countplot(x='Family_Size_Grouped', hue='Survived', data=df_all, ax=axs[1][1])



axs[1][0].set_title('Family Size Feature Value Counts After Grouping', size=20, y=1.05)

axs[1][1].set_title('Survival Counts in Family Size After Grouping', size=20, y=1.05)





for i in range(2):

    axs[i][1].legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 20})

    for j in range(2):

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

        axs[i][j].set_xlabel('')

        axs[i][j].set_ylabel('')



plt.show()
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')
fig, axs = plt.subplots(figsize=(12, 9))

sns.countplot(x='Ticket_Frequency', hue='Survived', data=df_all)



plt.xlabel('Ticket Frequency', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y=1.05)



plt.show()
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

# https://www.w3schools.com/python/ref_string_split.asp



df_all['Is_Married'] = 0

df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1
fig, axs = plt.subplots(nrows=2, figsize=(20, 20))

sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[0])



axs[0].tick_params(axis='x', labelsize=10)

axs[1].tick_params(axis='x', labelsize=15)



for i in range(2):    

    axs[i].tick_params(axis='y', labelsize=15)



axs[0].set_title('Title Feature Value Counts', size=20, y=1.05)



df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')

df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')



sns.barplot(x=df_all['Title'].value_counts().index, y=df_all['Title'].value_counts().values, ax=axs[1])

axs[1].set_title('Title Feature Value Counts After Grouping', size=20, y=1.05)



plt.show()
df_train,df_test= divide_df(df_all)

dfs=[df_train,df_test]
df_all['Name'].sample(10)
def extract_surname(data):    

    

    families = []

    

    for i in range(len(data)):  

        name = data.iloc[i]



        if '(' in name:

            name_no_bracket = name.split('(')[0] 

        else:

            name_no_bracket = name

            

        family = name_no_bracket.split(',')[0]

        title = name_no_bracket.split(',')[1].strip().split(' ')[0]

        

        for c in string.punctuation:

            family = family.replace(c, '').strip()

            

        families.append(family)

            

    return families



df_all['Family'] = extract_surname(df_all['Name'])

df_train = df_all.loc[:890]

df_test = df_all.loc[891:]

dfs = [df_train, df_test]
# Creating a list of families and tickets that are occuring in both training and test set

non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]

non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]



df_family_survival_rate = df_train.groupby('Family')['Survived', 'Family','Family_Size'].median()

df_ticket_survival_rate = df_train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()



family_rates = {}

ticket_rates = {}



for i in range(len(df_family_survival_rate)):

    # Checking a family exists in both training and test set, and has members more than 1

    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:

        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]



for i in range(len(df_ticket_survival_rate)):

    # Checking a ticket exists in both training and test set, and has members more than 1

    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:

        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]
mean_survival_rate = np.mean(df_train['Survived'])



train_family_survival_rate = []

train_family_survival_rate_NA = []

test_family_survival_rate = []

test_family_survival_rate_NA = []



for i in range(len(df_train)):

    if df_train['Family'][i] in family_rates:

        train_family_survival_rate.append(family_rates[df_train['Family'][i]])

        train_family_survival_rate_NA.append(1)

    else:

        train_family_survival_rate.append(mean_survival_rate)

        train_family_survival_rate_NA.append(0)

        

for i in range(len(df_test)):

    if df_test['Family'].iloc[i] in family_rates:

        test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])

        test_family_survival_rate_NA.append(1)

    else:

        test_family_survival_rate.append(mean_survival_rate)

        test_family_survival_rate_NA.append(0)

        

df_train['Family_Survival_Rate'] = train_family_survival_rate

df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA

df_test['Family_Survival_Rate'] = test_family_survival_rate

df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA



train_ticket_survival_rate = []

train_ticket_survival_rate_NA = []

test_ticket_survival_rate = []

test_ticket_survival_rate_NA = []



for i in range(len(df_train)):

    if df_train['Ticket'][i] in ticket_rates:

        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])

        train_ticket_survival_rate_NA.append(1)

    else:

        train_ticket_survival_rate.append(mean_survival_rate)

        train_ticket_survival_rate_NA.append(0)

        

for i in range(len(df_test)):

    if df_test['Ticket'].iloc[i] in ticket_rates:

        test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])

        test_ticket_survival_rate_NA.append(1)

    else:

        test_ticket_survival_rate.append(mean_survival_rate)

        test_ticket_survival_rate_NA.append(0)

        

df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate

df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA

df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate

df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA
for df in [df_train, df_test]:

    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2

    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2    
non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'Family_Size_Grouped', 'Age', 'Fare']



for df in dfs:

    for feature in non_numeric_features:        

        df[feature] = LabelEncoder().fit_transform(df[feature])
onehot_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'Family_Size_Grouped']

encoded_features = []



for df in dfs:

    for feature in onehot_features:

        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()

        n = df[feature].nunique()

        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df.index

        encoded_features.append(encoded_df)



# *encoded_features gives all encoded features of each of Six onehot_features         

df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)

df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)
df_all = concat_df(df_train, df_test)



# Dropping Un-needed feature

drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Survived',

             'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title',

            'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']



df_all.drop(columns=drop_cols, inplace=True)

df_all.head()
X = df_train.drop(columns=drop_cols)
X_train = StandardScaler().fit_transform(X)

Y_train = df_train['Survived'].values

X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))



print('X_train shape: {}'.format(X_train.shape))

print('Y_train shape: {}'.format(Y_train.shape))

print('X_test shape: {}'.format(X_test.shape))
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)



sgd.score(X_train, Y_train)



acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# KNN 

knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
gaussian = GaussianNB() 

gaussian.fit(X_train, Y_train)  

Y_pred = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)



Y_pred = perceptron.predict(X_test)



acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)  

Y_pred = decision_tree.predict(X_test)  

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
# Link ---> ttps://stackoverflow.com/questions/25006369/what-is-sklearn-cross-validation-cross-val-score

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100,oob_score=True)

scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
rf.fit(X_train, Y_train)

Y_prediction = rf.predict(X_test)



rf.score(X_train, Y_train)



acc_random_forest = round(rf.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
importances = pd.DataFrame({'feature':X.columns,'importance':np.round(rf.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(26)
importances.plot.bar()
random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)

random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)



acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
print("oob score:", round(rf.oob_score_, 4)*100, "%")
random_forest = RandomForestClassifier(criterion='gini',

                                           n_estimators=1750,

                                           max_depth=7,

                                           min_samples_split=6,

                                           min_samples_leaf=6,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=42,

                                           n_jobs=-1,

                                           verbose=1) 

random_forest.fit(X_train, Y_train)

Y_prediction = (random_forest.predict(X_test)).astype(int)



random_forest.score(X_train, Y_train)



print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
from sklearn.model_selection import StratifiedKFold

N = 5

oob = 0

probs = pd.DataFrame(np.zeros((len(X_test), N * 2)), columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])

fprs, tprs, scores = [], [], []



skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)



for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train), 1):

    print('Fold {}\n'.format(fold))

    

    # Fitting the model

    random_forest.fit(X_train[trn_idx], Y_train[trn_idx])

    

    # Computing Train AUC score

    trn_fpr, trn_tpr, trn_thresholds = roc_curve(Y_train[trn_idx], random_forest.predict_proba(X_train[trn_idx])[:, 1])

    trn_auc_score = auc(trn_fpr, trn_tpr)

    # Computing Validation AUC score

    val_fpr, val_tpr, val_thresholds = roc_curve(Y_train[val_idx],random_forest.predict_proba(X_train[val_idx])[:, 1])

    val_auc_score = auc(val_fpr, val_tpr)  

      

    scores.append((trn_auc_score, val_auc_score))

    fprs.append(val_fpr)

    tprs.append(val_tpr)

    

    # X_test probabilities

    probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = random_forest.predict_proba(X_test)[:, 0]

    probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = random_forest.predict_proba(X_test)[:, 1]

        

    oob += random_forest.oob_score_ / N

    print('Fold {} OOB Score: {}\n'.format(fold, random_forest.oob_score_))   

    

print('Average OOB Score: {}'.format(oob))
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100,oob_score=True)

scores = cross_val_score(random_forest, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)

confusion_matrix(Y_train, predictions)
from sklearn.metrics import precision_score, recall_score



print("Precision:", precision_score(Y_train, predictions))

print("Recall:",recall_score(Y_train, predictions))
from sklearn.metrics import f1_score

f1_score(Y_train, predictions)
from sklearn.metrics import roc_curve



# getting the probabilities of our predictions

y_scores = random_forest.predict_proba(X_train)

y_scores = y_scores[:,1]



# compute true positive rate and false positive rate

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)

# plotting them against each other

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
from sklearn.metrics import roc_auc_score

r_a_score = roc_auc_score(Y_train, y_scores)

print("ROC-AUC-Score:", r_a_score)
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": Y_prediction

    })



submission.to_csv('submission.csv', index=False)
data=pd.read_csv("submission.csv")

data.head(10)