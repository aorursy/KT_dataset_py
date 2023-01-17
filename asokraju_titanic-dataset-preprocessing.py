import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import random as rnd



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, log_loss

from sklearn.pipeline import Pipeline



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

#warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



dataset = [train,test]



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



pass_id = test_df["PassengerId"]
CORR_dir = []

OBS_dir = []

TASKS_dir = []



def CORR(x= ''):

    if x not in CORR_dir:

        CORR_dir.append(x)

    return CORR_dir

def OBS(x=''):

    if x not in OBS_dir:

        OBS_dir.append(x)

    return OBS_dir

def TASKS(x=''):

    if x not in TASKS_dir:

        TASKS_dir.append(x)

    return TASKS_dir
print("Basic information on TRAIN data")

print('='*40)

print(train.info())

print('='*40)

print('='*40)

print("Basic information on TEST data")

print('='*40)

print(test.info())

print('='*40)
print("Basic Statistics of TRAIN data")

print('='*40)

print(train.describe())

print('='*40)

print('='*40)

print("Basic Statistics of TEST data")

print('='*40)

print(test.describe())
print('='*40)

missing = train.isna().sum().sort_values(ascending = False)

missing_percent = round(train.isna().sum().sort_values(ascending = False)*100/train.shape[0],2)

missing_train = pd.concat([missing, missing_percent], axis = 1, keys = ['Total', 'Percent'])

missing_train = missing_train[missing_train.Total != 0]

print("Missing values in train data set")

print(missing_train)

print('='*40)

print('='*40)

missing = test.isna().sum().sort_values(ascending = False)

missing_percent = round(train.isna().sum().sort_values(ascending = False)*100/train.shape[0],2)

missing_test = pd.concat([missing, missing_percent], axis = 1, keys = ['Total', 'Percent'])

missing_test = missing_test[missing_test.Total != 0]

print("Missing values in test data set")

print(missing_test)

print('='*40)

#print('Our Primary task is to either drop/fill the missing values')

OBS('Train Missing Values Age-177,Cabin-687 and Embarked-2')

OBS('Test Missing Values Age-86,Cabin-327 and Fare-1')

TASKS('Fill/Drop Age/Cabin/Embarked missing value')

print(TASKS(''))

print(OBS(''))
Sex_train = pd.pivot_table(train, values='Survived', index='Sex', columns=None, aggfunc=[np.sum,'mean'])

Sex_train.columns = ['Total_sur','Percent_sur']

print(Sex_train)



fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.barplot(

    x='Sex',

    y='Survived',

    #hue='Sex',

    data=train,

    #palette=pal

)

ax.set_title('Sex vs Survived')

plt.show()

print("This shows that Sex is an very important feature.")

OBS('75% of women survived, and 19% men survived')
print("Correlation between Embarked and Survival")

Embarked_tot = train.Embarked.value_counts(normalize=False,ascending=False,dropna=True)

Embarked_per = train.Embarked.value_counts(normalize=True,ascending=False,dropna=True)*100

Embarked_sur = pd.pivot_table(

    train,

    values='Survived',

    index='Embarked',

    #columns='Sex',

    aggfunc=np.sum,

    fill_value=None,

    margins=False,

    dropna=True,

    margins_name='All')

Embarked_train = pd.concat([Embarked_tot,Embarked_per,Embarked_sur.Survived, Embarked_sur.Survived/Embarked_tot],

                           axis=1,

                           keys=['Tot_Emb', 'Percent_Emb','Tot_Sur', 'Percent_Sur'],

                           sort=True).sort_values('Tot_Emb',ascending=False)

print(Embarked_train)

#pal = {'S':"Red", 'C':"Green",'Q':'Blue'}



fig, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.barplot(

    x='Embarked',

    y='Survived',

    #hue='Sex',

    data=train,

#palette=pal

)

ax.set_title('Embarked vs Survived')

plt.show()

print("This shows that Embarked is an important feature to use: We should fill the missing values.")

OBS('Embarked is an important feature to use: We should fill the missing values.')
temp = round(pd.pivot_table(train, values='Survived', index=['Embarked'], columns=None, aggfunc=[np.sum,'mean']),2)

temp.columns = ['Total_sur','Percent_sur']

print(temp)

print("Over all people embarked from C has higher survival rates")

print('_'*40+'\n')

print('let us now see if this varies with Sex \n')

Embarked_sex_train = pd.pivot_table(train, values='Survived', index=['Embarked','Sex'], columns=None, aggfunc=[np.sum,'mean'])

Embarked_sex_train.columns = ['Total_sur','Percent_sur']

Embarked_sex_train.reindex(index = ['S','C','Q'], level = 0)

print(round(Embarked_sex_train,2))



fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.barplot(

    x='Embarked',

    y='Survived',

    hue='Sex',

    data=train,

    #palette=pal

)

ax.set_title('Embarked/Sex vs Survived')

plt.show()

print("31% of the males Embarked from C survived (this is very high compared to males embarked from other ports)")



OBS("31% of the males Embarked from C survived (this is very high compared to males embarked from other ports)")
Pclass_train = pd.pivot_table(train, values='Survived', index='Pclass', columns=None, aggfunc=[np.sum,'mean'])

Pclass_train.columns = ['Total_sur','Percent_sur']

print(round(Pclass_train,2))



fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.barplot(

    x='Pclass',

    y='Survived',

    #hue='Sex',

    data=train,

    #palette=pal

)

ax.set_title('Pclass vs Survived')

plt.show()

print("This shows that Pclass is also an important feature.")

OBS('Approx: 63% (136) class 1, 47% (87) class 2, 24% (119) class 3 passengers survived')
Pclass_sex_train = pd.pivot_table(train, values='Survived', index=['Pclass','Sex'], columns=None, aggfunc=[np.sum,'mean'])

Pclass_sex_train.columns = ['Total_sur','Percent_sur']

print(round(Pclass_sex_train,2))



fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.barplot(

    x='Pclass',

    y='Survived',

    hue='Sex',

    data=train,

    #palette=pal

)

ax.set_title('Pclass/Sex vs Survived')

plt.show()

print("96.8% of the 1st class females survived!")

print("Let us check if there is a correlation between Embarked, Sex, Survival")

OBS("Survived Females: 1st: 96.8% (91), 2nd: 92% (70), and 3rd: 50% (72)")

OBS("Survived Males: 1st: 37% (45), 2nd: 16% (17), and 3rd: 14% (47)")
Pclass_Emb_train = pd.pivot_table(train, values='Fare', index=['Pclass'], columns=['Embarked'], aggfunc=[lambda x:len(x)])

Pclass_Emb_train.rename(columns={'sum':'survived','<lambda>': 'Count'}, inplace=True)

Pclass_Emb_train.columns=['C','Q','S']

print(round(Pclass_Emb_train,2))



fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.barplot(

    x='Pclass',

    y='Survived',

    hue='Embarked',

    data=train,

    #palette=pal

)

ax.set_title('Pclass/Embarked vs Survived')

plt.show()

fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.barplot(

    x='Embarked',

    y='Survived',

    hue='Pclass',

    data=train,

    #palette=pal

)

ax.set_title('Embarked/Pclass vs Survived')

plt.show()
temp = round(pd.pivot_table(train, values='Survived', index=['Embarked'], columns='Pclass', aggfunc=['mean']),2)

print(temp)



print("Let us Visualize this with point plots")

print('_'*40+'\n')



grid =sns.FacetGrid(

    train,

    row='Embarked',

    height=3,

    palette=None,

    row_order=None,

    col_order=['1','2','3'],

    dropna=True,

    legend_out=True,

    despine=True,

    margin_titles=False,size=2.2, aspect=1.6

)

grid.map(sns.pointplot, 'Pclass','Survived',order = [1,2,3], alpha=.5)

grid.add_legend();

plt.show()

print("I assumed that as the Pclass increases the survival rate decreases. However, this changes for people embarked from Q.")

temp = round(pd.pivot_table(train, values='Survived', index=['Embarked'], columns='Pclass', aggfunc=['mean']),2)

print(temp)

OBS("For S,C as Pclass increses Survival decreses. However, for Q Class 2 had higher \nsurvival chances. (There might be some other factors)")
print("the fllowing table illustrate the correlations between all the four Catagorical Feature:")

print("Pclass, Embarked, Sex, Survival")

print('_'*80+'\n')



temp = round(pd.pivot_table(train, 

                            values='Survived', 

                            index=['Embarked','Sex'], 

                            columns='Pclass', 

                            aggfunc=[lambda x: len(x), np.sum,'mean'] ),2)



temp.rename(columns={'sum':'survived','<lambda>': 'travelled'}, inplace=True)

print(temp)

print('_'*80+'\n')

print("Above, on the left we have total number of passengers, on the middle the survived ones, \nand finally on the right we have the (mean)propabilities for survival.")

print('_'*80+'\n')



#grid = sns.FacetGrid(train_df, col='Embarked')

grid = sns.FacetGrid(train, row='Embarked', height=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order = [1,2,3], hue_order = ['male','female'])

grid.add_legend()

plt.show()



print('Only 2 first class and 3 second class passengers Embarked from Q ')

OBS('Only 2 first class and 3 second class passengers Embarked from Q ')

OBS("Out of 41 males Embarked from Q  only 3 had survived")

print("Out of 41 males Embarked from Q  only 3 had survived")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

train[train.Survived==1].Age.hist( ax = axes[0], color = 'darkblue',bins=20)

train[train.Survived==0].Age.hist(ax = axes[1], color = 'darkblue',bins=20)

axes[0].set_title("Survived")

axes[1].set_title("Expired")

axes[0].set_xlabel("Age")

axes[1].set_xlabel("Age")

axes[0].set_ylabel("Frequency")

axes[1].set_ylabel("Frequency")

# Setting the ylabel to '% change'

# ... YOUR CODE FOR TASK 7 ...

#axes[0].set_ylabel('# change')

#axes[1].set_ylabel('% change')

plt.suptitle('Age/Survived histograms')

plt.show()



print("it looks like there is a pattern with kids and old people")

print("Let us accuratly calculate the bin widths, by looking at the age bin widths using traditional histograms")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

temp1 = axes[0].hist(train[train.Survived==1].Age, bins=16)

temp2 = axes[1].hist(train[train.Survived==0].Age, bins=16)

axes[0].set_xlabel('Age')

axes[1].set_xlabel('Age')

axes[1].set_title('Expired')

axes[0].set_title('Survived')

axes[0].set_xlim([0,80])

axes[1].set_xlim([0,80])

axes[0].set_ylim([0,80])

axes[1].set_ylim([0,80])

plt.suptitle('Age/Survived Distributions')

plt.show()



print("# people Survived - # of people Expired")

print(temp1[0])

print(temp2[0])

print(temp1[0]-temp2[0])

print('-'*80)



plt.bar(np.linspace(0,80,16),temp1[0]-temp2[0])

plt.title("# people Survived - # of people Expired")

plt.xlabel("Age")

plt.ylabel('# of people')

print('There are three import observations here')

print('(1): An 80year old person survived, I think its safe to assume he/she is an outlier.')

print('(2): After removing him, we can see that everyone above 65years have expired.')

print('(3): Kids below 5years had a great chance of survival')

print('* Apart from these catogories I dont see any clear differences.')

print('** If we can find the moms of these babies, mostly she should have survived. But we dont have the data.')

OBS('An 80year old person survived, I think its safe to assume he/she is an outlier.')

OBS('everyone above 65years have expired (after removing the outlier)')

OBS('Kids below 5years had a great chance of survival')

print('-'*80)
# Make a strip plot of 'hp' grouped by 'cyl'

#plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

#sns.swarmplot(x='Embarked', y='Age',hue='Survived', data=train)

#plt.show()



pal = {'male':"green", 'female':"Pink"}



plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

sns.violinplot(x='Embarked', y='Age', data=train, inner =None, color='lightgray')

sns.swarmplot(x='Embarked',y='Age',data=train, size=4,hue='Sex',palette=pal)

plt.show()



print("All the people Embarked from Q with Age>35 did not survive")

print("All the people Embarked from Q with Age<10 did not survive")

print("All the people Embarked from C with Age<10 survived")

OBS('All the people Embarked from Q with Age>35 did not survive')

OBS('All the people Embarked from Q with Age<10 did not survive')

OBS('All the people Embarked from C with Age<10 survived')

print("-"*40)
fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.boxplot(

    x='Embarked',

    y='Age',

    hue='Pclass',

    data=train,

    #palette=pal

)

ax.set_title('Embarked/Pclass vs Age')

plt.show()

OBS('Class 3 has the Youngest people')

print("Class 3 has the Youngest people")
fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.boxplot(

    x='Embarked',

    y='Age',

    hue='Survived',

    data=train,

    #palette=pal

)

ax.set_title('Embarked/Survived vs Age')

plt.show()

print("Over all youger people has a Higher chance of survivel")
pal = {'male':"green", 'female':"Pink"}

fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.boxplot(

    x='Pclass',

    y='Age',

    hue='Sex',

    data=train,

    palette=pal

)

ax.set_title('Embarked/Survived vs Age')

plt.show()
# Make a strip plot of 'hp' grouped by 'cyl'

#plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

#sns.swarmplot(x='Pclass', y= 'Age', data=train,hue='Survived', size =3)

# Display the plot

#plt.show()

pal = {'male':"green", 'female':"Pink"}



plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

sns.violinplot(x='Pclass', y= 'Age', data=train,  inner =None, color='lightgray')

sns.swarmplot(x='Pclass',y='Age',data=train, size=5,hue='Sex',palette=pal)

plt.show()



print("All the second class people with age< 18 (approx) have survived")

print("-"*40)

print("Most class 3 have died and most people embarked from S have died. There might be a good correlation.")

print("Most of the old people with age > 65 (approx) have not survived")

print("A lot of Class 1 people with Age < 20 have survived")

OBS('All the 2nd class people with age< 18 (approx) have survived')

OBS('A lot of Class 1 people with Age < 20 have survived')

print("Particularly no new information")
fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.boxplot(

    x='Pclass',

    y='Age',

    hue='Survived',

    data=train,

    #palette=pal

)

ax.set_title('Pclass/Survived vs Age')

plt.show()

OBS("In every Class younger people had higher chance of survival")

print("In every Class younger people had higher chance of survival")
# Make a strip plot of 'hp' grouped by 'cyl'

#plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

#sns.swarmplot(x='Pclass', y= 'Age', data=train,hue='Survived', size =3)

# Display the plot

#plt.show()

plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

sns.violinplot(x='Pclass', y= 'Age', data=train,  inner =None, color='lightgray')

sns.swarmplot(x='Pclass',y='Age',data=train, size=5,hue='Survived')

plt.show()

temp = round(pd.pivot_table(train, 

                            values='Age', 

                            index=['Pclass'], 

                            columns=['Survived'], 

                            aggfunc=['count',lambda x: len(x)] ),2)

temp.rename(columns={'count':'Excluding NAN','<lambda>': 'Including NAN'}, inplace=True)

print(temp)

print('-'*40)



print("Histograms")

grid = sns.FacetGrid(train, row='Pclass', col = 'Survived', size=2.5, aspect=1.6)

grid.map(plt.hist, 'Age',bins =20)

grid.add_legend()

plt.show()

print('-'*40)

print("Normalized distributions")

grid = sns.FacetGrid(train, row='Pclass', col = 'Survived', size=2.5, aspect=1.6)

grid.map(sns.distplot, 'Age',color = 'g',bins=20,rug=True)

grid.add_legend()

plt.show()

OBS("All the kids Survived from PClass 2")

print("All the kids Survived from PClass 2")
print("No of people travelled")

temp = round(pd.pivot_table(train, 

                            values='Age', 

                            index=['Pclass','Sex'], 

                            columns=['Embarked'], 

                            aggfunc=[(lambda x: len(x))] ),2)

temp.rename(columns={'<lambda>': '#of obervations '}, inplace=True)

temp.columns=['C','Q','S']

temp_col = pd.Categorical(['S','C','Q'],ordered =True)

temp.reindex(temp_col, axis = 'columns')

print(temp)

print('-'*40)

print("Missing values in Age column")

temp = round(pd.pivot_table(train, 

                            values='Age', 

                            index=['Pclass','Sex'], 

                            columns=['Embarked'], 

                            aggfunc=[(lambda x: sum(x.isnull()))] ),2)

temp.rename(columns={'<lambda>': '#NAN-in-Age '}, inplace=True)

temp.columns=['C','Q','S']

temp.reindex(['S','C','Q'],axis = 'columns')

print(temp)

print('-'*40)

print("Median Age")

temp = round(pd.pivot_table(train, 

                            values='Age', 

                            index=['Pclass','Sex'], 

                            columns=['Embarked'], 

                            aggfunc=[(lambda x: x.quantile()),'mean'] ),2)

temp.rename(columns={'<lambda>': 'Median'}, inplace=True)

#temp.columns=['C','Q','S']

#temp.reindex(['S','C','Q'],axis = 'columns')

print(temp)

print('-'*40)

# Generate a joint plot of 'hp' and 'mpg'

sns.jointplot(x='Age',y='Fare',data=train, kind='hex')

sns.set(style="ticks")

# Display the plot

plt.show()

sns.jointplot(x='Age',y='Fare',data=train, kind='kde')



# Display the plot

plt.show()

OBS("there are quite a lot of people concetrated at 0<fare<30 and 16<age<25")

print("there are quite a lot of people concetrated at 0<fare<30 and 16<age<25")
sns.pairplot(train[['Fare','Pclass','Age']], hue='Pclass',hue_order = [1,2,3],kind='scatter')



OBS("Pclass doest completly depend on Fare")

OBS(" There is cluster where are all three classes coexist for the same fair (50$-100$)")

print("Pclass doest completly depend on Fare, There is cluster where are all three classes coexist for the same fair (50$-100$)")



sns.pairplot(train[['Fare','Survived','Age']], hue='Survived',hue_order = [0,1],kind='scatter')
sns.pairplot(train[['Fare','Sex','Age']], hue='Sex',hue_order = ['male','female'],kind='scatter')

OBS("there are no females above 65y age and all of tese have expired.")

print("there are no females above 65y age and all of tese have expired. ")
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()



print("Survival chances of C>S>Q")
# Generate a joint plot of 'hp' and 'mpg'

sns.jointplot(x='Age',y='Survived',data=train, kind='hex')



# Display the plot

plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

train[train.Survived==1].Fare.hist( ax = axes[0], color = 'darkblue',bins=20)

train[train.Survived==0].Fare.hist(ax = axes[1], color = 'darkblue',bins=20)

axes[0].set_title("Survived")

axes[1].set_title("Expired")

axes[0].set_xlabel("Fare")

axes[1].set_xlabel("Fare")

axes[0].set_ylim([0,380])

axes[1].set_ylim([0,380])

axes[0].set_ylabel("Frequency")

axes[1].set_ylabel("Frequency")

# Setting the ylabel to '% change'

# ... YOUR CODE FOR TASK 7 ...

#axes[0].set_ylabel('# change')

#axes[1].set_ylabel('% change')

plt.suptitle('Fare/Survived histograms')

plt.show()



OBS('Remove the outlier with 500 fare')

OBS("people with low fare has low chance of surival")

print("There is an outlier with fare 500!")

print("people with low fare has low chance of surival")

train = train[train.Fare < 500]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

temp1 = axes[0].hist(train[train.Survived==1].Fare, bins=16)

temp2 = axes[1].hist(train[train.Survived==0].Fare, bins=16)

axes[0].set_xlabel('Fare')

axes[1].set_xlabel('Fare')

axes[1].set_title('Expired')

axes[0].set_title('Survived')

axes[0].set_ylim([0,380])

axes[1].set_ylim([0,380])

axes[0].set_xlim([0,300])

axes[1].set_xlim([0,300])

plt.suptitle('Fare/Survived Distributions')

plt.show()



print("# people Survived - # of people Expired")

print(temp1[0])

print(temp2[0])

print(temp1[0]-temp2[0])

print('-'*80)



plt.bar(np.linspace(0,80,16),temp1[0]-temp2[0])

plt.title("# people Survived - # of people Expired")

plt.xlabel("Fare")

plt.ylabel('# of people')

print('There are three import observations here')

OBS('Kids below 5years had a great chance of survival')

print('-'*80)



print("This looks like the #passengers survived ossillates with the price")

print("Banding the price might help")




fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

train[train.Sex=='male'].Fare.hist( ax = axes[0], color = 'darkblue',bins=20)

train[train.Sex=='female'].Fare.hist(ax = axes[1], color = 'darkblue',bins=20)

axes[0].set_title("male")

axes[1].set_title("female")

axes[0].set_xlabel("Fare")

axes[1].set_xlabel("Fare")

axes[0].set_ylim([0,380])

axes[1].set_ylim([0,380])

axes[0].set_ylabel("Frequency")

axes[1].set_ylabel("Frequency")

# Setting the ylabel to '% change'

# ... YOUR CODE FOR TASK 7 ...

#axes[0].set_ylabel('# change')

#axes[1].set_ylabel('% change')

plt.suptitle('Fare/Sex histograms')

plt.show()

#train = train[train.Fare < 500]



print("A lot people are in between 0-50 dollars")

print("let us try some logathamic binning")



bins=np.logspace(np.log10(1),np.log10(300), 5)

bins = [0,8,16,30,300]

print(np.round(bins,1))



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

train[train.Sex=='male'].Fare.hist( ax = axes[0], color = 'darkblue',bins=bins)

train[train.Sex=='female'].Fare.hist(ax = axes[1], color = 'darkblue',bins=bins)

axes[0].set_title("male")

axes[1].set_title("female")

axes[0].set_xlabel("Fare")

axes[1].set_xlabel("Fare")

axes[1].set_xticks(bins)

axes[0].set_ylim([0,380])

axes[1].set_ylim([0,380])

axes[0].set_ylabel("Frequency")

axes[1].set_ylabel("Frequency")

# Setting the ylabel to '% change'

# ... YOUR CODE FOR TASK 7 ...

#axes[0].set_ylabel('# change')

#axes[1].set_ylabel('% change')

plt.suptitle('Fare/Sex histograms')

plt.show()



print("new bins sizes (exponential)")

bins = [0,8,16,30,300]

train.hist('Fare',bins=bins)

plt.show()





bins = [0,16,32, 48, 64,300]

train.hist('Fare',bins=bins)

plt.show()



bins = [0,8,15, 31,300]

train.hist('Fare',bins=bins)

plt.show()
fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (8,6))

sns.barplot(

    x='Survived',

    y='Fare',

    data=train,

)

ax.set_title('Survived vs Fare')

plt.show()
pal = {'male':"green", 'female':"Pink"}

fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (10,8))

sns.barplot(

    x='Embarked',

    y='Fare',

    hue='Sex',

    data=train,

    palette=pal

)

ax.set_title('Embarked/Sex vs Fare')

plt.show()
fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.barplot(

    x='Embarked',

    y='Fare',

    hue='Pclass',

    data=train,

    #palette=pal

)

ax.set_title('Embarked/Pclass vs Fare')

plt.show()
fix, ax = plt.subplots(nrows=1,ncols=1,figsize = (12,6))

sns.barplot(

    x='Embarked',

    y='Fare',

    hue='Survived',

    data=train,

    #palette=pal

)

ax.set_title('Embarked/Survived vs Fare')

plt.show()
print("No of people travelled")

temp = round(pd.pivot_table(train, 

                            values='Age', 

                            index=['Pclass','Sex'], 

                            columns=['Embarked'], 

                            aggfunc=[(lambda x: len(x))] ),2)

temp.rename(columns={'<lambda>': '#of obervations '}, inplace=True)

temp.columns=['C','Q','S']

temp_col = pd.Categorical(['S','C','Q'],ordered =True)

temp.reindex(temp_col, axis = 'columns')

print(temp)

print('-'*40)

print("Missing values in Age column")

temp = round(pd.pivot_table(train, 

                            values='Age', 

                            index=['Pclass','Sex'], 

                            columns=['Embarked'], 

                            aggfunc=[(lambda x: sum(x.isnull()))] ),2)

temp.rename(columns={'<lambda>': '#NAN-in-Age '}, inplace=True)

temp.columns=['C','Q','S']

temp.reindex(['S','C','Q'],axis = 'columns')

print(temp)

print('-'*40)

print("Median Age")

temp = round(pd.pivot_table(train, 

                            values='Age', 

                            index=['Pclass','Sex'], 

                            columns=['Embarked'], 

                            aggfunc=[(lambda x: x.quantile()),'mean'] ),2)

temp.rename(columns={'<lambda>': 'Median'}, inplace=True)

#temp.columns=['C','Q','S']

#temp.reindex(['S','C','Q'],axis = 'columns')

print(temp)

print('-'*40)

for i, temp in enumerate(OBS_dir):

    print('({}) '.format(i+1)+temp)
y_train = train.Survived

train.drop('Survived', axis = 1, inplace = True)
print(y_train.shape)

print(train.shape)
print('Before')

print(train.head())

train.drop('PassengerId', axis= 1, inplace = True)

test.drop('PassengerId', axis= 1, inplace = True)

print("After")

print(train.head())
train["Title"] = train.Name.str.extract(' ([A-Za-z]+)\.',expand = False)

train['Name_len'] = train.Name.map(lambda x: len(x))

test["Title"] = test.Name.str.extract(' ([A-Za-z]+)\.',expand = False)

test['Name_len'] = test.Name.map(lambda x: len(x))

train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')

train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')

train.drop('Name',axis=1,inplace=True)

print(train.head())

test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test['Title'] = test['Title'].replace(['Mlle', 'Ms'], 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')

test.drop('Name',axis=1,inplace=True)

print(test.head())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

train['Title'] = train['Title'].map(title_mapping)

train['Title'] = train['Title'].fillna(0)

test['Title'] = test['Title'].map(title_mapping)

test['Title'] = test['Title'].fillna(0)



print(y_train.shape)

print(train.shape)
train['Has_cabin'] = train['Cabin'].apply(lambda x: 0 if type(x)==float else 1)

train.drop('Cabin',axis=1,inplace=True)

print(train.head())

test['Has_cabin'] = test['Cabin'].apply(lambda x: 0 if type(x)==float else 1)

test.drop('Cabin',axis=1,inplace=True)

print(test.head())



print(y_train.shape)

print(train.shape)
train['Has_cabin'].value_counts()
train.drop('Has_cabin',axis=1,inplace =True)

test.drop('Has_cabin',axis=1,inplace =True)
train.drop('Ticket',axis=1,inplace =True)

test.drop('Ticket',axis=1,inplace =True)
print(train.head())

print(test.head())

print(y_train.shape)

print(train.shape)
test_train = pd.concat([train,test],axis =0)

test_train.index



print(y_train.shape)

print(train.shape)


print("No of people travelled")

temp = round(pd.pivot_table(test_train, 

                            values='Age', 

                            index=['Pclass','Sex'], 

                            columns=['Embarked'], 

                            aggfunc=[(lambda x: len(x))] ),2).astype(int)

temp.rename(columns={'<lambda>': '#of obervations '}, inplace=True)

temp.columns=['C','Q','S']

temp_col = pd.Categorical(['S','C','Q'],ordered =True)

temp.reindex(temp_col, axis = 'columns')

print(temp)

print('-'*40)

print("Number of Missing values in Age column")

temp = round(pd.pivot_table(test_train, 

                            values='Age', 

                            index=['Pclass','Sex',], 

                            columns=['Embarked'], 

                            aggfunc=[(lambda x: sum(x.isnull()))] ),2).astype(int)

temp.rename(columns={'<lambda>': '#NAN-in-Age '}, inplace=True)

temp.columns=['C','Q','S']

temp.reindex(['S','C','Q'],axis = 'columns')

print(temp)

print('-'*40)

print("Median Age")

temp = round(pd.pivot_table(test_train, 

                            values='Age', 

                            index=['Pclass','Sex'], 

                            columns=['Embarked'], 

                            aggfunc=[(lambda x: x.quantile()),'mean', 'std'] ),2)

temp.rename(columns={'<lambda>': 'Median'}, inplace=True)

#temp.columns=['C','Q','S']

#temp.reindex(['S','C','Q'],axis = 'columns')

print(temp)

print('-'*40)
test_train = pd.concat([train,test], axis =0, ignore_index=True)

test_train.head()
for sex in ['male','female']:

    for pclass in range(1,4):

        for emb in ['S','C','Q']:

            for title in range(1,6):

                new_train = test_train[(test_train['Sex']==sex) & 

                                 (test_train['Pclass'] == pclass) & 

                                 (test_train['Embarked'] == emb) & 

                                 (test_train['Title'] == title)]

                total_features = new_train.shape[0]

                null_features = new_train['Age'].isnull().sum()

                if null_features >0:

                    if total_features-null_features > 1:

                        new_train_dropped_na = new_train.dropna()

                        mean, std = new_train_dropped_na.Age.mean(), new_train_dropped_na.Age.std()

                        test_train.loc[list(new_train[new_train.Age.isna()].index),'Age'] = np.random.randint(mean-std, mean+std, size = null_features).astype(float)

                    elif total_features-null_features >= 0:

                        new_train = test_train[(test_train['Title'] == title)]

                        null_features = new_train['Age'].isnull().sum()

                        #print(null_features)

                        new_train_dropped_na = new_train.dropna()

                        mean, std = new_train_dropped_na.Age.mean(), new_train_dropped_na.Age.std()

                        test_train.loc[list(new_train[new_train.Age.isna()].index),'Age'] = np.random.randint(mean-std, mean+std, size = null_features).astype(float)

test_train.Age = test_train.Age.astype(int)
test_train.Age.isnull().sum()
assert test_train.Age.isnull().sum()==0
train = test_train.iloc[:888]

test = test_train.iloc[888:]

dataset = [train,test]

print(y_train.shape)

print(train.shape)

print(test.shape)

train.loc[ train['Age'] <= 16, 'Age'] = 0

train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1

train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2

train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3

train.loc[ train['Age'] > 64, 'Age'] = 4



test.loc[ test['Age'] <= 16, 'Age'] = 0

test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1

test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2

test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3

test.loc[ test['Age'] > 64, 'Age'] = 4

print(train.head())

print(test.head())
train.loc[:,'Sex'] = train.loc[:,'Sex'].map( {'female': 0, 'male': 1} ).astype(int)

test.loc[:,'Sex'] = test.loc[:,'Sex'].map( {'female': 0, 'male': 1} ).astype(int)



train.head()
print(train[train.Embarked.isna()])

print(test[test.Embarked.isna()])
train.loc[:,'Embarked'] = train.loc[:,'Embarked'].fillna('S')

test.loc[:,'Embarked'] = test.loc[:,'Embarked'].fillna('S')
print(train[train.Embarked.isna()])

print(test[test.Embarked.isna()])
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()
print(train.shape)

print(y_train.shape)

"We removed it earlier"
print(train.Fare.isnull().sum())

print(test.Fare.isnull().sum())

print("I will fill it with the median value")
test.loc[:,'Fare'] = test.loc[:,'Fare'].fillna(test.loc[:,'Fare'].median())

print(test.Fare.isnull().sum())
#train['FareBand'] = pd.qcut(train.loc[:,'Fare'], 4)
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0

train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1

train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare'] = 2

train.loc[(train['Fare'] > 31), 'Fare'] = 3

#train.drop('FareBand ', axis =1, inplace =True)

test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0

test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1

test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare'] = 2

test.loc[(test['Fare'] > 31), 'Fare'] = 3

print(train.head())

print(test.head())
train.loc[:,'Family_size'] = train.loc[:,'SibSp'] + train.loc[:,'Parch'] + 1

train.loc[:,'isAlone'] = 0

train.loc[train.loc[:,'Family_size'] == 1,'isAlone'] = 1

print(train.head())

test.loc[:,'Family_size'] = test.loc[:,'SibSp'] + test.loc[:,'Parch'] + 1

test.loc[:,'isAlone'] = 0

test.loc[test.loc[:,'Family_size'] == 1,'isAlone'] = 1

print(test.head())
train.drop(['SibSp','Parch'], axis =1 , inplace= True)

test.drop(['SibSp','Parch'], axis =1 , inplace= True)
print(train.head())

print(test.head())
train.to_csv('train_pp.csv')

test.to_csv('test_pp.csv')

y_train.to_csv('y_pp.csv')
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)

plt.show()
# Setup the pipeline steps: steps

steps = [('scaler', StandardScaler()),

        ('knn', KNeighborsClassifier())]

        

# Create the pipeline: pipeline

pipeline = Pipeline(steps)



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(train,y_train,test_size = 0.3,random_state=42)



# Fit the pipeline to the training set: knn_scaled

knn_scaled = pipeline.fit(X_train,y_train)



# Instantiate and fit a k-NN classifier to the unscaled data

knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)



# Compute and print metrics

print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test,y_test)))

print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test,y_test)))

print("Scaling helps!")

y_pred = pipeline.predict(test)

submission = pd.DataFrame({

        "PassengerId": pass_id,

        "Survived": y_pred

    })



#submission.to_csv('submission.csv')