import numpy as np
import pandas as pd

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv('../input/train.csv')  # train set
test_df  = pd.read_csv('../input/test.csv')   # test  set
combine  = train_df + test_df
train_df.describe(include = 'all')
train_df.head()
test_df.describe(include = 'all')
test_df.head()
plt.title('Age distribution on Titanic',fontsize = 16)
plt.hist(train_df['Age'], bins=np.arange(train_df['Age'].min(), train_df['Age'].max()+1))
plt.show()
# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# code from https://stackoverflow.com/questions/31029560/plotting-categorical-data-with-pandas-and-matplotlib
train_df['Sex'].value_counts().plot(kind='bar')
# plt.hist(train_df['Fare'], bins=np.arange(train_df['Fare'].min(), train_df['Fare'].max()+1))

plt.boxplot(train_df['Fare'])
plt.show()
plt.title('Fare and survival',fontsize = 16)
plt.scatter(train_df['Fare'],train_df['Survived'])
plt.show()
# Following code from https://stackoverflow.com/questions/8202605/matplotlib-scatterplot-colour-as-a-function-of-a-third-variable
# Function to map the colors as a list from the input list of x variables
def pltcolor(lst):
    cols=[]
    for l in lst:
        if l == 1:
            cols.append('red')
        elif l == 2:
            cols.append('blue')
        else:
            cols.append('green')
    return cols
# Create the colors list using the function above
color_cols = pltcolor(train_df['Pclass'])

plt.scatter(x=train_df['Fare'],y=train_df['Survived'],s=20,c=color_cols) #Pass on the list created by the function here
plt.title('Fare and survival sorted by class',fontsize = 16)
plt.show()
# Following code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.5, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.75, bins=20)
grid.add_legend();
# How many people paid $0 fare
(train_df.Fare == 0).sum()
total = train_df.isnull().sum().sort_values(ascending = False)
percent = round(train_df.isnull().sum().sort_values(ascending = False)/len(train_df)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])

# from 2b in https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic/notebook
# train_df[train_df.Age.isnull()]
total = test_df.isnull().sum().sort_values(ascending = False)
percent = round(test_df.isnull().sum().sort_values(ascending = False)/len(test_df)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
train_df.info()
print('_'*40)
test_df.info()
combine = [train_df, test_df]
# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
# code from https://www.kaggle.com/startupsci/titanic-data-science-solutions
# guess_ages is an external array which we will fill in the guessed age
guess_ages = np.zeros((2,3))
guess_ages
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
total = train_df.isnull().sum().sort_values(ascending = False)
percent = round(train_df.isnull().sum().sort_values(ascending = False)/len(train_df)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin', 'Embarked'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Embarked'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape