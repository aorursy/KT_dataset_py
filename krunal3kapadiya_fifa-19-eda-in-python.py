# importing libraries
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# reading dataset
df_data = pd.read_csv('../input/data.csv')
df_data.head() #printing values in dataset
df_data.columns
def display_graph(ax, title, xlabel, ylabel, legend):
    '''
    Graph theme will be same throught the kernel
    '''
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(legend)
    plt.show()
print('In the age column there is total {} number of players and in dataset there is {} number of null in Age. Also in that data the mean (average age) is {}, maximum age is {}, and minimum age is {}, containing total {} numbers of countries. Now, lets display the other data related information based on Age.'.format(df_data['Age'].sum(), df_data['Age'].isna().sum(), format(df_data['Age'].mean(), '.2f'), df_data['Age'].max(), df_data['Age'].min(), len(df_data['Nationality'].unique())))
print('This dataset have {} players having age 16 and {} players who has age more than 42'.format(sum((df_data['Age'] == 16)),sum(df_data['Age'] >= 40)))
ax = sns.distplot(df_data[['Age']])
display_graph(ax, 'Age', 'Age count', '', ['Age'])
ax = sns.distplot(df_data[['Potential']])
display_graph(ax, 'Potential count', 'Potential', '', ['Potential'])
ax = sns.scatterplot(x = 'Age', y='Potential', data=pd.DataFrame(df_data, columns=['Age', 'Potential']))
display_graph(ax, 'Age vs Potential graph', 'Age', 'Potential', ['Age'])
df_age = pd.DataFrame(df_data, columns=['Name', 'Age', 'Potential', 'Nationality'])
df_age.sort_values(by='Age').head()
df_age.sort_values(by='Age').tail()
df_age.groupby('Age', as_index=False).count().head(5)
ax = df_age.groupby('Age').mean().plot.bar()
display_graph(ax, 'Average potential count', 'Age', 'Potential', ['Potential'])
df_joined = df_data['Joined']
df_joined.isna().sum()
df_joined.dropna(inplace = True)
df_joined = df_joined.apply(lambda x: datetime.strptime(x, '%b %d, %Y'))
# get the list of years
df_year = df_joined.apply(lambda x: x.year)
ax = df_year.value_counts().plot()
display_graph(ax, 'Players growth over the year', 'Years', 'Number of players', ['Players Growth'])
df_month = df_joined.apply(lambda x: x.month)
df_month.sort_values(ascending = True, inplace=True)
ax = df_month.value_counts(sort = False).plot.bar()
display_graph(ax, 'Enrolling players as per month', 'Month', 'Number of players', ['Player/Month'])
df_height = pd.DataFrame(df_data, columns=['Height', 'Weight', 'Strength', 'Aggression', 'Stamina', 'Dribbling'])
df_height.corr()
df_height.describe()
accuracy = pd.DataFrame(df_data, columns=['HeadingAccuracy', 'FKAccuracy'])
accuracy.head()
prefered_type = df_data['Preferred Foot'].value_counts()
prefered_type
sum(df_data['Preferred Foot'].isnull())
ax = prefered_type.plot.bar()
display_graph(ax, 'Righty/Lefty Players count', 'Preffered leg', 'Number of players', ['Right', 'Left'])
df_data['Contract Valid Until'].value_counts().head(10)
df_contract = pd.DataFrame(df_data, columns=['Contract Valid Until'])
df_contract.dropna(inplace = True)
def get_only_year(dates):
    '''
    some of the date in this df contains 21 Jul, 2018 and some have only names
    so, getting only years value
    '''
    newDates = []
    for i, date in enumerate(dates):
        if(len(date)>4):
            date = date[-4:]
        newDates.append(date)
    return newDates
df_contract_valid = get_only_year(df_contract['Contract Valid Until'])
df_contract_valid = pd.Series(df_contract_valid)
len(df_contract_valid.unique())
ax = df_contract_valid.value_counts().plot()
display_graph(ax, 'Contract valid until', 'Years', 
             'Players count', ['Contract'])
f = (df_data
         .loc[df_data['Position'].isin(['ST', 'GK'])]
         .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]
    )
f = f[f["Overall"] >= 80]
f = f[f["Overall"] < 85]
f['Aggression'] = f['Aggression'].astype(float)
ax = sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f)
display_graph(ax, 'Overall Aggression', 'Overall', 'Aggression', ['ST, GK'])