# basic operations

import numpy as np

import pandas as pd 



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns
# reading the data



data = pd.read_csv('../input/fifa19/data.csv',index_col=0)



print(data.shape)
# defining a function for cleaning the wage column



def extract_value_from(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)





# applying the function to the wage column



data['Value'] = data['Value'].apply(lambda x: extract_value_from(x))

data['Wage'] = data['Wage'].apply(lambda x: extract_value_from(x))



# filling the missing value for the continous variables for proper data visualization



data['ShortPassing'].fillna(data['ShortPassing'].mean(), inplace = True)

data['Volleys'].fillna(data['Volleys'].mean(), inplace = True)

data['Dribbling'].fillna(data['Dribbling'].mean(), inplace = True)

data['Curve'].fillna(data['Curve'].mean(), inplace = True)

data['FKAccuracy'].fillna(data['FKAccuracy'], inplace = True)

data['LongPassing'].fillna(data['LongPassing'].mean(), inplace = True)

data['BallControl'].fillna(data['BallControl'].mean(), inplace = True)

data['HeadingAccuracy'].fillna(data['HeadingAccuracy'].mean(), inplace = True)

data['Finishing'].fillna(data['Finishing'].mean(), inplace = True)

data['Crossing'].fillna(data['Crossing'].mean(), inplace = True)

data['Weight'].fillna('200lbs', inplace = True)

data['Contract Valid Until'].fillna(2019, inplace = True)

data['Height'].fillna("5'11", inplace = True)

data['Loaned From'].fillna('None', inplace = True)

data['Joined'].fillna('Jul 1, 2018', inplace = True)

data['Jersey Number'].fillna(8, inplace = True)

data['Body Type'].fillna('Normal', inplace = True)

data['Position'].fillna('ST', inplace = True)

data['Club'].fillna('No Club', inplace = True)

data['Work Rate'].fillna('Medium/ Medium', inplace = True)

data['Skill Moves'].fillna(data['Skill Moves'].median(), inplace = True)

data['Weak Foot'].fillna(3, inplace = True)

data['Preferred Foot'].fillna('Right', inplace = True)

data['International Reputation'].fillna(1, inplace = True)

data['Wage'].fillna('200000', inplace = True)



data.fillna(0, inplace = True)

data = data[['ID','Age', 'Overall', 'Position', 'Club', 'Wage', 'Nationality']]
# visualisasi 1: LINEPLOT 



plt.figure(figsize = (15,10))

x = data[data['Position'] != 'GK']['Age']

y = data[data['Position'] != 'GK']['Overall']

sns.set_style("whitegrid")

ax = sns.lineplot(x,y,ci=None)

ax.set_xlabel(xlabel = 'Age', fontsize = 15)

ax.set_ylabel(ylabel = 'Overall Score', fontsize = 15)

plt.title('Rating pemain cenderung menurun setelah umur 30', fontsize=25)



plt.show()
plt.figure(figsize = (15,10))

AverageRating = data.groupby("Club").mean().sort_values(by=['Overall'],ascending=False)

avr = AverageRating.head(20).sort_values(by=['Overall'],ascending=False).index

sns.set_style("whitegrid")

ax = sns.boxplot(x = data["Overall"], y=data["Club"], order = avr, color='lightgrey')

ax.set_xlabel(xlabel = 'Some Popular Clubs', fontsize = 15)

ax.set_ylabel(ylabel = 'Overall Score', fontsize = 15)

ax.set_title(label = 'Rata-rata rating dari 20 Klub dengan rating tertinggi', fontsize = 25)

plt.show()
# To show that there are people having same age

# Histogram: number of players's age



x = data.Age

plt.figure(figsize = (15,10))

# plt.rcParams['figure.figsize'] = (15, 10)

ax = sns.distplot(x, bins = 58, kde = False, color = 'g')

sns.set_style("whitegrid")

ax.set_xlabel(xlabel = "Umur", fontsize = 15)

ax.set_ylabel(ylabel = 'Jumlah Pemain', fontsize = 15)

ax.set_title(label = 'Histogram of players age', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (10, 7)



TotalWages = data.groupby("Club").sum().sort_values(by=['Wage'],ascending=False)

tow = TotalWages.head(20).sort_values(by=['Wage'],ascending=False).index



AverageRating = data.groupby("Club").mean().sort_values(by=['Overall'],ascending=False)

avr = AverageRating.head(20).sort_values(by=['Overall'],ascending=False).index



sns.set_style("whitegrid")

ax = sns.barplot(x = data["Wage"], y=data["Club"],order = tow, color='lightgrey', ci=None, estimator = sum)

ax.set_xlabel(xlabel = 'Total Gaji (dalam Euro)', fontsize = 15)

ax.set_ylabel(ylabel = 'Club', fontsize =15)

ax.set_title(label = 'Total gaji pemain per bulan', fontsize = 20)

# ax.text(3+0.2, 10, "An annotation", horizontalalignment='left', size='medium', color='black', weight='semibold')





plt.show()
# Every Nations' Player and their wages



some_countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia')

data_countries = data.loc[data['Nationality'].isin(some_countries) & data['Wage']]



plt.rcParams['figure.figsize'] = (15, 7)

ax = sns.barplot(x = data_countries['Nationality'], y = data_countries['Wage'], palette = 'Purples', ci =None)

ax.set_xlabel(xlabel = 'Countries', fontsize = 9)

ax.set_ylabel(ylabel = 'Wage', fontsize = 9)

ax.set_title(label = 'Distribution of Wages of players from different countries', fontsize = 15)

plt.show()