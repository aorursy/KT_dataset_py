import numpy as np

import pandas as pd

import seaborn as sns

import scipy as stats

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
data = pd.read_csv('../input/world-happiness/2019.csv')

print(data.info())

data.head(10)
data.describe()
data.rename(columns={'Country or region':'Country'}, inplace=True)

data.info()
if(data.isnull().values.any()):

    data.dropna(subset = [

        'Overall rank',

        'Country',

        'Score',

        'GDP per capita',

        'Social support',

        'Healthy life expectancy',

        'Freedom to make life choices',

        'Generosity',

        'Perceptions of corruption'

        

    ], inplace=True)

else:

    print('Tidak terdapat missing values')
#Menampilkan daftar benua dari dataset Country Mapping - ISO, Continent, Region

world_df = pd.read_csv('../input/country-mapping-iso-continent-region/continents2.csv')

world_df['region'].value_counts()
#Membuat list negara dari tiap-tiap benua

africa = world_df.loc[world_df['region'] == 'Africa', 'name'].tolist()

america = world_df.loc[world_df['region'] == 'Americas', 'name'].tolist()

asia = world_df.loc[world_df['region'] == 'Asia', 'name'].tolist()

europe = world_df.loc[world_df['region'] == 'Europe', 'name'].tolist()

oceania = world_df.loc[world_df['region'] == 'Oceania', 'name'].tolist()



#fungsi mendapatkan nama benua berdasarkan nama negara

def get_continent(country):

    if country in africa:

        return 'Africa'

    elif country in america:

        return 'Americas'

    elif country in asia:

        return 'Asia'

    elif country in europe:

        return 'Europe'

    elif country in oceania:

        return "Oceania"



continent = [get_continent(country) for country in data['Country']]

data['Region'] = continent

data['Region'].value_counts()
income_group_df = pd.read_csv('../input/world-bank-country-and-lending-groups/worldbank_classification.csv')

income_group_df['Income group'].value_counts()
high = income_group_df.loc[income_group_df['Income group'] == 'High income', 'Economy'].tolist()

upper_middle = income_group_df.loc[income_group_df['Income group'] == 'Upper middle income', 'Economy'].tolist()

lower_middle = income_group_df.loc[income_group_df['Income group'] == 'Lower middle income', 'Economy'].tolist()

lower = income_group_df.loc[income_group_df['Income group'] == 'Low income', 'Economy'].tolist()



def get_income_category(country):

    if country in high:

        return 'High income'

    elif country in upper_middle:

        return 'Upper middle income'

    elif country in lower_middle:

        return 'Lower middle income'

    else:

        return 'Low income'



income_list = [get_income_category(country) for country in data['Country']]

data['Income group'] = income_list

data['Income group'].value_counts()
fig, axes = plt.subplots(2, 3, figsize=(20,10))

sns.regplot(x='GDP per capita', y='Score', data=data, ax=axes[0][0], color='blue')

sns.regplot(x='Social support', y='Score', data=data, ax=axes[0][1], color='red')

sns.regplot(x='Healthy life expectancy', y='Score', data=data, ax=axes[0][2], color='green')

sns.regplot(x='Freedom to make life choices', y='Score', data=data, ax=axes[1][0], color='purple')

sns.regplot(x='Generosity', y='Score', data=data, ax=axes[1][1], color='crimson')

sns.regplot(x='Perceptions of corruption', y='Score', data=data, ax=axes[1][2], color='orange')

plt.show()

plt.subplots(figsize=(10, 10))

sns.heatmap(data.corr(), annot=True)

plt.show()
africa = data['Region'] == 'Africa'

africa_df = data[africa]



america = data['Region'] == 'Americas'

america_df = data[america]



asia = data['Region'] == 'Asia'

asia_df = data[asia]



europe = data['Region'] == 'Europe'

europe_df = data[europe]



oceania = data['Region'] == 'Oceania'

oceania_df = data[oceania]





fig, axes = plt.subplots(2, 3, figsize=(20, 13))

sns.barplot(x='Region', y='Score', data=data, ax=axes[0][0])

sns.barplot(x='Region', y='GDP per capita', data=data, ax=axes[0][1])

sns.barplot(x='Region', y='Social support', data=data, ax=axes[0][2])

sns.barplot(x='Region', y='Healthy life expectancy', data=data, ax=axes[1][0])

sns.barplot(x='Region', y='Generosity', data=data, ax=axes[1][1])

sns.barplot(x='Region', y='Perceptions of corruption', data=data, ax=axes[1][2])

plt.show()

def generate_linreg_plot(variable):

    fig, axes = plt.subplots(2, 2, figsize=(20,10))

    sns.regplot(x=variable, y='Score', data=africa_df, ax=axes[0][0], color='blue')

    sns.regplot(x=variable, y='Score', data=america_df, ax=axes[0][1], color='red')

    sns.regplot(x=variable, y='Score', data=asia_df, ax=axes[1][0], color='green')

    sns.regplot(x=variable, y='Score', data=europe_df, ax=axes[1][1], color='purple')

    axes[0][0].set_title('Africa')

    axes[0][1].set_title('America')

    axes[1][0].set_title('Asia')

    axes[1][1].set_title('Europe')

    plt.show()
generate_linreg_plot('GDP per capita')
generate_linreg_plot('Social support')
generate_linreg_plot('Healthy life expectancy')
generate_linreg_plot('Freedom to make life choices')
generate_linreg_plot('Generosity')
generate_linreg_plot('Perceptions of corruption')
continent_corrval_dict = {

    'Region':['Africa', 'America', 'Asia', 'Europe'],

    'GDP per capita': [

        africa_df['Score'].corr(africa_df['GDP per capita']),

        america_df['Score'].corr(america_df['GDP per capita']),

        asia_df['Score'].corr(asia_df['GDP per capita']),

        europe_df['Score'].corr(europe_df['GDP per capita']),    

    ],

    'Social support':[

        africa_df['Score'].corr(africa_df['Social support']),

        america_df['Score'].corr(america_df['Social support']),

        asia_df['Score'].corr(asia_df['Social support']),

        europe_df['Score'].corr(europe_df['Social support']),        

    ],

    'Healthy life expectancy':[

        africa_df['Score'].corr(africa_df['Healthy life expectancy']),

        america_df['Score'].corr(america_df['Healthy life expectancy']),

        asia_df['Score'].corr(asia_df['Healthy life expectancy']),

        europe_df['Score'].corr(europe_df['Healthy life expectancy']),        

    ],

    'Freedom to make life choices':[

        africa_df['Score'].corr(africa_df['Freedom to make life choices']),

        america_df['Score'].corr(america_df['Freedom to make life choices']),

        asia_df['Score'].corr(asia_df['Freedom to make life choices']),

        europe_df['Score'].corr(europe_df['Freedom to make life choices'])          

    ],

    'Generosity':[

        africa_df['Score'].corr(africa_df['Generosity']),

        america_df['Score'].corr(america_df['Generosity']),

        asia_df['Score'].corr(asia_df['Generosity']),

        europe_df['Score'].corr(europe_df['Generosity'])        

    ],

    'Perceptions of corruption':[

        africa_df['Score'].corr(africa_df['Perceptions of corruption']),

        america_df['Score'].corr(america_df['Perceptions of corruption']),

        asia_df['Score'].corr(asia_df['Perceptions of corruption']),

        europe_df['Score'].corr(europe_df['Perceptions of corruption'])                  

    ],

}



continent_corrval_df = pd.DataFrame.from_dict(continent_corrval_dict)



fig, axes = plt.subplots(2, 3, figsize=(20,10))

sns.barplot(x='Region', y='GDP per capita', data=continent_corrval_df, ax=axes[0][0])

sns.barplot(x='Region', y='Social support', data=continent_corrval_df, ax=axes[0][1])

sns.barplot(x='Region', y='Healthy life expectancy', data=continent_corrval_df, ax=axes[0][2])

sns.barplot(x='Region', y='Generosity', data=continent_corrval_df, ax=axes[1][0])

sns.barplot(x='Region', y='Perceptions of corruption', data=continent_corrval_df, ax=axes[1][1])

plt.show()

high_income = data['Income group'] == 'High income'

high_income_df = data[high_income]



upper_middle_income = data['Income group'] == 'Upper middle income'

upper_middle_income_df = data[upper_middle_income]



lower_middle_income = data['Income group'] == 'Lower middle income'

lower_middle_income_df = data[lower_middle_income]



low_income = data['Income group'] == 'Low income'

low_income_df = data[low_income]
def generate_income_linreg_plot(variable):

    fig, axes = plt.subplots(2, 2, figsize=(20,13))

    sns.regplot(x=variable, y='Score', data=high_income_df, ax=axes[0][0], color='blue')

    sns.regplot(x=variable, y='Score', data=upper_middle_income_df, ax=axes[0][1], color='red')

    sns.regplot(x=variable, y='Score', data=lower_middle_income_df, ax=axes[1][0], color='green')

    sns.regplot(x=variable, y='Score', data=low_income_df, ax=axes[1][1], color='purple')

    axes[0][0].set_title('High income')

    axes[0][1].set_title('Upper middle income')

    axes[1][0].set_title('Lower middle income')

    axes[1][1].set_title('Low income')

    plt.show()
generate_income_linreg_plot('GDP per capita')
generate_income_linreg_plot('Social support')
generate_income_linreg_plot('Healthy life expectancy')
generate_income_linreg_plot('Freedom to make life choices')
generate_income_linreg_plot('Generosity')
generate_income_linreg_plot('Perceptions of corruption')
income_corrval_dict = {

    'Income group':['High income', 'Upper middle income', 'Lower middle income', 'Low income'],

    'GDP per capita': [

        high_income_df['Score'].corr(high_income_df['GDP per capita']),

        upper_middle_income_df['Score'].corr(upper_middle_income_df['GDP per capita']),

        lower_middle_income_df['Score'].corr(lower_middle_income_df['GDP per capita']),

        low_income_df['Score'].corr(low_income_df['GDP per capita']),    

    ],

    'Social support':[

        high_income_df['Score'].corr(high_income_df['Social support']),

        upper_middle_income_df['Score'].corr(upper_middle_income_df['Social support']),

        lower_middle_income_df['Score'].corr(lower_middle_income_df['Social support']),

        low_income_df['Score'].corr(low_income_df['Social support']),        

    ],

    'Healthy life expectancy':[

        high_income_df['Score'].corr(high_income_df['Healthy life expectancy']),

        upper_middle_income_df['Score'].corr(upper_middle_income_df['Healthy life expectancy']),

        lower_middle_income_df['Score'].corr(lower_middle_income_df['Healthy life expectancy']),

        low_income_df['Score'].corr(low_income_df['Healthy life expectancy']),        

    ],

    'Freedom to make life choices':[

        high_income_df['Score'].corr(high_income_df['Freedom to make life choices']),

        upper_middle_income_df['Score'].corr(upper_middle_income_df['Freedom to make life choices']),

        lower_middle_income_df['Score'].corr(lower_middle_income_df['Freedom to make life choices']),

        low_income_df['Score'].corr(low_income_df['Freedom to make life choices'])          

    ],

    'Generosity':[

        high_income_df['Score'].corr(high_income_df['Generosity']),

        upper_middle_income_df['Score'].corr(upper_middle_income_df['Generosity']),

        lower_middle_income_df['Score'].corr(lower_middle_income_df['Generosity']),

        low_income_df['Score'].corr(low_income_df['Generosity'])        

    ],

    'Perceptions of corruption':[

        high_income_df['Score'].corr(high_income_df['Perceptions of corruption']),

        upper_middle_income_df['Score'].corr(upper_middle_income_df['Perceptions of corruption']),

        lower_middle_income_df['Score'].corr(lower_middle_income_df['Perceptions of corruption']),

        low_income_df['Score'].corr(low_income_df['Perceptions of corruption'])                  

    ],

}



income_corrval_df = pd.DataFrame.from_dict(income_corrval_dict)





fig, axes = plt.subplots(3, 2, figsize=(20,15))

sns.barplot(x='Income group', y='GDP per capita', data=income_corrval_df, ax=axes[0][0])

sns.barplot(x='Income group', y='Social support', data=income_corrval_df, ax=axes[0][1])

sns.barplot(x='Income group', y='Healthy life expectancy', data=income_corrval_df, ax=axes[1][0])

sns.barplot(x='Income group', y='Generosity', data=income_corrval_df, ax=axes[1][1])

sns.barplot(x='Income group', y='Perceptions of corruption', data=income_corrval_df, ax=axes[2][0])

plt.show()


