import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from sklearn.impute import SimpleImputer

# Basic packages

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, gc

from scipy import stats; from scipy.stats import zscore, norm, randint

import matplotlib.style as style; style.use('fivethirtyeight')



# Display settings

pd.options.display.max_rows = 400

pd.options.display.max_columns = 100

pd.options.display.float_format = "{:.2f}".format



random_state = 42

np.random.seed(random_state)



# Suppress warnings

import warnings; warnings.filterwarnings('ignore')



data = pd.read_csv('../input/malnutrition-across-the-globe/malnutrition-estimates.csv')

data_by_country = pd.read_csv('../input/malnutrition-across-the-globe/country-wise-average.csv')

data.drop(['Unnamed: 0','ISO code','Survey Year','Source','Report Author','Notes','Short Source'], axis=1, inplace=True)



def income_map(val):

    mapper = {0:'Low Income', 1:'Lower Middle Income', 2:'Upper Middle Income',3:'High Income'}

    return mapper[val]

def lldc_map(val):

    mapper = {0:'Others', 2:'SIDS', 1:'LLDC'}

    return mapper[val]



data['Income Classification'] = data['Income Classification'].apply(income_map)

data['LLDC or SID2'] = data['LLDC or SID2'].apply(lldc_map)
#from IPython.display import Image

#Image("../input/nifty50/Capture.PNG")

#source: https://www.slideshare.net/souravgoswami11/epidemiology-of-childhood-malnutrition-in-india-and-strategies-of-control
data.head()
data.columns
data.info()
data.describe().T
# Check missing values in the dataframe

data.isnull().sum()
columns = list(['Severe Wasting', 'Wasting','Overweight', 'Stunting', 'Underweight'])



print('Descriptive Stats before imputation for columns with missing values: \n', '--'*35)

display(data[columns].describe().T)



data['Wasting'].fillna(data['Wasting'].mean(), inplace=True)

data['Severe Wasting'].fillna(data['Severe Wasting'].mean(), inplace=True)

data['Overweight'].fillna(data['Overweight'].mean(), inplace=True)

data['Stunting'].fillna(data['Stunting'].mean(), inplace=True)

data['Underweight'].fillna(data['Underweight'].mean(), inplace=True)



print('Descriptive Stats after imputation: \n', '--'*35)

display(data[columns].describe().T)



# Functions that will help us with EDA plot

def odp_plots(df, col):

    f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 7.2))

    

    # Boxplot to check outliers

    sns.boxplot(x = col, data = df, ax = ax1, orient = 'v', color = 'darkslategrey')

    

    # Distribution plot with outliers

    sns.distplot(df[col], ax = ax2, color = 'teal', fit = norm).set_title(f'{col} with outliers')

    

    # Removing outliers, but in a new dataframe

    upperbound, lowerbound = np.percentile(df[col], [1, 99])

    y = pd.DataFrame(np.clip(df[col], upperbound, lowerbound))

    

    # Distribution plot without outliers

    sns.distplot(y[col], ax = ax3, color = 'tab:orange', fit = norm).set_title(f'{col} without outliers')

    

    kwargs = {'fontsize':14, 'color':'black'}

    ax1.set_title(col + ' Boxplot Analysis', **kwargs)

    ax1.set_xlabel('Box', **kwargs)

    ax1.set_ylabel(col + ' Values', **kwargs)



    return plt.show()
# Outlier, distribution for columns with outliers

boxplotcolumns = ['Severe Wasting', 'Wasting', 'Overweight', 'Stunting',

                  'Underweight']

for cols in boxplotcolumns:

    Q3 = data[cols].quantile(0.75)

    Q1 = data[cols].quantile(0.25)

    IQR = Q3 - Q1



    print(f'{cols.capitalize()} column', '--'*40)

    count = len(data.loc[(data[cols] < (Q1 - 1.5 * IQR)) | (data[cols] > (Q3 + 1.5 * IQR))])

    print(f'no of records with outliers values: {count}')

    

    display(data.loc[(data[cols] < (Q1 - 1.5 * IQR)) | (data[cols] > (Q3 + 1.5 * IQR))].head())

    print(f'EDA for {cols.capitalize()} column', '--'*40)

    odp_plots(data, cols)



del cols, IQR, boxplotcolumns
corr = data.corr()

mask = np.zeros_like(corr, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask = mask,  linewidths = .5 )#, annot = True)

# Filter for correlation value greater than threshold

sort = corr.abs().unstack()

sort = sort.sort_values(kind = "quicksort", ascending = False)

display(sort[(sort > 0.7) & (sort < 1)])
country = data.loc[:,['Country','Underweight']]

country['percunder'] = country.groupby('Country')['Underweight'].transform('max')

country = country.drop('Underweight',axis=1).drop_duplicates().sort_values('percunder', ascending=False).head()



fig = px.pie(country, names='Country', values='percunder', template='seaborn')

fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)

fig.show()
country = data.loc[:,['Country','Overweight']]

country['percunder'] = country.groupby('Country')['Overweight'].transform('max')

country = country.drop('Overweight',axis=1).drop_duplicates().sort_values('percunder', ascending=False).head()



fig = px.pie(country, names='Country', values='percunder', template='seaborn')

fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)

fig.show()
f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 7.2))

sns.distplot(data['Underweight'], ax=ax1)



df_LM = data.loc[data['Income Classification'] == 'Lower Middle Income']

df_UM = data.loc[data['Income Classification'] == 'Upper Middle Income']

df_Low = data.loc[data['Income Classification'] == 'Low Income']

df_High = data.loc[data['Income Classification'] == 'High Income']



sns.distplot( df_LM['Underweight'],ax = ax2 , color = 'r')

sns.distplot( df_UM['Underweight'],ax = ax2, color = 'g')

sns.distplot( df_Low['Underweight'],ax = ax2, color = 'b')

sns.distplot( df_High['Underweight'],ax = ax2, color = 'y')



df = data.loc[:,['Income Classification','Underweight']]

df['maxunder'] = df.groupby('Income Classification')['Underweight'].transform('mean')

df = df.drop('Underweight', axis=1).drop_duplicates()

df = data.loc[:,['Income Classification','Underweight']]

df['maxunder'] = df.groupby('Income Classification')['Underweight'].transform('mean')

df = df.drop('Underweight', axis=1).drop_duplicates()



fig = sns.barplot(data=df, x='Income Classification', y='maxunder')

fig.set(xticklabels = ['LM', 'UM', 'Low', "High"])

plt.show()
df = data.loc[:,['Income Classification','Underweight']]

df['maxunder'] = df.groupby('Income Classification')['Underweight'].transform('max')

df = df.drop('Underweight', axis=1).drop_duplicates()



fig = px.pie(df, names='Income Classification', values='maxunder', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label", showlegend=False)

fig.show()
f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 7.2))

df_with_LDC = data.loc[data['LDC'] == 1]

df_with_DC = data.loc[data['LDC'] == 0]



sns.distplot(data['Underweight'], ax=ax1)

sns.distplot( df_with_LDC['Underweight'],ax = ax2 , color = 'r')

sns.distplot( df_with_DC['Underweight'],ax = ax2, color = 'g')



df = data.loc[:,['LIFD','Underweight']]

df['maxunder'] = df.groupby('LIFD')['Underweight'].transform('mean')

df = df.drop('Underweight', axis=1).drop_duplicates()



fig = sns.barplot(data=df, x='LIFD', y='maxunder', ax=ax3)

fig.set(xticklabels = ['Not LIFD', 'LIFD'])

plt.show()

f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 7.2))

df_with_LIFD = data.loc[data['LIFD'] == 1]

df_with_NLIFD = data.loc[data['LIFD'] == 0]



sns.distplot(data['Underweight'], ax=ax1)

sns.distplot( df_with_LIFD['Underweight'],ax = ax2 , color = 'r')

sns.distplot( df_with_NLIFD['Underweight'],ax = ax2, color = 'g')



df = data.loc[:,['LIFD','Underweight']]

df['maxunder'] = df.groupby('LIFD')['Underweight'].transform('mean')

df = df.drop('Underweight', axis=1).drop_duplicates()

df = data.loc[:,['LIFD','Underweight']]

df['maxunder'] = df.groupby('LIFD')['Underweight'].transform('mean')

df = df.drop('Underweight', axis=1).drop_duplicates()



fig = sns.barplot(data=df, x='LIFD', y='maxunder')

fig.set(xticklabels = ['Not LIFD', 'LIFD'])

plt.show()
data["Income Classification"].value_counts()
df = data.loc[:,['LLDC or SID2','Underweight']]

df['maxunder'] = df.groupby('LLDC or SID2')['Underweight'].transform('max')

df = df.drop('Underweight', axis=1).drop_duplicates()



fig = px.pie(df, names='LLDC or SID2', values='maxunder', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label", showlegend=False)

fig.show()
sns.kdeplot(data=data['Severe Wasting'], shade=True)

plt.title('Distribution of Sever Wasting percentages in countries')

plt.show()
sns.pairplot(data[['Severe Wasting','Overweight','Underweight', 'Stunting']])

plt.show()
sns.kdeplot(data=data['U5 Population (\'000s)'], shade=True)

plt.title('Distribution of U5 Population')

plt.show()
fig = sns.scatterplot(data=data, x='Underweight', y='U5 Population (\'000s)')

fig.set(yticklabels=[])

plt.show()
df = data.loc[:,['Country','Underweight','U5 Population (\'000s)']]

df['underweight_count'] = (df['U5 Population (\'000s)'] * df['Underweight'])/100

df.drop(['Underweight','U5 Population (\'000s)'], axis=1, inplace=True)

df['undermean'] = df.groupby('Country')['underweight_count'].transform('mean')

df = df.drop('underweight_count', axis=1).drop_duplicates().sort_values('undermean', ascending=False).head()



fig = px.pie(df, names='Country', values='undermean', template='seaborn')

fig.update_traces(rotation=90, pull=[0.2,0.03,0.1,0.03,0.1], textinfo="percent+label", showlegend=False)

fig.show()