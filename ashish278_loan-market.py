# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Data-Set.csv",header = 0)

df.head()

pd.set_option('max_columns', 120)

pd.set_option('max_colwidth', 5000)
import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams['figure.figsize'] = (12,8)



# skip row 1 so pandas can parse the data properly.

df = pd.read_csv('../input/Data-Set.csv',low_memory=False)



half_count = len(df) / 2

df = df.dropna(thresh=half_count,axis=1) # Drop any column with more than 50% missing values

df.head(2)

df.shape
def data_completeness(df):

    colnames = list(df)

    null_list = []

    for col in colnames:

        null_count = sum(df[col].isnull())

        null_list.append(null_count)

    completeness = pd.DataFrame({'Column_Name': colnames, 'Null_Count': null_list})

    return completeness

loans_completeness_check = data_completeness(df)

loans_completeness_check
def data_uniqueness(data, dataname):

    if dataname == 'df':

        key_cols = ['SA_ID', 'CODE_ZIP_CODE', 'PROFESSION_CODE', 'EMP_ZIP']

        data_duplicated = df[df.duplicated(key_cols, keep=False)]

        return data_duplicated

    

loans_uniqueness_check = data_uniqueness(df, 'df')

loans_uniqueness_check

#no duplicated data as such
#def data_validity(loan_data):

 #   invalid_data = loan_data[(df['LOAN_AMOUNT'] > df['AMOUNT_CREDIT']) 

                        

   # if len(invalid_data.index) == 0

  #      return('Data in loans is valid.')

    #else

     #   return('There has invalid data in loans.')

#data_validity(df)
import plotly.graph_objs as go

import plotly.plotly as py

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



df_int=df.groupby(['HOUSE_TYPE_FROM_HOMER','LOAN_AMOUNT'], as_index=False)['INTEREST_RATE'].mean()

trace0=go.Scatter(

    x=df_int.LOAN_AMOUNT,

    y=df_int.INTEREST_RATE[df_int.HOUSE_TYPE_FROM_HOMER=='PARENTAL'],

    mode='lines',

    name='PARENTAL'

)

trace1=go.Scatter(

    x=df_int.LOAN_AMOUNT,

    y=df_int.INTEREST_RATE[df_int.HOUSE_TYPE_FROM_HOMER=='OWNED'],

    mode='lines',

    name='OWNED'

)

trace2=go.Scatter(

    x=df_int.LOAN_AMOUNT,

    y=df_int.INTEREST_RATE[df_int.HOUSE_TYPE_FROM_HOMER=='RENTED'],

    mode='lines',

    name='RENTED'

)

trace3=go.Scatter(

    x=df_int.LOAN_AMOUNT,

    y=df_int.INTEREST_RATE[df_int.HOUSE_TYPE_FROM_HOMER=='XNA'],

    mode='lines',

    name='XNA'

)





layout=go.Layout(title="Average Interest Rate",

                 font=dict(size=18),

                 xaxis={'title':'LOAN_AMOUNT',

                    'tickfont':dict(size=16)}, 

                 yaxis={'title':'Average interest rate (%)',

                       'tickfont':dict(size=16)},

                 showlegend=False)

annotations=[]

for i in df_int.index:

    if df_int.iloc[i,1]=='2014':

        annotations.append(dict(x=2014, y=df_int.iloc[i,2]+0.5, text="Grade "+df_int.iloc[i,0],

                                font=dict(family='Arial', size=14,

                                color='rgba(0, 0, 102, 1)'),

                                showarrow=False,))    

    layout['annotations']=annotations

data=[trace0, trace1, trace2, trace3,]

fig=go.Figure(data=data,layout=layout)

py.iplot(fig)  ## Riskier the loan higher the interest rate
# datatypes present into training dataset

def datatypes_insight(data):

    display(data.dtypes.to_frame().T)

    data.dtypes.value_counts().plot(kind="barh")



datatypes_insight(df)
# Missing value identification



def Nan_value(data):

    display(data.apply(lambda x: sum(x.isnull())).to_frame().T)

    ##data.apply(lambda x: sum(x.isnull())).plot(kind="barh")



Nan_value(df)

# Ploting the NAN values if any.

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')
# Unique values in features

def unique_data(data):

    display(data.apply(lambda x: len(x.unique())).to_frame().T)

    data.apply(lambda x: len(x.unique())).plot(kind="barh")



unique_data(df)
def distploting(df):

    col_value=df.columns.values.tolist()

    sns.set(context='notebook',style='whitegrid', palette='dark',font='sans-serif',font_scale=1.2,color_codes=True)

    

    fig, axes = plt.subplots(nrows=7, ncols=2,constrained_layout=True)

    count=0

    for i in range (7):

        for j in range (2):

            s=col_value[count+j]

            #axes[i][j].hist(df[s].values,color='c')

            sns.distplot(df[s].values,ax=axes[i][j],bins=30,color="c")

            axes[i][j].set_title(s,fontsize=17)

            fig=plt.gcf()

            fig.set_size_inches(8,20)

            plt.tight_layout()

        count=count+j+1

        

             

distploting(df)
df[['LOAN_AMOUNT', 'AMOUNT_CREDIT']].groupby(['LOAN_AMOUNT'], as_index=False).mean().sort_values(by='AMOUNT_CREDIT', ascending=False)
df[['HOUSE_TYPE_FROM_HOMER', 'LOAN_AMOUNT']].groupby(['HOUSE_TYPE_FROM_HOMER'], as_index=False).mean().sort_values(by='LOAN_AMOUNT', ascending=False)
df[['ECONOMICAL_STATUS_CODE', 'LOAN_AMOUNT']].groupby(['ECONOMICAL_STATUS_CODE'], as_index=False).mean().sort_values(by='LOAN_AMOUNT', ascending=False)
df[['POS_REGION', 'LOAN_AMOUNT']].groupby(['POS_REGION'], as_index=False).mean().sort_values(by='LOAN_AMOUNT', ascending=False)
df[['POS_CATEGORY', 'LOAN_AMOUNT']].groupby(['POS_CATEGORY'], as_index=False).mean().sort_values(by='LOAN_AMOUNT', ascending=False)
df[['CLIENT_AGE_FROM_HOMER', 'LOAN_AMOUNT']].groupby(['CLIENT_AGE_FROM_HOMER'], as_index=False).mean().sort_values(by='LOAN_AMOUNT', ascending=False)
df[['GENDER_FROM_HOMER', 'LOAN_AMOUNT']].groupby(['GENDER_FROM_HOMER'], as_index=False).mean().sort_values(by='LOAN_AMOUNT', ascending=False)
df[['MARITIAL_STATUS_FROM_HOMER', 'LOAN_AMOUNT']].groupby(['MARITIAL_STATUS_FROM_HOMER'], as_index=False).mean().sort_values(by='LOAN_AMOUNT', ascending=False)
df[['EDUCATION_FROM_HOMER', 'LOAN_AMOUNT']].groupby(['EDUCATION_FROM_HOMER'], as_index=False).mean().sort_values(by='LOAN_AMOUNT', ascending=False)
g = sns.FacetGrid(df, col='LOAN_AMOUNT',height=3,aspect=1)

g.map(plt.hist,'INCOME', bins=1)



grid = sns.FacetGrid(df, col='LOAN_AMOUNT', row='INTEREST_RATE', size=2.5, aspect=1.6)

grid.map(plt.hist, 'INCOME', alpha=.5, bins=20)

grid.add_legend();
df["PP_income_M"] = (((bank_df["INCOME"])/12)-((bank_df["INTEREST_RATE"])/12))



g = sns.FacetGrid(bank_df, col='LOAN_AMOUNT')

g.map(plt.hist,'PP_income_M', bins=20)