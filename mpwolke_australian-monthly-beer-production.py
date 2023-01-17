#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSO6RWpP3hYko4MX-v9cfqhe4Jzs-VLKLn4qg&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/monthly-beer-production/datasets_56102_107707_monthly-beer-production-in-austr.csv', encoding='ISO-8859-2')

df.head()
plt.style.use('fivethirtyeight')

df.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)

plt.show()
plt.style.use('fivethirtyeight')

sns.countplot(df['Monthly beer production'],linewidth=3,palette="Set2",edgecolor='black')

plt.show()
px.histogram(df, x='Month', color='Monthly beer production', title='Australian Montly Beer Production')
fig = px.bar(df, 

             x='Month', y='Monthly beer production', color_discrete_sequence=['#D63230'],

             title='Australian Monthly Beer Production', text='Monthly beer production')

fig.show()
fig = px.line(df, x="Month", y="Monthly beer production", 

              title="Australian Montly Beer Production")

fig.show()
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['Month']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Monthly beer production']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
import matplotlib.ticker as ticker

ax = sns.distplot(df['Monthly beer production'])

plt.xticks(rotation=45)

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

figsize=(10, 4)
from scipy.stats import norm, skew #for some statistics

import seaborn as sb

from scipy import stats #qqplot

#Lets check the ditribution of the target variable (Placement?)

from matplotlib import rcParams

# figure size in inches

rcParams['figure.figsize'] = 4,2



sb.distplot(df['Monthly beer production'], fit=norm)



#Get the QQ-plot

fig = plt.figure()

res = stats.probplot(df['Monthly beer production'], plot=plt)

plt.show()
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]

print(numerical_cols)
df.columns
df.values
data=list(df["Month"].apply(lambda x:x.split(',')))
data
from mlxtend.preprocessing import TransactionEncoder
te=TransactionEncoder()

te_data=te.fit(data).transform(data)

df=pd.DataFrame(te_data,columns=te.columns_)

df
from mlxtend.frequent_patterns import apriori
df1=apriori(df,min_support=0.01,use_colnames=True)

df1
df1.sort_values(by="support",ascending=False)
df1["length"]=df1["itemsets"].apply(lambda x:len(x))

df1
df1[(df1["length"]==2) & (df1["support"]>0.05)]
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR40HpvOWeOIwVot55RaCus5T9VRiNKfB0pHg&usqp=CAU',width=400,height=400)