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
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsinterferoncsv/interferon.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'cusersmarildownloadsinterferoncsv/interferon.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
na_percent = (df.isnull().sum()/len(df))[(df.isnull().sum()/len(df))>0].sort_values(ascending=False)



missing_data = pd.DataFrame({'Missing Percentage':na_percent*100})

missing_data
na = (df.isnull().sum() / len(df)) * 100

na = na.drop(na[na == 0].index).sort_values(ascending=False)



f, ax = plt.subplots(figsize=(12,8))

sns.barplot(x=na.index, y=na)

plt.xticks(rotation='90')

plt.xlabel('Features', fontsize=15)

plt.title('Percentage Missing', fontsize=15)
#for col in ('Age'):

 #   df[col] = df[col].fillna(0)

    

for col in ['Group', 'IFNG+874', 'IGRA result', 'TST result', 'Sex', 'Ethnic background']:

    df[col] = df[col].fillna('None')
for col in ('Ag normalized', 'PHA normalized', 'CD3+ number', 'Ag IFN-y pg/ml', 'PHA IFN-y pg/ml', 'unstimulated IFN-y pg/ml', 'Registry' ):

    df[col] = df[col].fillna(df[col].mode()[0])
categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]
print(categorical_cols)
print(numerical_cols)
plt.style.use('fivethirtyeight')

sns.countplot(df['IFNG+874'],linewidth=3,palette="Set2",edgecolor='black')

plt.show()
from sklearn.preprocessing import LabelEncoder

categorical_col = ('Group', 'IFNG+874', 'IGRA result', 'TST result', 'Sex', 'Ethnic background')

        

        

for col in categorical_col:

    label = LabelEncoder() 

    label.fit(list(df[col].values)) 

    df[col] = label.transform(list(df[col].values))



print('Shape all_data: {}'.format(df.shape))
plt.rcParams['figure.figsize'] = (14,5)

plt.subplot(1,2,1)

sns.kdeplot(df['IFNG+874'][df.Group == 1],shade = True,color = "red")

plt.title('IFNG+874')

plt.xlabel('IFNG+874 Distribution ')

plt.subplot(1,2,2)

sns.kdeplot(df['IGRA result'][df.Group == 0],shade = True,color = "green")

plt.title('IGRA result')

plt.xlabel('IGRA Result Distribution')
# Let's See The Correlation Among The Features .



# Below chart is used to visualize how one feature is correlated with every other Features Present in the dataset .

# if we have two highly correlated features then we will consider only one of them to avoid overfitting .



# since in our Dataset There is now two  features which are highly correlated ,

# hence we have consider all the features for training our Model .





plt.rcParams['figure.figsize'] = (10, 6)

sns.heatmap(df.corr(),annot = True ,cmap = 'rainbow_r',annot_kws = {"Size":14})

plt.title( "Chart Shows Correlation Among Features   : ")
from scipy.stats import norm, skew

num_features = df.dtypes[df.dtypes != 'object'].index

skewed_features = df[num_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_features})

skewness.head(15)
numerical_df = df.select_dtypes(exclude='object')



for i in range(len(numerical_df.columns)):

    f, ax = plt.subplots(figsize=(7, 4))

    fig = sns.distplot(numerical_df.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})

    plt.xlabel(numerical_df.columns[i])
from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['IFNG+874']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Age']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
df['IFNG+874'].hist(figsize=(10,4), bins=20)
ax = df['IFNG+874'].value_counts().plot.barh(figsize=(10, 4))

ax.set_title('IFNG+874 Distribution', size=18)

ax.set_ylabel('IFNG+874', size=10)

ax.set_xlabel('Group', size=10)
import matplotlib.ticker as ticker

ax = sns.distplot(df['IFNG+874'])

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



sb.distplot(df['IFNG+874'], fit=norm)



#Get the QQ-plot

fig = plt.figure()

res = stats.probplot(df['IFNG+874'], plot=plt)

plt.show()
from statsmodels.tsa.statespace.sarimax import SARIMAX

#The data is highly skewed, but since we'll be applying ARIMA, it's fine.

df['IFNG+874'].skew()
df.dtypes