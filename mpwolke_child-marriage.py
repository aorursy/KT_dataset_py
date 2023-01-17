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
df = pd.read_csv('../input/child-marriage/Child-marriage-database.csv', encoding='ISO-8859-2')

df.head()
print(f"data shape: {df.shape}")
df.describe()
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes=='O']

print(categorical_nan)
df[categorical_nan].isna().sum()
# replacing missing values in categorical features

for feature in categorical_nan:

    df[feature] = df[feature].fillna('None')
df[categorical_nan].isna().sum()
# Lets first handle numerical features with nan value

numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

numerical_nan
df[numerical_nan].isna().sum()
## Replacing the numerical Missing Values



for feature in numerical_nan:

    ## We will replace by using median since there are outliers

    median_value=df[feature].median()

    

    df[feature].fillna(median_value,inplace=True)

    

df[numerical_nan].isnull().sum()
#fill in mean for floats

for c in df.columns:

    if df[c].dtype=='float16' or  df[c].dtype=='float32' or  df[c].dtype=='float64':

        df[c].fillna(df[c].mean())



#fill in -999 for categoricals

df = df.fillna(-999)

# Label Encoding

for f in df.columns:

    if df[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(df[f].values))

        df[f] = lbl.transform(list(df[f].values))

        

print('Labelling done.')
from sklearn.model_selection import train_test_split

# Hot-Encode Categorical features

df = pd.get_dummies(df) 



# Splitting dataset back into X and test data

X = df[:len(df)]

test = df[len(df):]



X.shape
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count("Female Married by 15", "Female Married by 15", df,4)
plot_count("Male Married by 18", "Male Married by 18", df,4)
import plotly.express as px

fig = px.line(df, x="Reference year", y="Female Married by 15", color_discrete_sequence=['darksalmon'], 

              title="Female Married by 15 & Reference Year")

fig.show()
import plotly.express as px

fig = px.line(df, x="Male Reference year", y="Male Married by 18", color_discrete_sequence=['#2B3A67'], 

              title="Male Married by 18 & Reference Year")

fig.show()
fig = px.scatter(df, x="Reference year", y="Female Married by 15",color_discrete_sequence=['crimson'], title="Female Married by 15 by Year Reference" )

fig.show()
import matplotlib.gridspec as gridspec

from scipy.stats import skew

from sklearn.preprocessing import RobustScaler,MinMaxScaler

from scipy import stats

import matplotlib.style as style

style.use('seaborn-colorblind')
def plotting_3_chart(df, feature): 

    ## Creating a customized chart. and giving in figsize and everything. 

    fig = plt.figure(constrained_layout=True, figsize=(10,6))

    ## crea,ting a grid of 3 cols and 3 rows. 

    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    #gs = fig3.add_gridspec(3, 3)



    ## Customizing the histogram grid. 

    ax1 = fig.add_subplot(grid[0, :2])

    ## Set the title. 

    ax1.set_title('Histogram')

    ## plot the histogram. 

    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)



    # customizing the QQ_plot. 

    ax2 = fig.add_subplot(grid[1, :2])

    ## Set the title. 

    ax2.set_title('QQ_plot')

    ## Plotting the QQ_Plot. 

    stats.probplot(df.loc[:,feature], plot = ax2)



    ## Customizing the Box Plot. 

    ax3 = fig.add_subplot(grid[:, 2])

    ## Set title. 

    ax3.set_title('Box Plot')

    ## Plotting the box plot. 

    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );

 



print('Skewness: '+ str(df['Female Married by 15'].skew())) 

print("Kurtosis: " + str(df['Female Married by 15'].kurt()))

plotting_3_chart(df, 'Female Married by 15')
plt.style.use('fivethirtyeight')

df.plot(subplots=True, figsize=(4, 4), sharex=False, sharey=False)

plt.show()
#Codes from Mario Filho https://www.kaggle.com/mariofilho/live26-https-youtu-be-zseefujo0zq



from category_encoders import OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



cols_selected = ['Female Married by 15']

ohe = OneHotEncoder(cols=cols_selected, use_cat_names=True)

df_t = ohe.fit_transform(df[cols_selected+['Reference year']])



#scaler = MaxAbsScaler()

X = df_t.iloc[:,:-1]

y = df_t.iloc[:, -1].fillna(df_t.iloc[:, -1].mean()) / df_t.iloc[:, -1].max()



mdl = Ridge(alpha=0.1)

mdl.fit(X,y)



pd.Series(mdl.coef_, index=X.columns).sort_values().head(10).plot.barh()
ax = df['Female Married by 15'].value_counts().plot.barh(figsize=(10, 4))

ax.set_title('Female Married by 15 by Reference Year', size=18)

ax.set_ylabel('Female Married by 15', size=10)

ax.set_xlabel('count', size=10)
fig=sns.lmplot(x='Reference year', y="Female Married by 15",data=df)
ax = df.groupby('Reference year')['Female Married by 15', 'Female Married by 18'].sum().plot(kind='bar', rot=45, figsize=(12,6), logy=True,

                                                                 title='Female Married by 15')

plt.xlabel('Reference Year')

plt.ylabel('Log')



plt.show()
ax = df.groupby('Reference year')['Female Married by 15', 'Female Married by 18'].sum().plot(kind='barh', figsize=(14,8),

                                                                 title='Female Married by 15', logx=True, linewidth=3)

plt.xlabel('Log')

plt.ylabel('Year Reference')

plt.show()
df = df.rename(columns={'Female Married by 15':'fmarried15', 'Female Married by 18': 'fmarried18', 'Male Married by 18': 'mMarried18'})
# Distribution of different type of amount

fig , ax = plt.subplots(1,2,figsize = (12,5))



fmarried15 = df.fmarried15.values

fmarried18= df.fmarried18.values

#mMarried18 = df.mMarried18.values



sns.distplot(fmarried15 , ax = ax[0] , color = 'pink').set_title('Female Married by 15' , fontsize = 14)

sns.distplot(fmarried18 , ax = ax[1] , color = 'cyan').set_title('Female Married by 18' , fontsize = 14)

#sns.distplot(mMarried18 , ax = ax[2] , color = 'purple').set_title('Male Married by 18' , fontsize = 14)





plt.show()