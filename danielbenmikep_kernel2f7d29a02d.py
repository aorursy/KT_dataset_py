# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.set_option('display.max_columns', 500) # setting display options to show max columns

pd.set_option('display.width', 1000)



pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Supress Warnings



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns



#importing sklearn

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score 



import statsmodels

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor

df=pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df.head()
df.shape
df.info()
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df.describe()
df = df.drop(['id','date'], axis = 1)
with sns.plotting_context("notebook",font_scale=2.5):

    g = sns.pairplot(df[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 

                 hue='bedrooms', palette='tab20',size=6)

g.set(xticklabels=[]);
def categorical_variables(df,column):

    plt.figure(figsize=(10,7.5))

    ax=sns.countplot(df[column])

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    for i in ax.patches:

        annotation=str('{:1.1f}%'.format((i.get_height()*100)/float(len(df))))+'\n'+str(i.get_height())

        ax.annotate(annotation, (i.get_x()+0.05, i.get_height()+1))       

    plt.title(column,fontsize=20,weight="bold")

    plt.show()

    

def quantitative_variables(df,column):

    plt.figure(figsize=(20,10))

    plt.subplot(121)

    sns.set(style="whitegrid")

    sns.boxplot(df[column],orient='v')

    plt.title(column,fontsize=20,weight="bold")

    plt.subplot(122)

    sns.set(style="whitegrid")

    sns.distplot(df[column])

    plt.title(column,fontsize=20,weight="bold")

    

def cat(df,columns):

    for i in columns:

        categorical_variables(df,i)

        



def quat(df,columns):

    for i in columns:

        quantitative_variables(df,i)        
df.columns
df
quantitative=['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','grade', 'floors', 'sqft_above',

              'sqft_basement', 'yr_built','yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

quat_var=['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above','sqft_basement','grade']
def printColsVsUniqueValues(df,maxuniqueCount):

    for col in df.columns:

        if len(df[col].unique()) <=maxuniqueCount:

            print(str(col)+" - "+str(df[col].unique()))

            

printColsVsUniqueValues(df,10)

df.floors.unique()

df.grade.unique()

df.zipcode.unique()
categorical=['waterfront','view','condition']
cat(df,categorical)
quat(df,quat_var)
plt.figure(figsize=[15,15])

sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
# data=df.copy()

# cols_to_drop=data.corr()[(data.corr()['price']<=0.1) & (data.corr()['price']>=-0.1)]

# cols_to_drop=cols_to_drop.reset_index()['index']

# cols_to_drop=list(cols_to_drop)

# data.drop(cols_to_drop,axis=1,inplace=True)

# df_corr=data



# plt.figure(figsize=(20,20))

# sns.heatmap(df_corr.corr(),annot=True,cmap='YlGnBu')

# b, t = plt.ylim() # discover the values for bottom and top

# b += 0.5 # Add 0.5 to the bottom

# t -= 0.5 # Subtract 0.5 from the top

# plt.ylim(b, t) # update the ylim(bottom, top) values

# plt.show()
# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
df.head()
scaler = MinMaxScaler()



quantitative_list=['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'grade', 'sqft_above', 

                   'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']



# quantitative_list=['price', 'bedrooms', 'bathrooms', 'sqft_living','floors', 'view','grade', 'sqft_above','yr_renovated','lat','sqft_living15']



df_train[quantitative_list] = scaler.fit_transform(df_train[quantitative_list])
df_train.describe()
y_train = df_train.pop('price')

X_train = df_train
#selecting variables by rfe

lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
def build_model(X,y):

    X = sm.add_constant(X) #Adding the constant

    lm = sm.OLS(y,X).fit() # fitting the model

    print(lm.summary()) # model summary

    return X

    

def checkVIF(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
X_train_new = build_model(X_train_rfe,y_train)

checkVIF(X_train_new)
X_train_new = X_train_new.drop(["sqft_living"], axis = 1)

X_train_new = build_model(X_train_new,y_train)

checkVIF(X_train_new)
# Residual analysis

lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)              

plt.xlabel('Errors', fontsize = 18)   