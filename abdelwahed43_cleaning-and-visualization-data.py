# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import Library 

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Read data from csv file

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print('The shape of our training set: ',df_train.shape[0], 'houses', 'and', df_train.shape[1], 'features')

print('The shape of our testing set: ',df_test.shape[0], 'houses', 'and', df_test.shape[1], 'features')

print('The testing set has 1 feature less than the training set, which is SalePrice, the target to predict  ')
df_train.head()
df_test.head()
df_train.describe()
df_test.describe()
#Header name Columns 

df_train.columns
df_test.columns
numeric = df_train.select_dtypes(exclude='object')

categorical = df_train.select_dtypes(include='object')
print("\nNumber of numeric features : ",(len(numeric.axes[1])))

print("\n", numeric.axes[1])
print("\nNumber of categorical features : ",(len(categorical.axes[1])))

print("\n", categorical.axes[1])
# Isolate the numeric features and check his relevance



num_corr = numeric.corr()

table = num_corr['SalePrice'].sort_values(ascending=False).to_frame()

cm = sns.light_palette("green", as_cmap=True)

tb = table.style.background_gradient(cmap=cm)

tb


f, ax = plt.subplots(figsize=(15,15))

sns.heatmap(df_train.corr(),annot=True, linewidths=.1, fmt= '.1f',ax=ax, cmap="YlGnBu")
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = df_train.corr().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
#missing data in Traing examples

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_data.head(20)
na = df_train.shape[0] #na is the number of rows of the original training set

nb = df_test.shape[0]  #nb is the number of rows of the original test set

y_train = df_train['SalePrice'].to_frame()

#Combine train and test sets

c1 = pd.concat((df_train, df_test), sort=False).reset_index(drop=True)

#Drop the target "SalePrice" and Id columns

c1.drop(['SalePrice'], axis=1, inplace=True)

c1.drop(['Id'], axis=1, inplace=True)

print("Total size for train and test sets is :",c1.shape)
##msv1 method to visualize missing values per columns

def msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=3): 

    """

    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking

    """

    

    plt.figure(figsize=(width,height))

    percentage=(data.isnull().mean())*100

    percentage.sort_values(ascending=False).plot.bar(color=color, edgecolor=edgecolor)

    plt.axhline(y=thresh, color='r', linestyle='-')

    plt.title('Missing values percentage per column', fontsize=20, weight='bold' )

    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh+12.5, 'Columns with more than %s%s missing values' %(thresh, '%'), fontsize=12, color='crimson',

         ha='left' ,va='top')

    plt.text(len(data.isnull().sum()/len(data))/1.7, thresh - 5, 'Columns with less than %s%s missing values' %(thresh, '%'), fontsize=12, color='green',

         ha='left' ,va='top')

    plt.xlabel('Columns', size=15, weight='bold')

    plt.ylabel('Missing values percentage')

    plt.yticks(weight ='bold')

    

    return plt.show()
msv1(c1, 20, color=('silver', 'gold', 'lightgreen', 'skyblue', 'lightpink'))
# drop columns (features ) with > 80% missing vales

c=c1.dropna(thresh=len(c1)*0.8, axis=1)

print('We dropped ',c1.shape[1]-c.shape[1], ' features in the combined set')
print('The shape of the combined dataset after dropping features with more than 80% M.V.', c.shape)
allna = (c.isnull().sum() / len(c))*100

allna = allna.drop(allna[allna == 0].index).sort_values()



def msv2(data, width=12, height=8, color=('silver', 'gold','lightgreen','skyblue','lightpink'), edgecolor='black'):

    """

    SOURCE: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking

    """

    fig, ax = plt.subplots(figsize=(width, height))



    allna = (data.isnull().sum() / len(data))*100

    tightout= 0.008*max(allna)

    allna = allna.drop(allna[allna == 0].index).sort_values().reset_index()

    mn= ax.barh(allna.iloc[:,0], allna.iloc[:,1], color=color, edgecolor=edgecolor)

    ax.set_title('Missing values percentage per column', fontsize=15, weight='bold' )

    ax.set_xlabel('Percentage', weight='bold', size=15)

    ax.set_ylabel('Features with missing values', weight='bold')

    plt.yticks(weight='bold')

    plt.xticks(weight='bold')

    for i in ax.patches:

        ax.text(i.get_width()+ tightout, i.get_y()+0.1, str(round((i.get_width()), 2))+'%',

            fontsize=10, fontweight='bold', color='grey')

    return plt.show()
msv2(c)
NA=c[allna.index.to_list()]
NAcat=NA.select_dtypes(include='object')

NAnum=NA.select_dtypes(exclude='object')

print('We have :',NAcat.shape[1],'categorical features with missing values')

print('We have :',NAnum.shape[1],'numerical features with missing values')
NAnum.head()
NANUM= NAnum.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



NANUM = NANUM.style.background_gradient(cmap=cm)

NANUM
#MasVnrArea: Masonry veneer area in square feet, the missing data means no veneer so we fill with 0

c['MasVnrArea']=c.MasVnrArea.fillna(0)

#LotFrontage has 16% missing values. We fill with the median

c['LotFrontage']=c.LotFrontage.fillna(c.LotFrontage.median())

#GarageYrBlt:  Year garage was built, we fill the gaps with the median: 1980

c['GarageYrBlt']=c["GarageYrBlt"].fillna(1980)

#For the rest of the columns: Bathroom, half bathroom, basement related columns and garage related columns:

#We will fill with 0s because they just mean that the hosue doesn't have a basement, bathrooms or a garage
bb=c[allna.index.to_list()]

nan=bb.select_dtypes(exclude='object')

N= nan.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



N= N.style.background_gradient(cmap=cm)

N
NAcat.head()
NAcat1= NAcat.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



NAcat1 = NAcat1.style.background_gradient(cmap=cm)

NAcat1
fill_cols = ['Electrical', 'SaleType', 'KitchenQual', 'Exterior1st',

             'Exterior2nd', 'Functional', 'Utilities', 'MSZoning']



for col in c[fill_cols]:

    c[col] = c[col].fillna(method='ffill')
dd=c[allna.index.to_list()]

w=dd.select_dtypes(include='object')

a= w.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



a= a.style.background_gradient(cmap=cm)

a
#we will just 'None' in categorical features

#Categorical missing values

NAcols=c.columns

for col in NAcols:

    if c[col].dtype == "object":

        c[col] = c[col].fillna("None")

#we will just fill 0s in the numerical features 

#Numerical missing values

for col in NAcols:

    if c[col].dtype != "object":

        c[col]= c[col].fillna(0)
c.isnull().sum().sort_values(ascending=False).head()






FillNA=c[allna.index.to_list()]







FillNAcat=FillNA.select_dtypes(include='object')



FC= FillNAcat.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



FC= FC.style.background_gradient(cmap=cm)

FC





FillNAnum=FillNA.select_dtypes(exclude='object')



FM= FillNAnum.isnull().sum().to_frame().sort_values(by=[0]).T

cm = sns.light_palette("lime", as_cmap=True)



FM= FM.style.background_gradient(cmap=cm)

FM


