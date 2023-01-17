# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

copied_train = train.copy()

test = pd.read_csv('../input/home-data-for-ml-course/test.csv')

copied_test = test.copy()
all_data= pd.concat([train,test])

C_all_data = all_data.copy()
num_col = C_all_data.select_dtypes(exclude=['object'])

cat_col = C_all_data.select_dtypes(exclude=['int64','float64'])
fig = plt.figure(figsize=(18,16))

for index,col in enumerate(num_col.columns):

    plt.subplot(10,4,index+1)

    sns.distplot(num_col.loc[:,col].dropna(), kde=False)

fig.tight_layout(pad=1.0)
plt.figure(figsize=(15,5))

sns.heatmap(num_col.isnull(),yticklabels=0,cbar=False,cmap='viridis')
num_col[[col for col in num_col.columns if num_col[col].isnull().any()]].isnull().sum().reset_index()
fig = plt.figure(figsize=(25,8))

D_Nun_Catcolumns = pd.DataFrame({'features':cat_col.isnull().sum().reset_index().iloc[:,0],

                                'sum':cat_col.isnull().sum().reset_index().iloc[:,1]}).sort_values(

    by= ['sum'], ascending=False).head(20)

sns.barplot(x='features', y='sum' ,data=D_Nun_Catcolumns)
cat_col[[col for col in cat_col.columns if cat_col[col].isnull().any()]].isnull().sum().reset_index()
# Must Be The Result Of Multi "plt.subplot(10,4,Id+1)" 10*4 > len(num_col.columns)

len(num_col.columns)
fig = plt.figure(figsize=(20,36)).tight_layout(pad=1.0)

for  Id ,col in enumerate(num_col.drop(columns='Id',inplace=False).columns):

    plt.subplot(10,4,Id+1)

    plt.scatter(num_col[col], num_col.SalePrice)

    plt.xlabel(col)

    #plt.ylabel('SalePrice')
len(cat_col.columns)
fig = plt.figure(figsize=(20,38)).tight_layout(pad= 1.0)

for Id, col in enumerate(cat_col.columns):

    plt.subplot(11,4,Id+1)

    sns.boxplot(cat_col[col], num_col.SalePrice)

    plt.xlabel(col)
plt.figure(figsize=(14,12))

correlation = num_col.corr()

sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.5, cmap='Blues')
num_col.loc[:,['GarageYrBlt','YearBuilt']]
num_col[['GarageYrBlt','YearBuilt']].isnull().sum()
# TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

# GrLivArea: Above grade (ground) living area square feet

num_col.loc[:,['TotRmsAbvGrd','GrLivArea','SalePrice']][:30]
# 1stFlrSF: First Floor square feet

# TotalBsmtSF: Total square feet of basement area

num_col.loc[:,['1stFlrSF','TotalBsmtSF']][:30]
num_col[['1stFlrSF','TotalBsmtSF']].isnull().sum()
num_col.loc[:,['GarageArea','GarageCars','SalePrice']][:50]
num_col[['GarageArea','GarageCars']].isnull().sum()
import pandas as pd

import matplotlib.pyplot as plt



def catscatter(df,colx,coly,cols,color=['grey','black'],ratio=10,font='Helvetica',save=False,save_name='Default'):

    '''

    Goal: This function create an scatter plot for categorical variables. It's useful to compare two lists with elements in common.

    Input:

        - df: required. pandas DataFrame with at least two columns with categorical variables you want to relate, and the value of both (if it's just an adjacent matrix write 1)

        - colx: required. The name of the column to display horizontaly

        - coly: required. The name of the column to display vertically

        - cols: required. The name of the column with the value between the two variables

        - color: optional. Colors to display in the visualization, the length can be two or three. The two first are the colors for the lines in the matrix, the last one the font color and markers color.

            default ['grey','black']

        - ratio: optional. A ratio for controlling the relative size of the markers.

            default 10

        - font: optional. The font for the ticks on the matrix.

            default 'Helvetica'

        - save: optional. True for saving as an image in the same path as the code.

            default False

        - save_name: optional. The name used for saving the image (then the code ads .png)

            default: "Default"

    Output:

        No output. Matplotlib object is not shown by default to be able to add more changes.

    '''

    # Create a dict to encode the categeories into numbers (sorted)

    colx_codes=dict(zip(df[colx].sort_values().unique(),range(len(df[colx].unique()))))

    coly_codes=dict(zip(df[coly].sort_values(ascending=False).unique(),range(len(df[coly].unique()))))

    

    # Apply the encoding

    df[colx]=df[colx].apply(lambda x: colx_codes[x])

    df[coly]=df[coly].apply(lambda x: coly_codes[x])

    

    

    # Prepare the aspect of the plot

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False

    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    plt.rcParams['font.sans-serif']=font

    plt.rcParams['xtick.color']=color[-1]

    plt.rcParams['ytick.color']=color[-1]

    plt.box(False)



    

    # Plot all the lines for the background

    for num in range(len(coly_codes)):

        plt.hlines(num,-1,len(colx_codes)+1,linestyle='dashed',linewidth=1,color=color[num%2],alpha=0.5)

    for num in range(len(colx_codes)):

        plt.vlines(num,-1,len(coly_codes)+1,linestyle='dashed',linewidth=1,color=color[num%2],alpha=0.5)

        

    # Plot the scatter plot with the numbers

    plt.scatter(df[colx],

               df[coly],

               s=df[cols]*ratio,

               zorder=2,

               color=color[-1])

    

    # Change the ticks numbers to categories and limit them

    plt.xticks(ticks=list(colx_codes.values()),labels=colx_codes.keys(),rotation=90)

    plt.yticks(ticks=list(coly_codes.values()),labels=coly_codes.keys())

    plt.xlim(xmin=-1,xmax=len(colx_codes))

    plt.ylim(ymin=-1,ymax=len(coly_codes))

    

    # Save if wanted

    if save:

        plt.savefig(save_name+'.png')
#catscatter(cat_col,'BsmtQual', 'BsmtCond', 'BsmtExposure')
num_col.columns
C_all_data.drop(columns=['GarageYrBlt','TotalBsmtSF','GarageCars'],inplace=True,axis=1)
C_all_data.loc[:,['OpenPorchSF','EnclosedPorch','ScreenPorch','SalePrice']][:50]
C_all_data.loc[:,['BsmtFullBath','BsmtHalfBath','SalePrice']][:50]
C_all_data.drop(columns=['OpenPorchSF','EnclosedPorch','ScreenPorch','BsmtFullBath','BsmtHalfBath'],inplace=True,axis=1)
C_all_data.loc[:,['FullBath','HalfBath','SalePrice']][:50]
C_all_data.loc[:,['BsmtFinSF1','BsmtFinSF2','SalePrice']][:50]
C_all_data.loc[:,['LotFrontage','LotArea','SalePrice']][:50]
C_all_data['TotalLot'] = C_all_data['LotFrontage'] + C_all_data['LotArea']

C_all_data['TotalBsmtFin'] = C_all_data['BsmtFinSF1'] + C_all_data['BsmtFinSF2']

C_all_data['TotalBath'] = C_all_data['FullBath'] + C_all_data['HalfBath']
C_all_data.drop(columns=['LotFrontage','LotArea','BsmtFinSF1','BsmtFinSF2','FullBath','HalfBath'],inplace=True,axis=1)
colum = ['MasVnrArea','TotalBsmtFin','2ndFlrSF','WoodDeckSF']



for col in colum:

    col_name = col+'_bin'

    C_all_data[col_name] = C_all_data[col].apply(lambda x: 1 if x > 0 else 0)
C_all_data.select_dtypes(exclude=['object']).isnull().any()
for col in ['MasVnrArea','BsmtUnfSF','GarageArea','TotalLot','TotalBsmtFin'] :

    C_all_data[col] = C_all_data.fillna(C_all_data[col].describe()['50%'])
C_all_data.drop(columns=['PoolQC','MiscFeature','Alley','Fence'],inplace=True,axis=1)
X = C_all_data.drop("SalePrice", axis=1)

y = C_all_data[['SalePrice']]
y["SalePrice"] = np.log(y['SalePrice'])
X = pd.get_dummies(X)
C_all_data.shape
C_all_data.head()
C_all_data = C_all_data.dropna(how='any')

C_all_data.shape
C_all_data.drop_duplicates()

print("shape of dataframe after dropping duplicates"), C_all_data.drop_duplicates().shape
plt.figure(figsize=(10,6))

plt.title("Before transformation of SalePrice")

dist = sns.distplot(train['SalePrice'],norm_hist=False)
plt.figure(figsize=(10,6))

plt.title("After transformation of SalePrice")

dist = sns.distplot(np.log(C_all_data['SalePrice']),norm_hist=False)
C_all_data = pd.get_dummies(C_all_data)
all_train = C_all_data[C_all_data['SalePrice'].notna()]

all_test = C_all_data[C_all_data['SalePrice'].isna()]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['Id','SalePrice'],axis=1), 

                                                    all_train['SalePrice'], test_size=0.30, 

                                                    random_state=101)
from sklearn.tree import DecisionTreeRegressor

logmodel = DecisionTreeRegressor()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import mean_absolute_error

mea = mean_absolute_error(y_test,predictions)

print(mea)
TestForPred = C_all_data.drop(['Id', 'SalePrice'], axis = 1)

logSub = pd.DataFrame({'Id': C_all_data['Id'], 'SalePrice':logmodel.predict(TestForPred).astype(int) })

logSub.to_csv("House Prediction Final.csv", index = False)