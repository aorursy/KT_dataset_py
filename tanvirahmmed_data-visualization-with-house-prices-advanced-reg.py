import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline
tx = pd.read_csv('/kaggle/input/house-prices-data/train.csv')
tx.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)
year_all = ['YearBuilt', 'YearRemodAdd','YrSold','MoSold','GarageYrBlt']
for i in tx:
  if (tx[i].dtypes == object and i != 'FireplaceQu') or i in year_all:
    tx[i] = tx[i].fillna(tx[i].mode()[0])
cat_data = []
for i in tx:
  if len(tx[i].unique()) <=10:
    cat_data.append(i)
#BarPlot
nr_rows = 16
nr_cols = 3
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3), squeeze=False)
i = 0
for r in range(0,nr_rows):
    for c in range(0, nr_cols):
      if i< len(cat_data):
        x = tx[cat_data[i]]
        sns.barplot(x, tx['SalePrice'], alpha=0.9,ax = axs[r][c])
        plt.ylabel('SalePrice', fontsize=12)
        plt.xlabel(cat_data[i], fontsize=12)
        i+=1
plt.tight_layout()    
plt.show()
g = sns.FacetGrid(tx, col="SaleType", height=4, aspect=.5)
g.map(sns.barplot, "SaleCondition", "SalePrice")
#CountPlot
nr_rows = 16
nr_cols = 3
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3), squeeze=False)
i = 0
for r in range(0,nr_rows):
    for c in range(0, nr_cols):
      if i< len(cat_data):
        x = tx[cat_data[i]]
        sns.countplot(x, data = tx, ax = axs[r][c])
        #plt.ylabel('SalePrice', fontsize=12)
        plt.xlabel(cat_data[i], fontsize=12)
        i+=1
plt.tight_layout()    
plt.show()
#BoxPlot
nr_rows = 16
nr_cols = 3
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3), squeeze=False)
i = 0
for r in range(0,nr_rows):
    for c in range(0, nr_cols):
      if i< len(cat_data):
        sns.boxplot(x=tx[cat_data[i]], y = tx['SalePrice'], data=tx, ax = axs[r][c])
        plt.ylabel('SalePrice', fontsize=12)
        plt.xlabel(cat_data[i], fontsize=12)
        i+=1
plt.tight_layout()    
plt.show()
#ScatterPlot
g = sns.FacetGrid(tx, col="LotShape", hue="Street")
g.map(sns.scatterplot, "LotFrontage", "SalePrice", alpha=.7)
g.add_legend()
#g.fig.subplots_adjust(wspace=.1, hspace=.1 )

g = sns.FacetGrid(tx, col="LotShape", hue="Street" )
g.map(sns.scatterplot, "LotArea", "SalePrice", alpha=.7  )
g.add_legend()
#DistPlot
for i in cat_data:
  g = sns.FacetGrid(tx, col=i)
  g.map(sns.distplot, "SalePrice")
g = sns.FacetGrid(tx, col="TotRmsAbvGrd", hue="KitchenAbvGr" )
g.map(sns.scatterplot, "BedroomAbvGr", "SalePrice", alpha=.7  )
g.add_legend()
g = sns.FacetGrid(tx, col="OverallQual", height=8, aspect=.5)
g.map(sns.barplot, "OverallCond", "SalePrice")
g = sns.FacetGrid(tx, col="LotShape", height=8, aspect=.5)
g.map(sns.barplot, "LotConfig", "LotArea")
g = sns.FacetGrid(tx, col="Condition2", height=8, aspect=.5)
g.map(sns.barplot, "OverallCond", "SalePrice")
g = sns.FacetGrid(tx, col="MSZoning", height=8, aspect=.5)
g.map(sns.barplot, "MSSubClass", "SalePrice")
int_col = tx.select_dtypes(exclude=object).columns
for i in int_col:
    tx[i] = tx[i].fillna(tx[i].mean())
len(int_col)

#Univariate Analysis
nr_rows = 13
nr_cols = 3
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3), squeeze=False)
i = 0
for r in range(0,nr_rows):
    for c in range(0, nr_cols):
      if i< len(int_col):
        #sns.jointplot(x = np.log1p(tx[int_col[i]]), y = np.log1p(tx['SalePrice']), data = tx, kind = 'reg', ax = axs[r][c])
        sns.scatterplot(x = np.log1p(tx[int_col[i]]), y = np.log1p(tx['SalePrice']), data = tx, ax = axs[r][c])
        plt.ylabel('SalePrice', fontsize=12)
        plt.xlabel(int_col[i], fontsize=12)
        i+=1
      else:
        break
plt.tight_layout()    
plt.show()

#BoxPlot
nr_rows = 13
nr_cols = 3
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3), squeeze=False)
i = 0
for r in range(0,nr_rows):
    for c in range(0, nr_cols):
      if i< len(int_col):
        sns.boxplot(x=tx[int_col[i]], ax = axs[r][c])
        #plt.ylabel('SalePrice', fontsize=12)
        plt.xlabel(int_col[i], fontsize=12)
        i+=1
plt.tight_layout()    
plt.show()
# Skewness & Kurtosis

int_feature = np.log1p(tx.select_dtypes(exclude=object).copy())

f, axes = plt.subplots(12, 3, figsize=(50, 100), sharex=True)
c = 0
for i in range(10):
  for j in range(3):
    #sns.histplot(int_feature.iloc[:,c], kde=True, ax=axes[i,j])#skyblue
    #sns.displot(int_feature.iloc[:,c], color="red",kind = 'hist', ax=axes[i,j])#skyblue
    sns.kdeplot(int_feature.iloc[:,c], color="red", cumulative=True, bw=1.5, ax=axes[i,j])
    #sns.displot(data=int_feature, x=int_feature.iloc[:,c], kde=True)
    c+=1
for i, ax in enumerate(axes.reshape(-1)):
    ax.text(x=0.97, y=0.97, transform=ax.transAxes, s="Skewness: %f" % int_feature.iloc[:,i].skew(),\
        fontweight='demibold', fontsize=20, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:poo brown')
    ax.text(x=0.97, y=0.91, transform=ax.transAxes, s="Kurtosis: %f" % int_feature.iloc[:,i].kurt(),\
        fontweight='demibold', fontsize=20, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:dried blood')
plt.tight_layout()
