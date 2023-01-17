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
dataset=pd.read_excel('/kaggle/input/vegetable-and-fruits-price-in-india/Vegetable and Fruits Prices  in India.xlsx')
dataset['year'] = pd.DatetimeIndex(dataset['Date']).year
dataset.head()
dataset['price']=dataset['price'].replace(0,np.nan)
#dataset['datesk']=dataset['datesk'].replace(0,np.nan)
#dataset['Item Name']=dataset['Item Name'].replace(0,np.nan)
dataset.dropna(inplace=True)
dataset['year'].unique()
from sklearn.metrics import mean_squared_error
def plotting_3_chart(dataset, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(dataset.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(dataset.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(dataset.loc[:,feature], orient='v', ax = ax3 );
    
plotting_3_chart(dataset, 'price')
print("Skewness: " + str(dataset['price'].skew()))
dataset=dataset.drop(['Date'], axis=1)
dataset=dataset.drop(['datesk'], axis=1)
df_dummies = pd.get_dummies(dataset['Item Name'])
del df_dummies[df_dummies.columns[-1]]
df_new = pd.concat([dataset, df_dummies], axis=1)
del df_new['Item Name']
df_new.head()
dataset.head()
def fixing_skewness(df):
    from scipy.stats import skew
    from scipy.special import boxcox1p
    from scipy.stats import boxcox_normmax
    
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index

    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))

fixing_skewness(df_new)
import seaborn as sns
sns.distplot(df_new['price']);
x=df_new.drop(['price'],axis=1)
y=df_new['price']
x.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
mean_squared_error(y_pred,y_test)
import seaborn as sns
plt.figure(figsize=(20,10))
sns.lineplot(data=dataset, x='year', y='price', color='red')
plt.ylabel("Price")
dataset.tail()
dataset[dataset['Item Name']=='Tamarind seedless'].drop_duplicates()
