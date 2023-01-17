# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from matplotlib import rcParams
import warnings 
warnings.filterwarnings("ignore")
%matplotlib inline
plt.style.use('fivethirtyeight')
#plt.style.use('bmh')
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['text.color'] = 'k'
wine = pd.read_csv('../input/winemag-data-130k-v2.csv',index_col=None)
wine.head()
wine.drop('Unnamed: 0',axis=1,inplace=True)
wine = wine.reset_index(drop=True)
wine.head()
wine.describe()
print("Total number of examples",wine.shape[0])
print("Total number of examples with the same title and description", wine[wine.duplicated(['description','title'])].shape[0])
wine.drop_duplicates(['title','description'],inplace=True)
wine = wine.reset_index(drop=True)
wine.head()
wine.info()
total = wine.isnull().sum().sort_values(ascending=False)
percent = (wine.isnull().sum()/wine.isnull().count()*100).sort_values(ascending = False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data
wine.dropna(subset=['variety'],axis=0,inplace=True)
wine = wine.reset_index(drop=True)
wine['price'] = wine.groupby('variety')['price'].transform(lambda x: x.fillna(x.mean()))
wine.info()
wine.dropna(subset = ['price'],axis=0,inplace=True)
wine = wine.reset_index(drop=True)
f,(ax1,ax2) = plt.subplots(ncols=2)
sns.distplot(wine['price'],hist=True,ax=ax1)
ax1.set_title("Distribution of Wine",fontsize = 20)
sns.distplot(wine['price'],hist=True,bins=1000,ax=ax2)
ax2.set_title("Distribution of Wine within 0-200$",fontsize = 20)
ax2.set(xlim=(0,200))
f.set_size_inches(15,5)

f,ax1 = plt.subplots(ncols=1)
sns.distplot(wine['points'],hist=True,color='r',ax=ax1)
f.set_size_inches(15,5)
f,ax1 = plt.subplots(ncols=1)
sns.scatterplot(y = 'points',x='price',data=wine,ax=ax1,color='g')
f.set_size_inches(15,5)
country = pd.DataFrame(wine.groupby(by = 'country')['country'].count().sort_values(ascending=False))
country.head(10).plot.bar(color = 'y')
plt.title("Top 10 countries having the highest count of wine")
wine_price = pd.DataFrame(wine.groupby(by = 'country')['price'].mean().sort_values(ascending=False))
wine_price.head(10).plot.bar(color = 'r')
plt.title("Top 10 countries having the highest prize of wine")
top_variety_count = wine.groupby(by = 'variety')['variety'].count().sort_values(ascending=False)
top_variety_count.head(10).plot.bar(color = 'b')
plt.title("Top 10 varieties according to count")
expensive_wines = wine.groupby(by = 'variety')['price'].mean().sort_values(ascending=False)
expensive_wines.head(10).plot.bar(color = 'y')
plt.title("Top 10 varieties of most expensive wines")
plt.figure(figsize = (14,6))
sns.boxplot(
    x = 'variety',
    y = 'points',
    data = wine[wine.variety.isin(wine.variety.value_counts().head(6).index)]
)
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostRegressor, cv
X = wine.drop(columns=['points','description','title',])
X = X.reset_index(drop=True)
X = X.fillna(-1)
y = wine['points']
X.columns
categorical_features_indices = [0,1,3,4,5,6,7,8,9]
model = CatBoostRegressor(
        random_seed = 400,
        loss_function = 'RMSE',
        iterations=400,
    )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 
                                                    random_state=52)
def perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test):
    model = CatBoostRegressor(
        random_seed = 400,
        loss_function = 'RMSE',
        iterations=400,
    )
    
    model.fit(
        X_train, y_train,
        cat_features = categorical_features_indices,
        eval_set=(X_valid, y_valid),
        verbose=False
    )
    
    print("RMSE on training data: "+ model.score(X_train, y_train).astype(str))
    print("RMSE on test data: "+ model.score(X_test, y_test).astype(str))
    
    return model
    

model=perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test)
feature_score = pd.DataFrame(list(zip(X.dtypes.index, model.get_feature_importance(Pool(X, label=y, cat_features=categorical_features_indices)))),
                columns=['Feature','Score'])

feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')
plt.rcParams["figure.figsize"] = (12,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')

rects = ax.patches

labels = feature_score['Score'].round(2)

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

plt.show()

X=X.drop(columns=['designation','province','region_2','variety','region_1'])
X=X.fillna(-1)
X = X.reset_index(drop=True)

print(X.columns)
categorical_features_indices =[0,2,3,4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 
                                                    random_state=52)
model=perform_model(X_train, y_train,X_valid, y_valid,X_test, y_test)
