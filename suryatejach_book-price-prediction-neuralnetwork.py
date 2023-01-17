import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.api import OLS
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, accuracy_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df= pd.read_excel('../input/processed-data/Data_Train.xlsx')
train_df.head()
test_df= pd.read_excel('../input/processed-data/Data_Test.xlsx')
test_df.head()
train_df.shape
test_df.shape
train_df.dtypes
train_df.isnull().sum()
train_df.BookCategory.nunique()
train_df.BookCategory.unique()
train_df['BookCategory']= train_df['BookCategory'].replace(
    {
      'Action & Adventure':1, 'Biographies, Diaries & True Accounts':2,
       'Humour':3, 'Crime, Thriller & Mystery':4, 'Arts, Film & Photography':5,
       'Sports':6, 'Language, Linguistics & Writing':7,
       'Computing, Internet & Digital Media':8, 'Romance':9,
       'Comics & Mangas':10, 'Politics':11  
    }
)
train_df.BookCategory.unique()
train_df['BookCategory']= train_df['BookCategory'].astype('category')
train_df['BookCategory'].dtypes
train_df= train_df.drop(["Title", "Author","Edition", "Synopsis","Genre"], axis=1)
train_df.head()
## Checking outliers using Box plot
#pos=1
#plt.figure(figsize=(7,3))
#for i in train_df.columns:
  #plt.subplot(1,3,pos)
  #sns.boxplot(train_df[i],color="red")
  #pos+=1
## Detect outliers using IQR and Handling outliers
#Q1= train_df.quantile(0.25)
#Q3= train_df.quantile(0.75)
#IQR= Q3-Q1
#print(IQR)
#boolean_out= (train_df < (Q1 - 1.5 * IQR)) | (train_df > (Q3 + 1.5 * IQR))
#print(boolean_out)
#train_data= train_df[~boolean_out.any(axis=1)]
#print('Shape of outliers dataset:', train_df.shape)
#print('\n Shape of non outliers dataset:', train_data.shape)
## Checking outliers using Box plot
#pos=1
#plt.figure(figsize=(9,3))
#for i in train_data.columns:
  #plt.subplot(1,3,pos)
  #sns.boxplot(train_data[i],color="red")
  #pos+=1
#from scipy.stats import zscore
#z = np.abs(zscore(train_data))
#print(z)
#threshold = 1.5
#print(np.where(z > 1.5))
#book_data= train_data[(z < 1.5). all(axis=1)]
#print('Shape of outliers dataset:', train_data.shape)
#print('\n Shape of non-outliers dataset:', book_data.shape)
## Checking outliers using Box plot
#pos=1
#plt.figure(figsize=(9,3))
#for i in book_data.columns:
  #plt.subplot(1,3,pos)
  #sns.boxplot(book_data[i],color="green")
  #pos+=1
train_df= pd.get_dummies(train_df)
train_df= train_df[['Reviews','Ratings','BookCategory_1','BookCategory_2','BookCategory_3','BookCategory_4','BookCategory_5','BookCategory_6','BookCategory_7','BookCategory_8','BookCategory_9','BookCategory_10','BookCategory_11','Price']]
train_df.head()
corrl= train_df.corr()
fig = plt.figure(figsize=(7,5))
sns.heatmap(corrl,  annot=True,cmap ='RdYlGn',linewidths=1.5)
X= train_df.drop(["Price"], axis=1)
y= train_df.Price.values
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3, random_state=0)
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledRidge', Pipeline([('Scaler', StandardScaler()),('Ridge', Ridge())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
pipelines.append(('ScaledAda', Pipeline([('Scaler', StandardScaler()),('Ada', AdaBoostRegressor())])))
pipelines.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBRegressor())])))
pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('MLP', MLPRegressor())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Algorithm comparison based on recall
fig = plt.figure(figsize=(15,5))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
xgb_model= XGBRegressor(n_estimators=100).fit(X_train, y_train)
preds= xgb_model.predict(X_test)
from sklearn.metrics import mean_squared_error
print('Mean Squared Logg Error:', mean_squared_log_error(y_test, preds))
mlp_model= MLPRegressor(solver='sgd').fit(X_train, y_train)
mlp_preds= mlp_model.predict(X_test)
print('Mean Squared Logg Error of Neural Networks:', mean_squared_log_error(y_test, mlp_preds))
test_df.head()
test_df.BookCategory.unique()
test_df['BookCategory']= test_df['BookCategory'].replace(
    {
      'Action & Adventure':1, 'Biographies, Diaries & True Accounts':2,
       'Humour':3, 'Crime, Thriller & Mystery':4, 'Arts, Film & Photography':5,
       'Sports':6, 'Language, Linguistics & Writing':7,
       'Computing, Internet & Digital Media':8, 'Romance':9,
       'Comics & Mangas':10, 'Politics':11  
    }
)
test_df.BookCategory.unique()
test_df= test_df.drop(["Title", "Author",	"Edition", "Synopsis",	"Genre"], axis=1)
test_df.head()
test_df.isnull().sum()
test_df['BookCategory']= test_df['BookCategory'].astype('category')
test_df['BookCategory'].dtypes
test_df= pd.get_dummies(test_df)
test_df= test_df[['Reviews','Ratings','BookCategory_1','BookCategory_2','BookCategory_3','BookCategory_4','BookCategory_5','BookCategory_6','BookCategory_7','BookCategory_8','BookCategory_9','BookCategory_10','BookCategory_11']]
test_df.head()
test_preds= mlp_model.predict(test_df)
len(test_preds)
prediction_df= pd.DataFrame(test_preds, columns=["Price"])
prediction_df.head()
prediction_df.to_excel('/kaggle/working/Submission_MLP.xlsx')
