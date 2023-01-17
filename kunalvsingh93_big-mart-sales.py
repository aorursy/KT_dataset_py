import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
train=pd.read_csv('../input/big-mart-sales-regression/Trainsales.csv')
train.head(5)
test=pd.read_csv('../input/big-mart-sales-regression/Testsales.csv')
test.head(5)
train['Data']='train'
test['Data']='test'
test['Item_Outlet_Sales']=np.nan
# Combining train and test data
combined=pd.concat([train,test],ignore_index=True,sort=False)
combined.head()
# Target variable: Item_outlet_sales
sns.distplot(train['Item_Outlet_Sales'],color='hotpink')
plt.xlabel='Sales'
plt.ylabel('Frequency')
plt.title('Histogram-Item_Outlet_Sales')
plt.show()
#Inference
#1.Sales appears to be skewed
#2.In +ve skew-mean>median>mode and vice-versa
#seaborn library cannot use missing values
sns.distplot(combined['Item_Weight'].dropna(),color='hotpink')
#Item_weight is uniform,neither +ve,-ve skewed,so we cant use mean,median or mode for missing values
#Item_visibility
sns.distplot(combined['Item_Visibility'],color='hotpink')
#positively skewed
#Item_mrp
sns.distplot(combined['Item_MRP'],color='hotpink')
# It clearly shows that there is a cluster and 4 different groups are created
plt.scatter(combined.Item_MRP,combined.Item_Outlet_Sales,color='red')
combined.Item_Type.value_counts().plot(kind='bar',color='red')
combined.Outlet_Identifier.value_counts().plot(kind='bar',color='red')
combined.Outlet_Location_Type.value_counts().plot(kind='bar',color='red')
combined.Outlet_Type.value_counts().plot(kind='bar',color='red')
combined.Outlet_Size.value_counts().plot(kind='bar',color='red')
#Item wt vs Sales:
plt.scatter(combined.Item_Weight,combined.Item_Outlet_Sales,color='coral')
#No pattern
#Item visibility vs Item Outlet sales:
plt.scatter(combined.Item_Visibility,combined.Item_Outlet_Sales,color='coral')
# Item MRP vs Item outlet sales:
plt.scatter(combined.Item_MRP,combined.Item_Outlet_Sales,color='coral')
plt.figure(figsize=[5,5]) #Modifying the dimensions
#Bivariate: Categorical vs Numerical : Item_Fat_Content vs sales:
sns.boxplot(x='Item_Fat_Content',y='Item_Outlet_Sales',data=combined)
#IQR=Q3-Q1
#min=q1-1.5*iqr
#max=q3+1.5*iqr
# Item type vs Item outlet sales:
plt.figure(figsize=[10,5])
sns.boxplot(x='Item_Type',y='Item_Outlet_Sales',data=combined)
plt.xticks(rotation=90)
combined.isnull().sum()[combined.isnull().sum()!=0]
combined.groupby('Item_Identifier')['Item_Weight'].mean()
combined['Item_Weight']=combined.groupby('Item_Identifier')['Item_Weight'].transform(lambda x:x.fillna(x.mean()))
combined.groupby('Item_Identifier')['Item_Weight'].mean()
combined.isnull().sum()[combined.isnull().sum()!=0]
pd.DataFrame(combined.groupby(['Outlet_Location_Type','Outlet_Type'])['Outlet_Size'].value_counts())
a=combined[combined['Outlet_Size'].isnull()]
a
pd.DataFrame(a.groupby(['Outlet_Location_Type','Outlet_Type'])['Outlet_Identifier'].value_counts())
combined.loc[(combined.Outlet_Location_Type=='Tier 2')&(combined.Outlet_Type=='Supermarket Type1'),'Outlet_Size']='Small'
combined.loc[(combined.Outlet_Location_Type=='Tier 3')&(combined.Outlet_Type=='Grocery Store'),'Outlet_Size']='Small'
combined.isnull().sum()[combined.isnull().sum()!=0]
combined.head()
combined['Item_Fat_Content'].value_counts()
combined['Item_Fat_Content']=combined['Item_Fat_Content'].replace(['low fat'],['Low Fat'])
combined['Item_Fat_Content']=combined['Item_Fat_Content'].replace(['LF'],['Low Fat'])
combined['Item_Fat_Content']=combined['Item_Fat_Content'].replace(['reg'],['Regular'])
combined['Item_Fat_Content'].value_counts()
combined.Item_Type.value_counts()
combined.Item_Type.value_counts()
combined.head()
combined['Years']=2013-combined.Outlet_Establishment_Year
def size(x):
    if x=='Small':
        x=0
    elif x=='Medium':
        x=1
    elif x=='High':
        x=2
    return x
combined['Outlet_Size']=combined['Outlet_Size'].apply(size)
combined.Item_Visibility=combined.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x:x.replace(0,x.median()))
combined.head()
combined['Price_Per_Unit']=combined.Item_MRP/combined.Item_Weight
combined.Item_Identifier[0][:2]
ids=[]
for i in combined.Item_Identifier:
    ids.append(i[:2])
combined['ID_Cat']=pd.Series(ids)
combined.head()
combined.Item_Type.value_counts()
combined.Item_Fat_Content.value_counts()
combined.ID_Cat.value_counts()
pd.DataFrame(combined.groupby('ID_Cat')['Item_Type'].value_counts())
combined.loc[combined.ID_Cat=='NC','Item_Type']='Non Eatables'
pd.DataFrame(combined.groupby('ID_Cat')['Item_Type'].value_counts())
combined.loc[combined.ID_Cat=='FD','Item_Type']='Food'
combined.loc[combined.ID_Cat=='DR','Item_Type']='Drinks'
pd.DataFrame(combined.groupby('ID_Cat')['Item_Type'].value_counts())
combined.loc[combined.ID_Cat=='NC','Item_Fat_Content']='Non Edible'
pd.DataFrame(combined.groupby('ID_Cat')['Item_Fat_Content'].value_counts())
combined.head()
df=combined.copy()
df.drop(['Outlet_Establishment_Year','Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
df.head()
df=pd.get_dummies(df,columns=['Item_Fat_Content','Item_Type','Outlet_Location_Type','Outlet_Type','ID_Cat'],drop_first=True)
df.head()
df.columns
df.skew()
train=df.loc[df['Data']=='train']
train.shape
test=df.loc[df['Data']=='test']
test.shape
train=train.drop('Data',axis=1)
test=test.drop(['Data','Item_Outlet_Sales'],axis=1)
train.head()
test.head()
train.shape,test.shape
X=train.drop('Item_Outlet_Sales',axis=1)
y=train['Item_Outlet_Sales']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
import lightgbm as lgb
from tpot import TPOTRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
test_data = sc.transform(test.copy().values)
pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('lr',LinearRegression())])
pipeline_dtr=Pipeline([('scalar2',StandardScaler()),
                     ('dtr',DecisionTreeRegressor())])
pipeline_rfr=Pipeline([('scalar3',StandardScaler()),
                     ('rfr',RandomForestRegressor())])
pipeline_knn=Pipeline([('scalar4',StandardScaler()),
                     ('knn',KNNR())])
pipeline_svm=Pipeline([('scalar5',StandardScaler()),
                     ('svm',SVR())])
pipeline_ada=Pipeline([('scalar6',StandardScaler()),
                     ('ada',AdaBoostRegressor())])
pipeline_gbr=Pipeline([('scalar7',StandardScaler()),
                     ('gbr',GradientBoostingRegressor())])
pipeline_sgd=Pipeline([('scalar8',StandardScaler()),
                     ('sgd',SGDRegressor())])
pipelines=[pipeline_lr,pipeline_dtr,pipeline_rfr,pipeline_knn,pipeline_svm,pipeline_ada,pipeline_gbr,pipeline_sgd]
best_accuracy=0.0
best_regressor=0
best_pipeline=""
pipe_dict={0:'Linear Regression',1:'Decision Tree Regressor',2:'Random Forest Regressor',3:'KNN',4:'SVM',5:'ADA',6:'GBR',7:'SGD'}
for i in pipelines:
    i.fit(X_train,y_train)
    predictions=i.predict(X_test)
for i,model in enumerate(pipelines):
    print('{} Test Accuracy {}'.format(pipe_dict[i],model.score(X_test,y_test)))
for i,model in enumerate(pipelines):
    if model.score(X_test,y_test)>best_accuracy:
        best_accuracy=model.score(X_test,y_test)
        best_regressor=i
        best_pipeline=model
print("Regressor with best accuracy:{}".format(pipe_dict[best_regressor]))
# Create a pipeline
pipe = Pipeline([("regressor",LinearRegression())])
# Create dictionary with candidate learning algorithms and their hyperparameters
r_param = [{"regressor": [DecisionTreeRegressor()],
            "regressor__criterion":['mse','mae'],
            "regressor__max_depth":[5,8,15,25,30,None],
            "regressor__min_samples_leaf":[1,2,5,10,15,100],
            "regressor__max_leaf_nodes": [2, 5,10]},
               
           {"regressor": [RandomForestRegressor()],
            "regressor__criterion":['mse','mae'],
             "regressor__min_samples_leaf":[1,2,5,10,15,100],
             "regressor__max_leaf_nodes": [2, 5,10]},
           
           {'regressor':[lgb.LGBMRegressor()],
            'regressor__n_estimators':np.arange(50,250,5),
            'regressor__max_depth':np.arange(2,15,5),
            'regressor__num_leaves':np.arange(2,60,5)},
           
            {'regressor':[XGBRegressor()],
              "regressor__learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
            "regressor__max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
            "regressor__min_child_weight" : [ 1, 3, 5, 7 ],
            "regressor__gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            "regressor__colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]},
           
           {'regressor':[SGDRegressor()],
            "regressor__alpha":np.arange(0.0001,0.005,10),
            "regressor__penalty": ['l2']},
           
           {'regressor':[KNNR()],
            "regressor__weights":['uniform','distance'],
            'regressor__n_neighbors':np.arange(1,40)
          }]
           
          
rsearch = RandomizedSearchCV(pipe, r_param, cv=5, verbose=0,n_jobs=-1)
best_model_r = rsearch.fit(X_train,y_train)
print(best_model_r.best_estimator_)
print("The mean accuracy of the model through randomized search is :",best_model_r.score(X_test,y_test))
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
rfr=RandomForestRegressor(n_estimators=20,criterion='mse',max_depth=50,min_samples_split=20,min_samples_leaf=10)
abs(cross_val_score(rfr,X,y,cv=10,scoring='neg_root_mean_squared_error').mean())
rfr=RandomForestRegressor(n_estimators=20,criterion='mse',max_depth=50,min_samples_split=20,min_samples_leaf=10)
model=rfr.fit(X_train,y_train)
predictions=model.predict(X_test)
predictions.shape
output=model.predict(test)
output.shape
Submission_bms=pd.DataFrame(output)
Submission_bms.to_csv('Submission_bms.csv',index=False)