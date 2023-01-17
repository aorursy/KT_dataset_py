# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import pandas_profiling

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/Train.csv")

df_train.head()
df_train.profile_report(style={'full_width':True},title='Training dataset Profiling Report')
#Checking the columns in Training dataset......

print("Columns in training dataset based on datatypes {}".format(df_train.columns.to_series().groupby(df_train.dtypes).groups))
#Checking the dimensions

df_train.shape
#Checking missing values..............

df_train.isnull().sum()
df_train['Item_Weight']=df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean())
df_train['Outlet_Size']=df_train['Outlet_Size'].fillna(df_train['Outlet_Size'].mode()[0])
#Lets check whether we still have missing values in our dataset!!

import seaborn as sns

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')
df_train.describe(include = 'all')
df_train.info()
#Visualizing the "Outlet_Identifier"

df_train['Outlet_Identifier'].value_counts().plot(kind='bar',color = 'Black')
df_train = df_train.drop(['Item_Identifier','Outlet_Identifier'],axis=1)

df_train.head()
#Visualizing the "Item_Fat_Content"

df_train['Item_Fat_Content'].value_counts().plot(kind='bar',color = 'black')
df_train =  df_train.replace(to_replace ="low fat",  value ="Low Fat") 

df_train =  df_train.replace(to_replace ="LF",  value ="Low Fat") 

df_train =  df_train.replace(to_replace ="reg",  value ="Regular") 
#Visualizing the "Item_Fat_Content"

df_train['Item_Fat_Content'].value_counts().plot(kind='bar',color = 'Green')
#Visualizing the "Item_Type"

df_train['Item_Type'].value_counts().plot(kind='bar',color = 'Green')
#Visualizing the "Outlet_Size"

df_train['Outlet_Size'].value_counts().plot(kind='bar',color = 'green')
#Visualizing the "Outlet_Location_Type"

df_train['Outlet_Location_Type'].value_counts().plot(kind='bar',color = 'Green')
#Visualizing the "Outlet_Type"

df_train['Outlet_Type'].value_counts().plot(kind='bar',color = 'green')
y = df_train['Item_Weight']

plt.figure(1); 

sns.distplot(y, kde=True,color = 'red')
y = df_train['Item_Visibility']

plt.figure(1); 

sns.distplot(y, kde=True,color = 'red')
y = df_train['Item_MRP']

plt.figure(1);

sns.distplot(y, kde=True,color = 'red')
y = df_train['Outlet_Establishment_Year']

plt.figure(1); 

sns.distplot(y, kde=True,color = 'red')
y = df_train['Item_Outlet_Sales']

plt.figure(1);

sns.distplot(y, kde=True,color = 'red')
df_train["Qty_Sold"] = (df_train["Item_Outlet_Sales"]/df_train["Item_MRP"])

df_train.head()
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))

sns.heatmap(df_train.corr(),annot=True,cmap="YlGnBu")
categorical_columns=[x for x in df_train.dtypes.index if df_train.dtypes[x]=='object']

categorical_columns
df_train.pivot_table(index='Outlet_Type',values='Item_Outlet_Sales')
#print frequencies of these categories

for col in categorical_columns:

    print('Frequency of categories for variable')

    print(df_train[col].value_counts())

    print("\n")
#Encoding Categorical Variables

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in df_train.columns:

    df_train[col] = labelencoder.fit_transform(df_train[col])





#Now one hot encoding

df_train=pd.get_dummies(df_train, columns=['Item_Fat_Content',

 'Item_Type',

 'Outlet_Size',

 'Outlet_Location_Type',

 'Outlet_Type'],drop_first=False)



print(df_train.shape)
df_train.columns
#Rearrangement of the columns......



df = df_train[['Item_Weight', 'Item_Visibility', 'Item_MRP',

       'Outlet_Establishment_Year',

       'Item_Fat_Content_0', 'Item_Fat_Content_1', 'Item_Type_0',

       'Item_Type_1', 'Item_Type_2', 'Item_Type_3', 'Item_Type_4',

       'Item_Type_5', 'Item_Type_6', 'Item_Type_7', 'Item_Type_8',

       'Item_Type_9', 'Item_Type_10', 'Item_Type_11', 'Item_Type_12',

       'Item_Type_13', 'Item_Type_14', 'Item_Type_15', 'Outlet_Size_0',

       'Outlet_Size_1', 'Outlet_Size_2', 'Outlet_Location_Type_0',

       'Outlet_Location_Type_1', 'Outlet_Location_Type_2', 'Outlet_Type_0',

       'Outlet_Type_1', 'Outlet_Type_2', 'Outlet_Type_3', 'Item_Outlet_Sales', 'Qty_Sold']]

df.head()
df.shape
# iterating the columns 

for col in df.columns: 

    print(col)
#Separating features and label

X = df.iloc[:,0:33].values

y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = None)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_



print(explained_variance)
len(explained_variance)
print("Sorted List returned :")

print(sorted(explained_variance,reverse = True))
with plt.style.context('dark_background'):

    plt.figure(figsize=(16, 8))

    

    plt.bar(range(33), explained_variance, alpha=0.5, align='center',label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()

    
# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 3)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

print(explained_variance)
#Model comparison

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor


#Fit Decision_tree

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
#Fit Decision_tree

tree = DecisionTreeRegressor()

tree.fit(X_train, y_train)
#Fit Random_forest

forest = RandomForestRegressor(n_jobs=-1)

forest.fit(X_train, y_train)
#Fit Ada_Boost_Regressor..........

Ada_boost = AdaBoostRegressor()

Ada_boost.fit(X_train, y_train)
#Fit Bagging_Regressor..........

Bagging = BaggingRegressor()

Bagging.fit(X_train, y_train)
#Fit Extra_tree_regressor........

Extra_trees = ExtraTreesRegressor()

Extra_trees.fit(X_train, y_train)
#Fit Gradient_Boosting_Regressor........

Gradient_boosting = GradientBoostingRegressor()

Gradient_boosting.fit(X_train, y_train)
models= [('lin_reg', lin_reg), ('forest', forest), ('dt', tree),('Ada_boost',Ada_boost),('Bagging',Bagging),('Extra_trees',Extra_trees),('Gradient_boosting',Gradient_boosting)]

scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']





#for each model I want to test three different scoring metrics. Therefore, results[0] will be lin_reg x MSE, 

# results[1] lin_reg x MSE and so on until results [8], where we stored dt x r2



results= []

metric= []

for name, model in models:

    for i in scoring:

        scores = cross_validate(model, X_train, y_train, scoring=i, cv=10, return_train_score=True)

        results.append(scores)



print(results[20])
###############################################################################



#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

LR_RMSE_mean = np.sqrt(-results[0]['test_score'].mean())

LR_RMSE_std= results[0]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

LR_MAE_mean = -results[1]['test_score'].mean()

LR_MAE_std= results[1]['test_score'].std()

LR_r2_mean = results[2]['test_score'].mean()

LR_r2_std = results[2]['test_score'].std()



#THIS IS FOR RF

RF_RMSE_mean = np.sqrt(-results[3]['test_score'].mean())

RF_RMSE_std= results[3]['test_score'].std()

RF_MAE_mean = -results[4]['test_score'].mean()

RF_MAE_std= results[4]['test_score'].std()

RF_r2_mean = results[5]['test_score'].mean()

RF_r2_std = results[5]['test_score'].std()



#THIS IS FOR DT

DT_RMSE_mean = np.sqrt(-results[6]['test_score'].mean())

DT_RMSE_std= results[6]['test_score'].std()

DT_MAE_mean = -results[7]['test_score'].mean()

DT_MAE_std= results[7]['test_score'].std()

DT_r2_mean = results[8]['test_score'].mean()

DT_r2_std = results[8]['test_score'].std()









#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

ADA_RMSE_mean = np.sqrt(-results[9]['test_score'].mean())

ADA_RMSE_std= results[9]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

ADA_MAE_mean = -results[10]['test_score'].mean()

ADA_MAE_std= results[10]['test_score'].std()

ADA_r2_mean = results[11]['test_score'].mean()

ADA_r2_std = results[11]['test_score'].std()







#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

BAGGING_RMSE_mean = np.sqrt(-results[12]['test_score'].mean())

BAGGING_RMSE_std= results[12]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

BAGGING_MAE_mean = -results[13]['test_score'].mean()

BAGGING_MAE_std= results[13]['test_score'].std()

BAGGING_r2_mean = results[14]['test_score'].mean()

BAGGING_r2_std = results[14]['test_score'].std()





#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

ET_RMSE_mean = np.sqrt(-results[15]['test_score'].mean())

ET_RMSE_std= results[15]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

ET_MAE_mean = -results[16]['test_score'].mean()

ET_MAE_std= results[16]['test_score'].std()

ET_r2_mean = results[17]['test_score'].mean()

ET_r2_std = results[17]['test_score'].std()





#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

GB_RMSE_mean = np.sqrt(-results[18]['test_score'].mean())

GB_RMSE_std= results[18]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

GB_MAE_mean = -results[19]['test_score'].mean()

GB_MAE_std= results[19]['test_score'].std()

GB_r2_mean = results[20]['test_score'].mean()

GB_r2_std = results[20]['test_score'].std()
modelDF = pd.DataFrame({

    'Model'       : ['Linear Regression', 'Random Forest', 'Decision Trees','Ada Boosting','Bagging','Extra trees','Gradient Boosting'],

    'RMSE_mean'    : [LR_RMSE_mean, RF_RMSE_mean, DT_RMSE_mean,ADA_RMSE_mean,BAGGING_RMSE_mean,ET_RMSE_mean,GB_RMSE_mean],

    'RMSE_std'    : [LR_RMSE_std, RF_RMSE_std, DT_RMSE_std,ADA_RMSE_std,BAGGING_RMSE_std,ET_RMSE_std,GB_RMSE_std],

    'MAE_mean'   : [LR_MAE_mean, RF_MAE_mean, DT_MAE_mean,ADA_MAE_mean,BAGGING_MAE_mean,ET_MAE_mean,GB_MAE_mean],

    'MAE_std'   : [LR_MAE_std, RF_MAE_std, DT_MAE_std, ADA_MAE_std, BAGGING_MAE_std, ET_MAE_std, GB_MAE_std],

    'r2_mean'      : [LR_r2_mean, RF_r2_mean, DT_r2_mean, ADA_r2_mean,BAGGING_r2_mean, ET_r2_mean, GB_r2_mean],

    'r2_std'      : [LR_r2_std, RF_r2_std, DT_r2_std, ADA_r2_std,BAGGING_r2_std, ET_r2_std, GB_r2_std],

    }, columns = ['Model', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 'r2_mean', 'r2_std'])



    

modelDF.sort_values(by='r2_mean', ascending=False)
import seaborn as sns



sns.factorplot(x= 'Model', y= 'RMSE_mean', data= modelDF, kind='bar',size=6, aspect=4)
from sklearn.model_selection import GridSearchCV,StratifiedKFold



ETC = ExtraTreesRegressor()

gb_param_grid = {'n_estimators' : [100,200,300,400,500],

              'max_depth': [4, 8,12,16],

              'min_samples_leaf' : [100,150,200,250],

              'max_features' : [0.3, 0.1] 

              }



gsETC = GridSearchCV(ETC,param_grid = gb_param_grid, cv=10, n_jobs= -1, verbose = 0)



gsETC.fit(X_train,y_train)



ETC_best = gsETC.best_estimator_
# Best score

gsETC.best_score_,gsETC.best_params_
# =============================================================================

# Model creation

# =============================================================================





ETC = ExtraTreesRegressor(max_depth= 8,max_features = 0.3,min_samples_leaf =  100,n_estimators= 500)

ETC.fit(X_train, y_train)





#predicting the test set

y_pred = ETC.predict(X_test)

from sklearn import metrics

print("MAE:", metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))