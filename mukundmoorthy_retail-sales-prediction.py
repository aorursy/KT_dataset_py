# Importing the packages and dataset



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import pandas_profiling



%matplotlib inline



import os

print(os.listdir("../input"))
data_Train = pd.read_csv("../input/Train_Retail.csv")

data_Test = pd.read_csv("../input/Test_Retail.csv")

#data_Train['Source'] = 'train'

#data_Test['Source'] = 'test'
data_Train.head()                                    # CHecking the first 5 rows
data_Test.head()
data_Train.dtypes
data_Test.dtypes
data_Train.isnull().sum()
data_Test.isnull().sum()
data_Train.describe()
data_Test.describe()
Report1 = pandas_profiling.ProfileReport(data_Train)

Report2 = pandas_profiling.ProfileReport(data_Test)

Report1.to_file('Report1.html')

Report2.to_file('Report2.html')
data_Train.mean()
data_Test.mean()
from scipy import stats



Train_Mean = stats.trim_mean(data_Train.Item_Weight,0.18,axis=-1)

Train_Mean
f =data_Train.groupby(['Item_Type'])

f.first()

f['Item_Weight'].mean()


gx = f.get_group('Dairy')

gx = gx.fillna(f.get_group('Dairy').mean())



ax = f.get_group('Fruits and Vegetables')

ax = ax.fillna(f.get_group('Fruits and Vegetables').mean())



bx = f.get_group('Snack Foods')

bx = bx.fillna(f.get_group('Snack Foods').mean())



cx = f.get_group('Baking Goods')

cx = cx.fillna(f.get_group('Baking Goods').mean())



dx = f.get_group('Breads')

dx = dx.fillna(f.get_group('Breads').mean())



ex = f.get_group('Breakfast')

ex = ex.fillna(f.get_group('Breakfast').mean())



fx = f.get_group('Canned')

fx = fx.fillna(f.get_group('Canned').mean())



hx = f.get_group('Starchy Foods')

hx = hx.fillna(f.get_group('Starchy Foods').mean())



ix = f.get_group('Soft Drinks')

ix = ix.fillna(f.get_group('Soft Drinks').mean())



jx = f.get_group('Frozen Foods')

jx = jx.fillna(f.get_group('Frozen Foods').mean())



kx = f.get_group('Hard Drinks')

kx = kx.fillna(f.get_group('Hard Drinks').mean())



lx = f.get_group('Health and Hygiene')

lx = lx.fillna(f.get_group('Health and Hygiene').mean())



mx = f.get_group('Household')

mx = mx.fillna(f.get_group('Household').mean())



nx = f.get_group('Meat')

nx = nx.fillna(f.get_group('Meat').mean())



ox = f.get_group('Others')

ox = ox.fillna(f.get_group('Others').mean())



px = f.get_group('Seafood')

px = px.fillna(f.get_group('Seafood').mean())



Train_Weight_Final = [ax,bx,cx,dx,ex,fx,gx,hx,ix,jx,kx,lx,mx,nx,ox,px]



Result1 = pd.concat(Train_Weight_Final)



Result1.apply(lambda x: x.isnull().sum())
Result1.describe()
e = data_Test.groupby(['Item_Type'])

e['Item_Weight'].mean()
e.apply(lambda x: x.isnull().sum())
gy = e.get_group('Dairy')

gy = gy.fillna(e.get_group('Dairy').mean())



ay = e.get_group('Fruits and Vegetables')

ay = ay.fillna(e.get_group('Fruits and Vegetables').mean())



by = e.get_group('Snack Foods')

by = by.fillna(e.get_group('Snack Foods').mean())



cy = e.get_group('Baking Goods')

cy = cy.fillna(e.get_group('Baking Goods').mean())



dy = e.get_group('Breads')

dy = dy.fillna(e.get_group('Breads').mean())



ey = e.get_group('Breakfast')

ey = ey.fillna(e.get_group('Breakfast').mean())



fy = e.get_group('Canned')

fy = fy.fillna(e.get_group('Canned').mean())



hy = e.get_group('Starchy Foods')

hy = hy.fillna(e.get_group('Starchy Foods').mean())



iy = e.get_group('Soft Drinks')

iy = iy.fillna(e.get_group('Soft Drinks').mean())



jy = e.get_group('Frozen Foods')

jy = jy.fillna(e.get_group('Frozen Foods').mean())



ky = e.get_group('Hard Drinks')

ky = ky.fillna(e.get_group('Hard Drinks').mean())



ly = e.get_group('Health and Hygiene')

ly = ly.fillna(e.get_group('Health and Hygiene').mean())



my = e.get_group('Household')

my = my.fillna(e.get_group('Household').mean())



ny = e.get_group('Meat')

ny = ny.fillna(e.get_group('Meat').mean())



oy = e.get_group('Others')

oy = oy.fillna(e.get_group('Others').mean())



py = e.get_group('Seafood')

py = py.fillna(e.get_group('Seafood').mean())



Test_Weight_Final = [ay,by,cy,dy,ey,fy,gy,hy,iy,jy,ky,ly,my,ny,oy,py]



Result2 = pd.concat(Test_Weight_Final)
Result2.apply(lambda x: x.isnull().sum())
pd.crosstab(Result1['Outlet_Size'], 'count')
pd.crosstab(Result2['Outlet_Size'], 'count')
Result1['Outlet_Size'] = Result1['Outlet_Size'].fillna('Other')

Result2['Outlet_Size'] = Result2['Outlet_Size'].fillna('Other')
pd.crosstab(Result1['Outlet_Size'], 'count')
pd.crosstab(Result2['Outlet_Size'], 'count')
Result1.isnull().sum()
Result2.isnull().sum()
Result1['Item_Type'] = data_Train['Item_Type']

Result2['Item_Type'] = data_Test['Item_Type']
# Lets visualize the relation between the variables



sns.pairplot(Result1)
corr = Result1.corr()
sns.heatmap(corr, annot=True)
sns.lmplot(x='Item_Outlet_Sales', y='Item_MRP', data=Result1)
plt.figure(figsize=(20,10))

plt.xticks(rotation=90)

sns.barplot(x='Item_Type', y = 'Item_Outlet_Sales', data = Result1 )
plt.figure(figsize=[20,6])

plt.xticks(rotation = 90)

sns.barplot(Result1.Item_Type,Result1.Item_MRP,hue=Result1.Outlet_Location_Type)
plt.figure(figsize=[15,6])

plt.xticks(rotation=90)

sns.countplot(Result1.Item_Type,hue=Result1.Outlet_Location_Type)
Result1.groupby(['Outlet_Location_Type'])['Item_Type'].value_counts()
sns.boxplot(x='Outlet_Size', y = 'Item_Outlet_Sales', data = Result1)
median_sales=Result1.Item_Outlet_Sales.groupby(Result1.Outlet_Size).median()

median_sales
sns.barplot(x='Outlet_Size', y = 'Item_Outlet_Sales', data = Result1)

plt.figure(figsize=(12,6))

sns.barplot(x='Outlet_Type', y = 'Item_Outlet_Sales', data = Result1)
plt.figure(figsize=(12,6))

sns.boxplot(x='Outlet_Establishment_Year', y = 'Item_Outlet_Sales', data = Result1)
plt.figure(figsize=(12,6))

sns.countplot(Result1.Outlet_Establishment_Year)
plt.figure(figsize=(12,6))

sns.barplot(x='Outlet_Location_Type', y = 'Item_Outlet_Sales', data = Result1)
Result1.Outlet_Location_Type.value_counts()
Result1.groupby('Outlet_Location_Type')['Item_MRP'].mean()
sns.boxplot(x='Outlet_Location_Type', y = 'Item_Outlet_Sales', hue='Outlet_Size', data = Result1)
Result1['Outlet_Size'] = Result1['Outlet_Size'].replace({'Small':0, 'Medium':1, 'High':2, 'Other':3})
Result2['Outlet_Size'] = Result2['Outlet_Size'].replace({'Small':0, 'Medium':1, 'High':2, 'Other':3})
Result1['Item_Category'] = Result1['Item_Identifier'].apply(lambda x: x[0:2])
Result1['Item_Category'].value_counts()
Result2['Item_Category'] = Result2['Item_Identifier'].apply(lambda x: x[0:2])
Result2['Item_Category'].value_counts()
Result1.Item_Fat_Content.unique()
Result2.Item_Fat_Content.unique()
Result1['Item_Fat_Content'] = Result1['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})

Result1.reset_index(level=0, inplace=True)



#Check if they are replaced properly

Result1.Item_Fat_Content.unique()
Result2['Item_Fat_Content'] = Result2['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})

Result2.reset_index(level=0, inplace=True)



#Check if they are replaced properly

Result2.Item_Fat_Content.unique()
Result1.loc[Result1['Item_Category']=="NC",'Item_Fat_Content'] = "Non-Edible"
Result2.loc[Result2['Item_Category']=="NC",'Item_Fat_Content'] = "Non-Edible"
Result1.Item_Fat_Content.unique()

Result2.Item_Fat_Content.unique()
Result1.head()

Result1.shape
Result2.head()

Result2.shape
Result2.drop(['index','Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'], axis=1, inplace=True)
Result1.drop(['index','Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'], axis=1, inplace=True)
print(Result1.shape)

print(Result2.shape)
Result1.dtypes
Result2.dtypes
# All the categorical columns



var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Category','Outlet_Type','Item_Type']
# Dataframe of all the categorical columns

categorical_cols=Result1[var_mod]

categorical_cols.head()
cat = Result2[var_mod]

cat.head()
# Lets convert 'Outlet_Size' from int to string data type



categorical_cols['Outlet_Size']=categorical_cols['Outlet_Size'].astype(str)

categorical_cols.dtypes
cat['Outlet_Size']=cat['Outlet_Size'].astype(str)

cat.dtypes
dummy_encoded_Train_data = pd.get_dummies(categorical_cols, drop_first = True)

dummy_encoded_Test_data = pd.get_dummies(cat, drop_first = True)
dummy_encoded_Train_data.shape
dummy_encoded_Test_data.shape
dummy_encoded_Train_data.columns
Result1.shape
Result1.columns
data_train = pd.concat([Result1,dummy_encoded_Train_data],1)

data_test = pd.concat([Result2,dummy_encoded_Test_data],1)
data_train.columns
print(data_train.shape)

print(data_test.shape)
drop_categorical_cols=Result1[var_mod]

drop_cat = Result2[var_mod]
data_train.drop(drop_categorical_cols,1,inplace=True)

data_train.shape
data_test.drop(drop_cat,1,inplace=True)

data_test.shape
data_train.columns
data_Train_Y = data_train['Item_Outlet_Sales']

del data_train['Item_Outlet_Sales']

data_Train_X = data_train
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_Train_X, data_Train_Y, random_state=1, test_size=0.3)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
# Setting to display all the columns of the dataframe

pd.options.display.max_columns=False                
from sklearn import model_selection, metrics                                     

from sklearn.linear_model import LinearRegression                                # Importing LinearRegression from sklearn



lin_model = LinearRegression()

lin_model.fit(x_train,y_train)                                                   # Fitting the model on X_train and y_train
lin_model.coef_
coef1 = pd.DataFrame(lin_model.coef_,index=x_train.columns)                       # Create a dataframe of all the columns and their coefficients          

coef1
lin_model_predictions = lin_model.predict(x_test)
# Lets have a look at some of the predictions

lin_model_predictions[:8]
print("RMSE:",(np.sqrt(metrics.mean_squared_error(y_test, lin_model_predictions))))
lin_model.score(x_test,y_test)
print(metrics.r2_score(y_test,lin_model_predictions))
from xgboost.sklearn import XGBRegressor                                          # Importing the XGBoost Regressor
xgb = XGBRegressor(n_estimators=75,                                              # Number of trees

                   learning_rate=0.08,                                            # Learning Rate

                   gamma=0,                                                       # Minimum reduction in loss(entropy) to make further branches

                   subsample=0.75,                                                # Subsample ratio of the training instance

                   colsample_bytree=1,                                            # Subsample ratio of columns when constructing each tree

                   max_depth=7)                                                   # Max depth of each learner (Decision Tree)

xgb.fit(x_train.values,y_train.values)                                          
y_pred = xgb.predict(x_test.values)                                            
print('RMSE \n',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))            
print(metrics.r2_score(y_test,y_pred))                                   
from sklearn.model_selection import GridSearchCV                                  # Importing Grid Search
# param_grid is a dictionary containing different parameters for XGBoost



param_grid = {"max_depth": [3,5,10,15],

              "n_estimators": [50,100,200,300] , 

              "gamma": [0.1, 0.2, 0,3], 

              "learning_rate": [0.08],

              "min_child_weight": [5], 

              "colsample_bytree": [0.8], 

              "subsample": [0.85]} 



# Performing Grid Search on xgb regressor and param_grid parameters



grid_search = GridSearchCV(xgb, 

                           param_grid=param_grid,

                           cv = 2,

                           n_jobs=-1,

                           scoring='neg_mean_squared_error',

                           verbose=2)
grid_search.fit(x_train.values,y_train.values)
#Displaying the best parameters for XGBoost

print(grid_search.best_params_)         
y_pred = grid_search.predict(x_test.values)
print('RMSE \n',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))    # Measuring the error
print(metrics.r2_score(y_test,y_pred))                  # Checking the R2 score.
# Save it to a numpy array before model prediction



X_test = data_test.values
X_test.shape
y_test_predict=grid_search.predict(X_test)
y_test_predict[:8]