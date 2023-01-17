import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import mode



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv("../input/walmart-sales/Train.csv")

test_df = pd.read_csv("../input/walmart-sales/Test.csv")
train_df.head(10)
train_df.info()
for cols in train_df.columns:

    print(cols, train_df[cols].nunique())
train_df.describe()
train_df['Item_Identifier']
corr = train_df[['Item_Weight', 'Item_Visibility', 'Item_MRP','Item_Outlet_Sales']].corr(method='pearson')

sns.heatmap(corr)
def relations(x):

    Outlet_Identifier_Pivot = train_df.pivot_table(index=x, values='Item_Outlet_Sales', aggfunc=np.median)

    Outlet_Identifier_Pivot.plot(kind='bar')

    plt.xlabel(x)

    plt.ylabel('Item_Outlet_Sales')

    plt.show()
relations('Outlet_Identifier')
train_df.pivot_table(values='Outlet_Type', columns='Outlet_Identifier', aggfunc=lambda x:x.mode())
train_df.pivot_table(values='Outlet_Type', columns='Outlet_Size',aggfunc=lambda x:x.mode())
relations('Outlet_Establishment_Year')
relations('Outlet_Location_Type')
train_df.pivot_table(values='Outlet_Type', columns='Outlet_Location_Type', aggfunc=lambda x:x.mode())
relations('Outlet_Type')
relations('Outlet_Size')
relations('Item_Fat_Content')
relations('Item_Type')
sns.scatterplot(data=train_df, x='Item_Weight', y='Item_Outlet_Sales', alpha=0.3)
sns.scatterplot(data=train_df, x='Item_Visibility', y='Item_Outlet_Sales', alpha=0.3)
train_df['source']='train'

test_df['source']='test'

data = pd.concat([train_df,test_df], ignore_index = True)

print(train_df.shape, test_df.shape, data.shape)
data.info()
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('reg', 'Regular')
data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
for i in range(len(data)):

    if pd.isna(data.loc[i,'Outlet_Size']):

        if (data.loc[i,'Outlet_Type']=='Grocery Store') or (data.loc[i,'Outlet_Type']=='Supermarket Type1') :

            data.loc[i, 'Outlet_Size'] = 'Small'

        elif (data.loc[i,'Outlet_Type']=='Supermarket Type2') or (data.loc[i,'Outlet_Type']=='Supermarket Type3') :

            data.loc[i, 'Outlet_Size'] = 'Medium'
data['Item_Type_Category'] = data['Item_Identifier'].apply(lambda x: x[0:2])

data['Item_Type_Category'] = data['Item_Type_Category'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})

data['Item_Type_Category'].value_counts()
data.loc[data['Item_Type_Category']=='Non-Consumable','Item_Fat_Content'] = "Non-Edible"
Item_Type_Mean = data.pivot_table(columns='Item_Type', values='Item_Weight', aggfunc=lambda x:x.mean())

Item_Type_Mean
for i in range(len(data)):

    if pd.isna(data.loc[i, 'Item_Weight']):

        item = data.loc[i, 'Item_Type']

        data.at[i, 'Item_Weight'] = Item_Type_Mean[item]
Item_Visibility_Mean = data[['Item_Type_Category', 'Item_Visibility']].groupby(['Item_Type_Category'], as_index=False).mean()

Item_Visibility_Mean.columns
for i in range(len(data)):

    if data.loc[i, 'Item_Visibility']==0:

        cat =  data.loc[i, 'Item_Type_Category']

        m = Item_Visibility_Mean.loc[Item_Visibility_Mean['Item_Type_Category'] == cat]['Item_Visibility']

        data.at[i, 'Item_Visibility'] = m
data['Operation_Years'] = 2013-data['Outlet_Establishment_Year']
data=data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1)
lb=LabelEncoder()

data['Outlet']=lb.fit_transform(data['Outlet_Identifier'])

var=['Item_Fat_Content','Outlet_Location_Type','Outlet_Type','Outlet_Size','Item_Type_Category']

lb=LabelEncoder()

for item in var:

    data[item]=lb.fit_transform(data[item])

data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Type','Outlet_Size','Item_Type_Category'])
data.head(10)
train = data.loc[data['source']=='train']

test = data.loc[data['source']=='test']

train = train.drop(['source'], axis=1)

test = test.drop(['source'], axis=1)
train.columns, test.columns, train.shape, test.shape
data_temp = train.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)

x_train = train.drop(['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier'], axis=1)

y_train = train['Item_Outlet_Sales']

x_test = test.drop(['Item_Outlet_Sales','Item_Identifier', 'Outlet_Identifier'], axis=1)



x_train.shape, y_train.shape, x_test.shape
x_train.columns, x_test.columns
def model_prediction(algo):

    model = algo

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    acc= round(model.score(x_train,y_train) * 100,2)

    output = pd.DataFrame({'Item_Identifier':test['Item_Identifier'], 'Outlet_Identifier':test['Outlet_Identifier'], 'Item_Outlet_Sales':y_pred}, columns=['Item_Identifier','Outlet_Identifier', 'Item_Outlet_Sales'])

    return acc, output
model = LinearRegression()

linreg_acc, linreg_output = model_prediction(model)

linreg_acc
model = RandomForestRegressor(n_estimators=200, max_depth=5,min_samples_leaf=100, n_jobs=4)

forestreg_acc, forestreg_output = model_prediction(model)

forestreg_acc
train_set, test_set = train_test_split(data_temp, test_size = 0.2)

x_train_temp = train_set.drop(['Item_Outlet_Sales'], axis=1)

y_train_temp = train_set['Item_Outlet_Sales']

x_test_temp = test_set.drop(['Item_Outlet_Sales'], axis=1)

y_test_temp = test_set['Item_Outlet_Sales']



scaler = MinMaxScaler(feature_range=(0,1))

x_train_scaled = scaler.fit_transform(x_train_temp)

x_train_temp = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test_temp)

x_test_temp = pd.DataFrame(x_test_scaled)



x_train_temp.shape, y_train_temp.shape, x_test_temp.shape, y_test_temp.shape
rmse=[]

for k in range(15):

    k = k+1

    model = KNeighborsRegressor(n_neighbors = k)

    model.fit(x_train_temp, y_train_temp)

    y_pred_temp=model.predict(x_test_temp) 

    error = np.sqrt(mean_squared_error(y_test_temp,y_pred_temp))

    rmse.append(error)

    print('RMSE value for k= ' , k , 'is:', error)

    

curve = pd.DataFrame(rmse)

curve.plot()
model=KNeighborsRegressor(n_neighbors=11)

knr_acc, knr_output = model_prediction(model)

knr_acc
model = Ridge(alpha=0.05, normalize=True)

ridge_acc, ridge_output = model_prediction(model)

ridge_acc
model_acc_list = [['Linear Regression', linreg_acc], ['Random Forest Regressor', forestreg_acc], ['K Neighbors Regressor', knr_acc], ['Ridge Regression', ridge_acc]]

model_acc_df = pd.DataFrame(data=model_acc_list, columns=['Model', 'Accuracy'])

model_acc_df