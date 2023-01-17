import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import mean_squared_error



sns.set_style({ 

    'axes.spines.bottom': False,

    'axes.spines.left': False,

    'axes.spines.right': False,

    'axes.spines.top': False,

    'xtick.bottom': False,

    'ytick.left': False

})
data = pd.read_csv('../input/car data.csv')

data.head()
data.isnull().sum()
data.info()
data.describe(include='all')
sns.pairplot(data)
fig,ax1 = plt.subplots(figsize=(15,10))

sns.scatterplot(x='Kms_Driven',y='Selling_Price',data=data,ax=ax1)
data['Y_S_L'] = 2019 - data.Year

data.head()
sns.catplot(x='Y_S_L',y='Selling_Price',data=data,kind='point',height=8,aspect=2)

sns.despine(left=True,bottom=True)
data.Transmission.value_counts()
data.loc[:,['Transmission','Selling_Price']].sort_values(by=['Selling_Price'],ascending =False)['Transmission'].head(15).value_counts().plot.pie(figsize=(15,15),subplots=True, autopct='%.1f%%',explode=[0,.08],shadow=True)
le = LabelEncoder()

df = pd.get_dummies(data['Fuel_Type'],prefix='FT',drop_first=True)

data['Seller_Type'] = le.fit_transform(data['Seller_Type'])

data['Transmission'] = le.fit_transform(data['Transmission'])

data = pd.concat([data,df],axis=1)

data.drop(['Fuel_Type'],axis=1,inplace=True)

data.head()
fig,ax1 = plt.subplots(figsize=(15,8))

sns.heatmap(data.corr(),annot=True,ax=ax1,cmap=sns.cm.vlag,cbar=False)
def gen_model(X,y):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)

    lr = LinearRegression()

    

    lr.fit(X_train,y_train)

    y_pred = lr.predict(X_test)

    

    coeffecients = pd.DataFrame(lr.coef_,X.columns)

    coeffecients.columns = ['Coeffecient']

    print(f' Coefficients : \n {coeffecients} \n')

    

    mse = mean_squared_error(y_test,y_pred)

    rmse = np.sqrt(mse)

    print(f'Mean Squared Error of Test Set : {mse}')

    print(f'Root Mean Square Error of Test Set : {rmse}')

    

    yt_pred = lr.predict(X_train)

    tmse = mean_squared_error(y_test,y_pred)

    trmse = np.sqrt(mse)

    print(f'Mean Squared Error of Train Set : {tmse}')

    print(f'Root Mean Square Error of Train Set : {trmse}')

    



    fig,ax1 = plt.subplots(figsize=(15,8))

    fig = sns.scatterplot(y_test,y_pred,ax=ax1)

    plt.xlabel('Y true')

    plt.ylabel('Y predicted')

    plt.title('True vs Predicted')

    plt.show(fig)

    

    fig,ax1 = plt.subplots(figsize=(15,8))

    fig = sns.distplot((y_test-y_pred),ax=ax1);

    plt.title('Residual Distrubution')

    plt.show(fig)
#train test_split

X = data.drop(['Car_Name','Selling_Price'],axis = 1)

y = data['Selling_Price']
gen_model(X,y)
#train test_split

X = data.drop(['Car_Name','Selling_Price','Kms_Driven','Seller_Type','Transmission',

               'Owner', 'Y_S_L','FT_Petrol'],axis = 1)

y = data['Selling_Price']
gen_model(X,y)