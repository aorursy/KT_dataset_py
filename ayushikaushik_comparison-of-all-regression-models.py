#display image using python

from IPython.display import Image

url = 'https://img.etimg.com/thumb/msid-71806721,width-650,imgsize-807917,,resizemode-4,quality-100/avocados.jpg'

Image(url,height=300,width=400)
#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')

#importing the dataset

data = pd.read_csv('../input/avocado-prices/avocado.csv',index_col=0)

# Check the data

data.info()
data.head(3)
sns.distplot(data['AveragePrice']);
sns.countplot(x='year',data=data,hue='type');
data.year.value_counts()
sns.boxplot(y="type", x="AveragePrice", data=data);
data.year=data.year.apply(str)

sns.boxenplot(x="year", y="AveragePrice", data=data);
data['type']= data['type'].map({'conventional':0,'organic':1})



# Extracting month from date column.

data.Date = data.Date.apply(pd.to_datetime)

data['Month']=data['Date'].apply(lambda x:x.month)

data.drop('Date',axis=1,inplace=True)

data.Month = data.Month.map({1:'JAN',2:'FEB',3:'MARCH',4:'APRIL',5:'MAY',6:'JUNE',7:'JULY',8:'AUG',9:'SEPT',10:'OCT',11:'NOV',12:'DEC'})
plt.figure(figsize=(9,5))

sns.countplot(data['Month'])

plt.title('Monthwise Distribution of Sales',fontdict={'fontsize':25});
# Creating dummy variables

dummies = pd.get_dummies(data[['year','region','Month']],drop_first=True)

df_dummies = pd.concat([data[['Total Volume', '4046', '4225', '4770', 'Total Bags',

       'Small Bags', 'Large Bags', 'XLarge Bags', 'type']],dummies],axis=1)

target = data['AveragePrice']



# Splitting data into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_dummies,target,test_size=0.30)



# Standardizing the data

cols_to_std = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags','Large Bags', 'XLarge Bags']

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X_train[cols_to_std])

X_train[cols_to_std] = scaler.transform(X_train[cols_to_std])

X_test[cols_to_std] = scaler.transform(X_test[cols_to_std])
#importing ML models from scikit-learn

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
#to save time all models can be applied once using for loop

regressors = {

    'Linear Regression' : LinearRegression(),

    'Decision Tree' : DecisionTreeRegressor(),

    'Random Forest' : RandomForestRegressor(),

    'Support Vector Machines' : SVR(gamma=1),

    'K-nearest Neighbors' : KNeighborsRegressor(n_neighbors=1),

    'XGBoost' : XGBRegressor()

}

results=pd.DataFrame(columns=['MAE','MSE','R2-score'])

for method,func in regressors.items():

    model = func.fit(X_train,y_train)

    pred = model.predict(X_test)

    results.loc[method]= [np.round(mean_absolute_error(y_test,pred),3),

                          np.round(mean_squared_error(y_test,pred),3),

                          np.round(r2_score(y_test,pred),3)

                         ]
# Splitting train set into training and validation sets.

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.20)



#importing tensorflow libraries

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping



#creating model

model = Sequential()

model.add(Dense(76,activation='relu',kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),

    bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)))

model.add(Dense(200,activation='relu',kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),

    bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)))

model.add(Dropout(0.5))

model.add(Dense(200,activation='relu',kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),

    bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)))

model.add(Dropout(0.5))

model.add(Dense(200,activation='relu',kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),

    bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)))

model.add(Dropout(0.5))

model.add(Dense(1))



model.compile(optimizer='Adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
model.fit(x=X_train.values,y=y_train.values,

          validation_data=(X_val.values,y_val.values),

          batch_size=100,epochs=150,callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot();
dnn_pred = model.predict(X_test)
results.loc['Deep Neural Network']=[mean_absolute_error(y_test,dnn_pred).round(3),mean_squared_error(y_test,dnn_pred).round(3),

                                    r2_score(y_test,dnn_pred).round(3)]

results
f"10% of mean of target variable is {np.round(0.1 * data.AveragePrice.mean(),3)}"
results.sort_values('R2-score',ascending=False).style.background_gradient(cmap='Greens',subset=['R2-score'])