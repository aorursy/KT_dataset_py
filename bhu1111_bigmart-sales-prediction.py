# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
### Importing Required Libraries
import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import metrics

from sklearn import model_selection



%matplotlib inline
train = pd.read_csv('/kaggle/input/bigmart-sales-data/Train.csv')

train.head()
train.info()
train.describe()  ## Univariate statistical analysis
# To check for null values

train.isnull().sum()
#to check wether outlet identifier should be used for modelling or not

train.Outlet_Identifier.nunique()

# As there are very few unique values we should consider it for modelling
#similarly for Item Identifier

train.Item_Identifier.nunique()

#It should be ignored
train.shape
#defining numerical columns and categorical columns

num_cols = ['Item_Weight','Item_Visibility','Item_MRP',]

cat_cols = ['Item_Type','Item_Fat_Content','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Establishment_Year']

target = 'Item_Outlet_Sales'
# 1:

train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].mean())
# 2 : Using Apply method

"""weights = {}

for key,df in train.groupby(['Item_Type','Item_Fat_Content']):

    

    weights[key] = df['Item_Weight'].mean()

    

print(weights)



train['Item_Weight'] = train.apply(lambda x:weights[(x['Item_Type'],x['Item_Fat_Content'])] 

                                   if np.isnan(x['Item_Weight']) else x['Item_Weight'] ,axis = 1)"""
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode().values[0])
train.isnull().sum()

# Now there are no null values
#finding the categories in categorical columns

def cat_cols_info(df,col):

    print("Unique categories in {}".format(col))

    print(df[col].unique())

    print("Distribution of categories: \n")

    print(df[col].value_counts())

    print('\n')
for col in cat_cols:

    cat_cols_info(train,col)
# observation #

# category Item_fat_content has values such as low fat, lf  or LOW FAT which means the same, hence replacing them #

"""

train.Item_Fat_Content.replace('Low Fat','LF',inplace = True)

train.Item_Fat_Content.replace('low fat','LF',inplace = True)

train.Item_Fat_Content.replace('reg','Regular',inplace = True)

"""
# 1) Numeric vs Numeric

sns.pairplot(train[num_cols+[target]])
## Categorical w.r.t target

for cat_col in cat_cols:

    fig = plt.figure(figsize=(15,4))

    ax = fig.add_subplot(1,1,1)

    j = 0

    for key,df in train.groupby([cat_col]):

        sns.kdeplot(df[target], label = key)

        ax.set_xlabel(target)

        ax.set_ylabel("Frequency")

        ax.legend(loc="best")

        ax.set_title('Frequency Distribution of {}'.format(col), fontsize = 10)

        j = j + 1
# Categorical univariate

fig = plt.figure(figsize = (15,50))



j = 1

for cat_col in cat_cols:

    ax = fig.add_subplot(len(cat_cols),1,j)

    sns.countplot(x = cat_col,

                  data = train,

                  ax = ax)

    ax.set_xlabel(cat_col)

    ax.set_ylabel("Frequency")

    ax.set_title('Frequency Distribution for individual classes in {}'.format(cat_col), fontsize = 10)

    j = j + 1
# categorical vs numerical

for cat_col in cat_cols:

    fig = plt.figure(figsize=(15,50))

    j = 1

    for num_col in num_cols:

        ax = fig.add_subplot(len(cat_cols),len(num_cols),j)

        sns.boxplot(y = num_col,

                   x = cat_col,

                   data = train,

                   ax = ax)

        ax.set_xlabel(cat_col)

        ax.set_ylabel(num_col)

        ax.set_title('Distribution of {} with respect to {}'.format(num_col,cat_col), fontsize = 10)

        j = j + 1
# Finding coefficient of correlation

train_corr = train[num_cols].corr()

train_corr.head()
# plotting coefficient of correlation

sns.heatmap(train_corr, annot= True, cmap='coolwarm_r')

plt.show()
def handle_outliers(df,var,target,tol):

    var_data = df[var].sort_values().values

    q25, q75 = np.percentile(var_data, 25), np.percentile(var_data, 75)

    

    print('Outliers handling for {}'.format(var))

    print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

    iqr = q75 - q25

    print('IQR {}'.format(iqr))

    

    cut_off = iqr * tol

    lower, upper = q25 - cut_off, q75 + cut_off

    print('Cut Off: {}'.format(cut_off))

    print('{} Lower: {}'.format(var,lower))

    print('{} Upper: {}'.format(var,upper))

    

    outliers = [x for x in var_data if x < lower or x > upper]



    print('Number of Outliers in feature {}: {}'.format(var,len(outliers)))



    print('{} outliers:{}'.format(var,outliers))



    print('----' * 25)

    print('\n')

    print('\n')

        

    return list(df[(df[var] > upper) | (df[var] < lower)].index)
outliers = []

for num_col in num_cols:

    outliers.extend(handle_outliers(train,num_col,target,1.5))

outliers = list(set(outliers))

print(len(outliers))
#dropping the outliers

train = train.drop(outliers)
train.shape
train = train[num_cols + cat_cols + [target]]
train.head()
train = pd.get_dummies(train,columns=cat_cols,drop_first=True)

train.head()
train_data,test_data = train_test_split(train, test_size = .2, random_state = 101)
X_train = train_data.iloc[:,:-1]

X_test = test_data.iloc[:,:-1]



y_train = train_data.iloc[:,-1]

y_test = test_data.iloc[:,-1]
X_train.shape, X_test.shape
y_train.shape, y_test.shape
sc = StandardScaler()

sc.fit(X_train[num_cols])

X_train[num_cols] = sc.transform(X_train[num_cols])
X_train[num_cols].head()
X_test[num_cols] = sc.transform(X_test[num_cols])
X_train.shape, X_test.shape
train_df = pd.concat([X_train,y_train],axis = 1)

train_df.to_csv('preprocessed_train.csv',index = False)



test_df = pd.concat([X_test,y_test], axis = 1)

test_df.to_csv('preprocessed_test.csv',index = False)
train = pd.read_csv('preprocessed_train.csv')

test = pd.read_csv('preprocessed_test.csv')
train.head()
X_train = train.drop(['Item_Outlet_Sales'], axis = 1).values

X_test = test.drop(['Item_Outlet_Sales'], axis = 1).values
y_train = train['Item_Outlet_Sales'].values

y_test = test['Item_Outlet_Sales'].values


model_dict = {"Linear Regression": linear_model.LinearRegression(),

            "SGDRegressor" : linear_model.SGDRegressor() }
for key,regressor in model_dict.items():

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)

    print('The evalution scores for: ',regressor.__class__.__name__, 'are:')

    mse = metrics.mean_squared_error(y_test,y_pred)

    rmse = mse ** 0.5

    mae = metrics.mean_absolute_error(y_test,y_pred)

    mdae = metrics.median_absolute_error(y_test,y_pred)

    print('MSE :', mse)

    print('RMSE: ', rmse)

    print('MAE: ', mae)

    print('MDAE: ', mdae)

    print('\n')
import os

from keras.utils import plot_model

from keras.models import Model, Sequential

from keras.layers import Input,Dense,Flatten,Dropout

from keras.layers.merge import concatenate

from keras.callbacks import Callback
model = Sequential()
model.add(Dense(200, input_shape=(X_train.shape[1],), kernel_initializer='glorot_normal', activation='relu'))

model.add(Dense(100, kernel_initializer='glorot_normal', activation='relu'))

model.add(Dense(50, kernel_initializer='glorot_normal', activation='relu'))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mse'])
print(model.summary())
X_train.shape
(200 * 43) + 200
history = model.fit(X_train,

                    y_train,

                    batch_size=200,

                    epochs=100,

                    verbose=1,

#                    callbacks=None,

                    validation_data=(X_test,y_test)).history
res_df = pd.DataFrame(history)

res_df.head()
# Plot training vs validation MAE

plt.plot(res_df['loss'],label="Training")

plt.plot(res_df['val_loss'],label="Validation")

plt.legend(loc='best')

plt.xlabel('Epochs')

plt.ylabel('MAE')

plt.title('Training vs Validation MAE')
# Plot training vs validation MSE

plt.plot(res_df['mse'],label="Training")

plt.plot(res_df['val_mse'],label="Validation")

plt.legend(loc='best')

plt.xlabel('Epochs')

plt.ylabel('SE')

plt.title('Training vs Validation MSE')