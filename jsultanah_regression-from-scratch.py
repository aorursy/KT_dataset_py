# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
def transform_data(data):

    '''

    Input:

        data - Pandas DataFrame. House prices train data without the target column (SalePrice)

        

    Output:

        data - Pandas DataFrame. Standardised numeric features, with imputed missing values

    

    Transforms the train data into a more suitable format using the 

    

    1) Remove the Id column

    2) Extract only numeric features

    3) Normalize the columns based on the train sets mean and standard deviation

    4) Replace any NA values with 0 which will be the column mean after standardisation

    

    '''

    # only use numeric columns

    data = data.drop('Id',axis=1)._get_numeric_data()

    

    # normalize the numeric columns

    for col in data.columns:

        data[col] = (train[col] - train[col].mean())/train[col].std()

    

    # replace NA entries with 0

    for col in data.isna().sum()[data.isna().sum() > 0].index:

        data.loc[data[col].isna(),col] = 0

        

    return data
# Shuffle the train for train and validation

train = train.iloc[np.random.permutation(len(train))]



# Create a copy of the train set and drop the SalePrice before transforming

data = train.copy()

data = data.drop('SalePrice',axis=1)

data = transform_data(data)
def get_X_and_y(data,validation=0.2):

    '''

    Input:

        data - numeric data with input features and target value 'SalePrice'

    Output:

        X - numpy array of input features for model of dimension (m,n+1) where n is the number of features and m the number of training examples. Column of 1s is prepended for a bias term

        y - numpy array of target values, 'SalePrice' of shape (m,1) 

    '''

    

    # Transform target values into a numpy array

    y = np.array(train['SalePrice'])

    y = y.reshape(y.shape[0],1)

    # Transform input features to numpy array, X. Add a column of ones as the first feature. This will allow for a bias term in the regression formula

    X = np.array(data)

    X = np.concatenate((np.ones(X.shape[0]).reshape(X.shape[0],1),X),axis=1)

    

    val_length = int(0.2*len(y))

    y_train = y[val_length:]

    X_train = X[val_length:]

    y_val = y[:val_length]

    X_val = X[:val_length]

    

    

    return X_train,y_train,X_val,y_val
# for test data

def get_X(data):

    X = np.array(data)

    X = np.concatenate((np.ones(X.shape[0]).reshape(X.shape[0],1),X),axis=1)

    return X
X_train, y_train, X_val, y_val = get_X_and_y(data)
theta = np.random.randn(X_train.shape[1]).reshape(X_train.shape[1],1)

y_hat = np.dot(X_train,theta)

loss_list = []

loss_list_val = []

alpha = 10

for i in range(100000):

    for j in range(X_train.shape[1]):

        theta[j] = theta[j] - alpha*(np.dot(np.transpose(y_hat-y_train),X_train[:,j].reshape(X_train.shape[0],1)).item()/X_train.shape[0])

    y_hat = np.dot(X_train,theta)

    

    # calculate the new loss based on updated theta

    loss = np.sum(np.square(y_train-y_hat))/(2*y_train.shape[0])

    y_hat_val = np.dot(X_val,theta)

    val_loss = np.sum(np.square(y_val-y_hat_val))/(2*y_val.shape[0])

    

    # append the current loss to our record

    loss_list.append(loss)

    loss_list_val.append(val_loss)

    

    # print the loss after every 100 iterations

    if i % 100 ==0:

        print('iteration',i,'alpha',alpha,'loss:',loss_list[-1])

    

    # if the loss is diverging, i.e. the current loss is bigger than the previous loss, make alpha smaller

    if i> 0:

        if loss_list[-1]-loss_list[-2] > 0:

            alpha = alpha/10



    # stop the descent once the current loss minus the previous loss is less than some value

    if i> 0:

        if abs(loss_list[-1]-loss_list[-2]) < 0.1:

            break

    
print(loss_list[-1]**(1/2))
print(loss_list_val[-1]**(1/2))
fig, ax = plt.subplots()

ax.plot(loss_list[35:],label='train loss')

ax.plot(loss_list_val[35:], label='val loss')

ax.legend()
plt.figure(figsize=(10,10))

sns.barplot(x=np.arange(theta.shape[0]), y=theta.reshape(37))
theta = np.random.randn(X_train.shape[1]).reshape(X_train.shape[1],1)

y_hat = np.dot(X_train,theta)

loss_list = []

loss_list_val = []

lambd = 1

alpha = 10

for i in range(100000):

    for j in range(X_train.shape[1]):

        theta[j] = theta[j] - alpha*(np.dot(np.transpose(y_hat-y_train),X_train[:,j].reshape(X_train.shape[0],1)).item()/X_train.shape[0] + lambd*theta[j])

    y_hat = np.dot(X_train,theta)

    

    # calculate the new loss based on updated theta

    loss = np.sum(np.square(y_train-y_hat))/(2*y_train.shape[0])

    y_hat_val = np.dot(X_val,theta)

    val_loss = np.sum(np.square(y_val-y_hat_val))/(2*y_val.shape[0])

    

    # append the current loss to our record

    loss_list.append(loss)

    loss_list_val.append(val_loss)

    

    # print the loss after every 100 iterations

    if i % 100 ==0:

        print('iteration',i,'alpha',alpha,'loss:',loss_list[-1])

    

    # if the loss is diverging, i.e. the current loss is bigger than the previous loss, make alpha smaller

    if i> 0:

        if loss_list[-1]-loss_list[-2] > 0:

            alpha = alpha/10



    # stop the descent once the current loss minus the previous loss is less than some value

    if i> 0:

        if abs(loss_list[-1]-loss_list[-2]) < 0.01:

            break

    
print(loss_list[-1]**(1/2))
print(loss_list_val[-1]**(1/2))
fig, ax = plt.subplots()

ax.plot(loss_list[25:],label='train loss')

ax.plot(loss_list_val[25:], label='val loss')



ax.legend()
plt.figure(figsize=(10,10))

sns.barplot(x=['constant']+list(data.columns), y=theta.reshape(37))

plt.xticks(rotation=90)
sns.pairplot(data=train,

                  y_vars=['SalePrice'],

                  x_vars=data.columns[:10]

            )
sns.pairplot(data=train,

                  y_vars=['SalePrice'],

                  x_vars=data.columns[10:20]

            )
sns.pairplot(data=train,

                  y_vars=['SalePrice'],

                  x_vars=data.columns[20:30]

            )
sns.pairplot(data=train,

                  y_vars=['SalePrice'],

                  x_vars=data.columns[30:]

            )
test_orig = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



test = transform_data(test_orig)



X = get_X(test)



y = np.dot(X,theta)
y_df = pd.DataFrame(y).rename(columns={0:'SalePrice'})
output = test_orig.join(y_df)[['Id','SalePrice']]
pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
output.to_csv('/kaggle/working/my_submission.csv', index=False)

print("Your submission was successfully saved!")