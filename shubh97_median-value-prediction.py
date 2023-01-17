import pandas as pd

import numpy as np

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

import torch
data_path = '../input/california-housing-prices/housing.csv'
!head -10 $data_path
housing_data = pd.read_csv(data_path)
housing_df_copy = housing_data.copy()



housing_df_copy['ocean_proximity'] = housing_df_copy.ocean_proximity.astype('category')

housing_df_copy['ocean_proximity'] = pd.get_dummies(housing_df_copy.ocean_proximity)

housing_df_copy.head(5)
housing_df_copy.info()
housing_df_copy.describe()
def eda(dataframe):

    assert type(dataframe) == pd.core.frame.DataFrame, 'Invalid Input Passed'

    

    column_headers = dataframe.columns

    number_rows = len(dataframe)

    

    print('{: >20} {: >10} {: >10} {: >10} {: >70}'.format('Column Name', 'DataType', 

                                                           '# Null', '# unique', 

                                                           'Top 5 unique'), end='\n\n')

    for column in column_headers:

        datatype = dataframe[column].dtype

        null_values = sum(dataframe[column].isnull())

        unique_values_count = dataframe[column].nunique()

        unique_values = dataframe[column].unique()[:5]

        

        

        print('{: >20} {: >10} {: >10} {: >10} {: >70}'.format(column, str(datatype), 

                                                               null_values, unique_values_count, 

                                                               str(unique_values)))



    fig = plt.gcf()

    fig.set_size_inches(12, 8)

    

    sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm')
eda(housing_df_copy)
type(housing_df_copy)
# Since total_bedrooms has missing values and its highly correlated 

# with other features but very less correlated with target value, hence dropping it.



housing_df_copy.drop('total_bedrooms', axis=1, inplace=True)



# Dropping Highly Correlated Features



housing_df_copy.drop(['latitude', 'total_rooms', 'households'], axis=1)
y = housing_df_copy.pop('median_house_value')

X = housing_df_copy
X.head(5)
X = torch.from_numpy(X.values)

y = torch.from_numpy(y.values)
learning_rate = 0.00002



X -= X.min(1, keepdim=True)[0]

X /= X.max(1, keepdim=True)[0]



y -= y.min(-1, keepdim=True)[0]

y /= y.max(-1, keepdim=True)[0]



W = torch.randn(X.size()[1], requires_grad=True)

b = torch.randn(1, requires_grad=True)



loss_list = []



weights = []



for epoch in range(1, 10001):

    weights.append(W)

    y_hat = W.float() @ X.t().float() + b.float()

    loss = torch.sum((y_hat - y)**2)

    loss_list.append(loss)

    loss.backward()



    with torch.no_grad():

        W -= learning_rate * W.grad

        b -= learning_rate * b.grad



    W.grad.zero_()

    b.grad.zero_()

    

    if epoch % 1000 == 0:

        print(f'Running {epoch} epoch')

        print('loss', loss.item())
plt.plot(loss_list)