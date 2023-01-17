# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



#import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library for and its numerical mathematics extension NumPy
import seaborn as sns #library for making statistical graphics 
import mpl_toolkits # 3D plotting 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
df.head()

#Data processing
df.isnull().sum()

#Data visualization
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(df['sqft_living'], bins=30)
plt.show()
#create a correlation matrix to measure linear relationships between variables
#using corr function and heatmap function from the seaborn library to plot the correlation matrix

correlation_matrix = df.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


#we can notice that price and bedroom have a high correlation with the target sqft_living
#based on that, we will consider those features as our X variables and using a scatter plot we'll check how them vary with sqft_living

plt.figure(figsize=(20, 5))

features = ['price', 'bedrooms']
target = df['sqft_living']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = df[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('sqft_living')
#preparing the data for training the model
X = pd.DataFrame(np.c_[df['price'], df['bedrooms']], columns = ['price','bedrooms'])
Y = df['sqft_living']

#preparing the data for training the model
X = pd.DataFrame(np.c_[df['price'], df['bedrooms']], columns = ['price','bedrooms'])
Y = df['sqft_living']


#Splitting the data into training and testing sets -- we use 80% of the samples to train the model and 20% to test it

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#Training and testing the model
#We use scikit-learn’s LinearRegression to train our model on both the training and test sets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


#Model evaluation -- will evaluate our model using RMSE and R2-score


y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set

y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))



# plotting the y_test vs y_pred
# ideally should have been a straight line
plt.scatter(Y_test, y_test_predict)
plt.show()




