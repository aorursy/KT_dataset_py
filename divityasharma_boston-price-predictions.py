 # Start by importing some libraries:

import pandas as pd





# Visualization



import matplotlib.pyplot as plt

import seaborn as sns



#To plot the graph embedded in the notebook

%matplotlib inline

# from sklearn library:



from sklearn import datasets

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error
# Loading data:

from sklearn.datasets import load_boston

boston=load_boston()

boston
dir(boston)
print(boston.data.shape)
df_x=pd.DataFrame(boston.data,columns=boston.feature_names)
df_x.head()
df_y=pd.DataFrame(boston.target)
df_y.head()
df_x.isnull().sum()
df_y.isnull().sum()
df_x.describe()
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3))

plt.hist(boston.target)

plt.xlabel('price ($1000s)')

plt.ylabel('count')

plt.tight_layout()
for index, feature_name in enumerate(boston.feature_names):

    plt.figure(figsize=(4, 3))

    plt.scatter(boston.data[:, index], boston.target)

    plt.ylabel('Price', size=15)

    plt.xlabel(feature_name, size=15)

    plt.tight_layout()
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.3,random_state=0)

reg.fit(x_train,y_train)
reg=LinearRegression()

reg.fit(x_train, y_train)




y_predicted=reg.predict(x_test)
y_predicted
# Plot a graph between Y_predicted and y_test values:
plt.figure(figsize=(4, 3))

plt.scatter(y_test, y_predicted)

plt.plot([0, 50], [0, 50])

plt.axis('tight')

plt.xlabel('True price ($1000s)')

plt.ylabel('Predicted price ($1000s)')

plt.tight_layout()
y_predicted[0]
y_test[0]
# for Mean Square Error we have to import numpy Library



import numpy as np



np.mean((y_predicted-y_test)**2)