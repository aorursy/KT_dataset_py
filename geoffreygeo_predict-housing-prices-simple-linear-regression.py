# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
np.set_printoptions(threshold=np.nan)

#Importing DataSet 
dataset = pd.read_csv("../input/kc_house_data.csv")
space=dataset['sqft_living']
price=dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

#Splitting the data into Train and Test
from sklearn.cross_validation import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)
df_train,df_test = train_test_split(dataset,test_size=1/3)

df_train = df_train.drop(['date','id'],axis=1)
df_test = df_test.drop(['date','id'],axis=1)
df_test_label = df_test['price']
df_train.head()
sns.lmplot(x="sqft_living", y="price",hue='condition' ,data=df_train);
def plot_corr(df):
    corr=df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plot_corr(df_train)
dataset.describe()
feature = list(df_train.columns.values)
feature.remove('price')
print(feature)
#model 
#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_predict
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting the prices
pred = regressor.predict(xtest)
prediction_results= cross_val_predict(regressor,xtest,pred)
#plotiing the cross_val_predict 
fig, ax = plt.subplots()
ax.scatter(ytest, prediction_results)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
plt.title("Linear Regression Model of Boston Housing")
plt.scatter(xtest, ytest,  color='black')
plt.plot(xtest, pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
regressor.summary()
from sklearn.metrics import mean_squared_error
from sklearn  import metrics
from math import sqrt

rmse = sqrt(mean_squared_error(ytest,pred))
print("The linear regression score is {}".format(regressor.score(xtrain,ytrain)))
print("The linear regression score is {}".format(regressor.score(xtest,ytest)))
print("The RMSE is {}".format(rmse))
print("The RMSE of the training set is {}".format(np.sqrt(metrics.mean_squared_error(ytrain,xtrain))))
print("The MAE is {}".format(metrics.mean_absolute_error(ytest,pred)))
print("The MSE is {}".format(metrics.mean_squared_error(ytest,pred)))
df_train.price.head()
#using tensorflow 

#creating input fn 
def make_input_fn(df,epochs):
    return tf.estimator.inputs.pandas_input_fn(
        x=df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']],
        y=df['price'],
        batch_size=128,
        num_epochs = epochs,
        shuffle= True,
        queue_capacity =1000,
        num_threads = 1
    )

def make_feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in feature]
  return input_columns
#tensorflow model 

#model.train(input_fn = make_input_fn(df_train, num_epochs = 10))

model = tf.estimator.LinearRegressor(
        feature_columns = make_feature_cols(),
        )
model.train(input_fn= make_input_fn(df_train,10))
def print_rmse(model, name, df):
  metrics = model.evaluate(input_fn = make_input_fn(df, 1))
  print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
print_rmse(model, 'validation', df_train)

preds_iter = model.predict(input_fn = make_input_fn(df_train, 1))
predicted_values = model.predict(input_fn = make_input_fn(df_train, 1))
pred_values = list()
for i in predicted_values:
    pred_values.append(i['predictions'][0])
    #print(i['predictions'][0])
test_label = df_test_label.tolist()
print(pred_values[2])
print(test_label[2])
#visualiztion of the result 

#Visualizing the training Test Results 
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
