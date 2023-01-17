from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://akm-img-a-in.tosshub.com/sites/btmt/images/stories/zomato-fact-sheet_505_052417055850_111517063712.jpg?size=1200:675")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')
df.head()
df.info()
pd.DataFrame(round(df.isnull().sum()/df.shape[0] * 100,3), columns = ['Missing'])
df.drop(['url', 'address', 'phone'], axis=1, inplace = True)
df.rename(columns = {"approx_cost(for two people)" : "cost_two", "listed_in(type)" : "service_type", "listed_in(city)" : "serve_to"}, inplace = True)
df.info()
df.columns
# Converting the cost_two variaible into integer

df.cost_two = df.cost_two.astype(str)

df.cost_two = df.cost_two.apply(lambda x : x.replace(',','')).astype(float)
df.rate.unique()
df['rate'] = df.rate.replace('NEW', np.NaN)

df['rate'] = df.rate.replace('-', np.NaN)

df.rate = df.rate.astype(str)
df.rate = df.rate.apply(lambda x : x.replace('/5','')).astype(float)

df.head()
plt.rcParams['figure.figsize'] = 14,7

sns.countplot(df['rate'], palette='Set1')

plt.title("Count plot of the rate variable")

plt.xticks(rotation = 90)

plt.show()
df.columns
plt.figure(figsize=(14,10))

sns.set_style("darkgrid")

sns.jointplot(x = 'rate', y = 'votes', data=df, color = 'darkgreen',height = 8, ratio = 4)
# Analyzing the number of locations with respect to the location



df.location.value_counts().nlargest(10).plot(kind='barh')

plt.title("Number of restaurants by location")

plt.xlabel("Restaurant counts")

plt.show()
df.columns
df.head()
# Plotting a pie chart for online orders



trace = go.Pie(labels = ['Online_orders', 'No_online_orders'], values = df['online_order'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['lightskyblue','gold'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of order variable')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
df.head()
# Restaurants to serve to



df.serve_to.value_counts().nlargest(10).plot(kind = 'barh', color = 'r')

plt.title("Number of restaurants listed in")

plt.xlabel("Count")

plt.legend()

plt.show()
sns.countplot(x = df['rate'], hue = df['online_order'], palette= 'Set1')

plt.title("Distribution of restaurant rating over online order facility")

plt.show()
df.rest_type.value_counts().nlargest(20).plot(kind = 'barh')

plt.title("Restaurant type")

plt.xlabel("Count")

plt.legend()

plt.show()
df.head()
df.dish_liked.value_counts().nlargest(20).plot(kind = 'barh')

plt.show()
df.head()
df.name.value_counts().nlargest(20).plot(kind = 'barh')

plt.legend()

plt.show()
df.head()
# Plotting a pie chart for online orders



trace = go.Pie(labels = ['Table_booking_available', 'No_table_booking_available'], values = df['book_table'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['lightskyblue','gold'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of order variable')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
plt.figure(figsize=(20,10))

sns.countplot(x = df['online_order'], hue = df['rate'], palette= 'Set1')

plt.title("Distribution of restaurant rating over table booking facility")

plt.show()
plt.rcParams['figure.figsize'] = 14,7

plt.subplot(1,2,1)



df.name.value_counts().head().plot(kind = 'barh', color = sns.color_palette("hls", 5))

plt.xlabel("Number Of Restaurants")

plt.title("Biggest Restaurant Chain (Top 5)")



plt.subplot(1,2,2)



df[df['rate'] >= 4.5]['name'].value_counts().nlargest(5).plot(kind = 'barh', color = sns.color_palette("Paired"))

plt.xlabel("Number Of Restaurants")

plt.title("Biggest Restaurant Chain (Top 5) - Rating more than 4.5")

plt.tight_layout()
# checking for null values

df.isnull().sum()
# Replacing the NaN values in rate feature



df['rate'] = df['rate'].fillna(df['rate'].mean())
# Plotting a distplot

sns.distplot(df['rate'], color = 'darkgreen')

plt.title('Rating Distribution')

plt.show()
# Replacing the NaN values for cost_two



df.cost_two.value_counts().mean()
# Replacing the NaN values for the cost_two feature with mean value



df['cost_two'] = df['cost_two'].fillna(df['cost_two'].mean())
# Plotting a distplot for cost_two feature

sns.distplot(df['cost_two'], color = 'darkgreen')

plt.title('Rating Distribution')

plt.show()
df.head()
df['online_order'] = pd.get_dummies(df['online_order'], drop_first=True)

df.head()
df['book_table'] = pd.get_dummies(df['book_table'], drop_first=True)

df.head()
# Performing One Hot Encoding on rest_type



get_dummies_rest_type = pd.get_dummies(df.rest_type)

get_dummies_rest_type.head(3)
# Performing One Hot Encoding on location



get_dummies_location = pd.get_dummies(df.location)

get_dummies_location.head(3)
# Performing One Hot Encoding on type



get_dummies_service_type = pd.get_dummies(df.service_type)

get_dummies_service_type.head(3)
# Concatinating the dataframes

final_df = pd.concat([df,get_dummies_rest_type,get_dummies_service_type, get_dummies_location], axis = 1)

final_df.head()
final_df.head(2)
final_df = final_df.drop(["name","rest_type","location", 'cuisines', 'dish_liked', 'reviews_list'],axis = 1)

final_df.head()
final_df.head()
final_df = final_df.drop(["menu_item","service_type","serve_to"],axis = 1)

final_df.head()
sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
# Splitting the features into independent and dependent variables



x = final_df.drop(['rate'], axis = 1)

x.head()
y = final_df['rate']
from sklearn.ensemble import ExtraTreesRegressor



model = ExtraTreesRegressor()

model.fit(x,y)
print(model.feature_importances_)
#plotting graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
sns.distplot(df['rate'])
#Spliting data into test and train



from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()



lr.fit(x_train, y_train)



lr_pred = lr.predict(x_test)
r2 = r2_score(y_test,lr_pred)

print('R-Square Score: ',r2*100)
# Calculate the absolute errors

lr_errors = abs(lr_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(lr_pred), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (lr_errors / y_test)

# Calculate and display accuracy

lr_accuracy = 100 - np.mean(mape)

print('Accuracy for Logistic Regression is :', round(lr_accuracy, 2), '%.')
sns.distplot(y_test-lr_pred)
#plotting the Random forest values predicated Rating



plt.figure(figsize=(12,7))



plt.scatter(y_test,x_test.iloc[:,2],color="blue")

plt.title("True rate vs Predicted rate",size=20,pad=15)

plt.xlabel('Rating',size = 15)

plt.ylabel('Frequency',size = 15)

plt.scatter(lr_pred,x_test.iloc[:,2],color="yellow")
from sklearn.metrics import mean_absolute_error,mean_squared_error
print('mse:',metrics.mean_squared_error(y_test, lr_pred))

print('mae:',metrics.mean_absolute_error(y_test, lr_pred))

from sklearn.tree import DecisionTreeRegressor



dtree = DecisionTreeRegressor(criterion='mse')

dtree.fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)
r2 = r2_score(y_test,dtree_pred)

print('R-Square Score: ',r2*100)



# Calculate the absolute errors

dtree_errors = abs(dtree_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(dtree_pred), 2), 'degrees.')



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (dtree_errors / y_test)

# Calculate and display accuracy

dtree_accuracy = 100 - np.mean(mape)

print('Accuracy for Decision tree regressor is :', round(dtree_accuracy, 2), '%.')
#plotting the Random forest values predicated Rating



plt.figure(figsize=(12,7))



plt.scatter(y_test,x_test.iloc[:,2],color="blue")

plt.title("True rate vs Predicted rate",size=20,pad=15)

plt.xlabel('Rating',size = 15)

plt.ylabel('Frequency',size = 15)

plt.scatter(dtree_pred,x_test.iloc[:,2],color="yellow")

plt.legend()
from sklearn.ensemble import RandomForestRegressor



random_forest_regressor = RandomForestRegressor()

random_forest_regressor.fit(x_train, y_train)
rf_pred = random_forest_regressor.predict(x_test)
r2 = r2_score(y_test,rf_pred)

print('R-Square Score: ',r2*100)



# Calculate the absolute errors

rf_errors = abs(rf_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(rf_pred), 2), 'degrees.')



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (rf_errors / y_test)

# Calculate and display accuracy

rf_accuracy = 100 - np.mean(mape)

print('Accuracy for random forest regressor is :', round(rf_accuracy, 2), '%.')
#plotting the Random forest values predicated Rating



plt.figure(figsize=(12,7))



plt.scatter(y_test,x_test.iloc[:,2],color="blue")

plt.title("True rate vs Predicted rate",size=20,pad=15)

plt.xlabel('Rating',size = 15)

plt.ylabel('Frequency',size = 15)

plt.scatter(rf_pred,x_test.iloc[:,2],color="yellow")
import pickle
# For Logistic Regression



# open a file where you want to store the data

file = open('logistic_regression.pkl', 'wb')



# dump information to that file

pickle.dump(lr, file)
# For Decision Tree Regressor



# open a file where you want to store the data

file = open('Decision_tree_model.pkl', 'wb')



# dump information to that file

pickle.dump(dtree, file)
# For Random Forest Regressor



# open a file where you want to store the data

file = open('Random_forest.pkl', 'wb')



# dump information to that file

pickle.dump(random_forest_regressor, file)