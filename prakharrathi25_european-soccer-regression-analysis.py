import sqlite3

import pandas as pd 

from sklearn.tree import DecisionTreeRegressor 

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from math import sqrt 
cnx = sqlite3.connect('../input/database.sqlite')

data = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

data.shape
data.head()
data.describe()
data.columns
# Let's look for null values 

data.isnull().sum()

#data[data.isnull().any(axis = 1)]
data = data.dropna()

data.shape
features =list(data.columns[~(data.columns.str.contains('id'))])

features

# We also do not need overall_rating as that is our target and date 

features.remove('date')

features.remove('overall_rating')

features

data[features].head()
remove = ['preferred_foot', 'attacking_work_rate', 'defensive_work_rate']

for i in remove:

    features.remove(i)

features
# Let's loop through all the features 



for feature in features:

    co = data['overall_rating'].corr(data[feature])

    dict[feature] = co

dict
%matplotlib inline

import matplotlib.pyplot as plt

x_values = []

y_values = []

for value in dict:

    x_values.append(value)

    y_values.append(dict[value])



# Plotting the values using matplotlib.pyplot

plt.xlabel('Features in the data')

plt.ylabel('Correlation Coefficient with Overall Rating')

plt.title('Correlation of Overall Rating with different features')

plt.yticks([0, 1])



#Adjusting the size of the image 

from matplotlib.pyplot import figure

figure(num = None, figsize = (30, 6), dpi=80, facecolor='w', edgecolor='k')



plt.plot(x_values, y_values)

plt.show()

%matplotlib inline



#Plotting a subplot

fig, axis = plt.subplots(figsize = (40, 8))

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('Overall Player Rating',fontsize=10)

axis.set_xlabel('Player Features',fontsize=10)   

axis.set_ylabel('Correlation Values',fontsize=10)

axis.set_yticks([0,1])

axis.set_yticklabels(['0', '1'])



# # We can also use this to set figure size 

# f.set_figheight(15)

# f.set_figwidth(15)



axis.plot(x_values, y_values)

plt.show()
# Let's also specifiy the target 

target = ['overall_rating']
# Obtain the X and y values for regression analysis

X = data[features]

y = data[target]
# Let us look at a typical row from our features:

X.head()

# X.iloc[2]
y.head() 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 324)
X_train.describe()
X_test.describe()
regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_prediction = regressor.predict(X_test)

y_prediction
#Let's explore the predictions. 

y_prediction.mean()
y_test.describe().transpose()
RMSE = sqrt(mean_squared_error(y_true = y_test, y_pred = y_prediction))
print(RMSE)
from sklearn.metrics import r2_score, accuracy_score
print(r2_score(y_test, y_prediction))
#print(accuracy_score(y_test, y_prediction))
decision_regressor = DecisionTreeRegressor(max_depth = 50)

decision_regressor.fit(X_train, y_train)
y_prediction = decision_regressor.predict(X_test)

y_prediction.mean()
RMSE = sqrt(mean_squared_error(y_test, y_prediction))

print(RMSE)