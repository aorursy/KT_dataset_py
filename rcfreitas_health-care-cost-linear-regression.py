import pandas as pd
import matplotlib.pyplot as plt
#loading the dataset
data = pd.read_csv("../input/insurance.csv")

#basic infos
data.info()

#changing data types
for column in ['sex', 'smoker', 'region']:
    data[column] = data[column].astype('category')

#note the memory usage reduction from 73.2 kB to 46.2 kB
data.info()
#the numerical features
data.describe()
#the categorical features
data.describe(include='category', exclude='float')
from pandas.plotting import scatter_matrix

scatter_matrix(data[['charges','age','bmi', 'children']], alpha=0.3, diagonal='kde')
plt.figure(1)
plt.subplot(2,2,1)
data.groupby(['sex'])['charges'].sum().plot.bar()
plt.subplot(2,2,2)
data.groupby(['smoker'])['charges'].sum().plot.bar()
plt.subplot(2,2,3)
data.groupby(['region'])['charges'].sum().plot.bar()

plt.figure(2)
plt.subplot(2,2,1)
data.groupby(['sex'])['bmi'].sum().plot.bar()
plt.subplot(2,2,2)
data.groupby(['smoker'])['bmi'].sum().plot.bar()
plt.subplot(2,2,3)
data.groupby(['region'])['bmi'].sum().plot.bar()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

numerical = ['age','bmi', 'children']
categorical = ['sex', 'smoker', 'region']
X_train, X_test, y_train, y_test = train_test_split(data[numerical], 
                                                    data['charges'], 
                                                    test_size=0.2,
                                                   random_state=42)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print("The mean squared error is {:.2f}".format(mean_squared_error(y_test,y_pred)))
print("R2-score: {:.2f}".format(r2_score(y_test,y_pred)))
for feature in numerical:
    X_train, X_test, y_train, y_test = train_test_split(data[feature].values.reshape(-1,1),
                                                       data['charges'],
                                                       test_size=0.2,
                                                       random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    print("Feature: {}".format(feature))
    print("Mean squared error: {:.2f}".format(mean_squared_error(y_test,y_pred)))
    print("R2-score: {:.2f}".format(r2_score(y_test,y_pred)))
    plt.scatter(X_train,y_train, color='black')
    plt.plot(X_test,y_pred, color='blue')
    plt.ylabel('Charges')
    plt.xlabel(feature)
    plt.show()
X_train, X_test, y_train, y_test = train_test_split(data[numerical[0]].values.reshape(-1,1),
                                                       data['charges'],
                                                       test_size=0.2,
                                                       random_state=42)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print("Features: {}".format(numerical[0]))
print("Mean squared error: {:.2f}".format(mean_squared_error(y_test,y_pred)))
print("R2-score: {:.2f}".format(r2_score(y_test,y_pred)))
for i in range(2,4):
    X_train, X_test, y_train, y_test = train_test_split(data[numerical[0:i]],
                                                       data['charges'],
                                                       test_size=0.2,
                                                       random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    print("Features: {}".format(numerical[0:i]))
    print("Mean squared error: {:.2f}".format(mean_squared_error(y_test,y_pred)))
    print("R2-score: {:.2f}".format(r2_score(y_test,y_pred)))
data = pd.get_dummies(data)
data.head()
features = list(data.columns)
features.remove('charges')

r2_scores = []

X_train, X_test, y_train, y_test = train_test_split(data[features[0]].values.reshape(-1,1),
                                                       data['charges'],
                                                       test_size=0.2,
                                                       random_state=42)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print("Feature added: {}. Total features: {}".format(features[0],len(features[0:1])))
print("Mean squared error: {:.2f}".format(mean_squared_error(y_test,y_pred)))
print("R2-score: {:.2f}".format(r2_score(y_test,y_pred)))
r2_scores.append(r2_score(y_test,y_pred))
for i in range(2,11):
    X_train, X_test, y_train, y_test = train_test_split(data[features[0:i]],
                                                       data['charges'],
                                                       test_size=0.2,
                                                       random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    print("Feature addes: {}. Total features: {}".format(features[i],len(features[0:i])))
    print("Mean squared error: {:.2f}".format(mean_squared_error(y_test,y_pred)))
    print("R2-score: {:.2f}".format(r2_score(y_test,y_pred)))
    r2_scores.append(r2_score(y_test,y_pred))
plt.plot(list(range(0,10)),r2_scores)
plt.ylabel('R2 score')
plt.xlabel('features')
plt.show()
best = ['age','bmi','sex_male','sex_female','smoker_yes', 'smoker_no']
X_train, X_test, y_train, y_test = train_test_split(data[best],
                                                   data['charges'],
                                                   test_size=0.2,
                                                   random_state=42)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

print("Mean squared error: {:.2f}".format(mean_squared_error(y_test,y_pred)))
print("R2-score: {:.2f}".format(r2_score(y_test,y_pred)))