import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
customer_data = pd.read_csv("../input/mall-customers/Mall_Customers.csv")
customer_data.shape
customer_data.head()
customer_data.columns = ["ID", "gender", "age", "income", "spending_score"]
customer_data.head(3)
customer_data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
customer_data["gender"] = encoder.fit_transform(customer_data["gender"])
customer_data.info()
customer_data.head(2)
plt.figure(figsize=(12, 7))
corr_mat = customer_data.corr()
sns.heatmap(corr_mat)
sns.scatterplot(customer_data["ID"], customer_data["income"])
customer_data["spending_score"].describe()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
sns.distplot(customer_data["spending_score"], bins=50, ax=ax[0])
sns.boxplot(customer_data["spending_score"], ax=ax[1])
customer_data["gender"].unique()
sns.countplot(customer_data["gender"])
customer_data["gender"].value_counts(normalize=True)
customer_data["age"].unique()
plt.figure(figsize=(20, 7))
sns.countplot(customer_data["age"])
customer_data["income"].describe()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
sns.distplot(customer_data["income"], bins=50, ax=ax[0])
sns.boxplot(customer_data["income"], ax=ax[1])
plt.figure(figsize=(12, 7))
corr_mat = customer_data.corr()
sns.heatmap(corr_mat, annot=True)
customer_data.groupby("gender")["spending_score"].describe()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
sns.boxplot(customer_data["gender"], customer_data["spending_score"], ax=ax[0])
sns.kdeplot(customer_data[customer_data["gender"] == 0]["spending_score"], color='r',ax=ax[1])
sns.kdeplot(customer_data[customer_data["gender"] == 1]["spending_score"], color='g',ax=ax[1])
plt.figure(figsize=(20, 7))
sns.boxplot(customer_data["age"], customer_data["spending_score"])
plt.figure(figsize=(20, 7))
sns.regplot(customer_data["age"], customer_data["spending_score"])
plt.figure(figsize=(20, 7))
sns.boxplot(customer_data["income"], customer_data["spending_score"])
plt.figure(figsize=(20, 7))
sns.jointplot(customer_data["income"], customer_data["spending_score"], kind="hex")
plt.figure(figsize=(20, 7))
sns.scatterplot(customer_data["income"], customer_data["spending_score"], hue=customer_data["age"])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

data = customer_data.copy()
data["age"] = scaler.fit_transform(data["age"].values.reshape(-1, 1))
data["income"] = scaler.fit_transform(data["income"].values.reshape(-1, 1))
data["spending_score"] = scaler.fit_transform(data["spending_score"].values.reshape(-1, 1))
data.head()

X = data.drop(columns=["ID", "spending_score"])
y = data["spending_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_pred = linear_reg.predict(X_test)
np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(linear_reg_pred)))
plt.scatter(range(0, len(y_test)), y_test, color='r')
plt.scatter(range(0, len(y_test)), linear_reg_pred, color='b')
X = data["income"]
y = data["spending_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
linear_reg = LinearRegression()
linear_reg.fit(np.array(X_train).reshape(-1, 1), y_train)
linear_reg_pred = linear_reg.predict(np.array(X_test).reshape(-1, 1))
np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(linear_reg_pred)))
plt.scatter(range(0, len(y_test)), y_test, color='r')
plt.scatter(range(0, len(y_test)), linear_reg_pred, color='b')
X = data["age"]
y = data["spending_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
linear_reg = LinearRegression()
linear_reg.fit(np.array(X_train).reshape(-1, 1), y_train)
linear_reg_pred = linear_reg.predict(np.array(X_test).reshape(-1, 1))
np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(linear_reg_pred)))
plt.scatter(range(0, len(y_test)), y_test, color='r')
plt.scatter(range(0, len(y_test)), linear_reg_pred, color='b')
X = data["gender"]
y = data["spending_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
linear_reg = LinearRegression()
linear_reg.fit(np.array(X_train).reshape(-1, 1), y_train)
linear_reg_pred = linear_reg.predict(np.array(X_test).reshape(-1, 1))
np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(linear_reg_pred)))
plt.scatter(range(0, len(y_test)), y_test, color='r')
plt.scatter(range(0, len(y_test)), linear_reg_pred, color='b')
from sklearn.svm import SVR
X = data.drop(columns=["ID", "spending_score"])
y = data["spending_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

svr = SVR(C=1)
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)
np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(svr_pred)))
plt.scatter(range(0, len(y_test)), y_test, color='r')
plt.scatter(range(0, len(y_test)), svr_pred, color='b')

from sklearn.model_selection import learning_curve

train_size, train_score, test_score = \
        learning_curve(svr, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, shuffle=True)
train_mean = np.mean(train_score, axis=1)
train_std = np.std(train_score, axis=1)
test_mean = np.mean(test_score, axis=1)
test_std = np.std(test_score, axis=1)
plt.plot(train_size, train_mean, color='blue', marker='o', markersize=5)
plt.fill_between(train_size, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
plt.plot(train_size, test_mean, color="red", marker='s')
plt.fill_between(train_size, test_mean+test_std, test_mean-test_std, alpha=0.15, color='red')


