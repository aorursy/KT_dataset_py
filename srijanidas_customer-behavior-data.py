import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
customer_df = pd.read_csv("../input/customer-behaviour/Customer_Behaviour.csv")
customer_df.head(5)
customer_df.info()
customer_df["Male"] = pd.get_dummies(customer_df["Gender"])["Male"]
for col in ["Gender", "User ID"]:

  customer_df.drop(col, axis=1, inplace=True)
sns.pairplot(customer_df)
sns.scatterplot(customer_df["Age"], customer_df["EstimatedSalary"], hue=customer_df["Purchased"])
sns.scatterplot(customer_df["Age"], customer_df["EstimatedSalary"], hue=customer_df["Male"])
sns.countplot(customer_df["EstimatedSalary"], hue=customer_df["Purchased"])
sns.countplot(customer_df["Male"], hue=customer_df["Purchased"])
X = customer_df.drop("Purchased", axis=1).values

y = customer_df["Purchased"].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def evaluate_model_performance(y_test, y_pred):

  print(accuracy_score(y_test, y_pred))

  print(confusion_matrix(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier
error_rate = []



for i in range(1,40):

    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(X_train_scaled,y_train)

    pred_i = model.predict(X_test_scaled)

    error_rate.append(np.mean(pred_i != y_test))

    

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)



evaluate_model_performance(y_test, y_pred)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



evaluate_model_performance(y_test, y_pred)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



evaluate_model_performance(y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



evaluate_model_performance(y_test, y_pred)