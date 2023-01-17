import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

fish_data = pd.read_csv("../input/fish-market/Fish.csv")
fish_data.head(2)
fish_data.info()
fish_data.groupby('Species').mean()
sns.scatterplot(fish_data['Weight'], fish_data['Height'], hue=fish_data['Species'])
X = fish_data.drop('Species', axis=1)
y = fish_data['Species']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def evaluate_model_performance(y_test, y_pred):
  print(accuracy_score(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))
  print(classification_report(y_test, y_pred))
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
from sklearn.neighbors import KNeighborsClassifier

# Calculating the K value for the best performance
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
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

evaluate_model_performance(y_test, y_pred)
comparing_result = pd.DataFrame(columns=['original species', 'predicted species'])
comparing_result['original species'] = y_test
comparing_result['predicted species'] = y_pred

for i in comparing_result.index:
    if comparing_result['original species'][i] != comparing_result['predicted species'][i]:
        print(comparing_result.loc[i])