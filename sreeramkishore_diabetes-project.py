import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('../input/diabetes/diabetes.csv')

df.head()
countNoDisease = len(df[df.Outcome == 0])
countHaveDisease = len(df[df.Outcome == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.Outcome))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.Outcome))*100)))
pd.crosstab(df.Age,df.Outcome).plot(kind="bar",figsize=(15,6),color=['red','black' ])
plt.title('diabetes with respective to age')
plt.xlabel('outcome (0 = Has disease, 1 = no diabetes)')
plt.xticks(rotation=0)
plt.legend(["No diabetes", "Have diabetes"])
plt.ylabel('age')
plt.show()
X = df.drop('Outcome', axis=1)
X = StandardScaler().fit_transform(X)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
model = SVC()

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)
roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)
print('Score: {}'.format(roc_auc))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 8)  # n_neighbors means k
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)

print("{} NN Score: {:.2f}%".format(2, knn.score(X_test, y_test)*100))
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
predictions = cross_val_predict(model, X_test, y_test, cv=5)
# print(predictions)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))
np.around(score, decimals=4)
