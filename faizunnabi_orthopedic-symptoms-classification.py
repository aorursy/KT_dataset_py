import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/column_2C_weka.csv')
df.head()
df.info()
df.describe()
df['class']=df['class'].apply(lambda x:1 if x=='Normal' else 0)
df.head()
sns.set_style('whitegrid')
sns.countplot(df['class'])
plt.figure(figsize=(14,6))
sns.heatmap(df.isnull(),cmap='viridis',yticklabels=False,cbar=False)
plt.figure(figsize=(14,6))
sns.heatmap(df.corr(),annot=True)
sns.lmplot('sacral_slope','pelvic_incidence',df,hue='class')
sns.pairplot(df,hue='class')
from sklearn.model_selection import train_test_split
X=df.drop('class',axis=1)
y=df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression()
lgr.fit(X_train,y_train)
predictions=lgr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rf_pred=rfc.predict(X_test)
print(classification_report(y_test,rf_pred))
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df.drop('class',axis=1))
scaled_features=scaler.transform(df.drop('class',axis=1))
df_scaled = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_scaled.head()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
print(classification_report(y_test,knn_pred))
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn_25=KNeighborsClassifier(n_neighbors=25)
knn_25.fit(X_train,y_train)
knn25_pred=knn_25.predict(X_test)
print(classification_report(y_test,knn25_pred))
knn_5=KNeighborsClassifier(n_neighbors=15)
knn_5.fit(X_train,y_train)
knn5_pred=knn_5.predict(X_test)
print(classification_report(y_test,knn5_pred))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))



