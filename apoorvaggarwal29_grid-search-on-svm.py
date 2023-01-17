import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
df=pd.read_csv('../input/fish-market/Fish.csv')
df.head()
df.groupby('Species').mean()
df.info()
sns.countplot(x='Species',data=df)
# Then you map to the grid
g = sns.PairGrid(df)
g.map(plt.scatter)
sns.pairplot(df,hue='Species',palette='rainbow')
sns.boxplot(x='Species',y='Weight',data=df,palette='rainbow')
sns.boxplot(x='Species',y='Height',data=df,palette='rainbow')

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#no missing value
from sklearn.preprocessing import StandardScaler
df.columns
scaler = StandardScaler()
scaler.fit(df[['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']])
scaled_data = scaler.transform(df[['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']])
scaled_df=pd.DataFrame(scaled_data,columns=['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width'])
scaled_df['Species']=df['Species']
scaled_df.head()
from sklearn.model_selection import train_test_split
X=scaled_df[['Weight', 'Length1', 'Length2', 'Length3', 'Height','Width']]
y=scaled_df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
