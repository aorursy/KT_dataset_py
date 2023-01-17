import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("../input/mobile-price-classification/train.csv")
df.head()
df.info()
df.isnull().sum()
df.describe()
sns.countplot(df['price_range'])
sns.scatterplot(x=df['price_range'],y=df['n_cores'],data=df)
sns.scatterplot(x=df['price_range'],y=df['clock_speed'],data=df)
sns.scatterplot(x=df['n_cores'],y=df['clock_speed'],data=df)
sns.jointplot(x=df['price_range'],y=df['battery_power'])
sns.countplot(df['four_g'])
sns.countplot(df['touch_screen'])
sns.countplot(df['dual_sim'])
sns.scatterplot(x=df['mobile_wt'],y=df['battery_power'],hue=df['price_range'],data=df)
fig, ax = plt.subplots(figsize=(19,21))
sns.heatmap(df.corr(),annot=True,ax=ax)
sns.pairplot(df)
from sklearn.model_selection import train_test_split
x=df.drop('price_range',axis=1)
y=df['price_range']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
predictions=knn.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid,verbose=3, scoring='accuracy')
grid.fit(x_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions=grid.predict(x_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
