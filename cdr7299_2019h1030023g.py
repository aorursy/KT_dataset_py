import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

pd.set_option('display.max_columns', 100)
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

df.head(20)
# df.info()

NAs = pd.concat([df.isnull().sum()], axis=1)

NAs[NAs.sum(axis=1) > 0]
df.fillna(value=df.mean(),inplace=True)
df.isnull().any()

df = pd.get_dummies(df, columns=["type"])
df.head()

# df.describe()
df.corr()
corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(14, 11))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
X = df[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11","type_new","type_old"]].copy()

y = df["rating"].copy()

X.head()

X.info()
X.head()
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.1,random_state=42)
# from sklearn.neighbors import KNeighborsClassifier

# Trying out KNN

from sklearn import metrics



# # knn = KNeighborsClassifier(n_neighbors = 20)

# knn = KNeighborsClassifier(n_neighbors = 50,leaf_size=5,algorithm="auto",n_jobs=-1,p =30, weights="distance",)



# knn.fit(X_train,y_train)

# y_pred = knn.predict(X_val)

#Trying out GridSearch to get best params for KNN

# from sklearn.model_selection import GridSearchCV

# model = KNeighborsClassifier(n_jobs=-1)

# #Hyper Parameters Set

# params = {'n_neighbors':[5,6,7,8,9,10],

#           'leaf_size':[1,2,3,5],

#           'weights':['uniform', 'distance'],

#           'algorithm':['auto', 'ball_tree','kd_tree','brute'],

#           'n_jobs':[-1]}

# #Making models with hyper parameters sets

# model1 = GridSearchCV(model, param_grid=params, n_jobs=1)

# #Learning

# model1.fit(X_train,y_train)

# #The best hyper parameters set

# print("Best Hyper Parameters:\n",model1.best_params_)
#Also checking for better accuracy in RandomForests

from sklearn.ensemble import RandomForestClassifier



clf2 = RandomForestClassifier(n_estimators = 100).fit(X_train,y_train)
#Finally settled on RandomClassifier

from sklearn.metrics import mean_absolute_error



y_pred_lr = clf2.predict(X_val)

print(metrics.accuracy_score(y_val,y_pred_lr))



mae_lr = mean_absolute_error(y_pred_lr,y_val)



# print(y_pred_lr)

# print("Mean Absolute Error of Linear Regression: {}".format(mae_lr))
from sklearn.metrics import mean_squared_error



from math import sqrt



rmse = sqrt(mean_squared_error(y_val, y_pred_lr))



print(rmse)
predict = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

predict = pd.get_dummies(predict, columns=["type"])
X_test_predict = predict[["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11","type_new","type_old"]].copy()
#Check the test data

X_test_predict.fillna(value=df.mean(),inplace=True)

# X_test_predict.isnull().any()
y_pred_lr_test = clf2.predict(X_test_predict)
# print(predict)

predict['rating'] = y_pred_lr_test
predict.head()
ans = predict[["id","rating"]].copy()

ans.to_csv('ans.csv',index=False,encoding ='utf-8' )