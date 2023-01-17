import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
training_mush = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
training_mush.head(3)
unique_data = training_mush.unstack().unique()
random.shuffle(unique_data)
stack = {x: y for x, y in zip(unique_data, range(1, len(unique_data) + 1))}
pd.DataFrame([pd.Series(stack)]) 
training_mush = training_mush.replace(stack)
training_mush.head(3)
X, y = training_mush.iloc[:, 1:], training_mush.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
params = {'n_estimators': range(10, 70, 10), 'max_depth': range(1, 12, 2), 'min_samples_leaf':  range(1, 7), 
         'min_samples_split': range(2, 9, 2)}
lenin = GridSearchCV(rf, params, cv=3, n_jobs=-1).fit(X_train, y_train)
pd.DataFrame(pd.Series(lenin.best_params_), columns=['best_params']).sort_values('best_params', ascending=False)
lenin = lenin.best_estimator_
pd.DataFrame(lenin.feature_importances_, index=X.columns, columns=['feature_importances']).\
    sort_values('feature_importances', ascending=True).\
    plot.barh(stacked=True, color='r', figsize=(10,7));
m = confusion_matrix(y_test, lenin.predict(X_test))
labels =  np.array([[f'True_negatives = {m[0][0]}',f'False_positives = {m[0][1]}'],\
                    [f'False_negatives = {m[1][0]}', f'True_positives = {m[1][1]}']])
sns.heatmap(m, annot=labels, fmt = '');
lenin.score(X_test, y_test)