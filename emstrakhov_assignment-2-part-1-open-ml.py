import pandas as pd
df = pd.read_csv('../input/cardiovascular-disease-dataset/cardio_train.csv', 

                 sep=';')

df.head()
df.info()
import numpy as np
df['age in years'] = np.floor(df['age'] / 365.25)

df.head()
df_1 = pd.get_dummies(df, columns=['gluc', 'cholesterol'])
df_2 = df_1.drop(['age', 'id'], axis=1)

df_2.head()
from sklearn.model_selection import train_test_split

X = df_2.drop('cardio', axis=1)

y = df_2['cardio']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                     test_size=0.3, 

                                                     random_state=17)
from sklearn.tree import DecisionTreeClassifier # импорт класса



tree = DecisionTreeClassifier(max_depth=3, random_state=17) # создали экземпляр класса

tree.fit(X_train, y_train) # обучили модель
# картинка

from sklearn.tree import export_graphviz

tree_dot = export_graphviz(tree)

print(tree_dot)
# Прогноз

y_pred = tree.predict(X_valid) # предсказания



# Точность прогнозов

from sklearn.metrics import accuracy_score

acc1 = accuracy_score(y_pred, y_valid)
from sklearn.model_selection import GridSearchCV, cross_val_score



tree_params = {'max_depth': range(2, 11)}



tree_grid = GridSearchCV(tree, tree_params,

                         cv=5, n_jobs=-1, verbose=True)



tree_grid.fit(X_train, y_train)
tree_grid.best_params_
best_tree = tree_grid.best_estimator_
pd.DataFrame(tree_grid.cv_results_).T
import matplotlib.pyplot as plt



df_cv = pd.DataFrame(tree_grid.cv_results_)



plt.plot(df_cv['param_max_depth'], df_cv['mean_test_score'])

plt.xlabel("max_depth")

plt.ylabel("accuracy");
y_best_pred = best_tree.predict(X_valid)

acc2 = accuracy_score(y_valid, y_best_pred)
(acc2-acc1)/acc1*100
df_2['age4050'] = ((df['age in years'] >= 40) & (df['age in years'] < 50)).astype(int)

df_2['age5055'] = ((df['age in years'] >= 50) & (df['age in years'] < 55)).astype(int)

df_2['age5560'] = ((df['age in years'] >= 55) & (df['age in years'] < 60)).astype(int)

df_2['age6065'] = ((df['age in years'] >= 60) & (df['age in years'] < 65)).astype(int)



df_2['aphi120140'] = ((df['ap_hi'] >= 120) & (df['ap_hi'] < 140)).astype(int)

df_2['aphi140160'] = ((df['ap_hi'] >= 140) & (df['ap_hi'] < 160)).astype(int)

df_2['aphi160180'] = ((df['ap_hi'] >= 160) & (df['ap_hi'] < 180)).astype(int)
df_2.head()
df_2['gender'] = df_2['gender'] - 1
new_df = df_2[['gender', 'smoke', 'age4050', 'age5055', 'age5560', 'age6065',

               'aphi120140', 'aphi140160', 'aphi160180', 

               'cholesterol_1', 'cholesterol_2', 'cholesterol_3']]

new_df.head()
new_df.shape
X = new_df

y = df_2['cardio']

new_tree = DecisionTreeClassifier(max_depth=3, random_state=17)

new_tree.fit(X, y)
# картинка

from sklearn.tree import export_graphviz

tree_dot = export_graphviz(new_tree)

print(tree_dot)