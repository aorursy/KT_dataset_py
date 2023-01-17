import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
human_df = pd.read_csv('../input/adult-income-dataset/adult.csv')

human_df.head().T
human_df.info()
human_df['income'].value_counts()
human_df['income'].value_counts().plot(kind='pie', ylabel='Percentage of people', title='Distribution of income values', autopct='%1.2f%%')
print((human_df["workclass"] == "?").value_counts()[1])

print((human_df["occupation"] == "?").value_counts()[1])

print((human_df["native-country"] == "?").value_counts()[1])
human_df = human_df[human_df["workclass"] != "?"]

human_df = human_df[human_df["occupation"] != "?"]

human_df = human_df[human_df["native-country"] != "?"]

                    

human_df.shape
human_df.replace(['Divorced', 'Married-AF-spouse', 

              'Married-civ-spouse', 'Married-spouse-absent', 

              'Never-married','Separated','Widowed'],

             ['Not Married','Married','Married','Married',

              'Not Married','Not Married','Not Married'], inplace = True)
category_features =['workclass', 'race', 'education','marital-status', 'occupation',

               'relationship', 'gender', 'native-country', 'income'] 



for feature in category_features:

    unique_value, index = np.unique(human_df[feature], return_inverse=True) 

    human_df[feature] = index



human_df.head().T
from sklearn.model_selection import train_test_split

X = human_df.drop(['income'], axis=1)

y = human_df['income']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25) 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_valid)



from sklearn.metrics import accuracy_score

print('Качество модели:', accuracy_score(y_valid, y_pred))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(knn, X_train, y_train, 

                         cv=kf, scoring='accuracy')

print(scores)

mean_score = scores.mean()

print(mean_score)
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51)}

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='accuracy',

                        cv=kf) 

knn_grid.fit(X_train, y_train)
knn_grid.best_params_
knn_grid.best_score_ 
mean_scores_array = knn_grid.cv_results_.get('mean_test_score')

params_arr = knn_grid.cv_results_.get('params')

params1_arr = []

for i in range(len(params_arr)):

    params1_arr.append(params_arr[i].get('n_neighbors'))

    

plt.plot(params1_arr, mean_scores_array)

plt.xlabel('Param Values')

plt.ylabel('Mean Scores')

plt.title('Graph of metric values depending on k')

plt.show()
kf1 = KFold(n_splits = 5, shuffle = True, random_state = 42)



max_score = 0

param_max_score_p = 0



for p in np.linspace(1, 10, 200):

    knn1 = KNeighborsClassifier(n_neighbors = knn_grid.best_params_.get('n_neighbors'), p = p, weights='distance', metric='minkowski')



    knn1.fit(X_train, y_train)

    y_pred1 = knn1.predict(X_valid)



    scores = cross_val_score(knn1, X_train, y_train, cv = kf1, scoring = 'accuracy')

    new_score = scores.mean()

    print("p:", p, "; Score:", new_score)

    if new_score > max_score:

        max_score = new_score

        param_max_score_p = p
print("Значение параметра p =",param_max_score_p,";", "значение качества =", max_score)
from sklearn.neighbors import RadiusNeighborsClassifier

radius_nn = RadiusNeighborsClassifier(radius=1.0)

radius_nn.set_params(outlier_label=2)

radius_nn.fit(X_train, y_train)

y_pred = radius_nn.predict(X_valid)



y_pred