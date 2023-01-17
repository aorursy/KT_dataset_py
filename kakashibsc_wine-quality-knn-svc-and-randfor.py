# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
wine_df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
wine_df.head()
wine_df['quality'].value_counts().plot(kind='pie', figsize = (10,10))
sns.countplot(data = wine_df, x='quality')
for j in wine_df.drop('quality', axis = 1):

    sns.distplot(wine_df[j], kde = False)

    plt.legend()

    plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(wine_df.drop('quality', axis = 1))
scaler_feat = scaler.transform(wine_df.drop('quality', axis = 1))
scaler_feat
df_feat = pd.DataFrame(scaler_feat, columns=wine_df.columns[:-1])
df_feat.head()
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



X = df_feat

y = wine_df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
error_rate = []



for i in range(1,100):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i!=y_test))
plt.figure(figsize = (15,10))

plt.plot(range(1, 100), error_rate, color = 'green', linestyle = 'dashed', 

         marker = 'o', markerfacecolor = 'red', markersize = 10)

plt.title('Error Rate vs K values')

plt.xlabel('K values')

plt.ylabel('Error Rate')

for i_x, i_y in zip(range(1,100), error_rate):

    plt.text(i_x, i_y, '({},{})'.format(round(i_x,2), round(i_y,2)))
knn_1 = KNeighborsClassifier(n_neighbors= 1)

knn_1.fit(X_train, y_train)

pred = knn.predict(X_test)
from sklearn.metrics import classification_report, r2_score, confusion_matrix



print(classification_report(y_test, pred))

print('\n')

print(r2_score(y_test, pred))

print('\n')

print(confusion_matrix(y_test, pred))
X_1 = wine_df.drop('quality', axis=1)

y_1 = wine_df['quality']



X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size = 0.3, random_state = 101)



from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



para_grid = { 'C':[0.1, 1, 10, 100, 1000, 10000, 100000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]}

grid = GridSearchCV(SVC(), para_grid, verbose = 3)

grid.fit(X_train_1, y_train_1)
grid.best_params_
grid_predictions = grid.predict(X_test_1)
print(classification_report(y_test_1, grid_predictions))

print('\n')

print(r2_score(y_test_1, grid_predictions))

print('\n')

print(confusion_matrix(y_test_1, grid_predictions))
X_2 = wine_df.drop('quality', axis= 1)

y_2 = wine_df['quality']



X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size = 0.3, random_state = 101)



from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators= 120)

rfc.fit(X_train_2, y_train_2)

rfc_pred = rfc.predict(X_test_2)
print(classification_report(y_test_2, rfc_pred))

print('\n')

print(r2_score(y_test_2, rfc_pred))
