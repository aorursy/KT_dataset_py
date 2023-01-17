# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
vodafone_subset_6 = pd.read_csv("../input/vodafone6nm/vodafone-subset-6.csv")
vodafone_subset_6.head(10)
df = vodafone_subset_6[['target', 'ROUM', 'phone_value', 'DATA_VOLUME_WEEKDAYS', 'DATA_VOLUME_WEEKENDS', 
                        'device_brand', 'software_os_vendor', 'software_os_name', 'software_os_version', 'device_type_rus',
                       'AVG_ARPU', 'lifetime', 'how_long_same_model', 'ecommerce_score',
                        'banks_sms_count', 'instagram_volume', 'viber_volume', 'linkedin_volume', 'tinder_volume', 'telegram_volume', 'google_volume', 'whatsapp_volume', 'youtube_volume']]
df.head()
df.iloc[:, :23].describe().T
df.info()
df['device_brand'].value_counts()
df['software_os_vendor'].value_counts()
df['software_os_name'].value_counts()
df['software_os_version'].value_counts()
df['device_type_rus'].value_counts()
df_1 = pd.get_dummies(df, columns=['phone_value', 'device_brand', 'software_os_vendor', 'software_os_name', 'software_os_version', 'device_type_rus'])
df_1.head()
df_1.dtypes.value_counts()
df_2 = df_1.dropna()
df_2.shape
X = df_1.drop('target', axis=1)
y = df_1['target']

# scaler = StandardScaler()
# # scaler.fit(X)
# X_scaled = scaler.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_valid)
print(accuracy_score(y_pred, y_valid))
print(confusion_matrix(y_valid, y_pred))
pd.DataFrame(confusion_matrix(y_valid, y_pred),
            index=['True_'+str(i+1) for i in range(6)],
            columns=['Pred'+str(i+1) for i in range(6)])
kf = KFold(n_splits=5, shuffle=True, random_state=22)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
print(scores)
mean_score = scores.mean()
print(mean_score)
knn = KNeighborsClassifier()
knn_params = {'n_neighbors': np.arange(1, 51)}
knn_grid = GridSearchCV(knn, knn_params, scoring='accuracy', cv=kf)
knn_grid.fit(X_train, y_train)
print(knn_grid.best_estimator_)
print(knn_grid.best_params_)
print(knn_grid.best_score_)
results_df = pd.DataFrame(knn_grid.cv_results_)
plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('param_n_neighbors')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
knn = KNeighborsClassifier()
knn_params = {'n_neighbors': np.arange(51, 101)}
knn_grid = GridSearchCV(knn, knn_params, scoring='accuracy', cv=kf)
knn_grid.fit(X_train, y_train)
print(knn_grid.best_estimator_)
print(knn_grid.best_params_)
print(knn_grid.best_score_)
results_df = pd.DataFrame(knn_grid.cv_results_)
plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])

# Подписываем оси и график
plt.xlabel('param_n_neighbors')
plt.ylabel('Test accuracy')
plt.title('Validation curve')
plt.show()
knn_model = KNeighborsClassifier(n_neighbors=56, weights='distance')
knn_params = {'p': np.linspace(1, 10, 20)}
knn_model_grid = GridSearchCV(knn_model, knn_params, scoring='accuracy', cv=5)
knn_model_grid.fit(X_train, y_train)
print(knn_model_grid.best_estimator_)
print(knn_model_grid.best_params_)
print(knn_model_grid.best_score_)
nc = NearestCentroid()
nc.fit(X_train, y_train)
y_pred = nc.predict(X_valid)
print(accuracy_score(y_pred, y_valid))
print(confusion_matrix(y_valid, y_pred))
rnc = RadiusNeighborsClassifier(radius=1)
rnc.fit(X_train, y_train)
y_pred = rnc.predict(X_valid)
print(accuracy_score(y_pred, y_valid))
print(confusion_matrix(y_valid, y_pred))
rnc = RadiusNeighborsClassifier(radius=100)
rnc.fit(X_train, y_train)
y_pred = rnc.predict(X_valid)
print(accuracy_score(y_pred, y_valid))
print(confusion_matrix(y_valid, y_pred))
rnc = RadiusNeighborsClassifier(radius=10000)
rnc.fit(X_train, y_train)
y_pred = rnc.predict(X_valid)
print(accuracy_score(y_pred, y_valid))
print(confusion_matrix(y_valid, y_pred))
rnc = RadiusNeighborsClassifier(radius=1)
rnc_params = {'radius': np.arange(6000, 8000, 200)}
rnc_grid = GridSearchCV(rnc, rnc_params, scoring='accuracy', cv=5)
rnc_grid.fit(X_train, y_train)
print(rnc_grid.best_estimator_)
print(rnc_grid.best_params_)
print(rnc_grid.best_score_)
# a = X[0, :]
# b = X[1, :]
# print(np.linalg.norm(a-b))
# a = X_scaled[35, :]
# b = X_scaled[36, :]
# print(np.linalg.norm(a-b))
rnc = RadiusNeighborsClassifier(radius=2000, outlier_label='most_frequent')
rnc.fit(X_train, y_train)
y_pred = rnc.predict(X_valid)
print(accuracy_score(y_pred, y_valid))
print(confusion_matrix(y_valid, y_pred))