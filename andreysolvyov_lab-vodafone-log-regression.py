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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
vodafone_subset_5 = pd.read_csv("/kaggle/input/03032020/vodafone-subset-5.csv")
temp_df = vodafone_subset_5[['lifetime', 
                             'CALCULATION_METHOD_ID', 
                             'instagram_volume',
                             'intagram_count', 
                             'DATA_VOLUME_WEEKENDS', 
                             'DATA_VOLUME_WEEKDAYS',
                             'google_volume', 
                             'VKcom', 
                             '1020', 
                             'viber_count',
                             'calls_count_in_weekdays', 
                             'fb_count', 
                             'calls_duration_out_weekdays',
                             'youtube_volume', 
                             'ecommerce_score', 
                             'itunes_volume',
                             'Кафе и рестораны', 
                             'NovaPoshta', 
                             'Дети_1']]
Target = vodafone_subset_5['target']
#90%
indexes = []
for col in temp_df.columns:
    if max(temp_df[col].value_counts()) > temp_df.shape[0]*0.9:
        indexes.append(col)
len(indexes)
temp_df = temp_df.dropna(axis='index', how='any')
Target = Target[temp_df.index]
objects = []
types = []
for j in temp_df.columns:
    types.append(temp_df[[j]].dtypes[0])
    if temp_df[[j]].dtypes[0] == 'object':
        objects.append(j)
print(set(types))
print(objects)
temp_df
b = list(temp_df.columns)
del b[1]
del b[13]
for B in b:
    temp_df[[B]] = MinMaxScaler().fit_transform(temp_df[[B]])
data = pd.get_dummies(temp_df, columns = ['CALCULATION_METHOD_ID', 'ecommerce_score'])
data
Target
X_train, X_valid, y_train, y_valid = train_test_split(data, 
                                                      Target, 
                                                      test_size=0.25, 
                                                      random_state=2303)
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_valid, y_pred))
lams_C = []
score_C = []
for lam in np.logspace(-3, 3, 10):
    log_reg = LogisticRegression(C = lam, penalty = 'l2')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_valid)

    print(lam, ':', accuracy_score(y_valid, y_pred))
    
    lams_C.append(lam)
    score_C.append(accuracy_score(y_valid, y_pred))
import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.plot(lams_C, score_C)
lams_C = []
score_C = []
for lam in np.logspace(3, 7, 10):
    log_reg = LogisticRegression(C = lam, penalty = 'l2')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_valid)

    print(lam, ':', accuracy_score(y_valid, y_pred))
    
    lams_C.append(lam)
    score_C.append(accuracy_score(y_valid, y_pred))
matplotlib.pyplot.plot(lams_C, score_C)
lams_C = []
score_C = []
for lam in [lam for lam in range(1000, 5001, 100)]:
    log_reg = LogisticRegression(C = lam, penalty = 'l2')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_valid)

    print(lam, ':', accuracy_score(y_valid, y_pred))
    
    lams_C.append(lam)
    score_C.append(accuracy_score(y_valid, y_pred))
matplotlib.pyplot.plot(lams_C, score_C)
print(lams_C[score_C.index(max(score_C))], max(score_C))
log_reg = LogisticRegression(C = 1800, penalty = 'l2', random_state=123)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)
print(accuracy_score(y_valid, y_pred))
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(log_reg, X_valid, y_valid, values_format='5g')
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision:', precision_score(y_valid, y_pred, average='macro'))
print('Recall:', recall_score(y_valid, y_pred, average='macro'))
print('F1 score:', f1_score(y_valid, y_pred, average='macro'))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=34, p=1).fit(X_train, y_train)
y_pred = knn.predict(X_valid)
print('accuracy_score:', knn.score(X_valid, y_valid))
print('Precision:', precision_score(y_valid, y_pred, average='macro'))
print('Recall:', recall_score(y_valid, y_pred, average='macro'))
print('F1 score:', f1_score(y_valid, y_pred, average='macro'))
plot_confusion_matrix(knn, X_valid, y_valid, values_format='5g')
plt.show()
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=123).fit(X_train, y_train)
y_pred = tree.predict(X_valid)
print('accuracy_score:', accuracy_score(y_valid, y_pred))
print('Precision:', precision_score(y_valid, y_pred, average='macro'))
print('Recall:', recall_score(y_valid, y_pred, average='macro'))
print('F1 score:', f1_score(y_valid, y_pred, average='macro'))
plot_confusion_matrix(tree, X_valid, y_valid, values_format='5g')
plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=140, max_features = 20, max_depth = 11, min_samples_leaf = 12, random_state=2019).fit(X_train, y_train)
y_pred = rf.predict(X_valid)
print('accuracy_score:', accuracy_score(y_valid, y_pred))
print('Precision:', precision_score(y_valid, y_pred, average='macro'))
print('Recall:', recall_score(y_valid, y_pred, average='macro'))
print('F1 score:', f1_score(y_valid, y_pred, average='macro'))
plot_confusion_matrix(rf, X_valid, y_valid, values_format='5g')
plt.show()
lams_C = []
score_C = []
for lam in np.logspace(3, 7, 10):
    log_reg = LogisticRegression(C = lam, penalty = 'l2')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_valid)

    print(lam, ':', f1_score(y_valid, y_pred, average='macro'))
    
    lams_C.append(lam)
    score_C.append(f1_score(y_valid, y_pred, average='macro'))
matplotlib.pyplot.plot(lams_C, score_C)
lams_C = []
score_C = []
for lam in [lam for lam in range(700, 6001, 200)]:
    log_reg = LogisticRegression(C = lam, penalty = 'l2')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_valid)

    print(lam, ':', f1_score(y_valid, y_pred, average='macro'))
    
    lams_C.append(lam)
    score_C.append(f1_score(y_valid, y_pred, average='macro'))
matplotlib.pyplot.plot(lams_C, score_C)
print(lams_C[score_C.index(max(score_C))], max(score_C))
log_reg = LogisticRegression(C = 4900, penalty = 'l2', random_state=123)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)
print('accuracy_score:', accuracy_score(y_valid, y_pred))
print('Precision:', precision_score(y_valid, y_pred, average='macro'))
print('Recall:', recall_score(y_valid, y_pred, average='macro'))
print('F1 score:', f1_score(y_valid, y_pred, average='macro'))
plot_confusion_matrix(log_reg, X_valid, y_valid, values_format='5g')
plt.show()