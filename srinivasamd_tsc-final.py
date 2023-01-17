# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb

%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/AirQualityUCI_req.csv')
data.index = pd.DatetimeIndex(data.Date, dayfirst=True).strftime('%Y-%m-%d')
data = data.drop(['Date' ], 1)
cols = data.columns
data = data[data[cols] > 0]
data = data.fillna(method='ffill')
data.head()
temperature = data[['T']]
temperature_by_day = temperature.groupby(temperature.index).mean()
t_values = temperature_by_day.values
t_values = t_values.reshape(-1)
len(t_values)
def make_data(data_array, window=7):
    col = ['t'+str(i) for i in range(window)]
    col.append('Class')
    dict_data = {}
    count = 0
    inc_count = 0
    dec_count = 0
    for i in range(len(data_array)-window):
        if data_array[i+window] >= data_array[i+window-1]:
            temp_class = int(1)
            inc_count += 1
        else:
            temp_class = int(0)
            dec_count += 1
        #print(temp_class)
        count = count + 1
        temp_data = data_array[i:i+window]
        temp_row = np.concatenate((temp_data, temp_class), axis=None)
        dict_data[i] = temp_row
    #print(data)
    print(count, inc_count, dec_count)
    dataframe = pd.DataFrame.from_dict(dict_data,orient='index', columns=col)
    return dataframe, col[:len(col)-1]

transformed_data, feature_cols = make_data(t_values)
transformed_data.head()
features = transformed_data[feature_cols].values
classes = transformed_data[['Class']].values
print(features.shape, classes.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools
X_train, X_test, Y_train, Y_test = train_test_split(features,classes, test_size=0.33, random_state=42)
Y_train = Y_train.reshape(-1)
Y_test = Y_test.reshape(-1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
cm_lr = confusion_matrix(Y_test, y_pred)
acc_lr = accuracy_score(Y_test, y_pred)
print("Accuracy of classification : {0} %".format(acc_lr*100))
print(cm_lr)
model = MLPClassifier(hidden_layer_sizes=(90), learning_rate='constant')
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
cm_mlp = confusion_matrix(Y_test, y_pred)
acc_mlp = accuracy_score(Y_test, y_pred)
print("Accuracy of classification : {0} %".format(acc_mlp*100))
print(cm_mlp)
NB_model = GaussianNB()
NB_model.fit(X_train, Y_train)

y_pred = NB_model.predict(X_test)
acc_nb = accuracy_score(Y_test, y_pred)
cm_nb = confusion_matrix(Y_test, y_pred)
print("Accuracy of classification : {0} %".format(acc_nb*100))
print(cm_nb)
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train,Y_train)

y_pred = DT_model.predict(X_test)
acc_dt = accuracy_score(Y_test, y_pred)
cm_dt = confusion_matrix(Y_test, y_pred)
print("Accuracy of classification : {0} %".format(acc_dt*100))
print(cm_dt)
sb.barplot(x=[acc_lr,acc_mlp,acc_nb,acc_nb],y=['Logistic Regression', 'MLP', 'Naive Bayes', 'Decision Tree'])
plt.title('Without Dimensionality Reduction')
plt.xlabel('Accuracy in %')
plt.ylabel('Classifiers')
from sklearn.manifold import TSNE, MDS
features_embedded_tsne = TSNE(n_components=2).fit_transform(features)
plt.scatter(features_embedded_tsne[:,0], features_embedded_tsne[:,1],c=classes.reshape(-1))
features_embedded_mds = MDS(n_components=2).fit_transform(features)
plt.scatter(features_embedded_mds[:,0], features_embedded_mds[:,1],c=classes.reshape(-1))
X_train, X_test, Y_train, Y_test = train_test_split(features_embedded_mds,classes, test_size=0.33, random_state=42)
Y_train = Y_train.reshape(-1)
Y_test = Y_test.reshape(-1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
LRmodel = LogisticRegression(solver='liblinear')
LRmodel.fit(X_train,Y_train)
y_pred = LRmodel.predict(X_test)
cm_lr1 = confusion_matrix(Y_test, y_pred)
acc_lr1 = accuracy_score(Y_test, y_pred)
print("Accuracy of classification : {0} %".format(acc_lr1*100))
print(cm_lr1)
MLPmodel = MLPClassifier(hidden_layer_sizes=(90), learning_rate='constant')
MLPmodel.fit(X_train, Y_train)
y_pred = MLPmodel.predict(X_test)
cm_mlp1 = confusion_matrix(Y_test, y_pred)
acc_mlp1 = accuracy_score(Y_test, y_pred)
print("Accuracy of classification : {0} %".format(acc_mlp1*100))
print(cm_mlp1)
NB_model = GaussianNB()
NB_model.fit(X_train, Y_train)

y_pred = NB_model.predict(X_test)
acc_nb1 = accuracy_score(Y_test, y_pred)
cm_nb1 = confusion_matrix(Y_test, y_pred)
print("Accuracy of classification : {0} %".format(acc_nb1*100))
print(cm_nb1)
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train,Y_train)

y_pred = DT_model.predict(X_test)
acc_dt1 = accuracy_score(Y_test, y_pred)
cm_dt1 = confusion_matrix(Y_test, y_pred)
print("Accuracy of classification : {0} %".format(acc_dt1*100))
print(cm_dt1)
sb.barplot(x=[acc_lr1,acc_mlp1, acc_nb1, acc_nb1], y= ['Logistic Regression', 'MLP', 'Naive Bayes', 'Decision Tree'])
plt.title('With Dimensionality Reduction')
plt.xlabel('Accuracy in %')
plt.ylabel('Classifiers')
plt.figure(figsize=(20,5))
plt.subplot(1,5,1)
sb.barplot(x=['With MDS', 'Without MDS'], y = [acc_lr1,acc_lr])
plt.title('Logistic Regression')

plt.subplot(1,5,2)
sb.barplot(x=['With MDS', 'Without MDS'], y = [acc_mlp1,acc_mlp])
plt.title('MLP')

plt.subplot(1,5,3)
sb.barplot(x=['With MDS', 'Without MDS'], y = [acc_nb1,acc_nb])
plt.title('Naive Bayes')

plt.subplot(1,5,4)
sb.barplot(x=['With MDS', 'Without MDS'], y = [acc_dt1, acc_dt])
plt.title('Decision Tree')