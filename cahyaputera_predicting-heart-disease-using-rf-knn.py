import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.head()
df.info()
for i in df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]:

    sns.distplot(df[i])

    plt.title('Distribution of ' + i)

    plt.show()
plt.figure(figsize = (10, 16))

sns.countplot(y = df[df['target']==1]['age'])

plt.title('Count Of Positive Cases by Age')
a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")



frames = [df, a, b, c]

df_heart = pd.concat(frames, axis = 1)

df_heart = df_heart.drop(columns = ['cp', 'thal', 'slope'])
df_heart.head()
df_heart.info()
x = df_heart.drop(['target'], axis = 1)

y = df_heart.target.values
# Split the data with 80% Train size



x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8,random_state=0)
x_train.head()
x_test.head()
y_train
y_test
# Model Fitting



RFC = RandomForestClassifier(n_estimators = 2000, min_samples_split= 2, min_samples_leaf = 1, max_depth = 25)

RFC.fit(x_train, y_train)
# Random Forest Classifier predict



yp_RFC = RFC.predict(x_test)
# Confusion Matrix



cm_RFC = confusion_matrix(y_test,yp_RFC)

cm_RFC
# Labels for Confusion Matrix



labels = ['No Disease', 'Have Disease']
# Printing Classification Report and Showing Confusion Matrix



print(classification_report(y_test, yp_RFC, target_names = labels))

f, ax = plt.subplots(figsize=(8,5))

sns.heatmap(cm_RFC, annot=True, fmt=".0f", ax=ax)



ax.xaxis.set_ticklabels(labels)

ax.yaxis.set_ticklabels(labels)



plt.title('Heart Prediction With Random Forest Classifier')

plt.xlabel("ACTUAL")

plt.ylabel("PREDICT")

plt.show()
# Printing Score



print(RFC.score(x_test,y_test))
# Classification Report for Summary



report_RFC = pd.DataFrame(classification_report(y_test, yp_RFC, target_names= labels, output_dict=True)).T
# Determining the K-Value



k = round(len(x_train)**0.5)+1

k
# Fitting Model



KNN = KNeighborsClassifier(n_neighbors = k)

KNN.fit(x_train, y_train)
# KNN Predict



yp_KNN = KNN.predict(x_test)
# Confusion Matrix



cm_KNN = confusion_matrix(y_test,yp_KNN)

cm_KNN
# Printing Classification Report and Showing Confusion Matrix 



print(classification_report(y_test, yp_KNN, target_names = labels))

f, ax = plt.subplots(figsize=(8,5))

sns.heatmap(cm_KNN, annot=True, fmt=".0f", ax=ax)



ax.xaxis.set_ticklabels(labels)

ax.yaxis.set_ticklabels(labels)



plt.title('Heart Prediction With KNearest Neighbors')

plt.xlabel("ACTUAL")

plt.ylabel("PREDICT")

plt.show()
# Printing Score



print(KNN.score(x_test,y_test))
# Classification Report for Summary



report_KNN = pd.DataFrame(classification_report(y_test, yp_KNN, target_names= labels, output_dict=True)).T
# Showing the Confusion Matrix for both models



fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,6))

sns.heatmap(cm_RFC, annot=True, fmt=".0f", ax=ax1)

sns.heatmap(cm_KNN, annot=True, fmt=".0f", ax=ax2)



ax1.xaxis.set_ticklabels(labels), ax1.yaxis.set_ticklabels(labels)

ax2.xaxis.set_ticklabels(labels), ax2.yaxis.set_ticklabels(labels)



ax1.set_title('RFC'), ax2.set_title('KNN')

ax1.set_xlabel('ACTUAL'), ax2.set_xlabel('ACTUAL')

ax1.set_ylabel('PREDICTED'), ax2.set_ylabel('PREDICTED')



plt.show()
print('RFC Model : ', RFC.score(x_test,y_test))

print('KNN Model : ', KNN.score(x_test,y_test))
# Printing Classification Report Summary

pd.concat([report_RFC, report_KNN], keys = ['RFC MODEL', 'KNN MODEL'])