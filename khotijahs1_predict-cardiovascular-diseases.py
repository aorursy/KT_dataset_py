import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

print('Dataset :',data.shape)

data.info()

data[0:10]
# Distribution of DEATH_EVENT

data.DEATH_EVENT.value_counts()[0:30].plot(kind='bar')

plt.show()
data1 = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure',

'platelets','serum_creatinine','serum_sodium','sex','smoking','time']] #Subsetting the data

cor = data1.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
sns.set_style("whitegrid")

sns.pairplot(data,hue="DEATH_EVENT",size=3);

plt.show()
from sklearn.model_selection import train_test_split

Y = data['DEATH_EVENT']

X = data.drop(columns=['DEATH_EVENT'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=9)
print('X train shape: ', X_train.shape)

print('Y train shape: ', Y_train.shape)

print('X test shape: ', X_test.shape)

print('Y test shape: ', Y_test.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix



# We define the model

rfcla = RandomForestClassifier(n_estimators=100,random_state=9,n_jobs=-1)



# We train model

rfcla.fit(X_train, Y_train)



# We predict target values

Y_predict5 = rfcla.predict(X_test)
test_acc_rfcla = round(rfcla.fit(X_train,Y_train).score(X_test, Y_test)* 100, 2)

train_acc_rfcla = round(rfcla.fit(X_train, Y_train).score(X_train, Y_train)* 100, 2)
model1 = pd.DataFrame({

    'Model': ['Random Forest'],

    'Train Score': [train_acc_rfcla],

    'Test Score': [test_acc_rfcla]

})

model1.sort_values(by='Test Score', ascending=False)
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict5)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
# The confusion matrix

rfcla_cm = confusion_matrix(Y_test, Y_predict5)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(rfcla_cm, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")

plt.title('Random Forest Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
from sklearn.ensemble import BaggingClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC



# We define the SVM model

svmcla = OneVsRestClassifier(BaggingClassifier(SVC(C=10,kernel='rbf',random_state=9, probability=True), 

                                               n_jobs=-1))



# We train model

svmcla.fit(X_train, Y_train)



# We predict target values

Y_predict2 = svmcla.predict(X_test)
test_acc_svm = round(svmcla.fit(X_train,Y_train).score(X_test, Y_test)* 100, 2)

train_acc_svm = round(svmcla.fit(X_train, Y_train).score(X_train, Y_train)* 100, 2)
model2 = pd.DataFrame({

    'Model': ['SVM'],

    'Train Score': [train_acc_svm],

    'Test Score': [test_acc_svm]

})

model2.sort_values(by='Test Score', ascending=False)
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict2)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
# The confusion matrix

svm = confusion_matrix(Y_test, Y_predict5)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(svm, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")

plt.title('SVM Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
Y1 = data['DEATH_EVENT']

X1 = data.drop(columns=['age','anaemia','diabetes','high_blood_pressure'])

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel



lsvc = LinearSVC(C=0.06, penalty="l1", dual=False,random_state=10).fit(X1, Y1)

model = SelectFromModel(lsvc, prefit=True)

X_new = model.transform(X1)

cc = list(X1.columns[model.get_support(indices=True)])

print(cc)

print(len(cc))
# Principal component analysis

from sklearn.decomposition import PCA



pca = PCA().fit(X1)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("'age','anaemia','diabetes','high_blood_pressure'")

plt.ylabel('% Variance Explained')

plt.title('PCA Analysis')

plt.grid(True)

plt.show()
# Percentage of total variance explained

variance = pd.Series(list(np.cumsum(pca.explained_variance_ratio_)), 

                        index= list(range(0,9))) 

print(variance[20:80])
X1 = data[cc] 

from sklearn.model_selection import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.05, random_state=10)
# Random forest classification

rfcla.fit(X1_train, Y1_train)

Y1_predict5 = rfcla.predict(X1_test)

rfcla_cm = confusion_matrix(Y1_test, Y1_predict5)

score1_rfcla = rfcla.score(X1_test, Y1_test)
test_acc_rfcla = round(rfcla.fit(X1_train,Y1_train).score(X1_test, Y1_test)* 100, 2)

train_acc_rfcla = round(rfcla.fit(X1_train, Y1_train).score(X1_train, Y1_train)* 100, 2)
# SVM classification

svmcla.fit(X1_train, Y1_train)

Y1_predict2 = svmcla.predict(X1_test)

svmcla_cm = confusion_matrix(Y1_test, Y1_predict2)

score1_svmcla = svmcla.score(X1_test, Y1_test)
test_acc_svm2 = round(svmcla.fit(X_train,Y_train).score(X_test, Y_test)* 100, 2)

train_acc_svm2 = round(svmcla.fit(X_train, Y_train).score(X_train, Y_train)* 100, 2)
model3 = pd.DataFrame({

    'Model': ['Random Forest','SVM'],

    'Train Score': [train_acc_rfcla,train_acc_svm2 ],

    'Test Score': [test_acc_rfcla, test_acc_svm2]

})

model3.sort_values(by='Test Score', ascending=False)
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3, 3, 1) 

ax1.set_title('Random Forest') 

ax2 = fig.add_subplot(3, 3, 2) 

ax2.set_title('SVM Classification')





sns.heatmap(data=rfcla_cm, annot=True, linewidth=0.7, linecolor='black',cmap="BuPu" ,fmt='g', ax=ax1)

sns.heatmap(data=svmcla_cm, annot=True, linewidth=0.7, linecolor='black',cmap="BuPu" ,fmt='g', ax=ax2)

plt.show()