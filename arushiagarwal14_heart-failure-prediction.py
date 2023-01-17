import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
dataset.head()
import seaborn as sns

sns.pairplot(dataset, kind="scatter", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))

plt.show()
correlation = dataset.corr()

sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)
dataset.corr(method ='pearson') 
dataset.corr(method ='kendall') 
dataset.corr(method ='spearman') 
correlation[abs(correlation['DEATH_EVENT']) > 0.2]['DEATH_EVENT']
final_df = dataset[['age', 'ejection_fraction', 'serum_creatinine', 'time', 'DEATH_EVENT']]
final_df.head()
X = final_df.iloc[:, :-1].values

y = final_df.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)
# For checking the actual values vs predicted values side by side



#y_pred = classifier.predict(X_test)

#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
## KFold Cross validation



from sklearn.model_selection import cross_val_score

svm_cvs_acc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(svm_cvs_acc.mean()*100))
from sklearn.ensemble import RandomForestClassifier

rfclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')

rfclassifier.fit(X_train, y_train)
rf_cvs_acc = cross_val_score(estimator = rfclassifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(rf_cvs_acc.mean()*100))
from sklearn.linear_model import LogisticRegression

lrclassifier = LogisticRegression(random_state = 0)

lrclassifier.fit(X_train, y_train)
lr_cvs_acc = cross_val_score(estimator = lrclassifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(lr_cvs_acc.mean()*100))
from sklearn.naive_bayes import GaussianNB

nbclassifier = GaussianNB()

nbclassifier.fit(X_train, y_train)
nb_cvs_acc = cross_val_score(estimator = nbclassifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(nb_cvs_acc.mean()*100))
from sklearn.neighbors import KNeighborsClassifier

knnclassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knnclassifier.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

knn_cvs_acc = cross_val_score(estimator = knnclassifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(knn_cvs_acc.mean()*100))