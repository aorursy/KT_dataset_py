import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn import metrics
data = pd.read_csv('../input/Bank_Personal_Loan_Modelling.csv')

data.head()
data.shape
data.info()
data[data['Experience'] < 0]['Experience'].count()
#Since ZIP Code has 1 variable with 4 digits hence we converted to 5 digits by using most frequent value - mode of that column

data= data.replace('9307', np.nan)

data = data.apply(lambda x: x.fillna(x.mode(), axis = 0))

#This step may not be necessary as we would be dropping this column later
#Converted all of these columns into their actual figures

data[['Income', 'CCAvg', 'Mortgage']] = data[['Income', 'CCAvg', 'Mortgage']] * 1000
data.head()
data.groupby('Personal Loan').count()
data.describe().transpose()
#Dropping ID column as it is not required

data = data.drop(['ID'], axis = 1)

data.head()
sns.pairplot(data)
plt.figure(figsize = (15,15))

sns.heatmap(data.corr(), annot = True, cmap = 'plasma' )
#Lets study the relation between Personal Loan, Income,CCAvg and CD Account as these share high correlation

sns.scatterplot(y = 'Income', x = 'Age', data = data, hue = 'Personal Loan')
sns.scatterplot(x = 'Income', y = 'CCAvg', data = data, hue = 'Personal Loan')
sns.scatterplot(x = 'Income', y = 'Experience', data = data, hue = 'Personal Loan')
sns.scatterplot(x = 'Income', y = 'Securities Account', data = data, hue = 'Personal Loan')
sns.scatterplot(x = 'Income', y = 'CD Account', data = data, hue = 'Personal Loan')
sns.scatterplot(x = 'CCAvg', y = 'CD Account', data = data, hue = 'Personal Loan')
sns.scatterplot(y = 'Education', x = 'Income', data = data, hue = 'Personal Loan')
sns.scatterplot(y = 'Education', x = 'CCAvg', data = data, hue = 'Personal Loan')
sns.scatterplot(y = 'Mortgage', x = 'Income', data = data, hue = 'Personal Loan')
sns.scatterplot(y = 'Family', x = 'Income', data = data, hue = 'Personal Loan')
sns.scatterplot(x = 'Age', y = 'Income', data = data, hue = 'Personal Loan')
#Find % of people who have availed for Loan

not_availed_Loan = len(data[data['Personal Loan'] == 0])

availed_Loan = len(data[data['Personal Loan'] == 1])



print("% of people who did not avail Loan {:.2f}%". format(not_availed_Loan/ (data['Personal Loan'].count()) * 100))



print("% of people availed Loan {:.2f}%". format(availed_Loan/ (data['Personal Loan'].count()) * 100))

                       
data.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



# To scale the dimensions we need scale function which is part of scikit preprocessing libraries

from sklearn import preprocessing

from scipy.stats import zscore





X = data.drop(['Personal Loan','Age','Experience','ZIP Code','Online','CreditCard'], axis = 1)

y = data['Personal Loan']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 1)



X_train_scaled = preprocessing.scale(X_train)

X_test_scaled = preprocessing.scale(X_test)





logreg = LogisticRegression()

logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)
print('Train score: {}'.format(logreg.score(X_train_scaled, y_train) * 100))

print('Test score: {}'.format(logreg.score(X_test_scaled, y_test) * 100))
print(metrics.confusion_matrix(y_test, y_pred))
accuracies = {}

acc_logreg = logreg.score(X_test_scaled, y_test) * 100

accuracies['Logistic Regression'] = acc_logreg

print("Logistic Regression Accuracy: {}".format(acc_logreg))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train_scaled, y_train)
expected  = y_test

predicted = nb.predict(X_test_scaled)

print(metrics.classification_report(expected, predicted))

print('Total accuracy:', np.round(metrics.accuracy_score(expected, predicted),2))
print('Train score: {}'.format(nb.score(X_train_scaled, y_train) * 100))

print('Test score: {}'.format(nb.score(X_test_scaled, y_test) * 100))
print(metrics.confusion_matrix(expected, predicted))
acc_nb = nb.score(X_test_scaled, y_test) * 100

accuracies['Naive Bayes'] = acc_nb

print("Naive Bayes Accuracy: {}".format(acc_nb))
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_validate





# creating odd list of K for KNN

myList = list(range(1,10))



# subsetting just the odd ones

neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold accuracy scores

ac_scores = []



# perform accuracy metrics for values from 1,3,5....19

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    # predict the response

    y_pred = knn.predict(X_test)

    # evaluate accuracy

    scores = accuracy_score(y_test, y_pred)

    ac_scores.append(scores)





# changing to misclassification error

MSE = [1 - x for x in ac_scores]



# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)





plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
#the above graph sometimes shows 7, and sometimes 1 however as 1 is the right answer hence used it below.

#If someone knows why this happens please share your inputs...

knn = KNeighborsClassifier(n_neighbors = 1, weights = 'distance')
X_z = X.apply(zscore)

X_z.describe()
X = np.array(X_z)

y = np.array(y)
print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

knn.score(X_test, y_test)
print('Train score: {}'.format(knn.score(X_train, y_train) * 100))

print('Test score: {}'.format(knn.score(X_test, y_test) * 100))
cm_knn = print(metrics.confusion_matrix(y_test, y_pred))

cm_knn
acc_knn = knn.score(X_test, y_test) * 100

accuracies['KNN'] = acc_knn
from sklearn import svm

svm = svm.SVC(gamma = 0.025, C = 3)

svm.fit(X_train_scaled, y_train)
svm.score(X_test_scaled, y_test)
y_pred = svm.predict(X_test_scaled)



print('Train score: {}'.format(svm.score(X_train_scaled, y_train) * 100))

print('Test score: {}'.format(svm.score(X_test_scaled, y_test) * 100))
acc_svm = svm.score(X_test_scaled, y_test) * 100

accuracies['SVM'] = acc_svm
print(metrics.confusion_matrix(y_test, y_pred))
sns.set_style('whitegrid')

plt.figure(figsize = (8,5))

plt.yticks(np.arange(0,100,10))

sns.barplot(x = list(accuracies.keys()), y = list(accuracies.values()))
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Naive Bayes','K-Nearest Neighbors', 'Support Vector Machines'],

    

    'Score': [acc_logreg, acc_nb, acc_knn, acc_svm]

    })



models.sort_values(by='Score', ascending=True)
y_cm_lr = logreg.predict(X_test_scaled)

y_cm_nb = nb.predict(X_test_scaled)

y_cm_knn = knn.predict(X_test)

y_cm_svm = svm.predict(X_test_scaled)
from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test, y_cm_lr)

cm_knn = confusion_matrix(y_test, y_cm_knn)

cm_svm = confusion_matrix(y_test, y_cm_svm)

cm_nb = confusion_matrix(y_test, y_cm_nb)

plt.figure(figsize = (8,8))

plt.suptitle("Confusion Matrices",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace = 0.4)



plt.subplot(2,2,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 12})



plt.subplot(2,2,2)

plt.title("KNN Confusion Matrix")

sns.heatmap(cm_knn, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(2,2,3)

plt.title("NB Confusion Matrix")

sns.heatmap(cm_nb, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(2,2,4)

plt.title("SVM Confusion Matrix")

sns.heatmap(cm_svm, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})
