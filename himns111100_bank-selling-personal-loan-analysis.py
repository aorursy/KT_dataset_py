import pandas as pd

import numpy as np

import seaborn as sns

from scipy.stats import zscore

import scipy.stats as stats

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from scipy.stats import zscore

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, auc

from sklearn.naive_bayes import GaussianNB

from sklearn import model_selection
import os

print(os.listdir("../input/bank-loan-modelling"))
bank_data = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx', 'Data')

bank_data.head()
bank_data.columns


bank_data.shape
bank_data.info()
# checking for any missing values

bank_data.isna().sum()
bank_data.dtypes
bank_data.describe().transpose()
bank_data.apply(lambda x: len(x.unique()))
bank_data.groupby(["Personal Loan"]).count()
sns.pairplot(bank_data.iloc[:,1:])
#check the amount of negative values

bank_data[bank_data['Experience'] < 0]['Experience'].value_counts()
ncol = ['Age', 'Income','CCAvg', 'Mortgage']

grid = sns.PairGrid(bank_data, y_vars = 'Experience', x_vars = ncol, height = 4)

grid.map(sns.regplot);
bank_data[bank_data['Experience'] < 0]['Age'].value_counts()
# Get a list of 'Age' values where we found some negative values in 'Experience'

ages = bank_data[bank_data['Experience'] < 0]['Age'].unique().tolist()

ages
indexes = bank_data[bank_data['Experience'] < 0].index.tolist()
for i in indexes:

    for x in ages:

        bank_data.loc[i,'Experience'] = bank_data[(bank_data.Age == x) & (bank_data.Experience > 0)].Experience.mean()


bank_data[bank_data['Experience'] < 0]['Age'].value_counts()


bank_data.Experience.describe()
# checking back again if there is any negative value present in the experience column

bank_data[bank_data['Experience'] < 0]['Experience'].value_counts()
#convert bank data DataFrame object to numpy array and sort

numpydf = np.asarray(bank_data['Age'])

numpydf = sorted(numpydf)



#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(numpydf, np.mean(numpydf), np.std(numpydf)) 



#plot both series on the histogram

plt.plot(numpydf ,fit,'-',linewidth = 2)

plt.hist(numpydf, normed=True, bins = 100)      

plt.show() 



print('Skew =', bank_data['Age'].skew())
#convert bank data DataFrame object to numpy array and sort

numpydf = np.asarray(bank_data['Experience'])

numpydf = sorted(numpydf)



#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(numpydf, np.mean(numpydf), np.std(numpydf)) 



#plot both series on the histogram

plt.plot(numpydf ,fit,'-',linewidth = 2)

plt.hist(numpydf, normed=True, bins = 100)      

plt.show() 



print('Skew =', bank_data['Experience'].skew())
#convert bank data DataFrame object to numpy array and sort

numpydf = np.asarray(bank_data['Income'])

numpydf = sorted(numpydf)



#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(numpydf, np.mean(numpydf), np.std(numpydf)) 



#plot both series on the histogram

plt.plot(numpydf ,fit,'-',linewidth = 2)

plt.hist(numpydf, normed=True, bins = 100)      

plt.show() 



print('Skew =', bank_data['Income'].skew())
#convert bank data DataFrame object to numpy array and sort

numpydf = np.asarray(bank_data['CCAvg'])

numpydf = sorted(numpydf)



#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(numpydf, np.mean(numpydf), np.std(numpydf)) 



#plot both series on the histogram

plt.plot(numpydf ,fit,'-',linewidth = 2)

plt.hist(numpydf, normed=True, bins = 100)      

plt.show() 



print('Skew =', bank_data['CCAvg'].skew())
#convert bank data DataFrame object to numpy array and sort

numpydf = np.asarray(bank_data['Mortgage'])

numpydf = sorted(numpydf)



#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(numpydf, np.mean(numpydf), np.std(numpydf)) 



#plot both series on the histogram

plt.plot(numpydf ,fit,'-',linewidth = 2)

plt.hist(numpydf, normed=True, bins = 100)      

plt.show() 



print('Skew =', bank_data['Mortgage'].skew())


#convert bank data DataFrame object to numpy array and sort

numpydf = np.asarray(bank_data['Family'])

numpydf = sorted(numpydf)



#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(numpydf, np.mean(numpydf), np.std(numpydf)) 



#plot both series on the histogram

plt.plot(numpydf ,fit,'-',linewidth = 2)

plt.hist(numpydf, normed=True, bins = 100)      

plt.show() 



numpydf1 = np.asarray(bank_data['Education'])

numpydf1 = sorted(numpydf1)



#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(numpydf1, np.mean(numpydf1), np.std(numpydf1)) 



#plot both series on the histogram

plt.plot(numpydf1,fit,'-',linewidth = 2)

plt.hist(numpydf1, normed=True, bins = 100)      

plt.show() 



print('Skewness in Education Column =', bank_data['Education'].skew())



print('Skewness in Family Column =', bank_data['Family'].skew())
# checking back again the 5 point Summary

#bank_data.describe().T

bank_data.isna().sum()
sns.countplot(x='CD Account',data=bank_data,hue='Personal Loan')


series = bank_data[bank_data['CD Account'] == 1]['Personal Loan'].value_counts()


plt.axis('equal')

plt.title('Proportion of Customers Who Have Personal Loan and Who Don\'t,\n among CD Account Holders', \

          fontsize = 14, y = 1.2)

labels = ['NO Personal Loan','Personal Loan']

plt.pie(series, labels = labels,autopct= '%1.1f%%', shadow = True,explode = (0.1, 0), radius = 1.6, startangle = 90)
sns.countplot(x="Education", data=bank_data,hue="Personal Loan")
education = bank_data[bank_data['Personal Loan'] == 1]['Education'].value_counts()

education


plt.axis('equal')

plt.title('Proportion of Customers With Different Levels of Education \n among Personal Loan Holders', \

          fontsize = 14, y = 1.3)

labels = ['Education Level  3',' Education Level 2','Education Level 1']

plt.pie(education, labels = labels, autopct= '%1.2f%%', shadow = True,explode = (0.1, 0, 0), radius = 1.6, startangle = 90);

plt.savefig('Proportion_edu_levels_among_PL.png', bbox_inches = 'tight');
sns.set_style("whitegrid", {'grid.linestyle': '--'})

plt.figure(figsize = (7,5))

sns.barplot(x = "Education", y = "Income", hue='Personal Loan', data = bank_data)

plt.xlabel("Education")

plt.ylabel("Income")

plt.title("Distribution of Education & Income in terms of Personal Loan")
sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=bank_data)


sns.boxplot(x='Family',data=bank_data,hue='Personal Loan',y='Income')

plt.legend(loc='upper center')

plt.title("Family and Income Boxplot")
sns.boxplot(x="Education", y='Mortgage', hue="Personal Loan", data=bank_data,color='yellow')
sns.countplot(x="Securities Account", data=bank_data,hue="Personal Loan")


sns.countplot(x='Family',data=bank_data,hue='Personal Loan',palette='Set1')


sns.distplot( bank_data[bank_data['Personal Loan'] == 0]['CCAvg'], color = 'r')

sns.distplot( bank_data[bank_data['Personal Loan'] == 1]['CCAvg'], color = 'g')


print('Credit card spending of Non-Loan customers: ',bank_data[bank_data['Personal Loan'] == 0]['CCAvg'].median()*1000)

print('Credit card spending of Loan customers    : ', bank_data[bank_data['Personal Loan'] == 1]['CCAvg'].median()*1000)
fig, ax = plt.subplots()

colors = {1:'red',2:'yellow',3:'green'}

ax.scatter(bank_data['Experience'],bank_data['Age'],c=bank_data['Education'].apply(lambda x:colors[x]))

plt.xlabel('Experience')

plt.ylabel('Age')
colormap = plt.cm.plasma

plt.figure(figsize=(27,17))

plt.title('Correlation of Thera Bank Dataset', y=1.05, size=15)

sns.heatmap(bank_data.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, 

            linecolor='white', annot=True)
X = bank_data.drop(["ZIP Code", "ID", "Personal Loan"] , axis = 1)

X.head()
y=bank_data[['Personal Loan']]

y.head()
X = X.apply(zscore)
test_size=0.30

seed=7

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=seed)
test_size=0.30

seed=7

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size, random_state=seed)

KNNH=KNeighborsClassifier()

KNNH.fit(X_train, y_train)

# For every test data point, predict it's label based on 5 nearest neighbours in this model. The majority class will 

# be assigned to the test data point

predicted_labels = KNNH.predict(X_train)

## Train Accuracy

print("Score for training data is =", KNNH.score(X_train, y_train))

## Test accuracy

print("Score for test data is =", KNNH.score(X_test, y_test))
maxK = int(np.sqrt(X_train.shape[0]))

# creating odd list of K for KNN

myList = list(range(1,60))

# subsetting just the odd ones

neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores

cv_scores = []



# perform 10-fold cross validation

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train.values.ravel(), cv=10, scoring='accuracy') #used ravel() as it was giving warnings

    cv_scores.append(scores.mean())



# changing to misclassification error

misError = [1 - x for x in cv_scores]



optimal_k = neighbors[misError.index(min(misError))]

print("The value of optimal K is = ",optimal_k)
NNH = KNeighborsClassifier(n_neighbors=optimal_k)

NNH.fit(X_train, y_train)

print(NNH.score(X_train, y_train))

print(NNH.score(X_test, y_test))
# Confusion matrix

knn_cm=metrics.confusion_matrix(y_test, NNH.predict(X_test))

knn_cm
#find avaible value of k

score_list=[]

k_value=[]

for i in myList:

    knn_available=KNeighborsClassifier(n_neighbors=i)

    knn_available.fit(X_train,y_train.values.ravel())

    

    score_list.append(knn_available.score(X_test,y_test))

    k_value.append(i)

plt.plot(myList,score_list)

plt.xlabel("K values")

plt.ylabel("Score(accracy)")

plt.show()
y_predictProb = KNNH.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_predictProb[::,1])

roc_auc = auc(fpr, tpr)

optimalF1 = 0

optimalTh = 0
print(roc_auc)
for th in thresholds:

    preds = np.where(KNNH.predict_proba(X_test)[:,1] > th, 1, 0)

    f1Score = f1_score(y_test, preds)

    if(optimalF1 < f1Score):

        optimalF1 = f1Score

        optimalTh = th

        

THRESHOLD = optimalTh

preds = np.where(KNNH.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)

pd.DataFrame(data=[accuracy_score(y_test, preds), 

                   recall_score(y_test, preds),

                   precision_score(y_test, preds),

                   f1_score(y_test, preds)], 

             index=["accuracy", "recall", "precision", "f1Score"])
model = LogisticRegression()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print("Logistic Regression Model Score for Train Data=", model.score(X_train, y_train)) #Train Classification accuracy

print("Logistic Regression Model Score for Test Data=",model.score(X_test, y_test)) #Test Classification accuracy
metrics.confusion_matrix(y_predict, y_test)
model.predict_proba(X_train)
y_predictProb = model.predict_proba(X_train)


fpr, tpr, thresholds = roc_curve(y_train, y_predictProb[::,1])



roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")
optimalF1 = 0

optimalTh = 0
thresholds[0:10]
for th in thresholds:

    preds = np.where(model.predict_proba(X_train)[:,1] > th, 1, 0)

    f1Score = f1_score(y_train, preds)

    if(optimalF1 < f1Score):

        optimalF1 = f1Score

        optimalTh = th


optimalF1
optimalTh
THRESHOLD = optimalTh


preds = np.where(model.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)
pd.DataFrame(data=[accuracy_score(y_test, preds), 

                   recall_score(y_test, preds), 

                   precision_score(y_test, preds),

                   f1_score(y_test, preds)], 

             index=["accuracy", "recall", "precision", "f1_score"])
model_naive = GaussianNB()

model_naive.fit(X_train, y_train)



# make predictions

predicted = model_naive.predict(X_test)

metrics.confusion_matrix(predicted, y_test)
print("Train Accuracy for Naive Bayes is = ", model_naive.score(X_train, y_train))

print("Test Accuracy for Naive Bayes is = ", model_naive.score(X_test, y_test))
y_predictProb = model_naive.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_predictProb[::,1])
roc_auc = auc(fpr, tpr)

roc_auc
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")
optimalF1 = 0

optimalTh = 0

for th in thresholds:

    preds = np.where(model_naive.predict_proba(X_test)[:,1] > th, 1, 0)

    f1Score = f1_score(y_test, preds)

    if(optimalF1 < f1Score):

        optimalF1 = f1Score

        optimalTh = th
optimalF1
optimalTh
THRESHOLD = optimalTh

preds = np.where(model_naive.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)
pd.DataFrame(data=[accuracy_score(y_test, preds), 

                   recall_score(y_test, preds),

                   precision_score(y_test, preds),

                   f1_score(y_test, preds)], 

             index=["accuracy", "recall", "precision", "f1Score"])
models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('NB', GaussianNB()))

models.append(('LR', LogisticRegression()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=12345)

	cv_results = model_selection.cross_val_score(model, X, y.values.ravel(), cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()