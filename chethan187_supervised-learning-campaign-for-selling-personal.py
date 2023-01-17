import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



plt.show()

%matplotlib inline



sns.set(style='ticks')

df = pd.read_csv('../input/Bank_Personal_Loan_Modelling.csv') #read the data as dataframe
df.columns #To list all the columns in the data
print("The data has Rows {}, Columns {}".format(*df.shape))     #Find the shape of the data
df.info()   #Info about the data
df.apply(lambda x: sum(x.isnull())) #Check for any null values 
df.describe().T   #Gets the 5 point sumary
df.head() #To view top 5 rows of the data 
sns.pairplot(df.iloc[:,1:]) #pairplot, leaving out the column 'ID'
plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

plt.hist(df.Age, color='deepskyblue', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Age')

plt.title('Age Distribution')



plt.subplot(3,3,2)

plt.hist(df.Experience, color='lightblue', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Experience')

plt.title('Experience Distribution')



plt.subplot(3,3,3)

plt.hist(df.Income, color='gold', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Income')

plt.title('Income Distribution')



print('\n')



plt.subplot(3,3,4)

plt.hist(df.CCAvg, color='lime', edgecolor = 'black', alpha = 0.7)

plt.xlabel('CCAvg')

plt.title('CCAvg Distribution')



plt.subplot(3,3,5)

plt.hist(df.Mortgage, color='salmon', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Mortgage')

plt.title('Mortgage Distribution')



plt.show()
sns.countplot(x='Family', hue='Personal Loan', data=df)

plt.title('Family Countplot')



plt.show()
Family3_personalloan = df[df['Family'] == 3]['Personal Loan'].value_counts()   #To get the frequency distribution counts 

Family4_personalloan = df[df['Family'] == 4]['Personal Loan'].value_counts()   #for Family size 3 and 4 
labels = 'Without Personal Loan', 'With Personal Loan'

colors = ['lightblue', 'orange']

explodeTuple = (0.1, 0.0)



fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,6))

fig.suptitle('Personal loan chart among people with different family sizes', fontsize=16, y=1.1)



ax1.pie(Family3_personalloan, labels=labels, autopct = '%.1f%%',colors=colors, startangle=90, explode = explodeTuple)

plt.tight_layout()

ax1.set_title('Family 3', fontsize=15) 



labels = 'Without Personal Loan', 'With Personal Loan'

colors = ['lightblue', 'orange']

explodeTuple = (0.1, 0.0)



ax2.pie(Family4_personalloan, labels=labels, autopct = '%.1f%%',colors=colors, startangle=90, explode = explodeTuple)

plt.tight_layout()

ax2.set_title('Family 4', fontsize=15)



plt.show()
Family1 = df[df['Family'] == 1]['Personal Loan'].value_counts()

Family1 = Family1[1]



Family2 = df[df['Family'] == 2]['Personal Loan'].value_counts()

Family2 = Family2[1]



Family3 = df[df['Family'] == 3]['Personal Loan'].value_counts()

Family3 = Family3[1]



Family4 = df[df['Family'] == 4]['Personal Loan'].value_counts()

Family4 = Family4[1]
FamilyandPersonalloan = Family1, Family2, Family3, Family4
FamilyandPersonalloan
plt.figure(figsize=(12,5))

labels = ['Family 1',' Family 2','Family 3', 'Family 4']

explodeTuple = (0.1, 0.1, 0.1, 0.1)



plt.pie(FamilyandPersonalloan, labels=labels, autopct = '%.1f%%',colors=colors, startangle=90, explode = explodeTuple)

plt.tight_layout()

plt.title('Proportion of customers having personal loans with different family size', fontsize=20)

plt.show()
sns.countplot(x='Education', hue='Personal Loan', data=df)

plt.title('Education Countplot')



plt.show()
PL_edu1 = df[df['Education'] == 1]['Personal Loan'].value_counts()

PL_edu2 = df[df['Education'] == 2]['Personal Loan'].value_counts()

PL_edu3 = df[df['Education'] == 3]['Personal Loan'].value_counts()
labels = 'With Personal Loan', 'Without Personal Loan'

colors = ['mediumturquoise', 'lightgreen']

explodeTuple = (0.2, 0.0)



fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15,6))

fig.suptitle('Personal loan chart among people for different education background', fontsize=16, y=1.1)



ax1.pie(PL_edu1, labels=labels, autopct = '%.1f%%',colors=colors, startangle=90, explode = explodeTuple)

plt.tight_layout()

ax1.set_title('Education Level 1', fontsize=15) 



ax2.pie(PL_edu2, labels=labels, autopct = '%.1f%%',colors=colors, startangle=90, explode = explodeTuple)

plt.tight_layout()

ax2.set_title('Education Level 2', fontsize=15) 



ax3.pie(PL_edu3, labels=labels, autopct = '%.1f%%',colors=colors, startangle=90, explode = explodeTuple)

plt.tight_layout()

ax3.set_title('Education Level 3', fontsize=15) 



plt.show()
Education_personalloan = df[df['Personal Loan'] == 1]['Education'].value_counts()
plt.figure(figsize=(12,5))

labels = ['Education Level  3',' Education Level 2','Education Level 1']

explodeTuple = (0.1, 0.0, 0.0)



plt.pie(Education_personalloan, labels=labels, autopct = '%.1f%%',colors=colors, startangle=90, explode = explodeTuple)

plt.tight_layout()

plt.title('Proportion of customers having personal with education levels', fontsize=20)

plt.show()
loan_counts = pd.DataFrame(df['Personal Loan'].value_counts()).reset_index()

loan_counts.columns = ['Labels', 'Personal Loan']
plt.figure(figsize=(13,6))

explode = (0,0.10)



plt.pie(loan_counts['Personal Loan'], labels=loan_counts['Labels'], explode=explode,  autopct='%1.1f%%',shadow=True, 

        startangle=90)

#fig.axis('equal')

plt.title('Personal Load Distribution')

plt.show()
corr = df.corr()

fig = plt.figure(figsize=(11,6))

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,len(df.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df.columns)

ax.set_yticklabels(df.columns)

plt.show()
df[['Income', 'CCAvg', 'Education', 'Mortgage', 'CD Account', 'Personal Loan']].corr() 
sns.boxplot(data=df, x='Education', y='Income', hue='Personal Loan', color='navy')

plt.title('Education and Income')



plt.show()
sns.boxplot(data=df, x='Education', y='Mortgage', hue='Personal Loan', color='forestgreen')

plt.legend(loc = 'upper center')

plt.title('Education and Mortgage')



plt.show()
sns.countplot(x='CD Account', data=df, hue='Personal Loan')

plt.title('CD Account Countplot')



plt.show()
sns.boxplot(data=df, x='CD Account', y='CCAvg', hue='Personal Loan', color='sienna')

plt.legend(loc= 'upper center')

plt.title('Family and CD Account Boxplot')



plt.show()
sns.boxplot(data=df, x='Family', y='Income', hue='Personal Loan', color='darkviolet')

plt.legend(loc = 'upper center')

plt.title('Family and Income for Personal Loan Boxplot')



plt.show()
sns.countplot(data=df, x='Securities Account', hue='Personal Loan')

plt.title('Securities Account Countplot')



plt.show()
sns.countplot(data=df, x='Online')

plt.title('Online Countplot')



plt.show()
sns.countplot(data=df, x='CreditCard')

plt.title('CreditCard Countplot')



plt.show()
X = df[['Income', 'CCAvg', 'Mortgage', 'Education', 'CD Account']]

y = df['Personal Loan']
from sklearn import preprocessing
X = preprocessing.scale(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=100)
from sklearn import metrics

from sklearn.metrics import confusion_matrix



from sklearn.naive_bayes import GaussianNB



naive_model = GaussianNB()

naive_model.fit(X_train, y_train)

NB_predict = naive_model.predict(X_test)
print('**Accuracy for Naive Bayes classifier is:', metrics.accuracy_score(y_test, NB_predict))

print('**F1_score for Logistic Regression classifier is:', metrics.f1_score(y_test, NB_predict))

print('\n')

print('**Confusion matrix', '\n', confusion_matrix(y_test, NB_predict))
confusion_matrix_NB = confusion_matrix(y_test, NB_predict)
from mlxtend.plotting import plot_confusion_matrix
plot_confusion_matrix(conf_mat=confusion_matrix_NB, show_absolute=True, show_normed=True, colorbar=True)

plt.title('Confusion Matrix for Naive Bayes')

plt.show()
from sklearn.neighbors import KNeighborsClassifier
error_rate = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',

        markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
#FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1



knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)

metrics.accuracy_score(y_test, pred)
#Now with K=5



KNN_model = KNeighborsClassifier(n_neighbors=5, weights = 'distance')

KNN_model.fit(X_train, y_train)

knn_predict = KNN_model.predict(X_test)



print('**Accuracy for KNN classifier is:', metrics.accuracy_score(y_test, knn_predict))

print('**F1_score for KNN classifier is:', metrics.f1_score(y_test, knn_predict))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, knn_predict))
confusion_matrix_knn = confusion_matrix(y_test, knn_predict)



plot_confusion_matrix(conf_mat=confusion_matrix_knn, show_absolute=True, show_normed=True, colorbar=True)

plt.title('Confusion Matrix for K-NN')



plt.show()
from sklearn.linear_model import LogisticRegression



logit_model = LogisticRegression()

logit_model.fit(X_train, y_train)



logit_predict = logit_model.predict(X_test)



print('Accuracy for Logistic Regression model is:', metrics.accuracy_score(y_test, logit_predict))

print('F1_score for Logistic Regression model is:', metrics.f1_score(y_test, logit_predict))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, logit_predict))
confusion_matrix_LR = confusion_matrix(y_test, logit_predict)



plot_confusion_matrix(conf_mat=confusion_matrix_LR, show_absolute=True, show_normed=True, colorbar=True)

plt.title('Confusion Matrix for Logistic Regression')



plt.show()