#from mlxtend.plotting import plot_decision_regions

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_style('whitegrid')



import warnings

warnings.filterwarnings('ignore')
diabetes_data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
# View top 5 rows of our dataset

diabetes_data.head()
diabetes_data.shape
## Lets check data types,columns names, null value counts, memory usage etc

diabetes_data.info(verbose=True)
# Get the details of each column

diabetes_data.describe().T
diabetes_data['Pregnancies'].value_counts()
fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (8,6))



plot00=sns.distplot(diabetes_data['Pregnancies'],ax=axes[0],color='b')

axes[0].set_title('Distribution of Pregnancy',fontdict={'fontsize':8})

axes[0].set_xlabel('No of Pregnancies')

axes[0].set_ylabel('Frequency')

plt.tight_layout()





plot01=sns.boxplot('Pregnancies',data=diabetes_data,ax=axes[1],orient = 'v', color='r')

plt.tight_layout()
# Replace zeros with NaN

diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
diabetes_data.head()
total = diabetes_data.isnull().sum().sort_values(ascending=False)

percent = ((diabetes_data.isnull().sum()/diabetes_data.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(9)
f, ax = plt.subplots(figsize=(12, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
diabetes_data.corr()
plt.figure(figsize=(12,10))  

# sns.heatmap(diabetes_data.corr(), annot=True,cmap ='RdYlGn')

sns.heatmap(diabetes_data.corr(),annot=True, cmap='viridis',linewidths=.1)

plt.show()
# Check the distribution of each column, so that we can find wich is best central tendency (mean, medium or mode) to replace missing values:

diabetes_data.hist(figsize = (20,20))

plt.show()
diabetes_data['Glucose'].fillna(diabetes_data['Glucose'].mean(), inplace = True)

diabetes_data['BloodPressure'].fillna(diabetes_data['BloodPressure'].mean(), inplace = True)

diabetes_data['SkinThickness'].fillna(diabetes_data['SkinThickness'].median(), inplace = True)

diabetes_data['Insulin'].fillna(diabetes_data['Insulin'].median(), inplace = True)

diabetes_data['BMI'].fillna(diabetes_data['BMI'].median(), inplace = True)
diabetes_data.hist(figsize = (20,20))

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(x='Outcome',data=diabetes_data, palette='bright')

plt.title("Emergency call category")



print(diabetes_data['Outcome'].value_counts())
sns.pairplot(diabetes_data,hue='Outcome')
plt.figure(figsize=(12,8))

sns.boxplot(x='Pregnancies', y='BMI',data=diabetes_data, hue='Outcome')
plt.figure(figsize=(12,8))

sns.boxplot(x='Outcome', y='BMI',data=diabetes_data, hue='Outcome')
plt.figure(figsize=(18,10))

sns.countplot(x='Pregnancies',data=diabetes_data,hue = 'Outcome', palette='bright')
plt.figure(figsize=(12,8))

sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction',data=diabetes_data)
plt.figure(figsize=(12,8))

sns.boxplot(x='Outcome', y='Pregnancies',data=diabetes_data)
plt.figure(figsize=(12,8))

sns.boxplot(x='Outcome', y='BMI',data=diabetes_data)
normalBMIData = diabetes_data[(diabetes_data['BMI'] >= 18.5) & (diabetes_data['BMI'] <= 25)]

normalBMIData['Outcome'].value_counts()
notNormalBMIData = diabetes_data[(diabetes_data['BMI'] < 18.5) | (diabetes_data['BMI'] > 25)]

notNormalBMIData['Outcome'].value_counts()
plt.figure(figsize=(12,8))

sns.boxplot(x='Outcome', y='BMI',data=notNormalBMIData)
plt.figure(figsize=(12,8))

sns.boxplot(x='Outcome', y='Age',data=diabetes_data)
diabetes_data['Age'].value_counts().head()
plt.figure(figsize=(18,10))

sns.countplot(x='Age',data=diabetes_data,hue = 'Outcome', palette='bright')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(diabetes_data.drop('Outcome',axis=1))
scaled_features = scaler.transform(diabetes_data.drop('Outcome',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=diabetes_data.columns[:-1])
df_feat.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,diabetes_data['Outcome'],

                                                    test_size=0.30,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []

test_scores = []

train_scores = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    

    error_rate.append(np.mean(pred_i != y_test))

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
plt.figure(figsize=(12,8))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
## score that comes from testing on the same datapoints that were used for training

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely

max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
# NOW WITH K=20

knn = KNeighborsClassifier(n_neighbors=20)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=20')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
plt.figure(figsize=(20,8))

sns.lineplot(range(1,40),train_scores,marker='*',label='Train Score')

sns.lineplot(range(1,40),test_scores,marker='o',label='Test Score')
#Setup a knn classifier with k neighbors

knn = KNeighborsClassifier(20)



knn.fit(X_train,y_train)

knn.score(X_test,y_test)
#import confusion_matrix

from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above

y_pred = knn.predict(X_test)

confusion_matrix(y_test,y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn import metrics



cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="viridis" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
#import classification_report

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# Printing the Overall Accuracy of the model

print("Accuracy of the model : {0:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))
from sklearn.metrics import roc_curve

y_pred_proba = knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8,6))

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=11) ROC curve')

plt.show()
#Area under ROC curve

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,y_pred_proba)