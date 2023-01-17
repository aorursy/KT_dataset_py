import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Image
import os
!ls ../input/
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.shape
df.describe()
df.isna().sum()
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.isna().sum()
df['Glucose'].fillna(df['Glucose'].median(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)
ax = sns.catplot('Outcome', data=df, kind='count')
ax.set(xticklabels = ["No", "Yes"])

plt.ylabel('Number of Case')
plt.title('Diabete Distribution')

#the data is good? should we get an equal distribution of the sick people
import matplotlib.pyplot as plt
%matplotlib inline

df.hist(figsize=(10,20))
corr=df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.set(font_scale=1.15)
plt.figure(figsize=(16, 9))

sns.heatmap(corr, mask=mask,square=True,annot=True,cmap='YlGnBu',)

plt.title('Correlation between features')
#### add some graph that compare the outcome to the most important feature
#pregnancy, glucose, BMI, age
X = df.drop(['Outcome'], axis = 1)
y = df['Outcome']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
column = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler(with_mean=True, with_std=True)
dfnumss = pd.DataFrame(ss.fit_transform(X_train[column]), columns=['ss_'+x for x in column], index = X_train.index)
X_train = pd.concat([X_train, dfnumss], axis=1)
X_train = X_train.drop(column, axis=1)
dfnumss = pd.DataFrame(ss.transform(X_test[column]), columns=['ss_'+x for x in column], index = X_test.index)
X_test = pd.concat([X_test, dfnumss], axis=1)
X_test = X_test.drop(column, axis=1)
from sklearn.svm import SVC

svc = SVC()     
svc.fit(X_train, y_train)                                                        
y_pred = svc.predict(X_test)    

from sklearn import metrics

print (metrics.accuracy_score(y_test, y_pred))
print (metrics.confusion_matrix(y_test, y_pred))
print (metrics.classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test) 

print (metrics.accuracy_score(y_test, y_pred))
print (metrics.confusion_matrix(y_test, y_pred))
print (metrics.classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier                                                            

knn = KNeighborsClassifier() 
knn.fit(X_train, y_train)                                                                                                            
y_pred = knn.predict(X_test) 
print (metrics.accuracy_score(y_test, y_pred))
print (metrics.confusion_matrix(y_test, y_pred))
print (metrics.classification_report(y_test, y_pred))

from sklearn.feature_selection import SelectFromModel

from sklearn.inspection import permutation_importance
from matplotlib import pyplot


knn.fit(X_train, y_train)
results = permutation_importance(knn, X_train, y_train, scoring='accuracy')
importance = results.importances_mean

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

pyplot.barh([x for x in range(len(importance))],importance, color = 'green', align='center')
plt.ylabel("The Variables")
plt.xlabel("Score")
plt.title("Feature Importance from KNN")
pyplot.show()

