import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

cancer = pd.read_csv("../input/data.csv")

cancer.head()
cancer.describe()
cancer.drop(['id','Unnamed: 32'], axis=1, inplace=True)

cancer.columns
cancer.isnull().sum().sort_values(ascending=False)
print('counts of Malignant and Benign \n',cancer['diagnosis'].value_counts())

sns.countplot(cancer['diagnosis'],palette="Blues")

plt.title('Distribution of Malignant & Benign')
features_mean = cancer[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]

features_se = cancer[['diagnosis','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se']]

features_worst = cancer[['diagnosis','radius_worst', 'texture_worst','perimeter_worst', 'area_worst', 

'smoothness_worst','compactness_worst', 'concavity_worst', 'concave points_worst','symmetry_worst',

                   'fractal_dimension_worst']]
mean_correlation = features_mean.corr()

plt.figure(figsize=(8,8))

sns.heatmap(mean_correlation,vmax=1,square=True,annot=True,cmap='Greens')
plt.subplot(221)

sns.violinplot(x='diagnosis',y='texture_mean',data=features_mean,palette="Greens",inner="quartile")

plt.subplot(222)

sns.violinplot(x='diagnosis',y='concavity_mean',data=features_mean,palette="Greens",inner="quartile")

plt.subplot(223)

sns.violinplot(x='diagnosis',y='radius_mean',data=features_mean,palette="Greens",inner="quartile")

plt.subplot(224)

sns.violinplot(x='diagnosis',y='fractal_dimension_mean',data=features_mean,palette="Greens",inner="quartile")

plt.show()
se_correlation = features_se.corr()

plt.figure(figsize=(8,8))

sns.heatmap(se_correlation,vmax=1,square=True,annot=True,cmap='Oranges')
plt.subplot(221)

sns.violinplot(x='diagnosis',y='texture_se',data=features_se,palette="Oranges",inner="quartile")

plt.subplot(222)

sns.violinplot(x='diagnosis',y='concavity_se',data=features_se,palette="Oranges",inner="quartile")

plt.subplot(223)

sns.violinplot(x='diagnosis',y='radius_se',data=features_se,palette="Oranges",inner="quartile")

plt.subplot(224)

sns.violinplot(x='diagnosis',y='fractal_dimension_se',data=features_se,palette="Oranges",inner="quartile")

plt.show()
worst_correlation = features_worst.corr()

plt.figure(figsize=(8,8))

sns.heatmap(worst_correlation,vmax=1,square=True,annot=True,cmap='Reds')
plt.subplot(221)

sns.violinplot(x='diagnosis',y='texture_worst',data=features_worst,palette="Reds",inner="quartile")

plt.subplot(222)

sns.violinplot(x='diagnosis',y='smoothness_worst',data=features_worst,palette="Reds",inner="quartile")

plt.subplot(223)

sns.violinplot(x='diagnosis',y='concavity_worst',data=features_worst,palette="Reds",inner="quartile")

plt.subplot(224)

sns.violinplot(x='diagnosis',y='concave points_worst',data=features_worst,palette="Reds",inner="quartile")

plt.show()
pairplot = cancer[['diagnosis','radius_worst','area_worst','perimeter_worst']]

sns.pairplot(pairplot,hue='diagnosis',palette="Blues_d")

plt.show()
plt.subplot(221)

sns.swarmplot(x='diagnosis',y='area_worst',data=pairplot,palette="Blues_d")

plt.subplot(222)

sns.swarmplot(x='diagnosis',y='radius_worst',data=pairplot,palette="Blues_d")

plt.subplot(223)

sns.swarmplot(x='diagnosis',y='perimeter_worst',data=pairplot,palette="Blues_d")

plt.show()
plt.subplot(221)

sns.swarmplot(x='diagnosis',y='compactness_worst',data=features_worst,palette="Blues_d")

plt.subplot(222)

sns.swarmplot(x='diagnosis',y='concavity_worst',data=features_worst,palette="Blues_d")

plt.subplot(223)

sns.swarmplot(x='diagnosis',y='concave points_worst',data=features_worst,palette="Blues_d")

plt.show()
features_corr = cancer[['diagnosis','radius_mean', 'texture_mean', 'smoothness_mean', 'concavity_mean','symmetry_mean',

       'radius_se', 'texture_se', 'smoothness_se','concavity_se', 'symmetry_se','radius_worst', 'texture_worst',

       'smoothness_worst','concavity_worst','symmetry_worst']]

features_correlation = features_corr.corr()

plt.figure(figsize=(10,10))

sns.heatmap(features_correlation,vmax=1,square=True,annot=True,cmap='Blues')

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



features_corr['diagnosis'] = [0 if x == 'B' else 1 for x in features_corr['diagnosis']]

X = features_corr.drop(['diagnosis'],axis = 1 )

y = features_corr.diagnosis



X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.3,random_state=42)

rfc = RandomForestClassifier(n_estimators=10)

rfc.fit(X_train,y_train)

print('Accuracy score',rfc.score(X_test,y_test))
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



y_pred = rfc.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)

sns.heatmap(confusion_matrix,annot=True,fmt='')

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('True')

print('Classification Report')

print(classification_report(y_test,y_pred))



from sklearn.model_selection import cross_val_score

import numpy as np

rfc = RandomForestClassifier()

cv_results = cross_val_score(rfc,X,y,cv=5)

print(cv_results)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_results)))
from sklearn.metrics import roc_curve

rfc.fit(X_train,y_train)

y_pred_prob  =  rfc.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr,label='random forest')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
df = pd.read_csv("../input/data.csv")

df['diagnosis'] = [0 if x == 'B' else 1 for x in df['diagnosis']]

y = df.diagnosis        

list = ['Unnamed: 32','id','diagnosis']

X = df.drop(list,axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

forest = RandomForestClassifier()

forest.fit(X_train,y_train)

print('train score',forest.score(X_train,y_train))

print('test score',forest.score(X_test,y_test))
corr=df.corr()['diagnosis']

corr[np.argsort(corr,axis=0)[::-1]]
features = X.columns

for name, importance in zip(features, forest.feature_importances_):

    print(name, "=", importance)



importances = forest.feature_importances_

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), features) ## removed [indices]

plt.xlabel('Relative Importance')

plt.show()
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

y_pred = forest.predict(X_test)

confusion_matrix = confusion_matrix(y_test,y_pred)

sns.heatmap(confusion_matrix,annot=True,fmt='')

plt.xlabel('Predicted')

plt.ylabel('True')

print('Classification Report')

print(classification_report(y_test,y_pred))



from sklearn.model_selection import cross_val_score

import numpy as np

cv_results = cross_val_score(forest,X,y,cv=5)

print(cv_results)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_results)))
from sklearn.metrics import roc_curve

rfc.fit(X_train,y_train)

y_pred_prob  =  forest.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr,label='random forest')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()