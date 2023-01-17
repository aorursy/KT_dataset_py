import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_raw  = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df_raw.head()

df_raw.shape

#checkinning if there is any null value

df_raw.isnull().sum()

plt.subplots(figsize=(10,7))

sns.heatmap(df_raw.corr(), annot= True, square= True)

sns.distplot(df_raw['pH'], label ='pH')


df_raw['quality'].value_counts()



plt.subplots(figsize=(5,7))

sns.countplot(df_raw['quality'])
# sns.kdeplot(df_raw['quality'], shade= True, label ='quality')

sns.kdeplot(df_raw['sulphates'], shade= True, label ='sulphates')

sns.distplot(df_raw['fixed acidity'], label ='fixed acidity')

# sns.distplot(df_raw['citric acid'], label ='citric acid')
sns.scatterplot(x= df_raw['fixed acidity'], y= df_raw['pH'])

sns.scatterplot(x= df_raw['sulphates'], y= df_raw['pH'],  sizes=(10, 200))

sns.scatterplot(x= df_raw['chlorides'], y= df_raw['pH'],  sizes=(10, 200))

sns.scatterplot(x= df_raw['chlorides'], y= df_raw['sulphates'],  sizes=(10, 200))

sns.scatterplot(x= df_raw['pH'], y= df_raw['sulphates'],  sizes=(10, 200))

x1 = pd.Series(df_raw['sulphates'], name="$X_1$")

x2 = pd.Series(df_raw['pH'], name="$X_2$")



# Show the joint distribution using kernel density estimation

sns.jointplot(x1, x2, kind="kde", height=7, space=0)
sns.scatterplot(x= df_raw['pH'], y= df_raw['quality'],  sizes=(10, 200))

# anova test to  see if these two are independet or not

sns.boxplot(x= df_raw['quality'], y= df_raw['pH'])
# sns.kdeplot(df_raw['sulphates'], shade= True, label ='sulphates')

# sns.kdeplot(df_raw['Chlorides'], shade= True, label ='chlorides')
# df_raw.groupby(['quality','fixed acidity','pH']).sum()

df_raw.groupby('quality').sum()

bins = (2, 6.5, 8)

group_names = ['bad', 'good']

df_raw['quality'] = pd.cut(df_raw['quality'], bins = bins, labels = group_names)

df_raw.head()

df_raw.groupby('quality').sum()
X = df_raw.drop('quality', axis= 1) #data

Y = df_raw.quality #label



#splitting the dataset 

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV 



X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=40)

###Random Forest



from sklearn.ensemble import RandomForestClassifier

model_forest = RandomForestClassifier(n_estimators =10, random_state=30)

model_forest.fit(X_train, Y_train)





# In[126]:





predict =model_forest.predict(X_test)



from sklearn import metrics

print ('Accuracy:', metrics.accuracy_score(Y_test,predict ))

print(model_forest.feature_importances_)



feature_list = list(X.columns)

feature_imp = pd.Series(model_forest.feature_importances_, index= feature_list,)



print(feature_imp)
 ###### K Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



model_knn = KNeighborsClassifier() #n_neighbors = 3

model_knn.fit(X_train, Y_train)



predict_knn =model_knn.predict(X_test)



print ('Accuracy:', metrics.accuracy_score(Y_test,predict_knn ))

# print(model_knn.feature_importances_)
##### Logistic regression 



from sklearn.linear_model import LogisticRegression

model_logreg = LogisticRegression(solver = 'lbfgs')

model_logreg.fit(X_train, Y_train)



predict_logreg = model_logreg.predict(X_test)

print ('Accuracy:', metrics.accuracy_score(Y_test,predict_logreg ))

##### decision Tree



# In[185]:





from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

model_dec3 = DecisionTreeClassifier()

'''

param_grid = {

    'criterion': ['gini','entropy'],

    'max_depth': [None, 1, 2, 3, 4, 5, 6],

    'max_features': ['auto', 'sqrt','log2'],

    'max_leaf_nodes': [None, 1, 2, 3, 4, 5, 6],

    'min_samples_leaf': [1,2,3,4,5,6,7],

    'min_samples_split': [2,3,4,5,6,7,8,9,10]

}

# Doing Gridsearch to find optimal parameters

grid_dec3 = GridSearchCV(estimator=model_dec3, param_grid=param_grid, scoring='accuracy',cv=5, n_jobs=-1)

'''

model_dec3.fit(X_train, Y_train)

predict_dec3 = model_dec3.predict(X_test)



print ('Accuracy:', metrics.accuracy_score(Y_test,predict_dec3 ))





# Claisisfication report

print(classification_report(Y_test, predict_dec3))



'''

Y_prob = grid_dec3.predict_proba(X_test)[:,1]



#Create true and false positive rates

false_positive_rate_log,true_positive_rate_log,threshold_log = roc_curve(Y_test,Y_prob)



#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_log,true_positive_rate_log, linewidth=2)

plt.plot([0,1],ls='--', linewidth=2)

plt.plot([0,0],[1,0],c='.5', linewidth=2)

plt.plot([1,1],c='.5', linewidth=2)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()



'''