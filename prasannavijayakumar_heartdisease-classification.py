#import necessary modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()
#import the read the dataset

heart= pd.read_csv('../input/heart-disease/heart.csv')

heart.head()
#shape of the dataset

heart.shape
heart.columns
#summary of the dataset

heart.info()
#Checking missing value in dataset

heart.isnull().sum()
#Check for unique values in columns

heart.nunique()
#statistical summary of the dataset

heart.describe()
heart['target'].value_counts()
#convert columns to category

columns_to_category= ['sex', 'cp', 'fbs', 'restecg','exang', 'slope', 'ca', 'thal', 'target']



heart[columns_to_category]= heart[columns_to_category].astype('category')
heart.describe()
fig, ax = plt.subplots(figsize=(10,6))



sns.kdeplot(heart[heart["target"]==1]["age"], shade=True, color="blue", label="Presence of Heart Disease", ax=ax)

sns.kdeplot(heart[heart["target"]==0]["age"], shade=True, color="green", label="Absence of Heart Disease", ax=ax)



fig.suptitle("Presence of Heart Disease by Age")



ax.legend();
fig, ax = plt.subplots(figsize=(10,6))



sns.kdeplot(heart[heart["target"]==1]["trestbps"], shade=True, color="blue", label="Presence of Heart Disease", ax=ax)

sns.kdeplot(heart[heart["target"]==0]["trestbps"], shade=True, color="green", label="Absence of Heart Disease", ax=ax)



fig.suptitle("Presence of Heart Disease by Blood Sugar")



ax.legend();
fig, ax = plt.subplots(figsize=(10,6))



sns.kdeplot(heart[heart["target"]==1]["chol"], shade=True, color="blue", label="Presence of Heart Disease", ax=ax)

sns.kdeplot(heart[heart["target"]==0]["chol"], shade=True, color="green", label="Absence of Heart Disease", ax=ax)



fig.suptitle("Presence of Heart Disease based on Cholestrol")



ax.legend();
fig, ax = plt.subplots(figsize=(10,6))



sns.kdeplot(heart[heart["target"]==1]["thalach"], shade=True, color="blue", label="Presence of Heart Disease", ax=ax)

sns.kdeplot(heart[heart["target"]==0]["thalach"], shade=True, color="green", label="Absence of Heart Disease", ax=ax)



fig.suptitle("Presence of Heart Disease based on thalach")



ax.legend();
fig, ax = plt.subplots(figsize=(10,6))



sns.kdeplot(heart[heart["target"]==1]["oldpeak"], shade=True, color="blue", label="Presence of Heart Disease", ax=ax)

sns.kdeplot(heart[heart["target"]==0]["oldpeak"], shade=True, color="green", label="Absence of Heart Disease", ax=ax)



fig.suptitle("Presence of Heart Disease based on OldPeak")



ax.legend();
# Create the correlation matrix

corr = heart.corr()

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(14,10))

# Add the mask to the heatmap

sns.heatmap(corr, mask=mask, cmap='YlGnBu',center=0, linewidths=1, annot=True, fmt=".2f")

plt.show()
fig, ax = plt.subplots(5, 1, figsize=(10,20))

sns.boxplot(x='target', y='age', data=heart, ax=ax[0])

ax[0].set_xlabel('Target', fontsize=12)



sns.boxplot(x='target', y='trestbps', data=heart, ax=ax[1])

ax[1].set_xlabel('Target', fontsize=12)



sns.boxplot(x='target', y='chol', data=heart, ax=ax[2])

ax[2].set_xlabel('Target', fontsize=12)



sns.boxplot(x='target', y='thalach', data=heart, ax=ax[3])

ax[3].set_xlabel('Target', fontsize=12)



sns.boxplot(x='target', y='oldpeak', data=heart, ax=ax[4])

ax[4].set_xlabel('Target', fontsize=12)



plt.show()
fig = plt.figure(figsize=(15,10))

sns.pairplot(data=heart, vars=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], hue='target')

plt.subplots_adjust(top=0.9)

plt.show();
heart.var()
#split independent and dependent variable

X= heart.iloc[:, :13]

y= heart.iloc[:,-1]
#import necessary scikit learn libraries

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
category_cols= X.columns[X.dtypes == 'category']



encoded= pd.get_dummies(X[category_cols], drop_first=True)

encoded.head()
concat_X= pd.concat([X,encoded], axis=1)

concat_X.shape
final_X= concat_X.drop(['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thalach'], axis=1)

final_X.shape
X_train, X_test, y_train, y_test= train_test_split(final_X, y, random_state=99)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
scaler = MinMaxScaler()



lr = LogisticRegression(solver='liblinear', random_state=99) # Other solvers have failure to converge problem



pipeline = Pipeline([('scale',scaler), ('lr', lr),])



pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print ('Training accuracy: %.4f' % pipeline.score(X_train, y_train))
print ('Test accuracy: %.4f' % pipeline.score(X_test, y_test))
print ('Training accuracy score: %.4f' % accuracy_score(y_test, y_pred))
# Confusion matrix

pd.DataFrame(confusion_matrix(y_test, y_pred))
# Cross validation

from sklearn.metrics import make_scorer



scorer= make_scorer(accuracy_score)



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=99)

cv_score = cross_val_score(lr, X_train, y_train, cv=cv, scoring=scorer)

print('Cross validation accuracy score: %.3f' %np.mean(cv_score))
## Feature selection

from sklearn.feature_selection import RFECV



steps = 20

n_features = len(X_train.columns)

X_range = np.arange(n_features - (int(n_features/steps)) * steps, n_features+1, steps)



rfecv = RFECV(estimator=lr, step=steps, cv=cv, scoring=scorer)



pipeline2 = Pipeline([('scale',scaler), ('rfecv', rfecv)])

pipeline2.fit(X_train, y_train)

plt.figure(figsize=(10,6))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(np.insert(X_range, 0, 1), rfecv.grid_scores_)

plt.show()
print ('Optimal no. of features: %d' % np.insert(X_range, 0, 1)[np.argmax(rfecv.grid_scores_)])
from sklearn.model_selection import GridSearchCV



grid={"C":np.logspace(-2,2,5), "penalty":["l1","l2"]}

searcher_cv = GridSearchCV(lr, grid, cv=cv, scoring = scorer)

searcher_cv.fit(X_train, y_train)



print("Best parameter: ", searcher_cv.best_params_)

print("accuracy score: %.3f" %searcher_cv.best_score_)
from sklearn.metrics import roc_curve, auc



#compute predicted probabilities: y_pred_prob

y_pred_prob= pipeline.predict_proba(X_test)[:,1]





#Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Calculate the AUC



roc_auc = auc(fpr, tpr)

print ('ROC AUC: %0.3f' % roc_auc )



#Plot ROC curve

plt.figure(figsize=(10,8))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC



nb= MultinomialNB()

dt= DecisionTreeClassifier()

rf= RandomForestClassifier(n_estimators=100)

gbr= GradientBoostingClassifier()

svm= SVC(kernel='linear')
classifiers = [('Naive Bayes', nb),('Decision Tree Classifier', dt), ('RandomForest Classifier', rf), ('Gradient Boost', gbr), ('SVC', svm)]



# Iterate over the pre-defined list of regressors

for classifier_name, classifier in classifiers:   

    # Fit clf to the training set

    classifier.fit(X_train, y_train)    

    y_pred = classifier.predict(X_test) 

    

    training_set_score = classifier.score(X_train, y_train)

    test_set_score = classifier.score(X_test, y_test)

    

    

    print('{:s} : {:.3f}'.format(classifier_name, training_set_score))

    print('{:s} : {:.3f}'.format(classifier_name, test_set_score))