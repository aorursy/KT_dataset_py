# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()
df= pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
print(df.shape)
df.columns
df.info()
df.describe(include='all')
df= df.drop(['Name', 'PassengerId', 'Ticket'], axis=1)

df.head()
# Let's get unique values for each category

unique_vals = {

    k: df[k].unique()

    for k in df.columns

}



unique_vals
df.dtypes
df['Survived'] = df['Survived'].astype('category')

df['Pclass'] = df['Pclass'].astype('category')

df['Sex'] = df['Sex'].astype('category')

df['Embarked'] = df['Embarked'].astype('category')
numerical_cols= df.columns[df.dtypes == 'float']



# Categorical columns are the following

categorical_cols = [x for x in df.columns if x not in numerical_cols]



print(f"numerical columns = {numerical_cols}")

print(f"categorical columns = {categorical_cols}")
df.isnull().sum()
df= df.drop(['Cabin'], axis=1)
# From above, it is clearly evident that missing values are present in many columns



# Filling the missing values of numeric columns with median value

df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())



# Filling the missing values of categorical column with the most frequent value from one column. 

#Embarked Column have 2 missing Values. We can fill it with maximum frequency value of 'S'

df['Embarked'].fillna('S', inplace=True)



# Checking no more NULLs in the data

all(df.isna().sum() == 0)
df.dtypes
df['Age'].hist()
# Plot for survival vs Age

age_survival = sns.catplot(x="Survived", y="Age", data=df)

age_survival.fig.suptitle('Survival based on age')
sns.pairplot(df, hue='Survived', diag_kind='hist')

plt.show()
df.hist(figsize=(12,6))

plt.show()
sns.countplot(x='Embarked', data=df, hue='Survived')

plt.show()
station=df.groupby('Sex')['Embarked'].value_counts().reset_index(name='count')

station
sns.barplot(x='Embarked', y='count', data=station, hue='Sex')

plt.show()
sns.countplot(x='Parch', hue='Survived', data=df)

plt.legend(loc='upper right')

plt.show()
sns.barplot(x='Pclass', y='Fare', data=df)

plt.show()
df.boxplot(figsize=(8,6))

plt.show()
df.describe(include='all')
df.corr()
df.var()
df['TravelAlone']=np.where((df["SibSp"]+df["Parch"])>0, 0, 1)

df.head()
print(df.shape)
#PClass, SibSp, Parch belongs to ordinal category

#Perform Onehode Encoding only on Sex and Embarked though it is nominal categorical colummns



encoded= pd.get_dummies(df[['Sex', 'Embarked']], drop_first=True)

encoded.head()
df_final= pd.concat([df, encoded], axis=1)

df_final= df_final.drop(['Sex','Embarked','SibSp','Parch'], axis=1)

df_final.head()
#seperate the training and test set



y= df_final['Survived']

X= df_final.drop('Survived', axis=1)



y.value_counts()
from sklearn.model_selection import train_test_split, cross_val_score



#Lets divide the data-set into training and test-set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30, test_size=0.30,stratify=y)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler



scaler= StandardScaler()



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)



X_train, X_test
from sklearn.linear_model import LogisticRegression



seed=3

#Instantiate Logistic Regression model

logreg= LogisticRegression(solver='lbfgs', max_iter=1800, random_state=seed)
# Compute 5-fold cross-validation scores: cv_scores

cv_scores= cross_val_score(logreg, X, y, cv=5)



# Print the 5-fold cross-validation scores

print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
#Fit the logistic regression model to training data

logreg.fit(X_train, y_train)
#Check Training and Test Set Accuracy



training_accuracy= logreg.score(X_train, y_train)

test_accuracy= logreg.score(X_test, y_test)



print(f"Training Set accuracy = {training_accuracy}")

print(f"Test Set accuracy = {test_accuracy}")
# Coefficients of the model and its intercept

print(logreg.coef_)

print(logreg.intercept_)
# Coefficients of the model and its intercept

print(dict(zip(X.columns, abs(logreg.coef_[0]).round(2))))

print(logreg.intercept_)
from sklearn.feature_selection import RFE

from sklearn.metrics import accuracy_score



# Create the RFE with a LogisticRegression estimator and 5 features to select

rfe = RFE(estimator=logreg, n_features_to_select=5, verbose=1)

# Fits the eliminator to the data

rfe.fit(X_train, y_train)

# Print the features and their ranking (high = dropped early on)

print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated

print(X.columns[rfe.support_])

# Calculates the test set accuracy

acc = accuracy_score(y_test, rfe.predict(X_test))

print("{0:.1%} accuracy on test set.".format(acc))
from sklearn.model_selection import GridSearchCV



# Instantiate the GridSearchCV object and run the search

searcher = GridSearchCV(logreg, {'C':[0.001, 0.01, 0.1, 1, 10]})

searcher.fit(X_train, y_train)

# Report the best parameters

print("Best CV params", searcher.best_params_)
selected_features= ['Pclass', 'Age', 'TravelAlone', 'Sex_male', 'Embarked_S']



#seperate the training and test set



X_RFE= df_final[selected_features]

y= df_final['Survived']
from sklearn.model_selection import train_test_split, cross_val_score



#Lets divide the data-set into training and test-set

train_X, test_X, train_y, test_y = train_test_split(X_RFE, y, random_state=30, stratify=y)



train_X.shape, test_X.shape, train_y.shape, test_y.shape
#Instantiate Logistic Regression model

model= LogisticRegression(C=0.1)



#Fit the logistic regression model to training data

model.fit(train_X, train_y)
#Predictions on Test set

y_pred= model.predict(test_X)

y_pred
# Making the confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



cm = confusion_matrix(test_y,y_pred)

acc_score = accuracy_score(test_y, y_pred)



print(f"Accuracy = {acc_score*100:.2f}%")

print(f"Confusion matrix = \n{cm}")
#Check Training and Test Set Accuracy



training_accuracy= model.score(train_X, train_y)

test_accuracy= model.score(test_X, test_y)



print(f"Training Set accuracy = {training_accuracy}")

print(f"Test Set accuracy = {test_accuracy}")
from sklearn.metrics import roc_curve, auc



#compute predicted probabilities: y_pred_prob

y_pred_prob= model.predict_proba(test_X)[:,1]





#Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(test_y, y_pred_prob)



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
test_df= pd.read_csv('/kaggle/input/titanic/test.csv')

test_df.head()
test_df.isnull().sum()
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

test_df.isnull().sum()
test_df['TravelAlone']=np.where((test_df["SibSp"]+test_df["Parch"])>0, 0, 1)



valid_data= test_df.drop(["SibSp", "Parch"], axis=1)

valid_data.head()
test_data=valid_data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

test_data.head()
encoded_test= pd.get_dummies(test_data[['Sex', 'Embarked']], drop_first=True)

encoded_test.head()
df_test= pd.concat([test_data, encoded_test], axis=1)

df_test= df_test.drop(['Sex','Embarked'], axis=1)

df_test.head()
df_test.isnull().sum()
df_test['Survived'] = model.predict(df_test[selected_features])

df_test


df_test['PassengerId'] = test_df['PassengerId']



submission= df_test[['PassengerId','Survived']]



submission.to_csv("submission.csv", index=False)

submission.head()