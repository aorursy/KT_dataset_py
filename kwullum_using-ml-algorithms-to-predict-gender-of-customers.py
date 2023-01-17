import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('darkgrid')
columns = ['CustomerID', 'Gender', 'Age', 'Income', 'Score']

df = pd.read_csv('../input/Mall_Customers.csv', index_col='CustomerID', names=columns, header=0)

df = df[['Age', 'Income', 'Score', 'Gender']] # Putting Gender (target variable) at the end

df.head()
df.shape
df.describe()
# % share of gender in dataset

df.Gender.value_counts(normalize=True)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,4))



female = df[df.Gender == 'Female']

male = df[df.Gender == 'Male']



sns.distplot(female.Age, bins=12 ,ax=ax1)

sns.distplot(male.Age, bins=12, ax=ax2)



ax1.set_title('Age distr among females')

ax2.set_title('Age distr among males')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,4))



female = df[df.Gender == 'Female']

male = df[df.Gender == 'Male']



sns.distplot(female.Income, bins=12 ,ax=ax1)

sns.distplot(male.Income, bins=12, ax=ax2)



ax1.set_title('Income distr among females')

ax2.set_title('Income distr among males')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,4))



female = df[df.Gender == 'Female']

male = df[df.Gender == 'Male']



sns.distplot(female.Score, bins=12 ,ax=ax1)

sns.distplot(male.Score, bins=12, ax=ax2)



ax1.set_title('Score distr among females')

ax2.set_title('Score distr among males')
# Map Gender to 1 for female and 0 for male



mapping = {'Female': 1, 'Male': 0}

df.Gender.replace(mapping, inplace=True)

df.head()
# Comparing pairwise correlations between variables

sns.pairplot(df[['Age', 'Income', 'Score']])
sns.lmplot('Score', 'Income', hue='Gender', data=df, fit_reg=False)
# Standardize the data to all be the same unit



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(df.drop('Gender', axis=1))



# Transforming the data

scaled_features = scaler.transform(df.drop('Gender', axis=1))

scaled_features
# Use the scaler to create scaler dataframe

# This gives us a standardized version of our data



df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split, GridSearchCV



X = df_feat

y = df['Gender']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# Training and Predictions



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5) # k=5

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

pred
# Evaluating the algorithm



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print (confusion_matrix(y_test, pred))

print (classification_report(y_test, pred))

print ('Accuracy Score: ' + str(accuracy_score(y_test, y_pred)))
error_rate = []



for i in range(1,40): # Checking every possible k value between 1-40

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    

error_rate
plt.figure(figsize=(10,6))

plt.plot(range(1,40), error_rate, color='grey', marker='o', markerfacecolor='red')

plt.title('Error rate vs K value')

plt.xlabel('K value')

plt.ylabel('Mean error rate')
knn = KNeighborsClassifier(n_neighbors=17)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)



print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print ('Accuracy Score: ' + str(accuracy_score(y_test, y_pred)))
# Training the algorithm



from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=100, random_state=101)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
# Evaluating the algorithm



print (confusion_matrix(y_test, y_pred))

print (classification_report(y_test, y_pred))

print ('Accuracy Score: ' + str(accuracy_score(y_test, y_pred)))
# Grid search



grid_param = {  

    'n_estimators': [50, 80, 100, 120],

    'criterion': ['gini', 'entropy'],

    'bootstrap': [True, False],

    'max_depth': [10,30,50],

    'max_features': ['auto', 'sqrt'],

    'min_samples_split': [3,9,20],

    'min_samples_leaf': [1, 2, 4]

    }



gs = GridSearchCV(estimator=forest,  

                     param_grid=grid_param,

                     scoring='accuracy',

                     cv=5,

                     n_jobs=-1)



gs.fit(X_train, y_train)
print(gs.best_params_)
# Training the tuned algorithm



forest_tuned = RandomForestClassifier(n_estimators=100,

                                      criterion= 'gini',

                                      bootstrap= False,

                                      max_depth= 10,

                                      max_features= 'auto',

                                      min_samples_split= 20,

                                      min_samples_leaf= 1,

                                      random_state=101)

forest_tuned.fit(X_train, y_train)

y_pred = forest_tuned.predict(X_test)
# Evaluating the tuned algorithm



print (confusion_matrix(y_test, y_pred))

print (classification_report(y_test, y_pred))

print ('Accuracy Score: ' + str(accuracy_score(y_test, y_pred)))
# Training the algorithm



from sklearn.svm import SVC



svm = SVC()

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
# Evaluating the algorithm



print (confusion_matrix(y_test, y_pred))

print (classification_report(y_test, y_pred))

print ('Accuracy Score: ' + str(accuracy_score(y_test, y_pred)))
# Grid search



# "C" controls the cost of misclassification on the training data. 

# A large C-value gives you low bias and high variance. Low bias causes you penalize the cost of misclassification a lot.



# Small "gamma" means a Gaussian of a large variance. Large gamma leads to high bias and low variance in the model. 



param_grid = {

    'C': [0.1, 1, 10, 100, 1000],

    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]

}



gs = GridSearchCV(SVC(), param_grid, verbose=3)

gs.fit(X_train, y_train)
print(gs.best_params_)
# Training the tuned algorithm



svm_tuned = SVC(C = 10, gamma = 0.1)

svm_tuned.fit(X_train, y_train)

y_pred = svm_tuned.predict(X_test)
# Evaluating the algorithm



print (confusion_matrix(y_test, y_pred))

print (classification_report(y_test, y_pred))

print ('Accuracy Score: ' + str(accuracy_score(y_test, y_pred)))