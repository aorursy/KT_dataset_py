import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

color_pallete = ['#5893d4', '#f7b633']

sns.set_palette(color_pallete, 2)

sns.set_style("whitegrid")



from sklearn.preprocessing import MinMaxScaler, StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
df = pd.read_csv('../input/diabetes.csv')

df.head()
df.info()
df.describe()
df.isnull().sum()
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(), annot=True, cmap='RdBu', vmin=-1, vmax=1)

plt.show()
sns.pairplot(df, hue='Outcome')

plt.show()
for i in ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Insulin']:

    sns.catplot(x="Outcome", y=i, kind="box", data=df, )

    plt.plot()
for i in ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Insulin']:

    plt.figure(figsize=(15, 5))

    ax = sns.kdeplot(df[df['Outcome']==0][i], shade=True)

    ax = sns.kdeplot(df[df['Outcome']==1][i], shade=True)

    ax.set_xlabel(i)

    plt.legend(['Absence', 'Presence'])

    plt.show()
print('number of entries with value 0')

for i in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:

    print(i, ':', df[df[i]==0].shape[0])
df = df.drop(df[(df.Glucose == 0) | (df.BloodPressure == 0) | (df.BMI == 0)].index)
for i in ['SkinThickness']:

    df = df.drop([i], axis=1)

df.head()
'''df['insulin_taking'] = ['no' if i==0 else 'yes' for i in df['Insulin']]

print(df.head())

sns.countplot(x='Outcome', hue='insulin_taking', data=df)'''
# df.columns
mms = MinMaxScaler()



for i in ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Insulin']:

    df[i] = df[i].astype('float64')

    df[i] = mms.fit_transform(df[i].values.reshape(-1,1))

    

df.head()
X = df.drop(['Outcome'], axis=1)

y = df['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
nbc = GaussianNB()

nbc.fit(X_train, y_train)

nbc_pred = nbc.predict(X_test)



scores = cross_val_score(nbc, X, y, cv=10)

accuracy = np.mean(scores)



print(scores)

print(accuracy)

print(confusion_matrix(nbc_pred, y_test))
lr = LogisticRegression(solver='lbfgs')

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)



scores = cross_val_score(lr, X, y, cv=10)

accuracy = np.mean(scores)



print(scores)

print(accuracy)

print(confusion_matrix(nbc_pred, y_test))
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)



scores = cross_val_score(dt, X, y, cv=10)

accuracy = np.mean(scores)



print(scores)

print(accuracy)

print(confusion_matrix(nbc_pred, y_test))
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier()



hyperparameters = {

    "n_neighbors": range(1,50,2)

}



grid = GridSearchCV(model, param_grid=hyperparameters, cv=10)

grid.fit(X, y)



best_params = grid.best_params_

best_score = grid.best_score_



best_model = grid.best_estimator_

pred = best_model.predict(X_test)



print(grid.best_params_)

print(grid.best_score_)

print(confusion_matrix(pred, y_test))
model = RandomForestClassifier(random_state=1)



hyperparameters = {"criterion": ["entropy", "gini"],

                   "max_depth": [5, 10],

                   "max_features": ["log2", "sqrt"],

                   "min_samples_leaf": [1, 5],

                   "min_samples_split": [3, 5],

                   "n_estimators": [6, 9]

}



grid = GridSearchCV(model, param_grid=hyperparameters, cv=10)

grid.fit(X, y)



best_params = grid.best_params_

best_score = grid.best_score_



best_model = grid.best_estimator_

pred = best_model.predict(X_test)



print(grid.best_params_)

print(grid.best_score_)

print(confusion_matrix(pred, y_test))
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(nbc, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())