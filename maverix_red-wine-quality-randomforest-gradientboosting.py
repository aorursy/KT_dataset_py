import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegressionCV



from imblearn.under_sampling import RandomUnderSampler

import numpy as np

df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head(5)
sns.pairplot(df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']])
sns.pairplot(df, x_vars=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol'], y_vars='quality')
fig, ax = plt.subplots(figsize=(10,8))



correlation = df.corr()

sns.heatmap(correlation, cmap='coolwarm', annot=True, fmt=".2f")



plt.xticks
sns.countplot(df['quality'], data=df)
labels = ['bad', 'good']

bins = [2,6.5,8]

df['quality'] = pd.cut(df['quality'], bins=bins, labels=labels)
label_enc = LabelEncoder()

df['quality'] = label_enc.fit_transform(df['quality'])
X = df.drop('quality', axis=1)

y = df['quality']
resampler = RandomUnderSampler(random_state=0, replacement=True)

resampler.fit(X,y)

X_resampled, y_resampled = resampler.fit_resample(X,y)
sns.countplot(y_resampled)
X_resampled.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
model2 = LogisticRegressionCV(cv=5,random_state=0)

model2.fit(X_train,y_train)

pred2 = model2.predict(X_test)

print(mean_absolute_error(y_test,pred2))

print(accuracy_score(y_test,pred2))
print(classification_report(y_test,pred2))
model1 = RandomForestClassifier(n_estimators=50, max_depth=10, max_features='log2')



param_grid = {

    'n_estimators' : [50,1000],

    'max_depth': [1,100],

    'max_features': ['auto', 'sqrt', 'log2']

}





CV_rfc = GridSearchCV(estimator=model1, param_grid=param_grid, cv= 5)

CV_rfc.fit(X, y)

print (CV_rfc.best_params_)
model1 = RandomForestClassifier(n_estimators=50, max_depth=100, max_features='log2')



model1.fit(X_train,y_train)

pred1 = model1.predict(X_test)

print(mean_absolute_error(y_test,pred1))

print(accuracy_score(y_test,pred1))
print(classification_report(y_test,pred1))
clf = GradientBoostingClassifier(n_estimators=40,  learning_rate=0.2, max_depth=1)



clf.fit(X_train,y_train)

pred3 = clf.predict(X_test)

print(mean_absolute_error(y_test,pred3))

print(accuracy_score(y_test,pred3))
print(classification_report(y_test,pred3))