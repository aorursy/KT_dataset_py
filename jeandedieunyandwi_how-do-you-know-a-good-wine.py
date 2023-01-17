import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

wine_data=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
wine_data.columns
print("Rows,columns:" + str(wine_data.shape))
wine_data.info()
wine_data.describe().transpose()
wine_data.head(5)
wine_data.isnull().sum()

#Same as wine_data.isna().sum()
#From this plot, we see that most wine types are in 5 and 6 category, which means most are considered bad. 

plt.figure(figsize = (11,6))

sns.countplot(data=wine_data, x='quality')

# Let us see the relationship between the label "quality" and other variables.

plt.figure(figsize = (11,6))

sns.heatmap(wine_data.corr(), 

            xticklabels=wine_data.corr().columns, 

            yticklabels=wine_data.corr().columns, 

            annot=True, 

            cmap=sns.diverging_palette(220,20,

            as_cmap=True))
#PH has a correlation of -0.058. Now plotting it is obvious that wine quality doesn't depend on pH. 

#PH is Power of Hydrogen which is a scale used to specify how acidic or basic a water-based solution is

plt.figure(figsize = (11,6))

sns.barplot(data=wine_data, x='quality',y='pH')
#Alcohol is the most correlated feature with the quality of the wine, hence a reason why here good wine has high alcohol

plt.figure(figsize = (11,6))

sns.barplot(data=wine_data, x='quality',y='alcohol')
#Citric acid also correlates with the quality of the wine, hence a reason why here good wine has high citric acid

plt.figure(figsize = (11,6))

sns.barplot(data=wine_data, x='quality',y='citric acid')
#Another feature to explore is sulphates. 

plt.figure(figsize = (11,6))

sns.barplot(data=wine_data, x='quality',y='sulphates')
#Also, we can see that the quality of the wine does not depend on the residual sugar

#Alcohol is the most correlated feature with the quality of the wine, hence a reason why here good wine has high alcohol

plt.figure(figsize = (11,6))

sns.barplot(data=wine_data, x='quality',y='residual sugar')
wine_data['good quality']=[1 if x>=7 else 0 for x in wine_data['quality']]
#plt.figure(figsize = (11,6))

sns.countplot(data=wine_data, x='good quality')
#The exact values of wine quality. 0 stands for bad wine, whereas 1 is for good wine.

wine_data['good quality'].value_counts()
#Let us drop quality off the dataset



wine_data=wine_data.drop('quality',axis=1)
#Now let us seperate the dataset as response variable and label or target variable

X = wine_data.drop('good quality', axis = 1)

y = wine_data['good quality']
#Splitting data into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#Applying Standard scaling to get optimized result

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
model1=SVC()
model1.fit(X_train, y_train)
pred_svc=model1.predict(X_test)
print(classification_report(y_test, pred_svc))
model2 = RandomForestClassifier(n_estimators=200)

model2.fit(X_train, y_train)

pred_rfc = model2.predict(X_test)
print(classification_report(y_test, pred_rfc))
#Confusion matrix

print(confusion_matrix(y_test, pred_rfc))
lrmodel=LogisticRegression()

lrmodel.fit(X_train,y_train)

logpred=lrmodel.predict(X_test)

print(classification_report(y_test, logpred))
#Finding best parameters for our SVC model

param = {

    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],

    'kernel':['linear', 'rbf'],

    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]

}

grid_svc = GridSearchCV(model1, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
grid_svc.best_params_
#Let's run SVC again with the best parameters.

model_svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')

model_svc2.fit(X_train, y_train)

pred_svc2 = model_svc2.predict(X_test)

print(classification_report(y_test, pred_svc2))
rand_forest_val = cross_val_score(estimator = model2, X = X_train, y = y_train, cv = 10)

rand_forest_val.mean()
svc_val = cross_val_score(estimator = model1, X = X_train, y = y_train, cv = 10)

svc_val.mean()
log_val = cross_val_score(estimator = lrmodel, X = X_train, y = y_train, cv = 10)

log_val.mean()