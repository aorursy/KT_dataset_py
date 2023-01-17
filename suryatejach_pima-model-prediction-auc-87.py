import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import zscore

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, RepeatedStratifiedKFold

from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve, plot_roc_curve

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.neural_network import MLPClassifier
## Load pima data into data frame for manipulation using Pandas

diabetic_df= pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

diabetic_df.head()
## Checking the shape of dataset

diabetic_df.shape
## Checking column information

diabetic_df.info()
## Checking for NULL's to handle

diabetic_df.isnull().sum()
## Using Pair plot

sns.pairplot(diabetic_df, diag_kind='kde', hue='Outcome')
## Checking 

diabetic_df.describe()
## Classifying patients using Violin plot on Glucose

sns.violinplot(data= diabetic_df, x= "Outcome", y= "Glucose")
## Classifying patients using Violin plot on BP

sns.violinplot(data=diabetic_df, x='Outcome', y='BloodPressure')
from scipy.stats import shapiro

stat, p = shapiro(diabetic_df['BloodPressure'])

print('Statistics=%.3f, p-value=%.3f' % (stat, p))
diabetic_df['BloodPressure']=diabetic_df['BloodPressure'].replace(0,diabetic_df['BloodPressure'].median())

diabetic_df['BloodPressure'].describe()
sns.distplot(diabetic_df['BloodPressure'])
sns.violinplot(data=diabetic_df, x='Outcome', y='BMI')
sns.violinplot(data=diabetic_df, x='Outcome', y='Glucose')
stats, pval = shapiro(diabetic_df['BMI'])

print('Statistics=%.3f, p-value=%.3f' % (stats, pval))
diabetic_df['BMI']=diabetic_df['BMI'].replace(0,diabetic_df['BMI'].median())

diabetic_df['BMI'].describe()
sns.distplot(diabetic_df['BMI'])
stats, pval = shapiro(diabetic_df['Glucose'])

print('Statistics=%.3f, p-value=%.3f' % (stats, pval))
diabetic_df['Glucose']=diabetic_df['Glucose'].replace(0,diabetic_df['Glucose'].median())

diabetic_df['Glucose'].describe()
sns.distplot(diabetic_df['Glucose'])
## Checking pregnant's uniqueness

diabetic_df['Pregnancies'].unique()
## Plotting bar plot for Pregnants having BP

sns.barplot(data=diabetic_df, y='BloodPressure', x='Pregnancies')
## Plotting bar plot for Pregnants having Diabetes

sns.barplot(data=diabetic_df, y='Glucose', x='Pregnancies')
## Checking Data Distribution for Class Variable (0/1)

f, ax = plt.subplots(1,1, figsize=(6,4))

sns.countplot(x="Outcome", data=diabetic_df, hue='Outcome')

total = float(len(diabetic_df))

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(100*height/total),

            ha="center") 

plt.show()
## Checking Correlation of features using Heatmap

corr= diabetic_df.corr()

fig = plt.figure(figsize=(10,6))

sns.heatmap(corr,  annot=True,cmap ='RdYlGn',linewidths=1.5)
## Checking outliers using Box plot

pos=1

plt.figure(figsize=(15,10))

for i in diabetic_df.drop(['Outcome'], axis=1).columns:

  plt.subplot(3,3,pos)

  sns.boxplot(diabetic_df[i],color="red")

  pos+=1
## Detect outliers using IQR and Handling outliers

Q1= diabetic_df.quantile(0.25)

Q3= diabetic_df.quantile(0.75)

IQR= Q3-Q1

print(IQR)
boolean_out= (diabetic_df < (Q1 - 1.5 * IQR)) | (diabetic_df > (Q3 + 1.5 * IQR))

print(boolean_out)
diabetic_data= diabetic_df[~boolean_out.any(axis=1)]

print('Shape of outliers dataset:', diabetic_df.shape)

print('\n Shape of non outliers dataset:', diabetic_data.shape)
pos=1

plt.figure(figsize= (15,10))

for i in diabetic_data.drop(['Outcome'], axis=1).columns:

  plt.subplot(3,3, pos)

  sns.boxplot(diabetic_data[i], color='orange')

  pos+=1
z = np.abs(zscore(diabetic_data))

print(z)
threshold = 2

print(np.where(z > 2))
z[17][1]
pima_data= diabetic_data[(z < 2). all(axis=1)]
print('Shape of outliers dataset:', diabetic_data.shape)

print('\n Shape of non-outliers dataset:', pima_data.shape)
pos=1

plt.figure(figsize= (15,10))

for i in pima_data.drop(['Outcome'], axis=1).columns:

  plt.subplot(3,3, pos)

  sns.boxplot(pima_data[i], color='green')

  pos+=1
pima_data.describe()
pima_data.head()
pima_data.hist(bins=8,figsize=(8,8))
X= pima_data.drop(["Outcome"], axis= 1)

y= pima_data["Outcome"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
Xtrain_scaled= MinMaxScaler(X_train)

ytrain_scaled= MinMaxScaler(y_train)
pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LogisticRegression())])))

pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',GaussianNB())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))

pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier())])))

pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))

pipelines.append(('ScaledAda', Pipeline([('Scaler', StandardScaler()),('Ada', AdaBoostClassifier())])))

pipelines.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))

pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBClassifier())])))

pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('MLP', MLPClassifier())])))
results = []

names = []

for name, model in pipelines:

    kfold = KFold(n_splits=10, random_state=21)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Algorithm comparison

fig = plt.figure(figsize=(18,5))

fig.suptitle('Model Selection by comparision')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)
lr_preds= lr_model.predict(X_test)
mlp_model= MLPClassifier(solver='sgd')

mlp_model.fit(X_train, y_train)
mlp_preds= mlp_model.predict(X_test)
print('Classification report of Logistic Regression model: \n')

print(classification_report(y_test, lr_preds))
print('Classification report of Neural Network model: \n')

print(classification_report(y_test, mlp_preds))
## Confusion Matrix of Logistic Regression

confusion_matrix = pd.crosstab(y_test, lr_preds, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True)

plt.show()
plot_roc_curve(lr_model, X_test, y_test)

plt.show()
#Accuracy Score

from sklearn.metrics import accuracy_score

print('Accuracy Score for Logistic Regression:',accuracy_score(lr_preds,y_test))
# Define models and parameters

model = LogisticRegression()

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

penalty = ['l1','l2','elasticnet']

c_values = [ 1.0, 0.1, 0.01,0.001,1e-5]

mul_class= ['auto','ovr','multinomial']



# define grid search

grid = dict(solver=solvers,penalty=penalty,C=c_values, multi_class=mul_class)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X_train, y_train)
means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print(len(y_test))

print(len(lr_preds))
diab_predict= pd.DataFrame(y_test, columns=['Actual Class'])

diab_predict["Predicted Class"]= lr_preds

diab_predict.insert(0, 'Modelname', 'Logistic Regression')

diab_predict.head()
diab_predict.to_csv('/kaggle/working/Submission_LR.csv')