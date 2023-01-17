import numpy as np

import pandas as pd

from pandas import read_csv

from pandas import DataFrame

from pandas.plotting import scatter_matrix

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

import seaborn as sns
dataset = read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

dataset.shape
dataset.head()
dataset.info()
dataset.describe().T
dataset.groupby('Outcome').size()
sns.countplot(x='Outcome', data=dataset)
# box and whisker plots

dataset.drop('Outcome', axis=1).plot(kind='box', figsize=(12, 12), subplots=True, layout=(3,3), sharex=False, sharey=False)

plt.show()
dataset.drop('Outcome', axis=1).hist(figsize=(10,10))

plt.show()
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

axs = axs.flatten()

i=0

for col in dataset.drop('Outcome', axis=1).columns:

    sns.swarmplot(x='Outcome', y=col, data=dataset, ax=axs[i])

    i+=1
corr = dataset.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
dataset_new = dataset.copy(deep = True)

dataset_new[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dataset[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)



## showing the count of Nans

print(dataset_new.isnull().sum())
dataset_new['Glucose'].fillna(dataset_new['Glucose'].dropna().mean(), inplace = True)

dataset_new['BloodPressure'].fillna(dataset_new['BloodPressure'].dropna().mean(), inplace = True)

dataset_new['SkinThickness'].fillna(dataset_new['SkinThickness'].dropna().median(), inplace = True)

dataset_new['Insulin'].fillna(dataset_new['Insulin'].dropna().median(), inplace = True)

dataset_new['BMI'].fillna(dataset_new['BMI'].dropna().mean(), inplace = True)
dataset_new.drop('Outcome', axis=1).hist(figsize=(10,10))

plt.show()
corr = dataset_new.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
dataset_new['Glucose * BMI'] = dataset['Glucose'] * dataset['BMI']

dataset_new.head()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(dataset_new.drop(["Outcome"],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose * BMI'])

y = dataset_new["Outcome"]
X.head()
y.head()
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
models = []

models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))

models.append(('K-Neighbors', KNeighborsClassifier()))

models.append(('Descision Tree', DecisionTreeClassifier()))

models.append(('Naive Bayes', GaussianNB()))

models.append(('Support Vector Classifier', SVC(gamma='auto')))



results = []

names = []

acc_mean = []

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    acc_mean.append(cv_results.mean())

    print(f'{name} : {cv_results.mean()} ({cv_results.std()})')
plt.boxplot(results, labels=names)

plt.title('Algorithm Comparison')

plt.xticks(rotation=45, ha='right')

plt.show()
sns.barplot(x=acc_mean, y=names, palette=sns.color_palette("muted"))
model = LogisticRegression(solver='liblinear', multi_class='ovr')

model.fit(X_train, Y_train)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
#import confusion_matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(Y_validation,predictions)

pd.crosstab(Y_validation, predictions, rownames=['True'], colnames=['Predicted'], margins=True)