import os

import pandas as pd

os.getcwd()

fname = '../input/diabetescsv/diabetes.csv'

patients = pd.read_csv(fname, low_memory=False)

df = pd.DataFrame(patients)



print(df.head())
print(df.shape)
print(df.size)
print(df.columns)
dfworking = df.drop('DiabetesPedigreeFunction', axis =1)

dfworking.mean()
dfworking.std()
dfworking.max()
dfworking.min()
dfworking.quantile(0.25)
dfworking.quantile(0.75)
dfworking.quantile(0.25)*1.5
dfworking.quantile(0.75)*1.5
dfworking.boxplot(figsize=(10,6))

dfworking.plot.box(vert=False)

dfworking.boxplot(column=['Pregnancies','Glucose','BloodPressure','Insulin','BMI','Age'],figsize=(10,6))

dfworking.skew()
dfworking.kurtosis()
dfworking.hist(figsize=(10,6))
dfworking.corr()
my_tab = pd.crosstab(index = df['Outcome'],columns="Count")

my_tab = my_tab.sort_values('Count',ascending=[False])

print(my_tab)
my_tab = pd.crosstab(index = df['Insulin'],columns="Count")

my_tab = my_tab.sort_values('Count',ascending=[False])

print(my_tab)
#Data Preparation Stage

diabetics = pd.DataFrame(dfworking['Outcome'])

features = pd.DataFrame(dfworking.drop('Outcome',axis=1))

diabetics.columns

features.columns
dfworking.dtypes
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numerical = ['Age','SkinThickness', 'BMI','Insulin','BloodPressure','Glucose','Pregnancies']

features[numerical] = scaler.fit_transform(dfworking[numerical])

# Show an example of a record with scaling applied

#print(features_raw[numerical])

display(features[numerical].head(n=1))

features = pd.get_dummies(features)

display(features.head(1),diabetics.head(1))
encoded = list(features.columns)

print("{} total features after one-hot encoding.".format(len(encoded)))

print(encoded)

display(features.head(1),diabetics.head(1))

print(encoded)
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

features=shuffle(features, random_state=0)

diabetics=shuffle(diabetics, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(features, diabetics,

test_size = 0.2, random_state = 0)

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

seed = 7

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RFC', RandomForestClassifier()))

results = []

names = []

scoring = 'accuracy'

import warnings

warnings.filterwarnings("ignore")

for name, model in models:

    

    

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, y_train,

cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)