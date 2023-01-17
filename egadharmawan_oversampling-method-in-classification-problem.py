import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(13,13)})
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/income-classification/income_evaluation.csv', skipinitialspace=True)
df.head()
df.info()
## Income By Age
ax = sns.boxplot(x="income", y="age", data=df)
## Income By age and race
ax = sns.boxplot(x="income", y="age", hue="race",
                 data=df, palette="Set2")
## Income by age, race and sex
g = sns.catplot(x="income", y="age",
                hue="sex", col="race",
                col_wrap=3,
                data=df, kind="box",
                palette="Set3");
## Income by age, native country and sex
g = sns.catplot(x="income", y="age",
                hue="sex", col="native-country",
                col_wrap=3,
                data=df, kind="box",
                palette="vlag");
## Income by age and hours per week
ax = sns.scatterplot(x="age", y="hours-per-week", hue="income",
                     data=df, palette='prism')
## Income by age, hours per week and occupation
g = sns.relplot(x="age", y="hours-per-week",
                 col="occupation", col_wrap=3, hue="income",
                 kind="scatter", data=df, palette='rocket')
## Income by age, hours per week and education
g = sns.relplot(x="age", y="hours-per-week",
                 col="education", col_wrap=3, hue="income",
                 kind="scatter", data=df, palette='seismic')
## Income by age, hours per week and marital status
g = sns.relplot(x="age", y="hours-per-week",
                 col="marital-status", col_wrap=3, hue="income",
                 kind="scatter", data=df, palette='gist_heat')
## Income by age, hours per week and relationship
g = sns.relplot(x="age", y="hours-per-week",
                 col="relationship", col_wrap=3, hue="income",
                 kind="scatter", data=df, palette='Oranges')
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df.income = labelencoder.fit_transform(df.income)
df = pd.get_dummies(df)
df.head()
## Split Data

X = df.drop('income', 1)
y = df.income

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
## Oversample

df.income.value_counts()
### Oversample with Random Oversmpling
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler()
XTRO, YTRO = oversample.fit_resample(X_train, y_train)

# Check value distribution
YTRO.value_counts()
### Oversample with SMOTE
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
XTS, YTS = oversample.fit_resample(X_train, y_train)

# Check value distribution
YTS.value_counts()
### Oversample with ADASYN
from imblearn.over_sampling import ADASYN

oversample = ADASYN()
XTA, YTA = oversample.fit_resample(X_train, y_train)

# Check value distribution
YTA.value_counts()
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
Oversampling = ['Random OverSampler','SMOTE','ADASYN']
## Random Forest
RF = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 12)
RFs = []

### Random Oversampling
RF.fit(XTRO, YTRO)
pred = RF.predict(X_test)
actual = np.array(y_test)
RF_RO = accuracy_score(actual, pred)
RFs.append(RF_RO)

print('Random Oversampling')
print('Accuracy Score :', RF_RO)
print('Report : ')
print(classification_report(actual, pred))

### SMOTE
RF.fit(XTS, YTS)
pred = RF.predict(X_test)
actual = np.array(y_test)
RF_S = accuracy_score(actual, pred)
RFs.append(RF_S)

print('SMOTE')
print('Accuracy Score :', RF_S)
print('Report : ')
print(classification_report(actual, pred))

### ADASYN
RF.fit(XTA, YTA)
pred = RF.predict(X_test)
actual = np.array(y_test)
RF_A = accuracy_score(actual, pred)
RFs.append(RF_A)

print('ADASYN')
print('Accuracy Score :', RF_A)
print('Report : ')
print(classification_report(actual, pred))
## Logistic Regression
LR = LogisticRegression()
LRs = []

### Random Oversampling
LR.fit(XTRO, YTRO)
pred = LR.predict(X_test)
actual = np.array(y_test)
LR_RO = accuracy_score(actual, pred)
LRs.append(LR_RO)

print('Random Oversampling')
print('Accuracy Score :', LR_RO)
print('Report : ')
print(classification_report(actual, pred))

### SMOTE
LR.fit(XTS, YTS)
pred = LR.predict(X_test)
actual = np.array(y_test)
LR_S = accuracy_score(actual, pred)
LRs.append(LR_S)

print('SMOTE')
print('Accuracy Score :', LR_S)
print('Report : ')
print(classification_report(actual, pred))

### ADASYN
LR.fit(XTA, YTA)
pred = LR.predict(X_test)
actual = np.array(y_test)
LR_A = accuracy_score(actual, pred)
LRs.append(LR_A)

print('ADASYN')
print('Accuracy Score :', LR_A)
print('Report : ')
print(classification_report(actual, pred))
## SGD Classifier
SGD = SGDClassifier(loss='modified_huber')
SGDs = []

### Random Oversampling
SGD.fit(XTRO, YTRO)
pred = SGD.predict(X_test)
actual = np.array(y_test)
SGD_RO = accuracy_score(actual, pred)
SGDs.append(SGD_RO)

print('Random Oversampling')
print('Accuracy Score :', SGD_RO)
print('Report : ')
print(classification_report(actual, pred))

### SMOTE
SGD.fit(XTS, YTS)
pred = SGD.predict(X_test)
actual = np.array(y_test)
SGD_S = accuracy_score(actual, pred)
SGDs.append(SGD_S)

print('SMOTE')
print('Accuracy Score :', SGD_S)
print('Report : ')
print(classification_report(actual, pred))

### ADASYN
SGD.fit(XTA, YTA)
pred = SGD.predict(X_test)
actual = np.array(y_test)
SGD_A = accuracy_score(actual, pred)
SGDs.append(SGD_A)

print('ADASYN')
print('Accuracy Score :', SGD_A)
print('Report : ')
print(classification_report(actual, pred))
result = pd.DataFrame({'Random Forest': RFs,
                       'Logistic Regression': LRs,'SGD Classifier': SGDs},
                         index = Oversampling)
result
result.plot(figsize=(12,8));