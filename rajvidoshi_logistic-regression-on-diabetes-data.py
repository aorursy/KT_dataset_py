import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
d = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
d.shape
type(d)
d.head()
d.describe()
d.isna().sum()
d.groupby('Outcome').size()
d.hist(figsize=(10,10));
sns.heatmap(d.corr(), annot=True);
invalid = pd.Series(index=d.columns[0:8], data=np.zeros(8), dtype='int32')

d.columns

for i in d.columns[0:8]:

  invalid[i]=d.loc[d[i]==0,i].count()

invalid
d1 = d[(d.BMI!=0) & (d.Glucose != 0) & (d.BloodPressure != 0)]

d1.shape
d1.head()
X = d1[d.columns[0:8]]

y = d1.Outcome
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = d1.Outcome, random_state=0)
[X_train.shape, X_test.shape]
means = np.mean(X_train, axis=0)

stds = np.std(X_train, axis=0)

X_train = (X_train - means)/stds

X_test = (X_test - means)/stds
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver= 'lbfgs', class_weight='balanced', random_state=1).fit(X_train, y_train)
l = list(d1.columns[0:8])

l.insert(0,'Intercept')

coeff = pd.DataFrame({'Variables': l,

                            'Coefficients': np.zeros(9)})

var_coeff = list(model.coef_[0])

var_coeff.insert(0, model.intercept_[0])

coeff['Coefficients'] = var_coeff

print(coeff)
coeff.drop(0, axis=0,inplace=True)

coeff.sort_values(by=['Coefficients'], ascending=True, inplace=True)

coeff.set_index('Variables', inplace=True)

coeff.Coefficients.plot(kind='barh', figsize=(11, 6))

plt.xlabel('Importance');
y_pred = model.predict(X_test)
import sklearn.metrics as sm

c = pd.DataFrame(sm.confusion_matrix(y_test, y_pred), index=['Actual non diabetic','Actual diabetic'])

c.columns = ['Predicted non diabetic','Predicted diabetic']

c['Actual Total'] = c.sum(axis=1)

c.loc['Predicted Total',:] = c.sum(axis = 0)

c
print(["The accuracy is " + str(round(sm.accuracy_score(y_test, y_pred)*100,ndigits = 2)) + "%"])
print("The sensitivity (true positive rate) is " + str(round(100*c.iloc[1,1]/c.iloc[1,2], ndigits=2)) + "%")
from matplotlib import pyplot

ns_fpr, ns_tpr, _ = sm.roc_curve(y_test, np.zeros(len(y_test)))

lr_probs = model.predict_proba(X_test)

# keep probabilities for the positive outcome only

lr_probs = lr_probs[:, 1]

lr_fpr, lr_tpr, _ = sm.roc_curve(y_test, lr_probs)

# plot the roc curve for the model

pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')

# axis labels

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

# show the legend

pyplot.legend()

# show the plot

pyplot.show()
print("The Area under ROC curve is " + str(round(100 * sm.roc_auc_score(y_test, y_pred), ndigits=2)) + "%")
report = sm.classification_report(y_test, y_pred)

print(report)