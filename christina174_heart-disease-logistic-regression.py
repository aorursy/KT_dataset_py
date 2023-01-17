import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from statistics import mode, mean



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
data = pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
fig, ax = plt.subplots(figsize=(9,9))



sns.heatmap(data.corr(), square=True, annot=True, cbar=False,  ax=ax);

# and we can see that here no height correlation
data.isnull().sum()
data = data.dropna(axis='rows', thresh=15)

data.isnull().sum()
data["education"]=data["education"].fillna(mode(data["education"]))

data["BPMeds"]=data["BPMeds"].fillna(mode(data["BPMeds"]))



data["cigsPerDay"]=data["cigsPerDay"].fillna((data["cigsPerDay"].mean()))

data["totChol"]=data["totChol"].fillna((data["totChol"].mean()))

data["BMI"]=data["BMI"].fillna((data["BMI"].mean()))

data["heartRate"]=data["heartRate"].fillna((data["heartRate"].mean()))

data["glucose"]=data["glucose"].fillna(data["glucose"].mean())
data.isnull().any()
for col in data.columns[:-1]:

    pd.crosstab(data[col], data.TenYearCHD).plot(kind='bar')

    plt.xlabel(col)
data = data.drop(columns='currentSmoker')
X = data[['male','age','education','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose']]

y = pd.Series(data['TenYearCHD'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
scaler = StandardScaler()

X_train = scaler.fit_transform(x_train)

X_test = scaler.transform(x_test)
np.around(X_train.mean(axis = 0), 10)
X_train.std(axis = 0)
model = LogisticRegression()
model.fit(X_train, y_train);
labels = model.predict(X_test)
accuracy_score(y_test, labels)
acc = np.array([])

for i in range(0, 100, 10):

    y_pred_new_threshold = (model.predict_proba(X_test)[:, 1]>= i/100).astype(int)

    newscore = accuracy_score(y_test, y_pred_new_threshold)

    acc = np.append(acc, [y_pred_new_threshold])

acc = acc.astype(int)

acc = acc.reshape(10,-1)
i=0

for l in acc:

    print('***', i, '***')

    matrix = confusion_matrix(y_test, l)

    print('\n', matrix)

    print(classification_report(y_test, l))

    i+=1
logit_roc_auc = roc_auc_score(y_test, labels)

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()