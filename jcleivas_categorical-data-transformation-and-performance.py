import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')

df.head()
df.rename(columns={'sales' : 'department',

                    'left' : 'turnover'}, inplace=True)
df.isnull().sum() #There is no null data
df.describe()
f, axes = plt.subplots(ncols=3, figsize=(15, 5))

i=0

for col in ["satisfaction_level","last_evaluation", "average_montly_hours"]:

    sns.distplot(df[col], ax=axes[i]);

    i+=1
cols = [i for i in df.columns if i not in ["satisfaction_level","last_evaluation", "average_montly_hours", "turnover"]]
f, axes = plt.subplots(ncols=2,nrows=3, figsize=(15, 20))

i,j= 0,0

for col in cols:

    if j>1:

        i+=1

        j=0

    sns.countplot(y=col, data=df, hue="turnover",palette="deep", ax=axes[i][j]);

    j+=1
plt.figure(figsize=(10,10))

sns.heatmap(df.corr(), annot=True);
from sklearn.preprocessing import LabelEncoder
dfEncoded = df.copy()
leDep = LabelEncoder()

dfEncoded["department"] = leDep.fit_transform(dfEncoded["department"])
leSal = LabelEncoder()

dfEncoded["salary"] = leSal.fit_transform(dfEncoded["salary"])
plt.figure(figsize=(10,10))

sns.heatmap(dfEncoded.corr(), annot=True);
dfDummy = pd.get_dummies(df, columns=["department","salary"])

dfDummy.head()
plt.figure(figsize=(18,18))

sns.heatmap(dfDummy.corr(), annot=True);
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
def get_var_and_fit(df, clf):

    y = df["turnover"]

    X = df.drop("turnover", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    print ("ROC AUC Score: {:.4f}".format(roc_auc_score(y_test, predictions)))

    print(classification_report(y_test, predictions))

    return y_test, predictions
clfR = RandomForestClassifier(random_state=12)

y_test, predictions = get_var_and_fit(dfEncoded, clfR)
clfRDummy = RandomForestClassifier(random_state=12)

y_testDummy, predictionsDummy = get_var_and_fit(dfEncoded, clfRDummy)
f, axes = plt.subplots(ncols=2, figsize=(15,5))

sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap="YlGnBu", ax=axes[0]).set_title("Random Forest")

sns.heatmap(confusion_matrix(y_testDummy, predictionsDummy), annot=True, cmap="YlGnBu", ax=axes[1]).set_title("Random Forest Dummy");
from sklearn.linear_model import LogisticRegression
clfLog = LogisticRegression(random_state=12)

y_testLog, predictionsLog = get_var_and_fit(dfEncoded, clfLog)
clfLogDummy = LogisticRegression(random_state=12)

y_testLogDummy, predictionsLogDummy = get_var_and_fit(dfDummy, clfLogDummy)
f, axes = plt.subplots(ncols=2, figsize=(15,5))

sns.heatmap(confusion_matrix(y_testLog, predictionsLog), annot=True, cmap="YlGnBu", ax=axes[0]).set_title("Logistic Regression")

sns.heatmap(confusion_matrix(y_testLogDummy, predictionsLogDummy), annot=True, cmap="YlGnBu", ax=axes[1]).set_title("Logistic Regression Dummy");