import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



pd.options.mode.chained_assignment = None
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
n_train = df_train.shape[0]

n_test = df_test.shape[0]

ratio = round(n_train / (n_train + n_test),1)



print("Number of observations in the training set: %d (%d%%)" % (n_train, ratio*100))

print("Number of observations in the test set: %d (%d%%)" % (n_test, (1-ratio)*100))
df_train.info()
df_train.sample(10)
df_titanic = df_train.drop(["Ticket", "Cabin", "Name"], axis=1)
df_titanic_na = df_titanic.dropna()
df_titanic_na.Sex = df_titanic.Sex.map({"female": 0, "male": 1})

df_titanic_na.Embarked = df_titanic.Embarked.map({"C": 0, "Q": 1, "S": 2})
df_titanic_na.head()
g = sns.distplot(df_titanic_na.Survived, color="red", hist_kws={"alpha": 0.3}, kde=None)

g.set_xticks([0,1])

g.autoscale()

g.set_xticklabels(["Dead", "Survived"])
corrmat = df_titanic_na[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',

       'Fare', 'Embarked']].corr()



f, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(corrmat, vmax=.8, square=True)
survived = df_titanic_na.Survived == 1

died = df_titanic_na.Survived == 0
g = sns.distplot(df_titanic_na.Embarked, color="darkgreen", hist_kws={"alpha": 0.3}, kde=None)

g.set_xticklabels(["Cherbourg", "", "Queenstown", "", "Southampton"])
sns.distplot(df_titanic_na[survived].Age, color="darkgreen", hist_kws={"alpha": 0.3})

sns.distplot(df_titanic_na[died].Age, color="darkred", hist_kws={"alpha": 0.3})
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=df_titanic_na,

                   size=6, kind="bar", palette="muted", ci=None)

g.despine(left=True)

g.set_ylabels("Survival Probability")

g.set_xlabels("Passenger Class")
g = sns.factorplot(x="Embarked", y="Survived", hue="Pclass", data=df_titanic_na,

                   size=6, kind="bar", palette="muted", ci=None)

g.despine(left=True)

g.set_ylabels("Survival Probability")

g.set_xlabels("Embarkation Port")

g.set_xticklabels(["Cherbourg", "Queenstown", "Southampton"])
df_titanic_ml = df_titanic.copy()
df_titanic_ml.Embarked = df_titanic_ml.Embarked.fillna("Southampton") 
df_titanic_ml[df_titanic_ml.Embarked.isnull()].shape
null_age = df_titanic_ml.Age.isnull()

df_titanic_ml[null_age].shape
df_titanic_ml = df_titanic_ml[np.invert(null_age)]
df_titanic_ml.info()
df_titanic_ml.Sex = df_titanic.Sex.map({"female": 0, "male": 1})

df_titanic_ml.Embarked = df_titanic.Embarked.map({"C": 0, "Q": 1, "S": 2})
emb_dummies = pd.get_dummies(df_titanic_ml.Embarked, prefix="Embarked")

df_titanic_ml = df_titanic_ml.join(emb_dummies)

df_titanic_ml.drop("Embarked", axis=1, inplace=True)
df_titanic_ml.head()
age_over_18 = df_titanic_ml.Age > 18

women = df_titanic_ml.Sex == 0

with_parch = df_titanic_ml.Parch > 0
df_titanic_ml["is_mother"] = 0

df_titanic_ml["is_mother"][women & age_over_18 & with_parch] = 1
from sklearn import cross_validation, metrics

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = cross_validation.train_test_split(

    df_titanic_ml.drop(["PassengerId", "Survived"], axis=1), df_titanic_ml.Survived, test_size=0.3, random_state=0)
forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

forest.fit(X_train, y_train)
def forest_metrics(X_test, y_test, clf):  

    

    f_preds = clf.predict_proba(X_test)[:, 1]

    f_fpr, f_tpr, _ = metrics.roc_curve(y_test, f_preds)



    fig, ax = plt.subplots()

    ax.plot(f_fpr, f_tpr)

    lims = [

        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes

        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes

    ]



    print("Model Accuracy: %.1f%%" % (clf.score(X_test,y_test) * 100))

    print ("Model ROC AUC: %.1f%%" % (metrics.roc_auc_score(y_test, f_preds)*100))

    print("ROC Curve")

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
forest_metrics(X_test,y_test, forest)
print(metrics.confusion_matrix(y_test, forest.predict(X_test)))
print(metrics.classification_report(y_test, forest.predict(X_test)))