import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
risk = pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv", index_col = "Unnamed: 0")

df = risk.copy()
df
df.info()
df.describe().T
df.isnull().sum()
df["Job"].value_counts().sort_index()
df["Housing"].value_counts()
df["Saving accounts"].value_counts()
df["Checking account"].value_counts()
df["Purpose"].value_counts()
df["Risk"].value_counts()
df.describe().T
df.corr()
for i in df["Age"]:

    if i < 25:

        df["Age"] = df["Age"].replace(i,"0-25")

    elif (i >= 25) and (i < 30):

        df["Age"] = df["Age"].replace(i,"25-30")

    elif (i >= 30) and (i < 35):

        df["Age"] = df["Age"].replace(i,"30-35")

    elif (i >= 35) and (i < 40):

        df["Age"] = df["Age"].replace(i,"35-40")

    elif (i >= 40) and (i < 50):

        df["Age"] = df["Age"].replace(i,"40-50")

    elif (i >= 50) and (i < 76):

        df["Age"] = df["Age"].replace(i,"50-75")
for i in df["Duration"]:

    if i < 12:

        df["Duration"] = df["Duration"].replace(i,"0-12")

    elif (i >= 12) and (i < 24):

        df["Duration"] = df["Duration"].replace(i,"12-24")

    elif (i >= 24) and (i < 36):

        df["Duration"] = df["Duration"].replace(i,"24-36")

    elif (i >= 36) and (i < 48):

        df["Duration"] = df["Duration"].replace(i,"36-48")

    elif (i >= 48) and (i < 60):

        df["Duration"] = df["Duration"].replace(i,"48-60")

    elif (i >= 60) and (i <= 72):

        df["Duration"] = df["Duration"].replace(i,"60-72")
df["Job"] = pd.Categorical(df["Job"], categories = [0,1,2,3], 

                           ordered = True)

df["Age"] = pd.Categorical(df["Age"], 

                           categories = ['0-25','25-30', '30-35','35-40','40-50','50-75'], 

                           ordered = True)

df["Duration"] = pd.Categorical(df["Duration"], 

                                categories = ['0-12','12-24', '24-36','36-48','48-60','60-72'], 

                                ordered = True)
fig,ax = plt.subplots(ncols = 2, nrows = 3,figsize = (15, 10))

cat_list = ["Age", "Sex", "Job","Housing", "Purpose", "Risk"]

palette = ["crimson", "dodgerblue", "fuchsia", "lime","yellow", "sandybrown"]

count = 0

for i in range(3):

    for j in range(2):

        sns.countplot(df[cat_list[count]], ax = ax[i][j],

                      palette = sns.dark_palette(palette[count],

                                                 reverse=True))

        ax[i][j].set_xticklabels(ax[i][j].get_xticklabels(),rotation = 30)

        count += 1
plt.figure(figsize = (16, 5))

sns.stripplot(x = "Age", y = "Credit amount", data = df)
plt.figure(figsize = (16, 5))

sns.boxplot(x = "Age", y = "Credit amount",data = df)
fig, ax = plt.subplots(ncols = 2, nrows = 2, figsize=(16, 7))

liste = ["Sex", "Risk"]

for i in range(2):

    count = 0

    sns.pointplot(x = "Age", y = "Credit amount", hue = liste[i], data = df, 

                  ax = ax[0][i], palette = ["#FFAE00","#000000"], ci = None)

    sns.pointplot(x = "Purpose", y = "Credit amount", hue = liste[i], data = df, 

                  ax = ax[1][i], palette = ["#FF00F0","#04FF00"], ci = None)

    ax[1][i].set_xticklabels(ax[1][i].get_xticklabels(), rotation = 40, size = 11)

    ax[i][count].legend(loc = "upper left")

    count += 1
import missingno as msno
msno.bar(df)
msno.heatmap(df)
msno.matrix(df)
df.isnull().sum()
df["Saving accounts"].value_counts()
df["Saving accounts"].fillna(df["Saving accounts"].mode()[0], inplace = True)
df["Saving accounts"].value_counts()
df["Checking account"].value_counts()
df["Checking account"].fillna(df["Checking account"].mode()[0], inplace = True)
df["Checking account"].value_counts()
df.isnull().sum()
import statsmodels.api as sm

from sklearn import preprocessing

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
df
df_fit = df.apply(preprocessing.LabelEncoder().fit_transform)
df_fit
df_fit.info()
y = df_fit["Risk"]

X = df_fit.drop(["Risk"], axis=1)
loj = sm.Logit(y, X)

loj_model= loj.fit()

loj_model.summary()
from sklearn.linear_model import LogisticRegression
loj = LogisticRegression(solver = "liblinear")

loj_model = loj.fit(X,y)

loj_model
loj_model.intercept_
loj_model.coef_
y_pred = loj_model.predict(X)
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
loj_model.predict(X)[0:10]
loj_model.predict_proba(X)[0:10][:,0:2]
y[0:10]
y_probs = loj_model.predict_proba(X)

y_probs = y_probs[:,1]
y_probs[0:10]
y_pred = [1 if i > 0.5 else 0 for i in y_probs]
y_pred[0:10]
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
logit_roc_auc = roc_auc_score(y, loj_model.predict(X))



fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Oranı')

plt.ylabel('True Positive Oranı')

plt.title('ROC')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size = 0.30, 

                                                    random_state = 42)
loj = LogisticRegression(solver = "liblinear")

loj_model = loj.fit(X_train,y_train)

loj_model
accuracy_score(y_test, loj_model.predict(X_test))
cross_val_score(loj_model, X_test, y_test, cv = 10).mean()
from sklearn.naive_bayes import GaussianNB
y = df_fit["Risk"]

X = df_fit.drop(["Risk"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.30, 

                                                    random_state=42)
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)

nb_model
nb_model.predict(X_test)[0:10]
nb_model.predict_proba(X_test)[0:10]
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
from sklearn.ensemble import GradientBoostingClassifier
y = df_fit["Risk"]

X = df_fit.drop(["Risk"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.30, 

                                                    random_state=42)
gbm_model = GradientBoostingClassifier().fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)

accuracy_score(y_test, y_pred)
from sklearn.model_selection import learning_curve, GridSearchCV
gbm_model
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],

             "n_estimators": [100,500,100],

             "max_depth": [3,5,10],

             "min_samples_split": [2,5,10]}
gbm = GradientBoostingClassifier()



gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv.fit(X_train, y_train)
print("En iyi parametreler: " + str(gbm_cv.best_params_))
gbm = GradientBoostingClassifier(learning_rate = 0.01, 

                                 max_depth = 5,

                                min_samples_split = 10,

                                n_estimators = 100)
gbm_tuned =  gbm.fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)

accuracy_score(y_test, y_pred)
cross_val_score(gbm_model, X_test, y_test, cv = 10).mean()