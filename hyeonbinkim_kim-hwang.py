import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

import matplotlib.pyplot as plt
df = pd.read_csv("../input/kaggl2/human.csv", encoding = "CP949")

df.head(2)
df2 = pd.read_csv("../input/kaggl2/human_new.csv", encoding = "CP949")

df2.head(2)
df.columns = ["id", "age", "working_class", "fnlwgt", "education", "education_num", "marriage_status", "job",

              "relationship", "race", "sex", "capital_gain", "capital_loss", "work_hour", "motherland"]

df.head(2)
#파생변수

df["age_cat"] = df.age.apply(lambda x : 1 if x <= 10

                            else 2 if 10 < x <= 20

                            else 3 if 20 < x <= 30

                            else 4 if 30 < x <= 40

                            else 5 if 40 < x <= 50

                            else 6 if 50 < x <= 60

                            else 7 if 60 < x <= 70

                            else 8 if 70 < x <= 80

                            else 9)
df2.columns = ["id", "age", "working_class", "fnlwgt", "education", "education_num", "marriage_status", "job",

              "relationship", "race", "capital_gain", "capital_loss", "work_hour", "motherland"]

df2.head(2)
df_impute = df.copy()
# 결측치 처리

cat = ["working_class", "job", "motherland"]

imputer_cat = SimpleImputer(strategy="most_frequent")

df_impute[cat] = imputer_cat.fit_transform(df_impute[cat])
df_impute_1 = df_impute.copy()
# 컬럼 카테고리화

obj = ["working_class", "education", "marriage_status", "job", "relationship", "race", "sex", "motherland"]

df_impute_1[obj] = df_impute_1[obj].apply(lambda x: x.astype('category').cat.codes)
df_impute_1
df_impute_1["sex"].value_counts()
del_col = ["id", "sex", "age", "education"]
df
from imblearn.combine import *
dfX = df_impute_1.drop(del_col, axis=1)

dfy = df_impute_1['sex']
X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=0.25, random_state=0)
from catboost import CatBoostClassifier
catboost = CatBoostClassifier()
catboost.fit(X_train, y_train).score(X_train, y_train)
from imblearn.combine import SMOTEENN

XX, yy = SMOTEENN(random_state=0).fit_sample(X_train, y_train)

tree3 = catboost

tree3.fit(XX, yy)

y_pred3 = tree3.predict(X_test)



print(classification_report(y_test, y_pred3))
from imblearn.combine import SMOTETomek

XX, yy = SMOTETomek(random_state=0).fit_sample(X_train, y_train)

tree3 = catboost

tree3.fit(XX, yy)

y_pred3 = tree3.predict(X_test)



print(classification_report(y_test, y_pred3))
X_resampled, y_resampled = SMOTETomek(random_state=0).fit_sample(dfX, dfy)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=0)

print(dfX.shape, X_resampled.shape, X_train.shape, X_test.shape)
catboost.fit(X_test, y_test).score(X_test, y_test)
best_model = catboost

best_model.score(X_test,y_test)
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

pred_dummy = dummy.predict(X_test)
def plot_roc_curve(fpr, tpr, model, color=None) :

    model = model + ' (auc = %0.3f)' % auc(fpr, tpr)

    plt.plot(fpr, tpr, label=model, color=color)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.axis([0,1,0,1])

    plt.xlabel('FPR (1 - specificity)')

    plt.ylabel('TPR (recall)')

    plt.title('ROC curve')

    plt.legend(loc="lower right")
fpr_dummy, tpr_dummy, _ = roc_curve(y_test, 

                                    dummy.predict_proba(X_test)[:,1])



plot_roc_curve(fpr_dummy, tpr_dummy, 'dummy model', 'hotpink')



fpr_tree, tpr_tree, _ = roc_curve(y_test, 

                                  catboost.predict_proba(X_test)[:,1])



plot_roc_curve(fpr_tree, tpr_tree, 'lgbm', 'darkgreen')
df2["age_cat"] = df2.age.apply(lambda x : 1 if x <= 10

                            else 2 if 10 < x <= 20

                            else 3 if 20 < x <= 30

                            else 4 if 30 < x <= 40

                            else 5 if 40 < x <= 50

                            else 6 if 50 < x <= 60

                            else 7 if 60 < x <= 70

                            else 8 if 70 < x <= 80

                            else 9)
df2_impute = df2.copy()
imputer_cat = SimpleImputer(strategy="most_frequent")

df2_impute[cat] = imputer_cat.fit_transform(df2_impute[cat])
obj2 = ["working_class", "education", "marriage_status", "job", "relationship", "race", "motherland"]

df2_impute[obj2] = df2_impute[obj2].apply(lambda x: x.astype('category').cat.codes)
df2_impute['sex_proba'] = best_model.predict_proba(df2_impute.loc[:,'working_class':'age_cat'])[:,1]
df2_impute
predict = df2_impute[["id", "sex_proba"]]
predict.columns = ["id", "sex"]
predict
predict.to_csv("predict.csv", index=False)