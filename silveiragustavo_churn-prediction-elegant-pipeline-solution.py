import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pd.set_option("display.max_columns", 500)
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()
df.shape
df.info()
df.describe()
df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan, regex=False).astype(float)
df.describe()
df["TotalCharges"].hist()
df["TotalCharges"] = df["TotalCharges"].apply(lambda x: np.log1p(x))
df["Churn"].value_counts(normalize=True)
map_labels = {"No": 0,

              "Yes": 1}



df["Churn"] = df["Churn"].map(map_labels)
df.isnull().sum()
df = df.drop(columns=["customerID"])
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=23)

for train_index, test_index in split.split(df, df["Churn"]):

    strat_train_set = df.loc[train_index]

    strat_test_set = df.loc[test_index]
X_train = strat_train_set.drop("Churn", axis=1)

y_train = strat_train_set["Churn"].copy()
X_train.corr()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder





num_attribs = ["MonthlyCharges", "TotalCharges", "tenure"]

cat_attribs = ["SeniorCitizen", "gender", "Partner", "Dependents",

               "PhoneService", "MultipleLines", "InternetService", 'OnlineSecurity', 

               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 

               'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']





num_pipeline = Pipeline([

        ("imputer", SimpleImputer(strategy="median")),

        ('std_scaler', StandardScaler()),

    ])



preprocessor = ColumnTransformer([

    ("num", num_pipeline, num_attribs),

    ("cat", OneHotEncoder(), cat_attribs),

])
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
from sklearn.linear_model import LogisticRegression



clf_lr = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', LogisticRegression(solver="lbfgs", max_iter=300, class_weight="balanced"))])



scores = cross_val_score(clf_lr, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)

scores.mean()
from sklearn.model_selection import learning_curve



train_sizes, train_scores, test_scores = learning_curve(estimator=clf_lr,

                                                        X=X_train,

                                                        y=y_train,

                                                        train_sizes=np.linspace(0.1, 1.0, 10),

                                                        cv=skf,

                                                        scoring="roc_auc")
train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,

          color='blue', marker='o',

          markersize=5, label='Training AUC')

plt.fill_between(train_sizes,

                  train_mean + train_std,

                  train_mean - train_std,

                  alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,

          color='green', linestyle='--',

          marker='s', markersize=5,

          label='Validation AUC')

plt.fill_between(train_sizes,

                  test_mean + test_std,

                  test_mean - test_std,

                  alpha=0.15, color='green')

plt.grid()

plt.xlabel('Number of training examples')

plt.ylabel('ROC_AUC')

plt.legend(loc='lower right')

plt.ylim([0.8, 0.9])

plt.show()
final_model = clf_lr.fit(X_train, y_train)
X_test = strat_test_set.drop("Churn", axis=1)

y_test = strat_test_set["Churn"].copy()





final_predictions = final_model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score



print(confusion_matrix(y_test, final_predictions))

print(accuracy_score(y_test, final_predictions))

print(roc_auc_score(y_test, final_predictions))