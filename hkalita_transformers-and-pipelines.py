import warnings

warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

column_transformer = ColumnTransformer("One_Hot_Encoder",OneHotEncoder(),[0,1,2],remainder="passthrough")
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.compose import make_column_transformer

from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold,train_test_split



%matplotlib inline

plt.style.use("ggplot")
train_data = pd.read_csv("../input/adult-census-income/adult.csv")

train = train_data.copy()
print("Train Data Shape : {}".format(train.shape))
train.head(2)
# remove dot(.) from column names and insert '_'

for col in train.columns:

    new_name = col.replace(".","_")

    train.rename(columns={col:new_name},inplace=True)
#train data statistics

train_stats = pd.DataFrame()

train_stats["Columns"] = train.columns

train_stats["Missing_Values"] = train.isna().sum().values

train_stats["Unique_values"] = [train[x].nunique() for x in train.columns]

train_stats["Column_Type"] = [train[x].dtypes for x in train.columns]

skewness = []

for col in train.columns:

    try:

        skew = train[col].skew()

        skewness.append(skew)

    except:

        skewness.append("NA")

train_stats["Skewness"] = skewness

train_stats
#visualizations

train.hist(figsize=(12,9))

plt.tight_layout()

plt.show()
## Categorical plots for inference. Uncomment and run.

# categorical = ["workclass","education","marital_status","occupation","relationship","race","sex"]

# cnt = 0

# for cat in categorical:

#     plt.figure(figsize=(12,7))

#     sns.countplot(cat,hue="income",data=train)

#     plt.show()
#mapping target variable to make it binary

income_map = {"<=50K":0,">50K":1}

train.income = train.income.map(income_map)
df = train[["age","workclass","fnlwgt","education","education_num","occupation","sex","capital_gain","capital_loss","hours_per_week","income"]].copy()
most_frequent = df.capital_gain.value_counts(ascending=False).index[0]

freq = df.capital_gain.value_counts(ascending=False)[0]

print("Percentage of frequency of {} value in /capital_gain/ column = {}%".format(

    most_frequent,round((freq/df.capital_gain.value_counts().sum())*100,2)))



most_frequent = df.capital_loss.value_counts(ascending=False).index[0]

freq = df.capital_loss.value_counts(ascending=False)[0]

print("Percentage of frequency of {} value in /capital_loss/ column = {}%".format(

    most_frequent,round((freq/df.capital_loss.value_counts().sum())*100,2)))
df = df.drop(["capital_gain","capital_loss"], axis=1)
X = df.drop("income",axis=1)

y = df.income
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,stratify=y,random_state=3)
X_train.reset_index(drop=True,inplace=True)

X_test.reset_index(drop=True,inplace=True)
from sklearn.base import BaseEstimator, TransformerMixin

class transformation(BaseEstimator, TransformerMixin):

    def __init__(self,method="log2"):

        self.method = method

    def fit(self,X,y=None):

        #In this case the fit function is empty.

        return self

    def transform(self,X):

        X_ = X.copy()

        for col in X_.columns:

            if self.method == "log2":

                X_[col] = np.log2(X_[col])

            elif self.method == "log10":

                X_[col] = np.log10(X_[col])

            elif self.method == "sqrt":

                X_[col] = np.sqrt(X_[col])

            elif self.method == "cbrt":

                X_[col] = np.cbrt(X_[col])

        return X_
# # code for testing the skewness when respective transformations are done on each column

# for col in ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]:

#     print("\nOriginal Skewness for {} is {}".format(col,df[col].skew()))

#     for method in ["log2","log10","sqrt","cbrt"]:

#         trans = transformation(method)

#         trans.fit(df[col])

#         print("\nSkewness after {} transformation = {}".format(method,trans.transform(df[col]).skew()))

#     print("\n")
col_trans = make_column_transformer(

            (transformation("sqrt"),["age","fnlwgt"]),

            (OneHotEncoder(sparse=False),["workclass","education","occupation","sex"]),

            (StandardScaler(),["age","fnlwgt","education_num","hours_per_week"]),

            remainder = "passthrough"

            )
#cross folds

cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=3)

#model

logreg = LogisticRegression(solver="lbfgs")
#defining the steps for the pipeline

steps = [

    ("Col_trans",col_trans),

    ("Model",logreg)    

]

pipe = Pipeline(steps)
score_with_pipeline = cross_val_score(pipe,X,y,cv=cv,scoring="accuracy")
print("Mean Accuracy: {:.2f}%".format(score_with_pipeline.mean()*100))

print("Standard Deviation among Accuracy scores: {:.6f}".format(score_with_pipeline.std()))
#fitting the pipeline

pipe.fit(X_train,y_train)
#predicting on the test data

y_pred = pipe.predict(X_test[["age","workclass","fnlwgt","education","education_num","occupation","sex",

                              "hours_per_week"]])

print("Accuracy on the Test Data: {:.2f}%".format(accuracy_score(y_test,y_pred)*100))