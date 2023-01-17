import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder, StandardScaler 

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import make_pipeline

from lightgbm import LGBMClassifier

from category_encoders import OneHotEncoder

from sklearn.model_selection import cross_val_predict

from warnings import filterwarnings

from sklearn.impute import SimpleImputer

filterwarnings('ignore')

import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

print("train shape", train.shape)

test = pd.read_csv("../input/test.csv")

print("test shape", test.shape)
target_column = "target"

id_column = "id"

categorical_cols = [c for c in test.columns if test[c].dtype in [np.object]]

numerical_cols = [c for c in test.columns if test[c].dtype in [np.float, np.int] and c not in [target_column, id_column]]

print("Number of features", len(categorical_cols)+len(numerical_cols))
classifier = make_pipeline(

    ColumnTransformer([

        ('num', StandardScaler(), numerical_cols),

        ('cat', OneHotEncoder(), categorical_cols),    

    ]),

    LGBMClassifier(n_jobs=-1)

)
%%time

oof_pred = cross_val_predict(classifier, 

                             train, 

                             train[target_column], 

                             cv=5,

                             method="predict_proba")
print("Cross validation AUC {:.4f}".format(roc_auc_score(train[target_column], oof_pred[:,1])))
sub = pd.read_csv("../input/sample_submission.csv")

sub.head()
%%time

classifier.fit(train, train[target_column])

test_preds = classifier.predict_proba(test)[:,1]

sub[target_column] = test_preds

sub.to_csv("submission.csv", index=False)