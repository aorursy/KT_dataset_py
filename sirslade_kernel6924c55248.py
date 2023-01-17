# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import eli5

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from eli5.sklearn import PermutationImportance

from eli5.xgboost import get_feature_importance_explanation

from imblearn.over_sampling import SMOTE

from category_encoders import OneHotEncoder



from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import IsolationForest

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# All unclean data

X = pd.read_csv("/kaggle/input/santander-customer-satisfaction/train.csv", index_col="ID")

Y = X["TARGET"]

X.describe()
#All negative labels

X[X["TARGET"] == 0].describe()
#All positive labels

X[X["TARGET"] == 1].describe()
# All columns that have just 1 large negative value in an otherwise positive set

desc = X.describe()

neg_outs = [x for x in desc.columns if (desc.loc["min", x] < 0) and (desc.loc['25%', x] >= 0)]

print(len(neg_outs))

mark_20 = X.shape[0] * .1

neg_outs = [x for x in neg_outs if (X[X[x] < 0][x].value_counts().shape[0] == 1)]

indexes = []

for column in neg_outs:

    indexes.extend((X[X[column] < 0]).index.tolist())

indexes

#For dropping outliers

X.drop(index=indexes, inplace=True)

print(str(len(indexes)) + " rows dropped")

#outliers = IsolationForest().fit_predict(X)

#X[outliers == -1]

#All columns that have binary values

binaries = [x for x in X.columns if (X[x].value_counts().shape[0] == 2) & (x != "TARGET")]

binaries
#All columns that contain only 1 value

single_values = [x for x in X.columns if X[x].value_counts().shape[0] == 1]

print("Length: " + str(len(single_values)))

X.drop(columns=single_values, inplace = True)
# One hot encoding for features that have cardinality of less than 20

low_cardinality = [x for x in X.columns if X[x].nunique() < 20]

low_cardinality.remove("TARGET")

print("Low cardinality features: " + str(len(low_cardinality)))

X = OneHotEncoder(cols=low_cardinality, return_df=True).fit_transform(X)

X
#We drop the target column aftere the dataset has been cleaned

Y = X["TARGET"]

X.drop(columns=["TARGET"], inplace=True)
from sklearn.model_selection import cross_val_score



def run_model(model, tr_x, tr_y, te_x, te_y, debug=True):

    model.fit(tr_x, tr_y)

    preds = model.predict(tr_x)

    training_error = (abs(tr_y-preds).sum())/tr_x.shape[0] *100

    preds = model.predict(te_x)

    error = (abs(te_y-preds).sum())/te_x.shape[0] *100 # MAE

    

    if debug:

        print("Pred satisfied: " + str((preds.sum()/te_x.shape[0])))

        print("Act satisfied:" + str((te_y.sum()/te_x.shape[0])))

        print("Score: " + str((te_y.sum()- preds.sum())/te_x.shape[0])) #The difference between the quantity predicted of how many survived

        print("Error: " + str(error)) # How many incorrect predictions divided by the number of predictions

        print("Accuracy: " + str(100 - error))

        print("Training accuracy: " + str(100 - training_error))

    

def cross_validation(model, X, y, debug=True):

    scores = cross_val_score(model, X=X, y=y, cv= 10)

    if debug:

        print(scores)

    return scores.mean()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



trainX, valX, trainY, valY = train_test_split(X, Y)



# Create synthetic values only from training data

synthX, synthY= SMOTE(sampling_strategy={"Satisfied":6000}).fit_resample(trainX, trainY.map({1: "Satisfied", 0: "NotSatisfied"}))

synthY = synthY.map({"Satisfied" : 1, "NotSatisfied" : 0}) # We have to remap the values of Y
rfc = RandomForestClassifier(n_estimators = 10, random_state=101)

run_model(rfc, trainX, trainY, valX, valY)

print("\nWith synthetic data:")

run_model(rfc, synthX, synthY, valX, valY)

cross_validation(rfc, X, Y)
xgb = XGBClassifier(n_estimators = 10, random_state= 101)

run_model(xgb, trainX, trainY, valX, valY)

print("\nWith synthetic data:")

run_model(xgb, synthX, synthY, valX, valY)

cross_validation(xgb, X, Y)
perm_rfc = PermutationImportance(rfc, random_state=2).fit(valX, valY)

eli5.show_weights(perm_rfc, feature_names=valX.columns.tolist())
keep_columns = ["var15", "num_var5_0", "saldo_medio_var8_ult3", "num_op_var41_efect_ult3", "ind_var9_ult1", "ID", "num_var42", "saldo_var37", 

                "num_var40_0", "saldo_var8", "ind_var25_cte"]
preprocessor = ColumnTransformer(

    transformers=[

    ("Important", "passthrough", keep_columns)

    ],

    remainder="drop")



pipeline = Pipeline(

    steps=[

        ("prep", preprocessor),

        ("RandomForest", rfc)

    ])



binaries
import itertools



import matplotlib.pyplot as plt



nonbinaries = X.drop(columns=binaries).columns

for tupla in itertools.combinations(nonbinaries, 2):

    sns.scatterplot(x=X[tupla[0]], y=X[tupla[1]], hue=X[binaries[0]])

    plt.show()



X["ind_var2"].value_counts().shape
len(indexes)