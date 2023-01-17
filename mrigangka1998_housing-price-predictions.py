import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv",index_col="Id")

train_df.head()
test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv",index_col="Id")

test_df.head()
plt.hist(train_df["SalePrice"])

plt.xlabel("House Prices")

plt.ylabel("No. of houses")

plt.title("Distribution of House prices in Ames, Iowa")

plt.show()
train_df.dropna(axis=0,subset=["SalePrice"],inplace=True)  #drop rows with no sale price

y_train=train_df.SalePrice  #define the target variable

X_train=train_df.drop("SalePrice",axis=1)  #drop sale price from predictor df
#Columns with numerical data

num_cols=[col for col in X_train.columns if X_train[col].dtype in ["int64","float64"]]

print("There are "+str(len(num_cols))+" numerical columns.")

num_cols
#Columns with categorical data

cat_cols=[col for col in X_train.columns if X_train[col].dtype=="object"]

print("There are "+str(len(cat_cols))+" categorical columns.")

cat_cols
X_train[cat_cols].nunique()
cols=num_cols+cat_cols

X_train=X_train[cols]

X_test=test_df[cols]
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



num_transformer=SimpleImputer(strategy="median")



cat_transformer=Pipeline(steps=[

    ("impute",SimpleImputer(strategy="most_frequent")),

    ("onehot",OneHotEncoder(handle_unknown="ignore",sparse=False))

])



preprocessor=ColumnTransformer(transformers=[

    ("num",num_transformer,num_cols),

    ("cat",cat_transformer,cat_cols)

])
from xgboost import XGBRegressor



model=XGBRegressor(n_estimators=1000,learning_rate=0.05,random_state=0)
pipe=Pipeline(steps=[

    ("preprocessor",preprocessor),

    ("model",model)

])
pipe.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score



scores=-cross_val_score(pipe,X_train,y_train,cv=5,scoring="neg_mean_absolute_error") #The negative sign is used because negative mse is calculated

print("The average mean absolute error is "+str(scores.mean()))
preds=pipe.predict(X_test)
output_df=pd.DataFrame({"Id":X_test.index,"SalePrice":preds})

output_df.head()
output_df.to_csv("test_predictions.csv",index=False)