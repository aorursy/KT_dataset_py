!pip install pycaret
import pandas as pd
import numpy as np
#Import the dataset
train = pd.read_csv("../input/dsn-ai-oau-july-challenge/train.csv")
test = pd.read_csv('../input/dsn-ai-oau-july-challenge/test.csv')
sub = pd.read_csv('../input/dsn-ai-oau-july-challenge/sample_submission.csv')
test['Product_Supermarket_Sales'] = 100
train.shape
df = pd.concat([train,test])
df = df.set_index('Product_Supermarket_Identifier')
df.head()
df['Product_Identifier'].nunique()
df[pd.isnull(df['Product_Weight'])]["Product_Identifier"].nunique()
no_nan = df.dropna()
no_nan.isnull().sum()
result = {}
  # Go through each letter in the text
for product in no_nan['Product_Identifier']:
  if product not in result:
    result[product] = no_nan[no_nan['Product_Identifier'] == product]["Product_Weight"].mean()

result
len(result)

def fill(row):
  for ke in result.keys():
    if row['Product_Identifier'] == ke:
      return result[ke]

df['Product_Weight'] = df.apply(lambda row : fill(row), axis=1) 

df.isna().sum()
df["Product_Weight"].describe()
df["Product_Weight"].fillna(12.933616, inplace = True)
df["Supermarket _Size"].fillna("Medium", inplace = True)
df['Supermarket_Opening_Year'] = 2017 - df['Supermarket_Opening_Year']
df.drop(['Product_Identifier','Supermarket_Identifier'], axis = 1, inplace = True)
train = df[0:3742]
test = df[3742:]
train.shape
from pycaret.regression import *

exp_reg = setup(train, target = "Product_Supermarket_Sales")
compare_models()
model = create_model("xgboost")
tuned_model = tune_model("xgboost")
predictions = predict_model(tuned_model)
finalize_model(tuned_model)
save_model(tuned_model, 'sales_prediction_gbr')
test.drop("Product_Supermarket_Sales", axis = 1, inplace = True)
predictions = predict_model(tuned_model, data = test)
sub["Product_Supermarket_Sales"] = predictions['Label']
sub.to_csv('submission_sales_prediction.csv',index= False)
sub.head()
