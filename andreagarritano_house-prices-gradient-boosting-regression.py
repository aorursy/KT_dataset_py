import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



X_train_origin = data.drop(['SalePrice'], axis=1)

y = data.SalePrice

X_test_origin = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

X_train = X_train_origin.copy()

X_test = X_test_origin.copy()
threshold = 0.1

total = X_train.isnull().sum().sort_values(ascending = False)

percent = (X_train.isnull().sum()/X_train.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis =1, keys=['Total', 'Percentage'])

display(missing_data[(missing_data.Percentage > 0)])

X_train = X_train.drop(missing_data[missing_data.Percentage > threshold].index, axis = 1)

X_test = X_test.drop(missing_data[missing_data.Percentage > threshold].index, axis = 1)
threshold = 0.95

total = X_train.nunique()

percent = (X_train.nunique()/X_train.count())#.sort_values(ascending = False)

idness_data = pd.concat([total, percent], axis =1, keys=['Total', 'Percentage'])

display(idness_data[(idness_data.Percentage > 0.2)].sort_values(by='Percentage', ascending = False))

X_train = X_train.drop(idness_data[idness_data.Percentage > threshold].index, axis = 1)

X_test = X_test.drop(idness_data[idness_data.Percentage > threshold].index, axis = 1)

# Combine Test and Training sets to maintain consistancy.

combined = pd.concat([X_train,X_test],axis=0)

combined.head()

combined.shape



# Missing Value Handling

def HandleMissingValues(df):

    # for Object columns fill using 'UNKOWN'

    # for Numeric columns fill using median

    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]

    values = {}

    for a in cat_cols:

        values[a] = 'UNKOWN'



    for a in num_cols:

        values[a] = df[a].median()

        

    df.fillna(value=values,inplace=True)

    

HandleMissingValues(combined)

combined.head()



# Check for any missing values

combined.isnull().sum().sum()
#Categorical Feature Encoding

def getObjectColumnsList(df):

    return [cname for cname in df.columns if df[cname].dtype == "object"]



def PerformOneHotEncoding(df,columnsToEncode):

    return pd.get_dummies(df,columns = columnsToEncode)



cat_cols = getObjectColumnsList(combined)

combined = PerformOneHotEncoding(combined,cat_cols)

combined.head()
#respliting the data into train and test datasets

X_train=combined.iloc[:1460,:]

X_test=combined.iloc[1460:,:]

print(X_train.shape)

print(X_test.shape)
import xgboost as xgb



model_xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)

model_xgb.fit(X_train, y)

xgb_preds=model_xgb.predict(X_test)
#make the submission data frame

submission = {

    'Id': X_test_origin.Id.values,

    'SalePrice': xgb_preds

}

solution = pd.DataFrame(submission)

solution.head()
#make the submission file

solution.to_csv('submission.csv',index=False)