import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import BaggingRegressor
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
def transformDF(data):
    data = data.fillna(0)
    for col in data:
        if data[col].dtype == object:
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))    
    return data
y = np.log1p(df["SalePrice"])
X = transformDF(df.drop(["SalePrice"], axis=1))
model = BaggingRegressor().fit(X, y)
preds = model.predict(transformDF(test_df))
preds = np.exp(preds)
submission = pd.DataFrame(preds)
submission.index.name = "Id"
submission.index = test_df.Id
submission.columns = ['SalePrice']
submission.to_csv("submission.csv")
