import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
import os
print(os.listdir("../input"))
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
#(1460, 81)
train_data.head(20)
fix, ax = plt.subplots(figsize=(18, 10))
sns.heatmap(train_data.isnull())
plt.show()
var = ['LotArea', 'YearBuilt', 'GrLivArea', 'GarageArea', 'SaleCondition', 'SalePrice']
var_test = ['LotArea', 'YearBuilt', 'GrLivArea', 'GarageArea', 'SaleCondition']
def get_aljja(df, test=False):
    if test:
        df_dict = {v: df[v] for v in var_test}
    else:
        df_dict = {v: df[v] for v in var}
    df_copy = pd.DataFrame(df_dict)
    condition = {
        'Family': 3,
        'Normal': 2,
        'Partial': 1,
        'AdjLand': 1.5,
        'Alloca': 0.5,
        'Abnormal': 0,
        'Abnorml': 0
    }
    df_copy['SaleCondition'] = df_copy['SaleCondition'].apply(lambda v: condition[v])
    df_copy.fillna(0)
    return df_copy
df_quantify = get_aljja(train_data)
X = df_quantify.drop(['SalePrice'], axis=1)
y = df_quantify['SalePrice']
clf = XGBClassifier()
clf.fit(X, y)
plot_importance(clf)
plt.show()
df_test_quantify = get_aljja(test_data, test=True)
y_pred = clf.predict(df_test_quantify)
my_submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': y_pred})
my_submission.to_csv('submission.csv', index=False)
my_submission.head()
