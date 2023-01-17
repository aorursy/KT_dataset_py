import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
print(df_train.shape)
print(df_test.shape)
df_train.head()
def is_categorical(data, key):
    return data[key].dtype == "object" or hasattr(data[key], 'cat')

dummy_train = df_train.drop('Id', axis=1)
for key in df_train.keys():
    if is_categorical(df_train, key):
        indexes = df_train[key].value_counts(dropna=True).index
        dummy_train[key] = dummy_train[key].astype('category')
        dummy_train[key] = dummy_train[key].cat.set_categories(indexes, ordered=True)
        
dummy_train.info()
def plot_count(x, data, ax):
    if is_categorical(data, x):
        sns.countplot(x=x, data=data, ax=ax)
    else:
        sns.distplot(data[x].dropna(), ax=ax) # dropna 良いのか

def plot_corr(x, data, ax):
    if is_categorical(data, x):
        sns.boxplot(x=x, y='SalePrice', data=data, ax=ax)
    else:
        sns.regplot(x=x,y='SalePrice',data=data, ax=ax, fit_reg=False, x_estimator=np.mean)

rows = dummy_train.shape[1]
axes = plt.figure(figsize=(15,8*rows)).subplots(nrows=rows, ncols=2)
for n in range(rows):
    key = dummy_train.keys()[n]
    plot_count(x=key, data=dummy_train,ax=axes[n, 0])
    plot_corr(x=key, data=dummy_train, ax=axes[n, 1])
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd

df_dummy_train = pd.get_dummies(df_train,drop_first=True)
df_dummy_test = pd.get_dummies(df_test,drop_first=True)

target = df_dummy_train["SalePrice"].copy()
conditions = ["OverallQual", 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'SaleCondition_AdjLand',
 'SaleCondition_Alloca',
 'SaleCondition_Family',
 'SaleCondition_Normal',
 'SaleCondition_Partial', 'GarageQual_Fa', 'GarageQual_Gd', 'GarageQual_Po', 'GarageQual_TA', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA',  "Street_Pave", "YearBuilt", "GrLivArea",'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA',"1stFlrSF", "2ndFlrSF"]
train = df_dummy_train[conditions].copy()
df_dummy_test = df_dummy_test.reindex(columns=conditions)

X_train, X_test, Y_train, Y_test = train_test_split(train, target,test_size=.3,random_state=0)
model = SVR(kernel='linear',C=1e3,epsilon=1.0).fit(X_train, Y_train)
model
predicted = model.predict(X_test)
model.score(X_test, Y_test)
# 0.727894717945872
test = df_dummy_test[conditions].copy()
test = test.fillna(0.0)
test_predicted = model.predict(test)
output = df_test.copy()
output["SalePrice"] = test_predicted
output[["Id","SalePrice"]].to_csv("submission.csv",index=False)
output[["Id","SalePrice"]]
