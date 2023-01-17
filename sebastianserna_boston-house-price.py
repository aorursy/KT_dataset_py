import pandas as pd
import seaborn as sns
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(df_train.info())
df_train["SalePrice"].describe()
#histogram
sns.distplot(df_train['SalePrice'])
#correlation matrix
import matplotlib.pyplot as plt
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#print(corrmat)