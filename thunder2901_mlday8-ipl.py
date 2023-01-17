import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/ipl-data/IPL2013.csv')

df.head(10)
df.columns
df = df.drop(['Sl.NO.','PLAYER NAME'],axis=1)
df.head()
for col in df.columns:

    print(col,":",df[col].dtype)
features = list(df.columns)

categorical_features = ['COUNTRY','TEAM','PLAYING ROLE']

numerical_features = []

for f in features:

    if f not in categorical_features:

        numerical_features.append(f)
print(numerical_features)
numerical_features.remove('SOLD PRICE')
print(numerical_features)
df3 = df[categorical_features]

df3.head()
df4 = pd.get_dummies(df)
df4.columns
from statsmodels.stats.outliers_influence import variance_inflation_factor



def get_vif_factors(x):

    x_matrix = x.as_matrix()

    vif = [variance_inflation_factor(x_matrix,i) for i in range(x_matrix.shape[1])]

    vif_factors = pd.DataFrame()

    vif_factors['column'] = x.columns

    vif_factors['vif'] = vif

    return vif_factors
vif_factors = get_vif_factors(df2)

vif_factors
large_vif = vif_factors[vif_factors.vif > 4].column
large_vif
selected_features = large_vif
df5 = df[selected_features]

corrmat = df5.corr()



f, ax = plt.subplots(figsize =(15, 12)) 

sns.heatmap(abs(corrmat), ax = ax, cmap ="YlGnBu", linewidths = 0.1)
corrmat = corrmat.abs()

s = corrmat.unstack()

s = s.sort_values(ascending=False).drop_duplicates()

print(s[:19])
df6 = df[selected_features]

df6.head()
df6 = pd.concat((df6,df['SOLD PRICE']),axis=1)
df6.head()
cat = df[categorical_features]

cat.head()
enc = pd.get_dummies(cat)

enc.head()
final_df = pd.concat((enc,df6),axis=1)

final_df.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



X = final_df.drop('SOLD PRICE',axis=1)

y = final_df['SOLD PRICE']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)





lr = LinearRegression()

lr.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error



predicts = lr.predict(X_test)

sc = mean_absolute_error(predicts,y_test)

sc