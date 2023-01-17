import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv("../input/craigslist-carstrucks-data/vehicles.csv")
df.shape
df.columns
df.head()
df.nunique(axis=0)
df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
NA_val = df.isna().sum()
def na_filter(na, threshold = .4):
    col_pass = []
    for i in na.keys():
        if na[i]/df.shape[0]<threshold:
            col_pass.append(i)
    return col_pass
df_cleaned = df[na_filter(NA_val)]
df_cleaned.shape
NA_val = df.isna().sum()
df_cleaned = df_cleaned[df_cleaned['price'].between(999.99, 250000)] # Computing IQR
Q1 = df_cleaned['price'].quantile(0.25)
Q3 = df_cleaned['price'].quantile(0.75)
IQR = Q3 - Q1
# Filtering Values between Q1-1.5IQR and Q3+1.5IQR
df_filtered = df_cleaned.query('(@Q1 - 1.5 * @IQR) <= price <= (@Q3 + 1.5 * @IQR)')
df_filtered.boxplot('price')
df_filtered.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
df_filtered = df_filtered[df_filtered['year'].between(1900, 2020)]# = 140000 + 1.5 * (140000-52379)
df_filtered = df_filtered[df_filtered['odometer'].between(0, 271431.5)]
df_final = df_filtered.copy().drop(['id','url','region_url','image_url','region','description','model','state','paint_color'], axis=1)
df_final.shape
import matplotlib.pylab as plt
import seaborn as sns
# calculate correlation matrix
corr = df_final.corr()# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
df_final['manufacturer'].value_counts().plot(kind='bar')
df_cleaned['type'].value_counts().plot(kind='bar')
df_final = pd.get_dummies(df_final, drop_first=True)
df_final.shape
df_final
from sklearn.preprocessing import StandardScaler
X_head = df_final.iloc[:, df_final.columns != 'price']
X = df_final.loc[:, df_final.columns != 'price']
y = df_final['price']
X
X.isnull().sum();X.drop(['lat','long'],axis=1,inplace = True)
X = StandardScaler().fit_transform(X)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
model = RandomForestRegressor(random_state=1)
X_train
y_train
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(mae(y_test, pred))
print(model.score(X_test,y_test))