import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
df.head()
df.head().T
df.info()
df.describe()
df.isna().sum()
max_unique = 60
high_unique = [col for col in df.select_dtypes(exclude=np.number)
                   if df[col].nunique() > max_unique]
df = df.drop(columns=high_unique)
df.info()
df.head().T
df['currency_buyer'].value_counts()
corr = df.corr()
corr
fig,ax = plt.subplots(figsize = (16,16))
ax = sns.heatmap(corr,
                 annot=True,
                 linewidths=1.2,
                 fmt=".2f",
                 cmap="YlGnBu");
sns.countplot(df['origin_country']);
sns.countplot(df['urgency_text']);
sns.barplot(x = df.origin_country,y = df.units_sold);
df['has_urgency_banner'].value_counts()
df = df.drop(['crawl_month','origin_country','rating_count','shipping_option_name','urgency_text'],axis = 1)
df.head().T
df.info()
le = LabelEncoder()
df['currency_buyer'] = le.fit_transform(df['currency_buyer'])
df['theme'] = le.fit_transform(df['theme'])
for label,content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isna(content).sum():
            print(label)
df['has_urgency_banner'].value_counts()
df['has_urgency_banner'] = df['has_urgency_banner'].fillna(0)
for label,content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isna(content).sum():
            df[label] = content.fillna(content.median())
df.info()
df.head()
df.isna().sum()
x = df.drop('units_sold',axis = 1)
y = df['units_sold']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2)
model = RandomForestRegressor(n_estimators = 1000,random_state = 42)
model.fit(x_train,y_train)
model.score(x_test,y_test)
model1 = LinearRegression()
model1.fit(x_train,y_train)
model1.score(x_test,y_test)
scores = pd.DataFrame({'RandomForest': model.score(x_test,y_test),
                       'LinearRegression': model1.score(x_test,y_test)},
                        index = [0])

scores.T.plot(kind = 'bar',
              figsize = (10,10))
plt.title('Scores of all Model')
plt.xlabel('Model Name')
plt.ylabel('Scores');
