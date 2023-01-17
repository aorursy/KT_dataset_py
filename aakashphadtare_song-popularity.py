import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
%matplotlib inline
df = pd.read_csv('../input/top-songs-eda/top_songs.csv')
df.head()
for i in df.columns:
    print(i,'\n', df[i].nunique(),'\n'*2)
cor = df.corr()
plt.figure(figsize = (12,12))
sns.heatmap(cor,annot= True, cmap = 'seismic')
plt.figure(figsize= (12,12))
sns.barplot(x='top genre',y = 'bpm', data = df)
plt.xticks(rotation = 90)
plt.show()
df1 = df.copy()
df1.head()
# feature engineering : combining genres
genre = ['hip hop','pop','rap','edm','rock', 'electro', 'edm+house' ,'r&b' ]
df['top genre']= df['top genre'].str.replace(' ',' , ')
df['top genre'].head()
df.head()
df.columns
df['dB'] = df['dB'].apply(lambda x: str(x).replace('-',''))
df.head()
df.drop(['id', 'title', 'artist', 'top genre'],axis =1, inplace =True)
df.dB = pd.to_numeric(df.dB, errors = 'coerce')
df.info()
df.year = df.year - 2019
df.head()
sdfdf
X=df.drop('pop',axis=1)
y=df['pop']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
#building a model use fit

lr = LinearRegression()
lr.fit(X, y)
print(lr.coef_)
print(f'Coefficients: {lr.coef_}')
print(lr.intercept_)
print(f'Intercept: {lr.intercept_}')
print(lr.score(X,y) )
print(f'R^2 score: {lr.score(X, y)}')
df.year.unique()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train.head()
