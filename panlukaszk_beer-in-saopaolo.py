import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/Consumo_cerveja.csv", 
                 names=['date','temp_avg','temp_min','temp_max',
                        'rain','finalsemana','consumption'], header=0, decimal=',')
# df.head()
df['consumption'] = df['consumption'].astype('float')
df.dropna(inplace=True)
df.info()
import seaborn as sns
corr = df.corr()
sns.heatmap(corr)
df.plot.scatter('temp_max','consumption')
df.plot.scatter('rain','consumption')
from sklearn.model_selection import train_test_split

X = df[['temp_max']]
y = df['consumption']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)
print("shapes of train (%s) and test (%s)" % (X_train.shape, X_test.shape))
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
reg.score(X_test, y_test)