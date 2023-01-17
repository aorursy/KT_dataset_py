# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#มีสมการ ขีด regression เอง หาค่า k เอง วิเคราะห์กราฟ

df=pd.read_csv("../input/video-games-sales-2019/vgsales-12-4-2019-short.csv") 
df.head()
df=pd.read_csv("../input/video-games-sales-2019/vgsales-12-4-2019-short.csv") 
df.head()
df=pd.read_csv("../input/video-games-sales-2019/vgsales-12-4-2019-short.csv",index_col="Genre") 
df.head()
df = pd.read_csv("../input/video-games-sales-2019/vgsales-12-4-2019-short.csv")
df.describe()
df = pd.read_csv("../input/video-games-sales-2019/vgsales-12-4-2019-short.csv")
print(df[100:105].Total_Shipped)
df = pd.read_csv("../input/video-games-sales-2019/vgsales-12-4-2019-short.csv")
print(df[:].Total_Shipped)
df = pd.read_csv("../input/video-games-sales-2019/vgsales-12-4-2019-short.csv")
cols=['Year','Total_Shipped']
sns.pairplot(data=df[cols])
df = pd.read_csv("../input/video-games-sales-2019/vgsales-12-4-2019-short.csv")
sns.scatterplot(x='User_Score',y='Critic_Score',data=df)
df = pd.read_csv("../input/video-games-sales-2019/vgsales-12-4-2019-short.csv")
cols=['Year','Total_Shipped']
sns.pairplot(df,kind='reg')
df = pd.read_csv('../input/video-games-sales-2019/vgsales-12-4-2019-short.csv')
df.head()
x=df.JP_Sales.values.reshape(-1,1)
y=df.Critic_Score.values.reshape(-1,1)
sns.lmplot(x='Total_Shipped',y='Critic_Score',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
x_input=[[5]]
y_predict=model.predict(x_input)
y_predict