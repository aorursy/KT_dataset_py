# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
file_path='../input/coronavirusdataset/Case.csv'
data = pd.read_csv(file_path,index_col="case_id", parse_dates=True)
data.head()
#группируем данные по городу
datas=data.groupby(['city']).confirmed.sum().reset_index()
plt.figure(figsize=(30,10))
sns.barplot(x=datas.city, y=datas.confirmed)



#нейросеть по приложения из гугл плей
file_path='../input/google-play-store-apps/googleplaystore_user_reviews.csv'
data = pd.read_csv(file_path,index_col="App", parse_dates=True)
#заменяем для обучения
data["Sentiment"]= data["Sentiment"].replace("Positive", 1)
data["Sentiment"]= data["Sentiment"].replace("Neutral", 0)
data["Sentiment"]= data["Sentiment"].replace("Negative", -1)
data.head()

for index, row in data.iterrows():
    if not isinstance(row['Sentiment_Polarity'], float) or not isinstance(row['Sentiment_Subjectivity'], float): 
        data.drop(index, inplace=True)


pd.DataFrame(data).fillna(0,inplace=True)   
#по чему будет учить
features = ['Sentiment_Polarity','Sentiment_Subjectivity']
X=data[features]
#то что ищем
y = data.Sentiment
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
val_X.reset_index(inplace=True)
#записываем данные
output = pd.DataFrame({'Sentiment_Polarity':val_X.Sentiment_Polarity,'Sentiment_Subjectivity':val_X.Sentiment_Subjectivity,'Sentiment': rf_val_predictions})
#для удобства
output["Sentiment"]= output["Sentiment"].replace(1,"Positive")
output["Sentiment"]= output["Sentiment"].replace(0,"Neutral")
output["Sentiment"]= output["Sentiment"].replace(-1,"Negative")
print(output)