# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#lets read dataset
df=pd.read_csv("/kaggle/input/alphabet-inc-stockprice-history-dataset/GOOG.csv")
df
import matplotlib.pyplot as plt
x=df["Date"]
y=df["Close"]
plt.plot(x,y)
plt.xlabel('Date')
plt.ylabel('Close Stock Values')
plt.show()
df.describe()
#making dataset using those two columns
data=df[["Date","Close"]]
#lets rename columns
data=data.rename(columns={"Date":"ds","Close":"y"})
data.head(5)
from fbprophet import Prophet
model=Prophet(daily_seasonality=True)
model.fit(data)
future=model.make_future_dataframe(periods=365) #specifying number of days for future
prediction=model.predict(future)
model.plot(prediction)

#plot predictions now
plt.title("prediction of Googl's future stock price using prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()
model.plot_components(prediction)
plt.show()

