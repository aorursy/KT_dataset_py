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



from fbprophet import Prophet

from fbprophet.plot import plot_plotly

import plotly.offline as py



import numpy  as np

import matplotlib.pyplot as plt   



import pandas as pd



import seaborn as sns
mtl_df=pd.read_csv("../input/montral-crimes/interventionscitoyendo.csv", encoding='ISO-8859-1')

mtl_df
mtl_df.info()
plt.figure(figsize=(10,10))

sns.heatmap(mtl_df.isnull(), cbar='Oranges', cmap='Oranges' )
mtl_df2=mtl_df.drop(["PDQ","X","Y","LONGITUDE","LATITUDE"], axis=1)

mtl_df2
mtl_df2.DATE=pd.to_datetime(mtl_df.DATE, format = "%Y/%m/%d")

mtl_df2.DATE
#lets convert our date column to index table

mtl_df2.index=pd.DatetimeIndex(mtl_df.DATE)

mtl_df2
mtl_df2["CATEGORIE"].value_counts()
#lets now how many crime in each year. we can't do it without making date as index

mtl_df3=mtl_df2.resample("Y").size()

mtl_df3
plt.plot(mtl_df3)

plt.title("crime per year")

plt.xlabel("Years")

plt.ylabel("nb of crim")
#as we see numbers of crimes going down over the years in Montreal City
mtl_df4=mtl_df2.resample("m").size()

mtl_df4
plt.plot(mtl_df4)

plt.title("crime per month")

plt.xlabel("months")

plt.ylabel("nb of crim")
#crims going up during summer time

mtl_df5=mtl_df2.resample("Q").size() #by quarter of year...every 3 months

mtl_df5

plt.plot(mtl_df5)

plt.title("crime per quarter")

plt.xlabel("quarter")

plt.ylabel("nb of crim")
mtl_prophet=mtl_df2.resample("m").size().reset_index() #resample the data frame based on month, and add index and number of crime

mtl_prophet
mtl_prophet.columns=["Date","Crime count"]

mtl_prophet
#to apply facebooke prophet we have to rename our column DS and Y only
mtl_prophet_df_final=mtl_prophet.rename(columns={"Date":"ds", "Crime count":"y"})

mtl_prophet_df_final
Pr=Prophet(weekly_seasonality=True, daily_seasonality=True)

Pr.fit(mtl_prophet_df_final)
future=Pr.make_future_dataframe(periods=365)

forecast=Pr.predict(future)

forecast
#lets visualize my crime rate

figure=Pr.plot(forecast,xlabel = "ds", ylabel = "Crime Rate")
fig2 = Pr.plot_components(forecast)


py.init_notebook_mode()



fig6 = plot_plotly(Pr, forecast)  # This returns a plotly Figure

py.iplot(fig6)