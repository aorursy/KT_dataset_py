# This Python 3 environment comes with many helpful analytics libraries installed







# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python







# For example, here's several helpful packages to load in 















import numpy as np # linear algebra







import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







import itertools







from subprocess import check_output















import seaborn as sns







import matplotlib.pyplot as plt







%matplotlib inline















# Input data files are available in the "../input/" directory.







# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory















print(check_output(["ls", "../input"]).decode("utf8"))















# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/attacks_data_UTF8.csv',







                 encoding='latin1', parse_dates=['Date'],







                 infer_datetime_format=True,







                 index_col=1,







                )







df.info()
df_BD=df.loc[df['Country'] == "Bangladesh"]

df_BD=df_BD.loc[df_BD['City'] != "Kolkata"]

df_BD=df_BD.loc[df_BD['City'] != "West Bengal"]

df_BD
df_BD.City.value_counts().plot(kind='bar', figsize=(15, 10))

plt.title('Number of attacks by Cities')
df_BD.City.value_counts()
df_BD.Killed.plot(kind="bar",figsize=(150, 20))

plt.title("Killed by occurrence")
df_BD.groupby(df_BD.index.year).sum()[['Killed', 'Injured']].plot(kind='bar', figsize=(17, 7), subplots=False)

plt.title("Killed and Injured by Year")
df_BD.Injured.plot(kind="bar",figsize=(150, 20))

plt.title("Injured by occurrence")
df_BD[['Unnamed: 0']].groupby(df_BD.index.year).agg(['count']).plot(kind='bar',label='none', figsize=(17, 7), subplots=False)

plt.title("Occurrence by year")