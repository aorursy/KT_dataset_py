# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option("display.max_rows",None)

pd.set_option("display.max_columns",None)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

print(os.listdir("../input"))
data=pd.read_csv("../input/videogamesales/vgsales.csv")

data2=pd.read_csv("../input/videogamesales/vgsales.csv")
data.head()
data.shape
data.isnull().sum()
data[data.Publisher.isnull()]
data.shape
data_fight=data[data.Platform=="PS2"][["Publisher","Genre","Year","Name"]]
data_fight=data_fight.reset_index()

data_fight
data_fight[data_fight.Genre=="Fighting"]["Publisher"].value_counts()
data_fight.Year.value_counts()
data_fight[data_fight.Publisher.isnull()]
data.Publisher.value_counts().sort_values()
data.Publisher.fillna("Unknown",inplace=True)
data.Publisher.isna().sum()
data.Year.value_counts().plot(kind="bar")
plt.figure(figsize=(100,50))



sns.countplot(x="Year",data=data,hue="Genre")

xyz=pd.DataFrame(data.groupby(by=["Platform","Genre"])["Year"].median())

xyz

xyz.reset_index(inplace=True)
xyz.reset_index(inplace=True)
bnm=pd.DataFrame(data[data[["Platform","Genre","Year"]]["Year"].isna()][["Platform","Genre","Year"]]).reset_index()

bnm
xyz.Platform
for i in range(0,len(xyz)):

    for j in range(0,len(bnm)):

        if (xyz["Platform"][i]== bnm["Platform"][j]) & (bnm["Genre"][j]  ==  xyz["Genre"][i]) :

            bnm["Year"][j]=xyz["Year"][i]
bnm.isnull().sum()
for i in range(0,len(data)):

    for j in range(0,len(bnm)):

        if (data["Platform"][i]== bnm["Platform"][j]) & (bnm["Genre"][j]  ==  data["Genre"][i]) :

            data["Year"][i]=bnm["Year"][j]
data.isnull().sum()
# Which platform has the highest sales overall ?



# Which year has the highest sales globally for each platform ?



# top ten best selling games ?



# top 
data.head()
# Which platform has the highest sales overall ?



data.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).head(5).plot(kind="bar",color="r")
# Which genre has the highest sales overall ?



data.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False).head(5).plot(kind="line",color="r")
# Which year has the highest sales globally for each platform ?



sale_year=pd.DataFrame(data.groupby(["Platform","Year"])["Global_Sales"].sum()).reset_index()
sale_year=sale_year.groupby(['Platform'])['Global_Sales','Year'].max().reset_index()
sale_year.head()