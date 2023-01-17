from IPython.display import Image

import os

!ls ../input/

Image("../input/finalimg/IMG_20200715_102837.jpg",width=950,height=50)
Image("../input/finalimg/IMG_20200715_102743.jpg",width=950,height=100)
Image("../input/finalimg/IMG_20200715_102954.jpg",width=950,height=50)
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
import numpy as np

import pandas as pd

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

import plotly.express as px

import plotly.io as pio

%matplotlib inline
df = pd.read_csv("/kaggle/input/bangladesh-weather-dataset/Temp_and_rain.csv")
df.head()
df.info()
df.describe()
profile = pandas_profiling.ProfileReport(df)

profile
df.isnull().sum()
sns.set()

n = msno.bar(df,color="purple")
df1 = df.loc[0:599,:]

df1.tail()
sns.set()

plt.figure(figsize=(16,8))

sns.regplot(x = "Year",y="rain",fit_reg = False,data=df1)

plt.show()
pio.templates.default = "plotly_dark"

fig = px.line(df1,x='Year',y='rain',color="Year",title="Measurement of Rain according to Year")

fig.show()
df1.rain.max()
pio.templates.default = "plotly_dark"

fig = px.line(df1,x='Month',y='rain',color="Year",title="Measurement of Rain according to Month")

fig.show()
pio.templates.default = "plotly_dark"

fig = px.bar(df1[0:100],x='Month',y='rain',color="Year",title="Measurement of Rain according to Month")

fig.show()
fig = px.pie(data_frame=df1,values="rain",names='Month',labels= {"Month"},

             title="Pie chart of  Rain according to  Month")

fig.update_traces(textposition ='inside',textinfo='percent+label')

fig.show()

pio.templates.default = "plotly_dark"

fig = px.bar(df1,x='Month',y='tem',color="Year",title="Measurement of Temperature according to Month")

fig.show()
pio.templates.default = "plotly_dark"

fig = px.scatter(df1,x='Month',y='tem',color="Year",title="Measurement of Temperature according to Month")

fig.show()
fig = px.pie(data_frame=df1,values="tem",names='Month',labels= {"Month"},

             title="Pie chart of Temperature according to  Month")

fig.update_traces(textposition ='inside',textinfo='percent+label')

fig.show()

pio.templates.default = "plotly_dark"

fig = px.scatter(df1,x='tem',y ='rain',color="Month",title="Measurement of Rain according to Temperature")

fig.show()
pio.templates.default = "plotly_dark"

fig = px.scatter(df1,x='tem',y ='rain',animation_frame ='Year',animation_group = 'Month',

                 title="Changes of Rain according to Temperature During the Year",size="rain",color="tem",

                 hover_name="Month",log_x = True,size_max = 55)

fig.show()
df2 = df.loc[600:,:]

df2.tail()
pio.templates.default = "plotly_dark"

fig = px.line(df2,x='Year',y='rain',color="Year",title="Measurement of Rain according to Year")

fig.show()
df2.rain.max()
pio.templates.default = "plotly_dark"

fig = px.line(df2,x='Month',y='rain',color="Year",title="Measurement of Rain according to Month")

fig.show()
pio.templates.default = "plotly_dark"

fig = px.bar(df2,x='Month',y='rain',color="Year",title="Measurement of Rain according to Month")

fig.show()
fig = px.pie(data_frame=df2,values="rain",names='Month',labels= {"Month"},

             title="Pie chart of  Rain according to  Month")

fig.update_traces(textposition ='inside',textinfo='percent+label')

fig.show()
pio.templates.default = "plotly_dark"

fig = px.bar(df2,x='Month',y='tem',color="Year",title="Measurement of Temperature according to Month")

fig.show()
fig = px.pie(data_frame=df2,values="tem",names='Month',labels= {"Month"},

             title="Pie chart of Temperature according to  Month")

fig.update_traces(textposition ='inside',textinfo='percent+label')

fig.show()

pio.templates.default = "plotly_dark"

fig = px.scatter(df1,x='tem',y ='rain',color="Month",title="Measurement of Rain according to Temperature")

fig.show()
pio.templates.default = "plotly_dark"

fig = px.scatter(df2,x='tem',y ='rain',animation_frame ='Year',animation_group = 'Month',

                 title="Changes of Rain according to Temperature During the Year",size="rain",color="tem",

                 hover_name="Month",log_x = True,size_max = 55)

fig.show()
mapping = {1 : "Winter",2:"Winter",12: "Winter",3:"Spring",4:"Spring",10:"Late autumn",

           11:"Late autumn",8:"Autumn",9:"Autumn",6:"Rainy",7:"Rainy",4:"Summer",5:"Summer"}

df["Season"] = df["Month"].map(mapping).astype(str)
df["Season"].value_counts()
df.head()
df3 = df.loc[0:599,:]

df3.tail()
pio.templates.default = "plotly_dark"

fig = px.bar(df3,x='Season',y='rain',color="Year",title="Rain according to Season(1901-1950)")

fig.show()
fig = px.pie(data_frame=df3,values="rain",names='Season',labels= {"Season"},

             title="Pie chart of Rain according to  Season")

fig.update_traces(textposition ='inside',textinfo='percent+label')

fig.show()

pio.templates.default = "plotly_dark"

fig = px.scatter(df3,x='tem',y ='rain',animation_frame ='Year',animation_group = 'Month',

                 title="Relation between rain,temperature & Season During the Year(1901-1950)",size="rain",color="Season",

                 hover_name="Month",log_x = True,size_max = 55)

fig.show()
df4 = df.loc[600:,:]

df4.tail()
pio.templates.default = "plotly_dark"

fig = px.bar(df4,x='Season',y='rain',color="Year",title="Rain according to Season(1951-2015)")

fig.show()
fig = px.pie(data_frame=df4,values="rain",names='Season',labels= {"Season"},

             title="Pie chart of Rain according to  Season (1951-2015)")

fig.update_traces(textposition ='inside',textinfo='percent+label')

fig.show()

pio.templates.default = "plotly_dark"

fig = px.scatter(df4,x='tem',y ='rain',animation_frame ='Year',animation_group = 'Month',

                 title="Relation between rain,temperature & Season During the Year(1951-2015)",size="rain",color="Season",

                 hover_name="Month",log_x = True,size_max = 55)

fig.show()