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
import pandas as pd

mpt_map = pd.read_excel('../input/mptmapv5/icg_cihan (4).xlsx') 

mpt_map.head(5)
mpt_map.head(5)

mpt_map.disTicHacmi /= 57

mpt_map.head(5)
import plotly.express as px

df = mpt_map



fig = px.scatter_geo(df, locations="kod",

                     hover_name="ulke", size="GSMH",

                     animation_frame="yil", size_max = 100, color= 'disTicHacmi',

                     projection="natural earth")

fig.update_layout(

    height=600,

    title_text='GSMH Değişimi (MPT Ülkeleri)', title_x=0.5

)

fig.show()
import plotly.express as px

df = mpt_map



fig = px.scatter_geo(df, locations="kod",

                     hover_name="ulke", size="disTicHacmi",

                     animation_frame="yil", size_max = 100, color= 'GSMH',

                     projection="natural earth")

fig.update_layout(

    height=600,

    title_text='Dış Ticaret Hacminin Değişimi (MPT Ülkeleri Arasında)', title_x=0.5

)

fig.show()
fig = px.choropleth(df, locations="kod",

                    color="disTicHacmi", 

                    hover_name="ulke", animation_frame=df["yil"],

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(transition = {'duration': 1000})

fig.show()
df2 = df

#df2['disTicDengesi'] = [0 if each < 0.006 else each for each in df.disTicDengesi]

fig = px.choropleth(df2, locations="kod",

                    color="disTicDengesi", 

                    hover_name="ulke", animation_frame=df["yil"],

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(transition = {'duration': 1000})

fig.show()
fig = px.bar(df, x="ulke", y="disTicHacmi", color="disTicHacmi",

  animation_frame="yil", animation_group="ulke")

fig.show()
fig = px.bar(df, x="ulke", y="GSMH", color="GSMH",

  animation_frame="yil", animation_group="ulke")

fig.show()
mpt_map.head(5)