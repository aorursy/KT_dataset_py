# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")

data.head()
countryand_rates=data[["year","country","suicides/100k pop"]]

countryand_rates.tail()
import plotly.express as px

fig = px.choropleth(countryand_rates, locations="country",

                    color="suicides/100k pop", # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig
df = px.data.gapminder().query("year==2007")

df.head()
isoalphas=df[["country","iso_alpha"]].drop_duplicates()

isoalphas

isoalphas=isoalphas.append(pd.DataFrame({"country":["Republic of Korea"],"iso_alpha":["KOR"]}), ignore_index = True) 

isoalphas=isoalphas.append(pd.DataFrame({"country":["Lithuania"],"iso_alpha":["LTU"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["Suriname"],"iso_alpha":["SUR"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["Estonia"],"iso_alpha":["EST"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["Russian Federation"],"iso_alpha":["RUS"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["Belarus"],"iso_alpha":["BLR"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["Saint Vincent and Grenadines"],"iso_alpha":["VCT"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["Kazakhstan"],"iso_alpha":["KAZ"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["India"],"iso_alpha":["IND"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["Israel"],"iso_alpha":["ISR"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["United Arab Emirates"],"iso_alpha":["ARE"]}), ignore_index = True)

isoalphas=isoalphas.append(pd.DataFrame({"country":["Canada"],"iso_alpha":["CAN"]}), ignore_index = True)

isoalphas[70:85]
isoalphacol=[]

for country in countryand_rates["country"]:

    this=isoalphas[isoalphas["country"]==country]

    #print(type(this["iso_alpha"].values))

    iso_names="".join(this["iso_alpha"].values)

    

    isoalphacol.append(iso_names)
str(isoalphacol[4])
countryand_rates["iso_alpha"]=isoalphacol
countryand_rates
import plotly.express as px

count_2014=countryand_rates[(countryand_rates["year"]==2014)&(countryand_rates["suicides/100k pop"]>0.5)]

dropped=count_2014.drop_duplicates(subset=["country"])

fig = px.choropleth(dropped, locations="iso_alpha",

                    color="suicides/100k pop",

                    hover_name="country",

                    color_continuous_scale=px.colors.sequential.Plasma)

fig
dropped_with_year=countryand_rates.drop_duplicates(subset=["country","year"])

fig = px.choropleth(dropped_with_year, 

                    locations="country", 

                    locationmode = "country names", 

                    color="suicides/100k pop", 

                    hover_name="country", 

                    animation_frame="year"

                   )



fig.update_layout(

    title_text = 'Suicide Rates',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

fig
isoalphas.to_csv("country_iso_alphas.csv",index=False)