import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



#plotly libraries

import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



#pycounter for country codes

import pycountry





#read the data: EDA starts here:

bank_data = pd.read_csv('../input/procurement-notices.csv')



#renaming the column name:

bank_data.columns = [str(v).replace(' ','_') for v in list(bank_data.columns)]



#coverting the dates:

bank_data.loc[:,'Publication_Date'] = pd.to_datetime(bank_data.Publication_Date)

bank_data.loc[:,'Deadline_Date'] = pd.to_datetime(bank_data.Deadline_Date)





#number of calls currently out:



no_of_due = bank_data[(bank_data.Deadline_Date > pd.datetime.today()) | (bank_data.Deadline_Date.isnull())]['Deadline_Date']

#fig,ax = plt.subplots()

#ax.bar('No. of due',no_of_due)

print('Number of projects in progress:',len(no_of_due))
# Cell to find the ISO3 country code:



all_country = [c.alpha_3 for c in list(pycountry.countries)]



def get_code(c='India'):

    #print(c)

    get_c = pycountry.countries.get(name = c)

    if get_c:

        code = get_c.alpha_3

        return code

    else:

        return np.NaN

    

#Adding the country code with ISO_3

bank_data.loc[:,'County_code_3'] = bank_data[bank_data.Country_Name.notna()]['Country_Name'].apply(get_code)



#Finding the number of on going prjects:

on_going_pro = bank_data[(bank_data.Deadline_Date > pd.datetime.today()) & (bank_data.Deadline_Date.notna())]

country_code = on_going_pro.groupby(by='County_code_3').Deadline_Date.count()



#making a dataframe with all the countries and adding the pending project numbers:

ser = pd.Series(0,index=all_country)

df = pd.DataFrame({'all': ser,

              'few': country_code})

df.loc[:,'Pending_projects'] = df['all'] + df.few

df.drop(['all','few'],axis=1,inplace= True)

df.fillna(0.0,inplace= True)



# adding the country name:

df.loc[:,'Country_name'] = [pycountry.countries.get(alpha_3 = code).name for code in df.index]
data = [dict(

    type='choropleth',

    locations=df.index,

    z=df.Pending_projects,

    text=df.Country_name,

    colorscale=[[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [0.5, "rgb(70, 100, 245)"],\

                [0.6, "rgb(90, 120, 245)"], [0.7, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]],

    autocolorscale=False,

    reversescale=True,

    marker=dict(

        line=dict(

            color='rgb(180,180,180)',

            width=0.5

        )),

    colorbar=dict(

        autotick=False,

        #tickprefix='$',

        title='Number of projects'),

)]



layout = dict(

    title='Countrywise ongoing projects:',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection=dict(

            type='Mercator'

        )

    )

)



fig = dict(data=data, layout=layout)

iplot(fig,validate=False, filename='d3-world-map')
#Dist on due date:

DD_distribution = on_going_pro.groupby(by='Deadline_Date')

DD_distribution_count = DD_distribution['Publication_Date'].count()



# sepcify that we want a scatter plot with, with date on the x axis and meet on the y axis

data = [go.Scatter(x=DD_distribution_count.index, y=DD_distribution_count)]



# specify the layout of our figure

layout = dict(title = "Future Due Dates",

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False))



# create and show our figure

fig = dict(data = data, layout = layout)

iplot(fig)
