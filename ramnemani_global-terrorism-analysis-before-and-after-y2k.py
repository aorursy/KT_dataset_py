import pandas as pd

import matplotlib

#import cufflinks as cf

import plotly

import plotly.offline as py

import plotly.graph_objs as go



#cf.go_offline()  

py.init_notebook_mode()



gtd = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1',

                         usecols=[ 1, 8, 10, 98])

gtd = gtd.rename(

    columns={ 'iyear':'year', 

             'country_txt':'country', 

             'region_txt' : 'region',

             'nkill':'fatalities'})



gtd['fatalities'] = gtd['fatalities'].fillna(-1).astype(int)



# eliminate rows with no data from the gtd df

ctry_ftlts =   gtd[gtd.fatalities != -1][["country","year","fatalities"]]



## Summarize Fatalities by country and year

ctry_yr_cnt = ctry_ftlts[["country","year","fatalities"]].groupby(["country", "year"]).sum().reset_index(["year","country"])



## Summarize Fatalities by country all-time

ctry_alltime  = ctry_yr_cnt.groupby(["country"]).sum().reset_index()[["country", "fatalities"]]

ctry_alltime_dsc  = ctry_alltime.sort_values(by="fatalities", ascending=False)

ctry_alltime_asc  = ctry_alltime.sort_values(by="fatalities")



## Summarize Fatalities by year all-time

year_alltime  = ctry_yr_cnt.groupby(["year"]).sum().reset_index()[["year", "fatalities"]]

year_alltime_dsc  = year_alltime.sort_values(by="fatalities", ascending=False)



## Summarize Fatalities by country from 1983 to 1999 

ctry_b2k  = ctry_yr_cnt[(ctry_yr_cnt .year <2000) & (ctry_yr_cnt .year >= 1983)].groupby(["country"]).sum().reset_index()[["country", "fatalities"]]

ctry_b2k_asc  = ctry_b2k.sort_values(by="fatalities")

ctry_b2k_dsc  = ctry_b2k.sort_values(by="fatalities", ascending=False)



## Summarize Fatalities by country after year 2000

ctry_a2k = ctry_yr_cnt[ctry_yr_cnt.year >= 2000].groupby(["country"]).sum().reset_index()[["country", "fatalities"]]

ctry_a2k_asc = ctry_a2k.sort_values(by="fatalities")

ctry_a2k_dsc = ctry_a2k.sort_values(by="fatalities", ascending=False)

t10 = ctry_b2k_dsc[:10]



data  = go.Data([

            go.Bar(

              y = t10.fatalities,

              x = t10.country,

              orientation='v'

        )])

layout = go.Layout(

        height = '600',

        margin=go.Margin(l=50),

        title = "Top 10 countries ranked by fatalities before Y2K",

        yaxis={'title':'Fatalities'},

        hovermode = 'closest'

    )

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
t10a2k = ctry_a2k_dsc[:10]



data  = go.Data([

            go.Bar(

              y = t10a2k.fatalities,

              x = t10a2k.country,

              orientation='v'

        )])

layout = go.Layout(

        height = '600',

        margin=go.Margin(l=50),

        title = "Top 10 countries ranked by fatalities after Y2K",

        yaxis={'title':'Fatalities'},

        hovermode = 'closest'

    )

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
t10all = ctry_alltime_dsc[:10]



data  = go.Data([

            go.Bar(

              y = t10all.fatalities,

              x = t10all.country,

              orientation='v'

        )])

layout = go.Layout(

        height = '600',

        margin=go.Margin(l=50),

        title = "Top 10 countries (all-time) ranked by fatalities",

        yaxis={'title':'Fatalities'},

        hovermode = 'closest'

    )

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
t10year = year_alltime_dsc[:10]



data  = go.Data([

            go.Bar(

              y = t10year.fatalities,

              x = t10year.year,

              orientation='v'

        )])

layout = go.Layout(

        height = '500',

        margin=go.Margin(l=50),

        title = "Top 10 years ranked by fatalities all-time",

        yaxis={'title':'Fatalities'},

        hovermode = 'closest'

    )

fig  = go.Figure(data=data, layout=layout)

py.iplot(fig)
safe_ctry = ctry_alltime_asc[ctry_alltime_asc.fatalities  == 0]

safe_ctry