# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

%matplotlib inline

import math



#import libraries for Choropleth

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import seaborn as sb

from datetime import datetime, timedelta



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import data

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

forecast = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")

country_pop = pd.read_csv("/kaggle/input/country-population/Country_pop.csv")

train
# What's the inverval of the data?

print(train["Date"].min())

print(train["Date"].max())



# Create variable for last day with format Day-Month-Year:

first_date = train["Date"].min()

last_date = train["Date"].max()

last_date_f = datetime.strptime(train["Date"].max(), "%Y-%m-%d").strftime("%d-%B-%Y")

second2last_date = (datetime.strptime(train["Date"].max(), "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
# How many countries are in the data and which ones are they?

print(len(train["Country_Region"].unique()))

countries = train["Country_Region"].unique()

countries
# Last worldwide results into variables:

worldwide_evolution_i = train.drop(["Id"], axis=1).groupby("Date").sum()



nbr_ww_confirmed_cases = worldwide_evolution_i.loc[train["Date"].max()][0]

new_daily_ww_confirmed_cases = worldwide_evolution_i.loc[last_date][0] - worldwide_evolution_i.loc[second2last_date][0]

growth_factor_ww_confirmed_cases = (((worldwide_evolution_i.loc[last_date][0] / worldwide_evolution_i.loc[second2last_date][0])-1)*100).round(2)

nbr_ww_fatalities = worldwide_evolution_i.loc[train["Date"].max()][1]

new_daily_ww_fatalities = worldwide_evolution_i.loc[last_date][1] - worldwide_evolution_i.loc[second2last_date][1] 

growth_factor_ww_fatalities = (((worldwide_evolution_i.loc[last_date][1] / worldwide_evolution_i.loc[second2last_date][1])-1)*100).round(2)

    

cfr_ww = (nbr_ww_fatalities/nbr_ww_confirmed_cases)*100
# Worldwide evolution with addition of new metrics:

worldwide_evolution = worldwide_evolution_i.reset_index()



CFR = pd.DataFrame({"CFR (%)": (worldwide_evolution["Fatalities"] / worldwide_evolution["ConfirmedCases"] * 100)})

CFR["CFR (%)"] = CFR["CFR (%)"].fillna(0)



days = worldwide_evolution_i.index



ww_new_daily_cases = []

ww_new_daily_fatalities = []

ww_growth_factor_cases = []

ww_growth_factor_fatalities = []



for i in days:

    yesterday = (datetime.strptime(i, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    

    if yesterday in days:

        delta_cc = worldwide_evolution_i.loc[i]["ConfirmedCases"]-worldwide_evolution_i.loc[yesterday]["ConfirmedCases"]

        delta_fa = worldwide_evolution_i.loc[i]["Fatalities"]-worldwide_evolution_i.loc[yesterday]["Fatalities"]

        growth_cc = ((worldwide_evolution_i.loc[i]["ConfirmedCases"]/worldwide_evolution_i.loc[yesterday]["ConfirmedCases"])-1)*100

        growth_fa = ((worldwide_evolution_i.loc[i]["Fatalities"]/worldwide_evolution_i.loc[yesterday]["Fatalities"])-1)*100

        

        ww_new_daily_cases.append(delta_cc)

        ww_new_daily_fatalities.append(delta_fa)

        ww_growth_factor_cases.append(growth_cc)

        ww_growth_factor_fatalities.append(growth_fa)



    else:

        delta = 0

        ww_new_daily_cases.append(delta)

        ww_new_daily_fatalities.append(delta)

        ww_growth_factor_cases.append(delta)

        ww_growth_factor_fatalities.append(delta)





new_cases = pd.DataFrame({"New Confirmed Cases": ww_new_daily_cases, "New Fatalities" : ww_new_daily_fatalities})

growth_evolution = pd.DataFrame({"Growth Factor Confirmed Cases (%)" : ww_growth_factor_cases, "Growth Factor Fatalities (%)" : ww_growth_factor_fatalities})



worldwide_evolution_wGF = worldwide_evolution.join(CFR.round(2), how="right").join(new_cases,how="right").join(growth_evolution.round(2),how="right")



# Adding perCapita comparison

worldwide_evolution_wGF["Confirmed Cases %pop"] = worldwide_evolution_wGF["ConfirmedCases"]/country_pop.set_index(["Country Name"]).loc["World"][1] *100

worldwide_evolution_wGF
perCountry_evolution_i = train.drop(["Id"], axis=1).groupby(["Country_Region", "Date"]).sum()



def selected_countries_evolution(countries):

    

    countries_list = []



    for c in countries:

        if c in train["Country_Region"].unique():

            countries_list.append(c)

    

    countries_list.sort()

    if(len(countries_list)>0): 

        selected_country_evolution = perCountry_evolution_i.loc[countries_list].reset_index()

        selected_country_evolution_i = selected_country_evolution.set_index(["Country_Region", "Date"])

        

        CFR_perCountry = pd.DataFrame({"CFR (%)": (selected_country_evolution["Fatalities"] / selected_country_evolution["ConfirmedCases"] * 100)})

        CFR_perCountry["CFR (%)"] = CFR_perCountry["CFR (%)"].fillna(0)

         

        pc_new_daily_cases = []

        pc_new_daily_fatalities = []

        pc_growth_factor_cases = []

        pc_growth_factor_fatalities = []

        

        for c in countries_list:



            for i in days:

                yesterday = (datetime.strptime(i, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")



                if yesterday in days:

                    delta_cc = selected_country_evolution_i.loc[c].loc[i]["ConfirmedCases"]-selected_country_evolution_i.loc[c].loc[yesterday]["ConfirmedCases"]

                    delta_fa = selected_country_evolution_i.loc[c].loc[i]["Fatalities"]-selected_country_evolution_i.loc[c].loc[yesterday]["Fatalities"]

                    

                    divisor_cc = selected_country_evolution_i.loc[c].loc[yesterday]["ConfirmedCases"]

                    divisor_fa = selected_country_evolution_i.loc[c].loc[yesterday]["Fatalities"]

                    

                    if divisor_cc != 0:

                        growth_cc = ((selected_country_evolution_i.loc[c].loc[i]["ConfirmedCases"]/selected_country_evolution_i.loc[c].loc[yesterday]["ConfirmedCases"])-1)*100

                    else:

                        growth_cc = 0

                    

                    if divisor_fa != 0:

                        growth_fa = ((selected_country_evolution_i.loc[c].loc[i]["Fatalities"]/selected_country_evolution_i.loc[c].loc[yesterday]["Fatalities"])-1)*100

                    else:

                        growth_fa=0

                        

                    pc_new_daily_cases.append(delta_cc)

                    pc_new_daily_fatalities.append(delta_fa)

                    pc_growth_factor_cases.append(growth_cc)

                    pc_growth_factor_fatalities.append(growth_fa)

                    #print("NUM: ", selected_country_evolution_i.loc[c].loc[i]["ConfirmedCases"])

                    #print("DEN: ", selected_country_evolution_i.loc[c].loc[yesterday]["ConfirmedCases"])

                    #print("delta_cc: " ,delta_cc)

                    #print("growth_cc: " ,growth_cc)

                    

                    #print("NUM: ", selected_country_evolution_i.loc[c].loc[i]["Fatalities"])

                    #print("DEN: ", selected_country_evolution_i.loc[c].loc[yesterday]["Fatalities"])

                    #print("delta_fa: " ,delta_fa)

                    #print("growth_fa: " ,growth_fa)

                    

                else:

                    delta = 0

                    pc_new_daily_cases.append(delta)

                    pc_new_daily_fatalities.append(delta)

                    pc_growth_factor_cases.append(delta)

                    pc_growth_factor_fatalities.append(delta)

        

        pc_new_cases = pd.DataFrame({"New Confirmed Cases": pc_new_daily_cases, "New Fatalities" : pc_new_daily_fatalities})

        pc_growth_evolution = pd.DataFrame({"Growth Factor Confirmed Cases (%)" : pc_growth_factor_cases, "Growth Factor Fatalities (%)" : pc_growth_factor_fatalities})

        pc_growth_evolution["Growth Factor Confirmed Cases (%)"] = pc_growth_evolution["Growth Factor Confirmed Cases (%)"].fillna(0)

        pc_growth_evolution["Growth Factor Fatalities (%)"] = pc_growth_evolution["Growth Factor Fatalities (%)"].fillna(0)

        selected_country_evolution_wGF = selected_country_evolution.join(CFR_perCountry.round(2), how="right").join(pc_new_cases,how="right").join(pc_growth_evolution.round(2),how="right")

        

        

        # Adding perCapita comparison

        cc_ratio_per_pop = []

        for i in range(0, len(selected_country_evolution_wGF)):

            country = selected_country_evolution_wGF.iloc[i][0]

            if country in country_pop["Country Name"].unique():

                population = country_pop.set_index(["Country Name"]).loc[country][1]

                cc_ratio_per_pop.append((selected_country_evolution_wGF.iloc[i][2]/ population) *100)

            else:

                cc_ratio_per_pop.append("N/A")

        

        selected_country_evolution_wGF["Confirmed Cases %pop"]=cc_ratio_per_pop

        

        return selected_country_evolution_wGF

    

    else:

        print("No valid countries in the list!")

        
#######################################################################################################################################################

###################################### CODE FOR DATA VIZUALIZATION OF COVID-19 into MAPS ##############################################################

#######################################################################################################################################################



perCountry_evolution = train.drop(["Id"], axis=1).set_index(["Date","Country_Region"])

last_day = perCountry_evolution.loc[train["Date"].max()]



# Group all provinces/states of each country

last_day_perCountry = last_day.groupby("Country_Region").sum()

last_day_perCountry_ri = last_day_perCountry.reset_index()

last_day_perCountry_ri





def confirmedCases_log_map():

    fig = go.Figure(data=go.Choropleth(locations = last_day_perCountry_ri['Country_Region'], locationmode = 'country names', z = np.log10(last_day_perCountry_ri['ConfirmedCases']).round(2),

        colorscale = 'viridis_r', marker_line_color = 'darkgray',marker_line_width = 0.5, colorbar_tickprefix = '10^', colorbar_title = 'Confirmed cases <br>(log10 scale)'

    ))



    fig.update_layout(

        title_text = "COVID-19 Confirmed Cases as {}".format(last_date_f),

        title_x = 0.5,

        autosize=True,

        geo=dict(

            showframe = False,

            showcoastlines = False,

            projection_type = 'equirectangular'

        )

    )

    fig.show()



    

    

def fatalities_log_map():

    fig = go.Figure(data=go.Choropleth(locations = last_day_perCountry_ri['Country_Region'], locationmode = 'country names', z = np.log10(last_day_perCountry_ri['Fatalities']).round(2),

        colorscale = 'viridis_r', marker_line_color = 'darkgray',marker_line_width = 0.5, colorbar_tickprefix = '10^', colorbar_title = 'Fatalities <br>(log10 scale)'

    ))



    fig.update_layout(

        title_text = "COVID-19 Fatalities as {}".format(last_date_f),

        title_x = 0.5,

        autosize=True,

        geo=dict(

            showframe = False,

            showcoastlines = False,

            projection_type = 'equirectangular'

        )

    )

    fig.show()

    

# Create interative maps with spread of Confirmed Cases over time

def confirmedCases_spread_map():

    countrydate_evolution = train[train['ConfirmedCases']>0]

    countrydate_evolution = countrydate_evolution.groupby(['Date','Country_Region']).sum().reset_index()

    countrydate_evolution["ConfirmedCases_LOG"] = np.log10(countrydate_evolution["ConfirmedCases"]).round(2)



# Creating the visualization

    fig = px.choropleth(countrydate_evolution, locations="Country_Region", locationmode = "country names", color="ConfirmedCases_LOG", 

                    hover_name="Country_Region", animation_frame="Date", color_continuous_scale="viridis_r",

                   )



    fig.update_layout(

        title_text = 'Global Spread of Coronavirus over time',

        title_x = 0.5,

        autosize=True,

        geo=dict(

            showframe = False,

            showcoastlines = False,

            projection_type = 'equirectangular'

        ),

        coloraxis_colorbar=dict(

            title="Confirmed cases <br>(log10 scale)",

            ticks="outside", tickprefix="10^",

            )

    )

    

    fig.show()

    

    

def confirmedCases_vs_Fatalities():

    last_day_perCountry_ri['CFR (%)'] = ((last_day_perCountry_ri["Fatalities"] / last_day_perCountry_ri["ConfirmedCases"])*100).round(2)

    last_day_perCountry_ri["CFR (%)"] = last_day_perCountry_ri["CFR (%)"].fillna(0)

    last_day_perCountry_ri['Fatalities_log'] = [np.log10(x).round(2) if x!=0 else 0 for x in last_day_perCountry_ri["Fatalities"]]

    last_day_perCountry_ri

    

    fig = px.scatter(last_day_perCountry_ri,

                 x='ConfirmedCases',

                 y='Fatalities',

                 color='CFR (%)',

                 size='Fatalities_log',

                 size_max=25,

                 hover_name='Country_Region',

                 color_continuous_scale='viridis_r'

    )

    fig = fig.update_layout(

        title_text="COVID-19 Deaths vs Confirmed Cases by Country as per {}".format(last_date_f),

        xaxis_type="log",

        yaxis_type="log"

    )

    fig.show()
def plot_country_comparison(country_list):

    

    perCountry_evolution_woDate = selected_countries_evolution(country_list).set_index(["Country_Region"])

    

    #Plot with Confirmed Cases numbers

    

    fig1 = go.Figure()

    for c in country_list:

        fig1.add_scatter(x=perCountry_evolution_woDate.loc[c]["Date"], y=perCountry_evolution_woDate.loc[c]["ConfirmedCases"], name="{}".format(c), mode = 'lines+markers')



    fig1.update_yaxes(tickfont_color="MediumSlateBlue", title="Count")

    fig1.update_layout(title_text= "COVID19 - Confirmed Cases Evolution", hovermode ='x')

    fig1.show()

    

    #Plot with Fatalities numbers

    fig2 = go.Figure()

    for c in country_list:

        fig2.add_scatter(x=perCountry_evolution_woDate.loc[c]["Date"], y=perCountry_evolution_woDate.loc[c]["Fatalities"], name="{}".format(c), mode = 'lines+markers')



    fig2.update_yaxes(tickfont_color="MediumSlateBlue", title="Count")

    fig2.update_layout(title_text= "COVID19 - Fatalities Evolution", hovermode ='x')

    fig2.show()

    

    #Plot with CFR %

    fig3 = go.Figure()

    for c in country_list:

        fig3.add_scatter(x=perCountry_evolution_woDate.loc[c]["Date"], y=perCountry_evolution_woDate.loc[c]["CFR (%)"], name="{}".format(c), mode = 'lines+markers')



    fig3.update_yaxes(tickfont_color="MediumSlateBlue", title="%")

    fig3.update_layout(title_text= "COVID19 - Case Fatality Rate Evolution (%)", hovermode ='x')

    fig3.show()

    

    #Plot with New Confirmed Cases

    fig4 = go.Figure()

    for c in country_list:

        fig4.add_scatter(x=perCountry_evolution_woDate.loc[c]["Date"], y=perCountry_evolution_woDate.loc[c]["New Confirmed Cases"], name="{}".format(c), mode = 'lines+markers')



    fig4.update_yaxes(tickfont_color="MediumSlateBlue", title="Count")

    fig4.update_layout(title_text= "COVID19 - New Confirmed Cases", hovermode ='x')

    fig4.show()

        

    #Plot with Growth Factor of Confirmed Cases (%)

    fig5 = go.Figure()

    for c in country_list:

        fig5.add_scatter(x=perCountry_evolution_woDate.loc[c]["Date"], y=perCountry_evolution_woDate.loc[c]["Growth Factor Confirmed Cases (%)"], name="{}".format(c), mode = 'lines+markers')



    fig5.update_yaxes(tickfont_color="MediumSlateBlue", title="%")

    fig5.update_layout(title_text= "COVID19 - Growth Factor of Confirmed Cases (%)", hovermode ='x')

    fig5.show()

    

    #Plot with New Fatalities

    fig6 = go.Figure()

    for c in country_list:

        fig6.add_scatter(x=perCountry_evolution_woDate.loc[c]["Date"], y=perCountry_evolution_woDate.loc[c]["New Fatalities"], name="{}".format(c), mode = 'lines+markers')



    fig6.update_yaxes(tickfont_color="MediumSlateBlue", title="Count")

    fig6.update_layout(title_text= "COVID19 - New Fatalities", hovermode ='x')

    fig6.show()

    

    #Plot with Growth Factor of Fatalities (%)

    fig7 = go.Figure()

    for c in country_list:

        fig7.add_scatter(x=perCountry_evolution_woDate.loc[c]["Date"], y=perCountry_evolution_woDate.loc[c]["Growth Factor Fatalities (%)"], name="{}".format(c), mode = 'lines+markers')



    fig7.update_yaxes(tickfont_color="MediumSlateBlue", title="%")

    fig7.update_layout(title_text= "COVID19 - Growth Factor of Fatalities (%)", hovermode ='x')

    fig7.show()

    

    #Plot with Confirmed Cases per Population (%)

    fig8 = go.Figure()

    for c in country_list:

        fig8.add_scatter(x=perCountry_evolution_woDate.loc[c]["Date"], y=perCountry_evolution_woDate.loc[c]["Confirmed Cases %pop"], name="{}".format(c), mode = 'lines+markers')



    fig8.update_yaxes(tickfont_color="MediumSlateBlue", title="%")

    fig8.update_layout(title_text= "COVID19 - Confirmed Cases per Population (%)", hovermode ='x')

    fig8.show()

def plot_country_comparison_sincePatient(patient, country_list):

    

    perCountry_evolution_woDate = selected_countries_evolution(country_list).set_index(["Country_Region"])

    perCountry_evolution_woDate_patient=perCountry_evolution_woDate[perCountry_evolution_woDate["ConfirmedCases"]>=patient]

    

    

    #Plot with Confirmed Cases numbers

    fig1 = go.Figure()

    for c in country_list:

        x= (len(perCountry_evolution_woDate_patient.loc[c])+1)

        fig1.add_scatter(x=np.arange(1,x,1), y=perCountry_evolution_woDate_patient.loc[c]["ConfirmedCases"], name="{}".format(c), mode = 'lines+markers')



    fig1.update_yaxes(tickfont_color="MediumSlateBlue", title="Count")

    fig1.update_layout(title_text= "COVID19 - Confirmed Cases Evolution", hovermode ='x')

    fig1.show()

    

    #Plot with Fatalities numbers

    fig2 = go.Figure()

    for c in country_list:

        x= (len(perCountry_evolution_woDate_patient.loc[c])+1)

        fig2.add_scatter(x=np.arange(1,x,1), y=perCountry_evolution_woDate_patient.loc[c]["Fatalities"], name="{}".format(c), mode = 'lines+markers')



    fig2.update_yaxes(tickfont_color="MediumSlateBlue", title="Count")

    fig2.update_layout(title_text= "COVID19 - Fatalities Evolution", hovermode ='x')

    fig2.show()

    

    #Plot with CFR %

    fig3 = go.Figure()

    for c in country_list:

        x= (len(perCountry_evolution_woDate_patient.loc[c])+1)

        fig3.add_scatter(x=np.arange(1,x,1), y=perCountry_evolution_woDate_patient.loc[c]["CFR (%)"], name="{}".format(c), mode = 'lines+markers')



    fig3.update_yaxes(tickfont_color="MediumSlateBlue", title="%")

    fig3.update_layout(title_text= "COVID19 - Case Fatality Rate Evolution (%)", hovermode ='x')

    fig3.show()

    

    #Plot with New Confirmed Cases

    fig4 = go.Figure()

    for c in country_list:

        x= (len(perCountry_evolution_woDate_patient.loc[c])+1)

        fig4.add_scatter(x=np.arange(1,x,1), y=perCountry_evolution_woDate_patient.loc[c]["New Confirmed Cases"], name="{}".format(c), mode = 'lines+markers')



    fig4.update_yaxes(tickfont_color="MediumSlateBlue", title="Count")

    fig4.update_layout(title_text= "COVID19 - New Confirmed Cases", hovermode ='x')

    fig4.show()

        

    #Plot with Growth Factor of Confirmed Cases (%)

    fig5 = go.Figure()

    for c in country_list:

        x= (len(perCountry_evolution_woDate_patient.loc[c])+1)

        fig5.add_scatter(x=np.arange(1,x,1), y=perCountry_evolution_woDate_patient.loc[c]["Growth Factor Confirmed Cases (%)"], name="{}".format(c), mode = 'lines+markers')



    fig5.update_yaxes(tickfont_color="MediumSlateBlue", title="%")

    fig5.update_layout(title_text= "COVID19 - Growth Factor of Confirmed Cases (%)", hovermode ='x')

    fig5.show()

    

    #Plot with New Fatalities

    fig6 = go.Figure()

    for c in country_list:

        x= (len(perCountry_evolution_woDate_patient.loc[c])+1)

        fig6.add_scatter(x=np.arange(1,x,1), y=perCountry_evolution_woDate_patient.loc[c]["New Fatalities"], name="{}".format(c), mode = 'lines+markers')



    fig6.update_yaxes(tickfont_color="MediumSlateBlue", title="Count")

    fig6.update_layout(title_text= "COVID19 - New Fatalities", hovermode ='x')

    fig6.show()

    

    #Plot with Growth Factor of Fatalities (%)

    fig7 = go.Figure()

    for c in country_list:

        x= (len(perCountry_evolution_woDate_patient.loc[c])+1)

        fig7.add_scatter(x=np.arange(1,x,1), y=perCountry_evolution_woDate_patient.loc[c]["Growth Factor Fatalities (%)"], name="{}".format(c), mode = 'lines+markers')



    fig7.update_yaxes(tickfont_color="MediumSlateBlue", title="%")

    fig7.update_layout(title_text= "COVID19 - Growth Factor of Fatalities (%)", hovermode ='x')

    fig7.show()

    

    #Plot with Confirmed Cases per Population (%)

    fig8 = go.Figure()

    for c in country_list:

        x= (len(perCountry_evolution_woDate_patient.loc[c])+1)

        fig8.add_scatter(x=np.arange(1,x,1), y=perCountry_evolution_woDate_patient.loc[c]["Confirmed Cases %pop"], name="{}".format(c), mode = 'lines+markers')



    fig8.update_yaxes(tickfont_color="MediumSlateBlue", title="%")

    fig8.update_layout(title_text= "COVID19 - Confirmed Cases per Population (%)", hovermode ='x')

    fig8.show()

#######################################################################################################################################################

############################### MAIN CHARTS FOR DATA VIZUALIZATION OF COVID-19 EVOLUTION, MAPS AND COMPARISON #########################################

#######################################################################################################################################################



#################################################### Worldwide Dashboard ##############################################################################

def worlwide_evolution():



    fig = make_subplots(rows=4, cols=2,specs=[[{"rowspan": 2}, {"secondary_y": True}],

                                              [None, {"secondary_y": True}],

                                              [{"colspan": 2, "secondary_y": True}, None],

                                              [{"colspan": 2, "secondary_y": True}, None]], 

                        subplot_titles=("","Confirmed Cases", "Fatalities & CFR %", "New Confirmed Cases & Growth Factor", "New Fatalities & Growth Factor"), column_widths=[0.2, 0.8], vertical_spacing=0.05, horizontal_spacing=0.07)

    

    fig.update_xaxes(range=[0,2],tick0=0,zeroline=False, showgrid=False, showticklabels=False,row=1, col=1)

    fig.update_yaxes(range=[0,20],nticks=10,zeroline=False, showgrid=False, showticklabels=False,row=1, col=1)

    

    fig.add_annotation(x=1, y=19.5,text="Number of Confirmed Cases", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

    fig.add_annotation(x=1, y=18.5,text="{:,.0f}".format(nbr_ww_confirmed_cases), font=dict(family="Arial", size=18, color="LightSeaGreen"),row=1, col=1)

    fig.add_annotation(x=1, y=16.5,text="New Daily Confirmed Cases", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

    fig.add_annotation(x=1, y=15.5,text="+{:,.0f}".format(new_daily_ww_confirmed_cases), font=dict(family="Arial", size=18, color="LightSeaGreen"),row=1, col=1)

    fig.add_annotation(x=1, y=13.5,text="Growth Factor", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

    fig.add_annotation(x=1, y=12.5,text="{:.2f}%".format(growth_factor_ww_confirmed_cases), font=dict(family="Arial", size=18, color="Gray"),row=1, col=1)

    fig.add_annotation(x=1, y=10.5,text="Number of Fatalities", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

    fig.add_annotation(x=1, y=9.5,text="{:,.0f}".format(nbr_ww_fatalities), font=dict(family="Arial", size=18, color="DarkRed"),row=1, col=1)

    fig.add_annotation(x=1, y=7.5,text="New Daily Fatalities", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

    fig.add_annotation(x=1, y=6.5,text="+{:,.0f}".format(new_daily_ww_fatalities), font=dict(family="Arial", size=18, color="DarkRed"),row=1, col=1)

    fig.add_annotation(x=1, y=4.5,text="Growth Factor", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

    fig.add_annotation(x=1, y=3.5,text="{:.2f}%".format(growth_factor_ww_fatalities), font=dict(family="Arial", size=18, color="Gray"),row=1, col=1)

    fig.add_annotation(x=1, y=1.5,text="Case Fatality Rate (%)", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

    fig.add_annotation(x=1, y=0.5,text="{:.1f}".format(cfr_ww), font=dict(family="Arial", size=18, color="Red"),row=1, col=1)



    fig.update_annotations(dict(showarrow=False, align="right"))

  

    fig.add_bar(secondary_y=False, x=worldwide_evolution_wGF["Date"], y=worldwide_evolution_wGF["ConfirmedCases"], marker_color="#35b772", name="#Confirmed Cases",row=1, col=2)

    fig.add_scatter(secondary_y=True,  x=worldwide_evolution_wGF["Date"], y=worldwide_evolution_wGF["Confirmed Cases %pop"], marker_color="#fdca26", name="%Confirmed Cases per100", mode = 'lines+markers',row=1, col=2)

    fig.update_yaxes(secondary_y=False, tickfont_color="#35b772", title="#ConfirmedCases", row=1, col=2)

    fig.update_yaxes(secondary_y=True, tickfont_color="#fdca26", title="% Cases per100", row=1, col=2)

        

    fig.add_bar(secondary_y=False, x=worldwide_evolution_wGF["Date"], y=worldwide_evolution_wGF["Fatalities"], marker_color="#26828e", name="#Fatalities", row=2, col=2)

    fig.add_scatter(secondary_y=True,  x=worldwide_evolution_wGF["Date"], y=worldwide_evolution_wGF["CFR (%)"], marker_color="#440154", name="%CFR", mode = 'lines+markers',row=2, col=2)

    fig.update_yaxes(secondary_y=False, tickfont_color="#26828e", title="#Fatalities", row=2, col=2)

    fig.update_yaxes(secondary_y=True, tickfont_color="#440154", title="%CFR", row=2, col=2)



    fig.add_bar(secondary_y=False, x=worldwide_evolution_wGF["Date"], y=worldwide_evolution_wGF["New Confirmed Cases"], marker_color="#35b772", name="#ConfirmedCases", row=3, col=1)

    fig.add_scatter(secondary_y=True,  x=worldwide_evolution_wGF["Date"], y=worldwide_evolution_wGF["Growth Factor Confirmed Cases (%)"], marker_color="#fdca26", name="%GrowthFactor", mode = 'lines+markers',row=3, col=1)

    fig.update_yaxes(secondary_y=False, tickfont_color="#35b772", title="New Confirmed Cases",row=3, col=1)

    fig.update_yaxes(secondary_y=True, tickfont_color="#fdca26", title="%Growth Factor",row=3, col=1)

    

    fig.add_bar(secondary_y=False, x=worldwide_evolution_wGF["Date"], y=worldwide_evolution_wGF["New Fatalities"], marker_color="#26828e", name="#Fatalities",row=4, col=1)

    fig.add_scatter(secondary_y=True,  x=worldwide_evolution_wGF["Date"], y=worldwide_evolution_wGF["Growth Factor Fatalities (%)"], marker_color="#440154", name="%GrowthFactor", mode = 'lines+markers',row=4, col=1)

    fig.update_yaxes(secondary_y=False, tickfont_color="#26828e", title="New Fatalities",row=4, col=1)

    fig.update_yaxes(secondary_y=True, tickfont_color="#440154", title="%Growth Factor",row=4, col=1)



    fig.update_layout(plot_bgcolor="whitesmoke", paper_bgcolor="whitesmoke",title_text="COVID-19 Coronavirus Pandemic - Worldwide Numbers", title_font_size=24, hovermode ='x',  height=1500)

    fig.show()

    

# Plot maps:

def maps_plot():

    confirmedCases_vs_Fatalities()

    confirmedCases_spread_map()

    confirmedCases_log_map()

    fatalities_log_map()

    

####################################################### Country Dashboard ##############################################################################

def country_evolution(country):

    

    if country in train["Country_Region"].unique():

    

        nbr_confirmed_cases_perCountry = perCountry_evolution_i.loc[country].loc[last_date][0]

        new_daily_confirmed_cases_perCountry = perCountry_evolution_i.loc[country].loc[last_date][0] - perCountry_evolution_i.loc[country].loc[second2last_date][0]

        growth_factor_confirmed_cases_perCountry = (((perCountry_evolution_i.loc[country].loc[last_date][0] / perCountry_evolution_i.loc[country].loc[second2last_date][0])-1)*100).round(2)

        nbr_fatalities_perCountry = perCountry_evolution_i.loc[country].loc[last_date][1]

        new_daily_fatalities_perCountry = perCountry_evolution_i.loc[country].loc[last_date][1] - perCountry_evolution_i.loc[country].loc[second2last_date][1] 

        growth_factor_fatalities_perCountry = (((perCountry_evolution_i.loc[country].loc[last_date][1] / perCountry_evolution_i.loc[country].loc[second2last_date][1])-1)*100).round(2)



        cfr_perCountry = (nbr_fatalities_perCountry/nbr_confirmed_cases_perCountry)*100



        

        fig = make_subplots(rows=4, cols=2,specs=[[{"rowspan": 2}, {"secondary_y": True}],

                                              [None, {"secondary_y": True}],

                                              [{"colspan": 2, "secondary_y": True}, None],

                                              [{"colspan": 2, "secondary_y": True}, None]], 

                        subplot_titles=("","Confirmed Cases", "Fatalities & CFR %", "New Confirmed Cases & Growth Factor", "New Fatalities & Growth Factor"), column_widths=[0.2, 0.8], vertical_spacing=0.05, horizontal_spacing=0.07)

    

        fig.update_xaxes(range=[0,2],tick0=0,zeroline=False, showgrid=False, showticklabels=False,row=1, col=1)

        fig.update_yaxes(range=[0,20],nticks=10,zeroline=False, showgrid=False, showticklabels=False,row=1, col=1)



        fig.add_annotation(x=1, y=19.5,text="Number of Confirmed Cases", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

        fig.add_annotation(x=1, y=18.5,text="{:,.0f}".format(nbr_confirmed_cases_perCountry), font=dict(family="Arial", size=18, color="LightSeaGreen"),row=1, col=1)

        fig.add_annotation(x=1, y=16.5,text="New Daily Confirmed Cases", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

        fig.add_annotation(x=1, y=15.5,text="+{:,.0f}".format(new_daily_confirmed_cases_perCountry), font=dict(family="Arial", size=18, color="LightSeaGreen"),row=1, col=1)

        fig.add_annotation(x=1, y=13.5,text="Growth Factor", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

        fig.add_annotation(x=1, y=12.5,text="{:.2f}%".format(growth_factor_confirmed_cases_perCountry), font=dict(family="Arial", size=18, color="Gray"),row=1, col=1)

        fig.add_annotation(x=1, y=10.5,text="Number of Fatalities", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

        fig.add_annotation(x=1, y=9.5,text="{:,.0f}".format(nbr_fatalities_perCountry), font=dict(family="Arial", size=18, color="DarkRed"),row=1, col=1)

        fig.add_annotation(x=1, y=7.5,text="New Daily Fatalities", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

        fig.add_annotation(x=1, y=6.5,text="+{:,.0f}".format(new_daily_fatalities_perCountry), font=dict(family="Arial", size=18, color="DarkRed"),row=1, col=1)

        fig.add_annotation(x=1, y=4.5,text="Growth Factor", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

        fig.add_annotation(x=1, y=3.5,text="{:.2f}%".format(growth_factor_fatalities_perCountry), font=dict(family="Arial", size=18, color="Gray"),row=1, col=1)

        fig.add_annotation(x=1, y=1.5,text="Case Fatality Rate (%)", font=dict(family="Arial", size=18, color="black"),row=1, col=1)

        fig.add_annotation(x=1, y=0.5,text="{:.1f}".format(cfr_perCountry), font=dict(family="Arial", size=18, color="Red"),row=1, col=1)



        fig.update_annotations(dict(showarrow=False, align="right"))



        country_list = [country]

        perCountry_evolution_woDate = selected_countries_evolution(country_list).set_index(["Country_Region"])



        #def plot_country_CC_evolution(country):

        fig.add_bar(secondary_y=False, x=perCountry_evolution_woDate["Date"], y=perCountry_evolution_woDate["ConfirmedCases"], marker_color="#35b772", name="#Confirmed Cases",row=1, col=2)

        fig.add_scatter(secondary_y=True,  x=perCountry_evolution_woDate["Date"], y=perCountry_evolution_woDate["Confirmed Cases %pop"], marker_color="#fdca26", name="%Confirmed Cases per100", mode = 'lines+markers',row=1, col=2)

        fig.update_yaxes(secondary_y=False, tickfont_color="#35b772", title="#ConfirmedCases",row=1, col=2)

        fig.update_yaxes(secondary_y=True, tickfont_color="#fdca26", title="%Confirmed Cases per100",row=1, col=2)



        #def plot_country_F_evolution(country):

        fig.add_bar(secondary_y=False, x=perCountry_evolution_woDate["Date"], y=perCountry_evolution_woDate["Fatalities"], marker_color="#26828e", name="#Fatalities",row=2, col=2)

        fig.add_scatter(secondary_y=True,  x=perCountry_evolution_woDate["Date"], y=perCountry_evolution_woDate["CFR (%)"], marker_color="#440154", name="%CFR", mode = 'lines+markers',row=2, col=2)

        fig.update_yaxes(secondary_y=False, tickfont_color="#26828e", title="#Fatalities",row=2, col=2)

        fig.update_yaxes(secondary_y=True, tickfont_color="#440154", title="%CFR",row=2, col=2)



        #def plot_country_nCC_evolution(country):

        fig.add_bar(secondary_y=False, x=perCountry_evolution_woDate["Date"], y=perCountry_evolution_woDate["New Confirmed Cases"], marker_color="#35b772", name="#ConfirmedCases",row=3, col=1)

        fig.add_scatter(secondary_y=True,  x=perCountry_evolution_woDate["Date"], y=perCountry_evolution_woDate["Growth Factor Confirmed Cases (%)"], marker_color="#fdca26", name="%GrowthFactor", mode = 'lines+markers', row=3, col=1)

        fig.update_yaxes(secondary_y=False, tickfont_color="#35b772", title="New Confirmed Cases",row=3, col=1)

        fig.update_yaxes(secondary_y=True, tickfont_color="#fdca26", title="%Growth Factor",row=3, col=1)



        #def plot_country_nF_evolution(country):

        fig.add_bar(secondary_y=False, x=perCountry_evolution_woDate["Date"], y=perCountry_evolution_woDate["New Fatalities"], marker_color="#26828e", name="#Fatalities",row=4, col=1)

        fig.add_scatter(secondary_y=True,  x=perCountry_evolution_woDate["Date"], y=perCountry_evolution_woDate["Growth Factor Fatalities (%)"], marker_color="#440154", name="%GrowthFactor", mode = 'lines+markers',row=4, col=1)

        fig.update_yaxes(secondary_y=False, tickfont_color="#26828e", title="New Fatalities",row=4, col=1)

        fig.update_yaxes(secondary_y=True, tickfont_color="#440154", title="%Growth Factor",row=4, col=1)



        fig.update_layout(plot_bgcolor="whitesmoke", paper_bgcolor="whitesmoke",height=1500, title_text="COVID-19 Coronavirus Pandemic - {} Numbers".format(country), title_font_size=24, hovermode ='x')

        fig.show()

    

    else:

        return "Country does not exist"



    

    #get_last_numbers_perCountry(country)

    #plot_country_evolution(country)

    

    

# Plot Counties comparaison

def countries_comparison(country_list):

    plot_country_comparison(country_list)

    

# Plot Counties comparaison since Patient X

def countries_comparison_sincePatient(patient, country_list):

    plot_country_comparison_sincePatient(patient, country_list)
worlwide_evolution()
maps_plot()
country_evolution("US")
plot_country_comparison(["Portugal","Spain","France","Germany","Italy","United Kingdom", "Netherlands"])
countries_comparison_sincePatient(100, ["Portugal","Spain","France","Germany","Italy","United Kingdom", "Netherlands"])