# Main packages



from datetime import datetime

import numpy as np

import pandas as pd
# Packages for graph



from matplotlib import pyplot as plt

import matplotlib.dates as mdates

import matplotlib.ticker as ticker

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.style.use('fivethirtyeight')
def create_graph_multiple(data, list_units, title_graph, foot_note, adjust_factor, figV=16, figH=10, minDate = "2020-04-01", type_agg="mean"):

    '''

    :param data: dataFrame with the following columns: text/unit named "unit", date and numeric values.

    :param list_units: List with the name of the unique units.

    :param title_graph: Graph title.

    :param foot_note: Text to show in footer.

    :param adjust_factor: usually something around -1. Useful to make the foot note in the right position.

    :param figV: Figure size vertical

    :param figH: Figure size horizontal

    :param minDate: Starting date of the analysis. Format: "2020-04-30"

    :param type_agg: Type of aggregation when grouping unit values by date.

    '''

    

    # Start the figure that will receive the graphs

    fig = plt.figure(figsize=(figV, figH))

    # The figure will have as many rows as the numbers of states

    grid = plt.GridSpec(len(list_units), 20, hspace=0.5, wspace=0.5)

    # Counter to fill the graph

    row = 0

    

    # Now create data for the specific State and create the graph

    for unit in list_units:

        # 1- Preparing the data

        # Selecting an specific state

        data_unit = data.loc[(data['unit'] == unit)].copy()

        # Drop if date is less than.

        data_unit = data_unit.loc[ (data_unit['date'] >= pd.to_datetime(minDate)) ]



        #### Type of aggregation. When data has several groups, one must set the type of agreggation. Mean or Sum

        if(type_agg == "mean"):

            # Group by day

            data_unit = data_unit.groupby(['date']).mean()

        else:

            data_unit = data_unit.groupby(['date']).sum()

        

        # 2- Creating the graph

        ax = fig.add_subplot(grid[row, 1:])

        ax.xaxis.set_major_locator(mdates.WeekdayLocator())

        # set major ticks format

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        # Set title

        ax.set_title(f"{title_graph} - {unit}")



        # Plot data

        ax.plot(data_unit)

        # Label

        labels = list(data_unit.columns)

        ax.legend(labels, loc='upper left', shadow=True, fontsize='x-large')    

        # Go to next graph row

        row+=1

        

    # Add footer at x = min X, y = min Y - 1.5.

    plt.text(ax.get_xlim()[0], ax.get_ylim()[0] + adjust_factor, foot_note)



    return None
def create_graph_single(data, list_units, list_label, title_graph, title_legend, foot_note, adjust_factor, figV=16, figH=10, minDate = "2020-04-01", type_graph="none", type_agg="mean"):

    '''

    :param data: dataFrame with the following columns: text/unit named "unit", date and numeric values.

    :param list_units: List with the name of the unique units.

    :param list_label: Dict that receives the label for each unit.

    :param title_graph: Graph title.

    :param title_legend: The title in the legend. Upper left.

    :param foot_note: Text to show in footer.

    :param adjust_factor: usually something around -1. Useful to make the foot note in the right position.

    :param figV: Figure size vertical

    :param figH: Figure size horizontal

    :param minDate: Starting date of the analysis. Format: "2020-04-30"

    :param type_graph: Type "pct" to dysplay as percentage.

    :param type_agg: Type of aggregation when grouping unit values by date.

    '''

    

    # Start the figure that will receive the graphs

    fig, ax = plt.subplots(1, figsize=(figV, figH))

    fig.suptitle(title_graph, fontsize=15)

    

    row = 0

    # Now create data for the specific State and create the graph

    for unit in list_units:

        

        # 1- Preparing the data

        # Selecting an specific unit (state, country, continent)

        data_each_unit = data.loc[(data['unit'] == unit)].copy()

        # Drop if date is less than minimum.

        data_each_unit = data_each_unit.loc[ (data_each_unit['date'] >= pd.to_datetime(minDate)) ]

        

        #### Type of aggregation. When data has several groups, one must set the type of agreggation. Mean or Sum

        if(type_agg == "mean"):

            # Group by day

            data_each_unit = data_each_unit.groupby(['date']).mean()

        else:

            data_each_unit = data_each_unit.groupby(['date']).sum()



        # 2- Creating the graph

        ax.xaxis.set_major_locator(mdates.WeekdayLocator())

        # set major ticks format

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        if(type_graph == "pct"):

            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

        else:

            pass

        # Plot data with legend

        ax.plot(data_each_unit, label=list_label[unit])

        # Go to next graph row

        row+=1



    # Add footer at x = min X, y = min Y - 1.5.

    plt.text(ax.get_xlim()[0], ax.get_ylim()[0] + adjust_factor, foot_note)

    

    # Legend title

    plt.legend(loc="upper left", title=title_legend, frameon=True)



    plt.show()

    

    return None
# Caminho para os dados

path = "../input/HIST_PAINEL_COVIDBR_10jun2020.csv"



# Lê os dados do CSV.

data = pd.read_csv(path, encoding='UTF-8', delimiter=';' , low_memory=False, dtype={'codmun': float} )



# Renaming according to previous analysis

data.rename(columns={"data": "date", "estado": "state", "municipio": "city", "regiao": "region",

                              "casosAcumulado": "totalConfirmed", "casosNovos": "newConfirmed", 

                             "obitosAcumulado": "totalDeaths", "obitosNovos": "newDeaths",

                              "emAcompanhamentoNovos": "newFollowing", "Recuperadosnovos": "totalRecoveredBrazil"},  inplace=True)



# Assign correct row to Brazil total

data["state"] = data["state"].replace(np.NaN, "Brazil")



# Assign date

data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
# 1- Filter only data after 22/march/2020, since this is the first day with all states having data

data = data.loc[data['date'] >= pd.to_datetime("2020-03-22")]



# 2- Selecting only cities from the state of SP

data = data[data['codmun'].notna()]

data = data[data['state'] == "SP"]



# 3- Reset index: drop = False

data.reset_index(inplace = True, drop = True)



# 4- Drop munic with code = 350000. This is not recognized city.

data = data[data['codmun'] != 350000]

data.reset_index(inplace = True, drop = True)



# 5- Convert pop data to int

data["populacaoTCU2019"] = data["populacaoTCU2019"].astype(int)
# Caminho para os dados

path = "../input/de_para_drs_sp.csv"



# Lê os dados do CSV.

de_para_sp_regions = pd.read_csv(path, encoding='UTF-8', delimiter=';' , low_memory=False,  dtype={'codmun': float})
# 1) Copy main data

data_munic_sp_init = data.copy()
# 2) Get useful columns

data_munic_sp = data_munic_sp_init[["date", "codmun", "newDeaths", "totalDeaths", "newConfirmed", "totalConfirmed", "populacaoTCU2019"]].copy()
# 3) Now Merge with population data

data_munic_sp = data_munic_sp.merge(de_para_sp_regions, on="codmun")



# Rearrange columns

data_munic_sp.insert(1, "regional", data_munic_sp.pop("regional"))
data_munic_sp
# 4) Now get deaths and confirmed cases per 1MM per city



# Deaths

data_munic_sp["newDeaths_1MM"] = (data_munic_sp["newDeaths"] / data_munic_sp["populacaoTCU2019"])*1000000

data_munic_sp["totalDeaths_1MM"] = (data_munic_sp["totalDeaths"] / data_munic_sp["populacaoTCU2019"])*1000000



# Confirmed Cases

data_munic_sp["newConfirmed_1MM"] = (data_munic_sp["newConfirmed"] / data_munic_sp["populacaoTCU2019"])*1000000

data_munic_sp["totalConfirmed_1MM"] = (data_munic_sp["totalConfirmed"] / data_munic_sp["populacaoTCU2019"])*1000000
data_munic_sp
# 5) Here we aggregate per region, bringing the sum of new deaths and confirmed cases in that region per day. We also get the new deaths and confirmed rates per 1MM, but considering the

# average rate within the cities in such region, which provides an approximation of the real number.



# Group columns

data_region_sp = data_munic_sp.groupby(['regional', 'date']).agg({'newDeaths': 'sum', 'newConfirmed': 'sum', 'newDeaths_1MM': 'mean', 'newConfirmed_1MM': 'mean'})

# Name aggregations

data_region_sp.columns = ['newDeaths', 'newConfirmed', 'newDeaths_1MM_average', 'newConfirmed_1MM_average']

# Reset index

data_region_sp = data_region_sp.reset_index()
data_region_sp
data_region_deaths = data_region_sp[['regional', 'date', "newDeaths", "newDeaths_1MM_average"]].copy()
# Creating SMA on new deaths



# First, send date to index. That way we can use 'date' as reference, instead of observations. So, if we have a missing that, it will be considered.

data_region_deaths.set_index(["date"], inplace = True, append = False, drop = True)



# SMA 7 days

data_region_deaths.loc[:,'newDeaths_1MM_sma7'] = data_region_deaths.groupby('regional')['newDeaths_1MM_average'].rolling('7D',min_periods=7).mean().reset_index(0,drop=True)



# SMA 15 days

data_region_deaths.loc[:,'newDeaths_1MM_sma15'] = data_region_deaths.groupby('regional')['newDeaths_1MM_average'].rolling('15D',min_periods=15).mean().reset_index(0,drop=True)



# SMA 30 days

data_region_deaths.loc[:,'newDeaths_1MM_sma30'] = data_region_deaths.groupby('regional')['newDeaths_1MM_average'].rolling('30D',min_periods=30).mean().reset_index(0,drop=True)
data_region_deaths
# Graph: newDeaths_1MM_average per region 



graph = data_region_deaths[['regional', "newDeaths_1MM_sma7"]].copy()



graph.reset_index(inplace = True, drop = False)



graph.rename(columns={"regional": "unit"}, inplace=True)



list_labels = { "DRS - Munic. de São Paulo - Fase laranja": "São Paulo - Stage 2", "Fase laranja": "Other cities - Stage 2", "DRS XI - Presidente Prudente - Fase Vermelha": "Presidente Prudente - Stage 1", 

               "DRS XIII - Ribeirão Preto - Fase Vermelha": "Ribeirão Preto - Stage 1", "DRS V - Barretos - Fase Vermelha": "Barretos - Stage 1"}



create_graph_single(graph, ["DRS - Munic. de São Paulo - Fase laranja", "Fase laranja", "DRS V - Barretos - Fase Vermelha", 

                                             "DRS XI - Presidente Prudente - Fase Vermelha", "DRS XIII - Ribeirão Preto - Fase Vermelha"], list_labels,

                                            "New deaths of COVID-19 per 1MM pop. SMA 7 days.", 

                                            "Stage from 08/jun to 15/jun", "Source: Ministry of Health of Brazil", -1.5, 20, 6, "2020-03-24", "normal", "mean")
data_region_confirmed = data_region_sp[['regional', 'date', "newConfirmed", "newConfirmed_1MM_average"]].copy()
# Creating SMA on new deaths



# First, send date to index. That way we can use 'date' as reference, instead of observations. So, if we have a missing that, it will be considered.

data_region_confirmed.set_index(["date"], inplace = True, append = False, drop = True)



# SMA 7 days

data_region_confirmed.loc[:,'newConfirmed_1MM_sma7'] = data_region_confirmed.groupby('regional')['newConfirmed_1MM_average'].rolling('7D',min_periods=7).mean().reset_index(0,drop=True)



# SMA 15 days

data_region_confirmed.loc[:,'newConfirmed_1MM_sma15'] = data_region_confirmed.groupby('regional')['newConfirmed_1MM_average'].rolling('15D',min_periods=15).mean().reset_index(0,drop=True)



# SMA 30 days

data_region_confirmed.loc[:,'newConfirmed_1MM_sma30'] = data_region_confirmed.groupby('regional')['newConfirmed_1MM_average'].rolling('30D',min_periods=30).mean().reset_index(0,drop=True)
data_region_confirmed
# Graph: newDeaths_1MM_average per region 



graph = data_region_confirmed[['regional', "newConfirmed_1MM_sma7"]].copy()



graph.reset_index(inplace = True, drop = False)



graph.rename(columns={"regional": "unit"}, inplace=True)



list_labels = { "DRS - Munic. de São Paulo - Fase laranja": "São Paulo - Stage 2", "Fase laranja": "Other cities - Stage 2", "DRS XI - Presidente Prudente - Fase Vermelha": "Presidente Prudente - Stage 1", 

               "DRS XIII - Ribeirão Preto - Fase Vermelha": "Ribeirão Preto - Stage 1", "DRS V - Barretos - Fase Vermelha": "Barretos - Stage 1"}



create_graph_single(graph, ["DRS - Munic. de São Paulo - Fase laranja", "Fase laranja", "DRS V - Barretos - Fase Vermelha", 

                                             "DRS XI - Presidente Prudente - Fase Vermelha", "DRS XIII - Ribeirão Preto - Fase Vermelha"], list_labels,

                                            "New confirmed cases of COVID-19 per 1MM pop. SMA 7 days.", 

                                            "Stage from 08/jun to 15/jun", "Source: Ministry of Health of Brazil", -1.5, 20, 6, "2020-03-24", "normal", "mean")
# Caminho para os dados

path = "../input/brazil_pop_2019.csv"



# Lê os dados do CSV.

data_additional = pd.read_csv(path, encoding='UTF-8', delimiter=';' , low_memory=False)
# Caminho para os dados

path = "../input/HIST_PAINEL_COVIDBR_10jun2020.csv"



# Lê os dados do CSV.

data = pd.read_csv(path, encoding='UTF-8', delimiter=';' , low_memory=False)



# Renaming according to previous analysis

data.rename(columns={"data": "date", "estado": "state", "municipio": "city", "regiao": "region",

                              "casosAcumulado": "totalConfirmed", "casosNovos": "newConfirmed", 

                             "obitosAcumulado": "totalDeaths", "obitosNovos": "newDeaths",

                              "emAcompanhamentoNovos": "newFollowing", "Recuperadosnovos": "totalRecoveredBrazil"}, inplace=True)



# Assign correct row to Brazil total

data["state"] = data["state"].replace(np.NaN, "Brazil")



# Assign date

data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)
# For each country/date, create column for new recovered



# Sort by country and date

data = data.sort_values(["state", "date"], ascending = (True, True))

data.reset_index(inplace = True, drop = True)



# A column to receive new recovered cases for each day

data["newRecoveredBrazil"] = 0



# This function will assign to each country/day the number of new cases, based on the difference among accumulated cases of the actual and last day.

#country_before = data.iloc[0,0]



for row in range(1, data.shape[0], 1):

    #country_actual = data.iloc[row,0]

    if(data.iloc[row,1] == "Brazil"):

        data.iloc[row,16] = data.iloc[row,14] - data.iloc[row-1,14]

    else:

        pass

# Getting only total data for states. Ignore total for cities.



# 1- Filter only data after 22/march/2020, since this is the first day with all states having data

data = data.loc[data['date'] >= pd.to_datetime("2020-03-22")]



# 2- Drop row referring to cities. (codmun not NaN) We will be using only the number for states.

data = data[data['codmun'].isna()]



# Reset index: drop = False

data.reset_index(inplace = True, drop = True)
# 1) Copy main data

data_state_br_init = data.copy()
# 2) Get useful columns

data_state_br = data_state_br_init[["date", "state", "newDeaths", "totalDeaths", "newConfirmed", "totalConfirmed", "newRecoveredBrazil", "totalRecoveredBrazil"]]
# 3) Now Merge with population data

data_state_br = data_state_br.merge(data_additional, on="state")

del data_state_br["gov_name"]

del data_state_br["gov_image"]
data_state_br
# 4) Now Create a column for total deaths by 1MM habitants from each state. 

# Important that this command can only be used AFTER grouping by states.



# Deaths

data_state_br["newDeaths_1MM"] = (data_state_br["newDeaths"] / data_state_br["pop_2019"])*1000000

data_state_br["totalDeaths_1MM"] = (data_state_br["totalDeaths"] / data_state_br["pop_2019"])*1000000



# Cases

data_state_br["newConfirmed_1MM"] = (data_state_br["newConfirmed"] / data_state_br["pop_2019"])*1000000

data_state_br["totalConfirmed_1MM"] = (data_state_br["totalConfirmed"] / data_state_br["pop_2019"])*1000000



# Recovered

data_state_br["newRecovered_Brazil_1MM"] = (data_state_br["newRecoveredBrazil"] / data_state_br["pop_2019"])*1000000

data_state_br["totalRecovered_Brazil_1MM"] = (data_state_br["totalRecoveredBrazil"] / data_state_br["pop_2019"])*1000000
data_state_br
# Checking to see if every state has the same ammount of available records.

data_state_br.groupby(['date', 'state']).size().unstack(fill_value=0).describe().T.sort_values(["count"], ascending =True)
# 1) Selecting data

data_state_br_deaths = data_state_br.copy()

data_state_br_deaths = data_state_br_deaths[["date", "state", "newDeaths_1MM"]]
data_state_br_deaths
# 2) Creating SMA on new deaths



# First, send date to index. That way we can use 'date' as reference, instead of observations. So, if we have a missing that, it will be considered.

data_state_br_deaths.set_index(["date"], inplace = True, append = False, drop = True)



# SMA 7 days

data_state_br_deaths.loc[:,'newDeaths_1MM_sma7'] = data_state_br_deaths.groupby('state')['newDeaths_1MM'].rolling('7D',min_periods=7).mean().reset_index(0,drop=True)



# SMA 15 days

data_state_br_deaths.loc[:,'newDeaths_1MM_sma15'] = data_state_br_deaths.groupby('state')['newDeaths_1MM'].rolling('15D',min_periods=15).mean().reset_index(0,drop=True)



# SMA 30 days

data_state_br_deaths.loc[:,'newDeaths_1MM_sma30'] = data_state_br_deaths.groupby('state')['newDeaths_1MM'].rolling('30D',min_periods=30).mean().reset_index(0,drop=True)
###### Graph



graph = data_state_br_deaths[['state', "newDeaths_1MM_sma7"]].copy()



graph.reset_index(inplace = True, drop = False)



graph.rename(columns={"state": "unit"}, inplace=True)



list_labels = {"Brazil": "Brazil", "RJ": "RJ", "CE": "CE", "PA": "PA", "PE": "PE", "AM": "AM"}



create_graph_single(graph, ["Brazil", "RJ", "CE", "PA", "PE", "AM"],

                    list_labels,

                    "New deaths of COVID-19 per 1MM pop. SMA 7 days.", 

                    "State", "Source: Ministry of Health of Brazil", 

                    -3.5, 20, 6, "2020-04-01", "normal", "mean")
# 1) Selecting data

data_state_br_confirmed = data_state_br.copy()

data_state_br_confirmed = data_state_br_confirmed[["date", "state", "newConfirmed_1MM"]]
data_state_br_confirmed
# 2) Creating SMA on new deaths



# First, send date to index. That way we can use 'date' as reference, instead of observations. So, if we have a missing that, it will be considered.

data_state_br_confirmed.set_index(["date"], inplace = True, append = False, drop = True)



# SMA 7 days

data_state_br_confirmed.loc[:,'newConfirmed_1MM_sma7'] = data_state_br_confirmed.groupby('state')['newConfirmed_1MM'].rolling('7D',min_periods=7).mean().reset_index(0,drop=True)



# SMA 15 days

data_state_br_confirmed.loc[:,'newConfirmed_1MM_sma15'] = data_state_br_confirmed.groupby('state')['newConfirmed_1MM'].rolling('15D',min_periods=15).mean().reset_index(0,drop=True)



# SMA 30 days

data_state_br_confirmed.loc[:,'newConfirmed_1MM_sma30'] = data_state_br_confirmed.groupby('state')['newConfirmed_1MM'].rolling('30D',min_periods=30).mean().reset_index(0,drop=True)
###### Graph



graph = data_state_br_confirmed[['state', "newConfirmed_1MM_sma7"]].copy()



graph.reset_index(inplace = True, drop = False)



graph.rename(columns={"state": "unit"}, inplace=True)



list_labels = {"Brazil": "Brazil", "RJ": "RJ", "CE": "CE", "PA": "PA", "PE": "PE", "AM": "AM"}



create_graph_single(graph, ["Brazil", "RJ", "CE", "PA", "PE", "AM"],

                    list_labels,

                    "New confirmed cases of COVID-19 per 1MM pop. SMA 7 days.", 

                    "State", "Source: Ministry of Health of Brazil", 

                    -60, 20, 6, "2020-04-01", "normal", "mean")
# 1) Selecting data

data_br_recovered = data_state_br.copy()

data_br_recovered = data_br_recovered[["date", "state", "newRecovered_Brazil_1MM"]]



# Selecting data only for the consolidate, which mean, for Brazil

data_br_recovered = data_br_recovered.loc[(data_br_recovered['state'] == "Brazil")]

data_br_recovered.reset_index(inplace = True, drop = True)
data_br_recovered
# 2) Creating SMA on new deaths



# First, send date to index. That way we can use 'date' as reference, instead of observations. So, if we have a missing that, it will be considered.

data_br_recovered.set_index(["date"], inplace = True, append = False, drop = True)



# SMA 7 days

data_br_recovered.loc[:,'newRecovered_Brazil_1MM_sma7'] = data_br_recovered.groupby('state')['newRecovered_Brazil_1MM'].rolling('7D',min_periods=7).mean().reset_index(0,drop=True)



# SMA 15 days

data_br_recovered.loc[:,'newRecovered_Brazil_1MM_sma15'] = data_br_recovered.groupby('state')['newRecovered_Brazil_1MM'].rolling('15D',min_periods=15).mean().reset_index(0,drop=True)



# SMA 30 days

data_br_recovered.loc[:,'newRecovered_Brazil_1MM_sma30'] = data_br_recovered.groupby('state')['newRecovered_Brazil_1MM'].rolling('30D',min_periods=30).mean().reset_index(0,drop=True)
graph = data_br_recovered[['state', "newRecovered_Brazil_1MM", "newRecovered_Brazil_1MM_sma7", "newRecovered_Brazil_1MM_sma15", "newRecovered_Brazil_1MM_sma30"]].copy()



graph.reset_index(inplace = True, drop = False)



graph.rename(columns={"state": "unit"}, inplace=True)



list_labels = {"Brazil": "Brazil"}



create_graph_multiple(graph, ["Brazil"], "New recovered cases of COVID-19 per 1MM pop. SMA 7 days",

                          "Source: Ministry of Health of Brazil", 

                          -15, 20, 6, "2020-04-01", "mean" )
data_br_recovered
# 1) Getting data from Brazil

# deaths

br_deaths = data_state_br_deaths.loc[(data_state_br_deaths['state'] == "Brazil")].copy()

del br_deaths["newDeaths_1MM_sma15"]

del br_deaths["state"]

    

# confirmed

br_confirmed = data_state_br_confirmed.loc[(data_state_br_confirmed['state'] == "Brazil")].copy()

del br_confirmed["newConfirmed_1MM_sma15"]

del br_confirmed["state"]



# recovered

br_recovered = data_br_recovered.loc[(data_br_recovered['state'] == "Brazil")].copy()

del br_recovered["newRecovered_Brazil_1MM_sma15"]

del br_recovered["state"]
br_mod1 = br_deaths.merge(br_confirmed, on="date")
br_mod2 = br_mod1.merge(br_recovered, on="date")
##### Graph



graph = br_mod2.copy()



graph.reset_index(inplace = True, drop = False)



graph["unit"]="Brazil"



# Rename columns

graph.rename(columns={"newConfirmed_1MM": "New Confirmed", "newConfirmed_1MM_sma7": "New Confirmed SMA7", "newConfirmed_1MM_sma15": "New Confirmed SMA15", "newConfirmed_1MM_sma30": "New Confirmed SMA30",

                    "newRecovered_Brazil_1MM": "New Recovered", "newRecovered_Brazil_1MM_sma7": "New Recovered SMA7", "newRecovered_Brazil_1MM_sma15": "New Recovered SMA 15", "newRecovered_Brazil_1MM_sma30": "New Recovered SMA30",

                    "newDeaths_1MM": "New Deaths", "newDeaths_1MM_sma7": "New Deaths SMA7", "newDeaths_1MM_sma15": "New Deaths SMA15", "newDeaths_1MM_sma30": "New Deaths SMA30"}, inplace=True)



del graph["New Confirmed SMA7"]

del graph["New Recovered SMA7"]

del graph["New Deaths SMA7"]



create_graph_multiple(graph, ["Brazil"], "New deaths, recovered and confirmed cases of COVID-19 per 1MM pop. \n New and Simple Moving Average for 30 days",

                          "Source: Ministry of Health of Brazil", 

                          -25, 20, 6, "2020-04-01", "mean" )
# Caminho para os dados

path = "../input/data_pop_2020.csv"



# Lê os dados do CSV.

data_pop_2020 = pd.read_csv(path, encoding='UTF-8', delimiter=';' , low_memory=False, dtype={"country_pop_2020": float})
# Caminho para os dados

path = "../input/flourish_flags.csv"



# Lê os dados do CSV.

flourish_flags = pd.read_csv(path, encoding='UTF-8', delimiter=';' , low_memory=False)
path = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

data = pd.read_csv(path, encoding='ISO-8859-1', delimiter=',' , low_memory=False)

data_deaths = data.copy()



# Remove lat & long values

del data_deaths["Lat"]

del data_deaths["Long"]

del data_deaths["Province/State"]
path = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

data = pd.read_csv(path, encoding='ISO-8859-1', delimiter=',' , low_memory=False)

data_confirmed = data.copy()



del data



# Remove lat & long values

del data_confirmed["Lat"]

del data_confirmed["Long"]
path = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

data = pd.read_csv(path, encoding='ISO-8859-1', delimiter=',' , low_memory=False)

data_recovered = data.copy()



del data



# Remove lat & long values

del data_recovered["Lat"]

del data_recovered["Long"]
# This function receives dataset from Johns Hopkins and returns treated data per 1MM for confirmed, recovered and deaths.

def get_sma_per1MM(data_inicial, name):

    '''

    :param data: Dataset from Johns hopkins for confirmed, recovered or deaths.

    :param name: The type of data analysed. Confirmed, Recovered or Deaths by COVID-19.

    '''

    data = data_inicial.copy()

    # 1) Group by country

    data = data.groupby(['Country/Region']).sum()

    data.reset_index(inplace = True, drop = False)

    

    # 2) Make data in rows

    data = data.melt(id_vars=["Country/Region"], var_name="date", value_name=f"total{name}")

    

    # 3) For each country/date, create column for new deaths

    # Assign date type

    data['date'] = pd.to_datetime(data['date']).dt.date

    # Sort by country and date

    data = data.sort_values(["Country/Region", "date"], ascending = (True, True))

    data.reset_index(inplace = True, drop = True)

    # A column to receive new cases for each day

    data[f"new{name}"] = 0

    # This function will assign to each country/day the number of new cases, based on the difference among accumulated cases of the actual and last day.

    country_before = data.iloc[0,0]

    for row in range(1, data.shape[0], 1):

        country_actual = data.iloc[row,0]

        if(country_actual == country_before):

            data.iloc[row,3] = data.iloc[row,2] - data.iloc[row-1,2]

        else:

            data.iloc[row,3] = data.iloc[row,2]

        country_before = country_actual

    # Reset index: drop = False

    data.reset_index(inplace = True, drop = True)

    

    # 4) Add population data

    data = data.merge(data_pop_2020, on="Country/Region")

    

    # 5) Now Create a column for total deaths by 1MM habitants from each country. 

    # Important that this command can only be used AFTER grouping by contries.

    data[f"new{name}_1MM"] = 0

    data[f"new{name}_1MM"] = (data[f"new{name}"] / data["country_pop_2020"])*1000000

    data[f"total{name}_1MM"] = 0

    data[f"total{name}_1MM"] = (data[f"total{name}"] / data["country_pop_2020"])*1000000

    # remove pop column

    del data["country_pop_2020"]

    

    ##### Getting SMA for new cases and total cases (7,15,30)

    

    # 1) Creating SMA per country - New cases per 1MM

    # SMA 7 days

    data.loc[:,f'new{name}_1MM_sma7'] = data.groupby('Country/Region')[f'new{name}_1MM'].rolling(window=7).mean().reset_index(0,drop=True)

    # SMA 15 days

    data.loc[:,f'new{name}_1MM_sma15'] = data.groupby('Country/Region')[f'new{name}_1MM'].rolling(window=15).mean().reset_index(0,drop=True)

    # SMA 30 days

    data.loc[:,f'new{name}_1MM_sma30'] = data.groupby('Country/Region')[f'new{name}_1MM'].rolling(window=30).mean().reset_index(0,drop=True)

    

    # 2) Creating SMA per country - Total cases per 1MM

    # SMA 7 days

    data.loc[:,f'total{name}_1MM_sma7'] = data.groupby('Country/Region')[f'total{name}_1MM'].rolling(window=7).mean().reset_index(0,drop=True)

    # SMA 15 days

    data.loc[:,f'total{name}_1MM_sma15'] = data.groupby('Country/Region')[f'total{name}_1MM'].rolling(window=15).mean().reset_index(0,drop=True)

    # SMA 30 days

    data.loc[:,f'total{name}_1MM_sma30'] = data.groupby('Country/Region')[f'total{name}_1MM'].rolling(window=30).mean().reset_index(0,drop=True)

        

    return data

data_deaths_world = get_sma_per1MM(data_deaths, "Deaths")
#graph



graph = data_deaths_world[["date", "Country/Region", "newDeaths_1MM_sma7"]].copy()



graph.rename(columns={"Country/Region": "unit"}, inplace=True)



list_labels = {"US": "US", "Brazil": "Brazil",  "Belgium": "Belgium", "United Kingdom": "United Kingdom",

               "Spain": "Spain",  "Sweden": "Sweden", "Mexico": "Mexico"}



create_graph_single(graph, ["US", "Brazil",  "Spain", "Sweden", "United Kingdom", "Belgium", "Mexico"],

                    list_labels,

                    "Brazil compared to other countries - New deaths of COVID-19 per 1MM pop. \n Simple Moving Average - 7 days", 

                    "Country", "Source: Johns Hopkins", 

                    -5, 20, 6, "2020-03-15", "normal", "mean")
# Countries with high rates of new deaths on average

graph.groupby("unit").mean().sort_values(["newDeaths_1MM_sma7"], ascending = False).head(15)
################################################

# Prepare data for visual

################################################

data_deaths_world_visual = data_deaths_world.pivot(index='Country/Region', columns='date', values='newDeaths_1MM_sma7')

data_deaths_world_visual.reset_index(inplace = True, drop = False)
# Adding flags and Regions

data_deaths_world_visual.rename(columns={"Country/Region": "Country"}, inplace=True)

data_deaths_world_visual = data_deaths_world_visual.merge(flourish_flags, on="Country")

# Rearrange columns

data_deaths_world_visual.insert(1, "region", data_deaths_world_visual.pop("region"))

data_deaths_world_visual.insert(2, "Image URL", data_deaths_world_visual.pop("Image URL"))
# Remove Djibouti

data_deaths_world_visual.drop(data_deaths_world_visual[data_deaths_world_visual["Country"] == "Djibouti"].index, inplace=True)

data_deaths_world_visual.drop(data_deaths_world_visual[data_deaths_world_visual["Country"] == "San Marino"].index, inplace=True)

data_deaths_world_visual.drop(data_deaths_world_visual[data_deaths_world_visual["Country"] == "Andorra"].index, inplace=True)
data_deaths_world_visual
# Export data

#data_deaths_world_visual.to_excel("output/data_deaths_world_visual.xlsx")
data_confirmed_world = get_sma_per1MM(data_confirmed, "Confirmed")
#graph



graph = data_confirmed_world[["date", "Country/Region", "newConfirmed_1MM_sma7"]].copy()



graph.rename(columns={"Country/Region": "unit"}, inplace=True)



list_labels = {"US": "US", "Brazil": "Brazil",  "Belgium": "Belgium", "Peru": "Peru",

               "Spain": "Spain",  "Sweden": "Sweden", "Chile": "Chile"}



create_graph_single(graph, ["US", "Brazil",  "Spain", "Sweden", "Peru", "Belgium", "Chile"],

                    list_labels,

                    "Brazil compared to other countries - New confirmed cases of COVID-19 per 1MM pop. \n Simple Moving Average - 7 days", 

                    "Country", "Source: Johns Hopkins", 

                    -5, 20, 6, "2020-03-15", "normal", "mean")
# States with high rates of new deaths on average

graph.groupby("unit").mean().sort_values(["newConfirmed_1MM_sma7"], ascending = False).head(15)
data_recovered_world = get_sma_per1MM(data_recovered, "Recovered")
#graph



graph = data_recovered_world[["date", "Country/Region", "newRecovered_1MM_sma7"]].copy()



graph.rename(columns={"Country/Region": "unit"}, inplace=True)



list_labels = {"US": "US", "Brazil": "Brazil",  "Belgium": "Belgium", "Italy": "Italy",

               "Spain": "Spain",  "Peru": "Peru"}



create_graph_single(graph, ["US", "Brazil",  "Spain", "Italy", "Peru", "Belgium"],

                    list_labels,

                    "Brazil compared to other countries - New recovered cases of COVID-19 per 1MM pop. \n Simple Moving Average - 7 days", 

                    "Country", "Source: Johns Hopkins", 

                    -5, 20, 6, "2020-03-15", "normal", "mean")
# States with high rates of new deaths on average

graph.groupby("unit").mean().sort_values(["newRecovered_1MM_sma7"], ascending = False).head(15)