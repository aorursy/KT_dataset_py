import pandas as pd

import geopandas as gpd

import requests

from datetime import datetime,timedelta
# Daily update at 16:00u. I'll run and save this script on a daily base





url = 'https://geodata.rivm.nl/covid-19/COVID-19_aantallen_gemeente_cumulatief.json'



try:

    data = requests.get(url, timeout=2).json()

except requests.exceptions.RequestException:

    raise Exception('Failed to connect to %s' % url) from None



rivm = pd.DataFrame.from_dict(data, orient='columns')

rivm.head()
#Yesterday's date 

today = datetime.today() - timedelta(days=1)

today = today.strftime('%Y-%m-%d')+ " 10:00:00"

print(today)
#Last week's date

weekAgo = datetime.today() - timedelta(days=7)

weekAgo = weekAgo.strftime('%Y-%m-%d')+ " 10:00:00"

print(weekAgo)
# maak een nieuwe df van vandaag en vorige week, de gegevens van de provincies mogen eruit

cases_Today = rivm.loc[(rivm.Date_of_report == today)& (rivm.Municipality_name.notnull())]

cases_weekAgo = rivm.loc[(rivm.Date_of_report == weekAgo)& (rivm.Municipality_name.notnull())]

cases_Today.head()
# maak een nieuwe df met het verschil van aantal gevallen van vorige week en deze week

cases_lastWeek = cases_Today.loc[:, ['Municipality_code', 'Municipality_name', 'Province','Total_reported']]

cases_lastWeek['Total_reported_LastWeek'] = cases_weekAgo['Total_reported'].values

cases_lastWeek['this_week_cases'] = cases_lastWeek['Total_reported'] - cases_lastWeek['Total_reported_LastWeek']

cases_lastWeek.head()
# Haal de kaart met gemeentegrenzen op van PDOK

geodata_url = 'https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_gemeente_2019_gegeneraliseerd&outputFormat=json'

kaart_gemeente = gpd.read_file(geodata_url)

# De covid-data kan nu gekoppeld worden aan de gemeentegrenzen met merge.

# Koppel covid-data aan geodata met gemeentecodes

kaart_gemeente = pd.merge(kaart_gemeente, cases_lastWeek,

                           left_on = "statcode", 

                           right_on = "Municipality_code")

kaart_gemeente.head()
#Tot slot kan de thematische kaart gemaakt worden met de functie plot.





p = kaart_gemeente.plot(column='this_week_cases', 

                         figsize = (14,12), legend=True, cmap="Reds")

p.axis('off')

p.set_title('Amount of absolute reported Covid cases in The Netherlands only last Week')