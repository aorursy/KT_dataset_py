import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go



pol_dic = {

                'NO2': [0.04, [0, 1, 2, 3, 4, 5, 6]],

                'NO': [0.06, [0, 2]],

                'SO2': [0.05, [0, 1, 2, 3, 4, 5, 6]],

                'HCOH': [0.003, [0, 3]],

                'NH3': [0.04, [1, 5]],

             }

stations = pd.read_csv("../input/bishkek-pollution-kyrgyzhydromet-data/bishkek_stations.csv")



pol = pd.read_csv("../input/bishkek-pollution-kyrgyzhydromet-data/bishkek_pollution_all_stations.csv")



start_date = pol.head(1).values[0][0]

end_date = pol.tail(1).values[0][0]

stations
fig_map = px.scatter_mapbox(stations, lon='lon', lat='lat', hover_name='id', text='name', zoom=8)

fig_map.update_layout(mapbox_style="open-street-map")

fig_map.show()
for p in ['NO2', 'NO', 'SO2', 'HCOH', 'NH3']:

    for id in pol_dic[p][1]:

        fig1 = px.bar(pol[(pol['station'] == id) & (pol['pollutant'] == p)], x='date', y='value', title=str(p) + ", " + str(stations.loc[id]['name']))

        fig1.add_trace(

            go.Scatter(

                x=[start_date, end_date],

                y=[pol_dic[p][0], pol_dic[p][0]],

                mode="lines",

                line=go.scatter.Line(color="red"),

                showlegend=False)

        )

        fig1.show()