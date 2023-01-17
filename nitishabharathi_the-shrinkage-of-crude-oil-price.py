import plotly.express as px

import plotly.graph_objects as go



country_producing = ['United States','Saudi Arabia','Russia','Canada','China','Iraq','UAE','Brazil','Iran','Kuwait']

oil_produced = [19.51,11.81,11.49,5.50,4.89,4.74,4.01,3.67,3.19,2.94]



fig = go.Figure(data=[go.Pie(labels=country_producing,

                             values=oil_produced)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20)

fig.update_layout(title=go.layout.Title(text="<b>World's Largest Oil Producers (mn barrels/day)</b>", font=dict(

                family="Courier New, monospace",

                size=22,

                color="black"

            )))



fig.update_layout(annotations=[

       go.layout.Annotation(

            showarrow=False,

            text='Source: EIA, 2019 data',

            xanchor='right',

            x=0.75,

            xshift=275,

            yanchor='top',

            y=0.05,

            font=dict(

                family="Courier New, monospace",

                size=12,

                color="black"

            )

        )])



fig.show()
country_consuming = ['United States','China','India','Japan','Russia','Saudi Arabia','Brazil','South Korea','Germany','Canada']

consumption = [19.96,13.57,4.32,3.92,3.69,3.33,3.03,2.63,2.45,2.42]



fig = go.Figure(data=[go.Pie(labels=country_consuming,

                             values=consumption)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20)

fig.update_layout(title=go.layout.Title(text="<b>World's Largest Oil Consumers (mn barrels/day)</b>", font=dict(

                family="Courier New, monospace",

                size=22,

                color="black"

            )))



fig.update_layout(annotations=[

       go.layout.Annotation(

            showarrow=False,

            text='Source: EIA, 2017 data',

            xanchor='right',

            x=0.75,

            xshift=275,

            yanchor='top',

            y=0.05,

            font=dict(

                family="Courier New, monospace",

                size=12,

                color="black"

            )

        )])



fig['layout']['xaxis'].update(side='top')



fig.show()
country_export = ['Saudi Arabia','Russia','Iraq','Canada','UAE','Kuwait','Iran','United States','Nigeria','Kazakhstan','Angola','Norway','Libya','Mexico','Venezuela']

export = [182.5,129,91.7,66.9,58.4,51.7,50.8,48.3,43.6,37.8,36.5,33.3,26.7,26.5,26.4]



fig = go.Figure(data=[go.Pie(labels=country_export,

                             values=export)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20)

fig.update_layout(title=go.layout.Title(text="<b>World's Largest Oil Exporters (US$ billion)</b>", font=dict(

                family="Courier New, monospace",

                size=22,

                color="black"

            )))



fig.update_layout(annotations=[

       go.layout.Annotation(

            showarrow=False,

            text='Source: OECD, 2019 data',

            xanchor='right',

            x=0.75,

            xshift=275,

            yanchor='top',

            y=0.05,

            font=dict(

                family="Courier New, monospace",

                size=12,

                color="black"

            )

        )])



fig['layout']['xaxis'].update(side='top')



fig.show()
affected_country = ['Iraq','Libya','Congo Republic','Kuwait','South Sudan','Saudi Arabia','Oman','Equatorial Guinea','Azerbaijan','Angola','Iran','Gabon','Timor-Leste','Qatar','UAE']

oil_rent = [37.8,37.3,36.7,36.6,31.3,23.1,21.8,19.2,17.9,15.8,15.3,15.3,14.5,14.2,13.1]

affected_country = affected_country[::-1]

oil_rent = oil_rent[::-1]





fig = go.Figure(go.Bar(

            x=oil_rent,

            y=affected_country,

            orientation='h',

            text = oil_rent,

            textposition='auto'))

fig.update_traces(marker_color='purple')



fig.update_layout(title=go.layout.Title(text="<b>Countries Heavily Dependent on Oil Profits to Power GDP</b>", font=dict(

                family="Courier New, monospace",

                size=22,

                color="black"

            )))

fig.update_layout(annotations=[

       go.layout.Annotation(

            showarrow=False,

            text='Source: World Bank',

            xanchor='right',

            x=35,

            xshift=275,

            yanchor='top',

            y=0.05,

            font=dict(

                family="Courier New, monospace",

                size=10,

                color="black"

            )

        )])



fig['layout']['xaxis'].update(side='top')



fig.show()