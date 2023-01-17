import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import plotly.graph_objects as go

import os

import pycountry

import seaborn as sns
def format_string(string):

    if string=='sealife':

        string = 'sealife_scheveningen'

    return  string.replace('_',' ').title()







def plot_lines(df,

              title='Visitor Split',

              subtitle='',

               x_title='',

               y_title='',

               hoverinfo='x+text',

                y=[2018,2019],

                y_text=None,

                textposition=[None]*10,

                labelposition=['middle']*10,

                label_size=18,

                x='weekday',

                modes=['lines']*10,

                colors=['grey','#33C4CA'],

                fill=[None,None],

                line_width=[1]*10,

                line_dash=['solid']*10,

                label_right=True,

                height=None,

                width=None,

                legend=dict(x=1, y=1),

                title_position=dict(x=0, y=1.1),

                font_family='Montserrat',

                y_range=None,

                x_autorange=None,

                x_side =None,

                show_x_axis=True,

                show_y_axis=False,

                opacity=[1]*42,

                margin=dict(

                            autoexpand=False,

                            l=100,

                            r=50,

                            t=100,

                        ),

               additional_traces=[],

               additional_annotations=[],

               comment=''

                  ):

    if not y_text:

        y_text=y

    data=additional_traces

    label_index= -1 if label_right else 0

    for k in range(len(y)):

        

        data.append(go.Scatter(name=y[k], 

                   x=df[x], 

                   y=df[y[k]],

                   mode=modes[k],

                   hoverinfo=hoverinfo,

                   fill=fill[k],

                   text=df[y_text[k]],

                   textposition=textposition[k],

                   textfont=dict(

                                family=font_family,

                                size=label_size,

                                color=colors[k]

                                ),

                   line=dict(width=line_width[k], 

                             dash=line_dash[k],

                             shape='spline'),

                   opacity=opacity[k],

                   marker=dict(color=colors[k]),

                    showlegend=False

                                ),

                   )

        additional_annotations.append(dict(xref='x' if label_right else 'paper', 

                                           x=df[[y[k],x]].dropna()[x].iloc[label_index] if label_right else 0, 

                                           y=df[y[k]].dropna().iloc[label_index],

                                           xanchor='left' if label_right else 'right', 

                                           yanchor=labelposition[k],

                                          text= format_string(y[k]),

                                          font=dict(family=font_family,

                                            size=label_size+4,

                                            color=colors[k]

                                        ),

                              showarrow=False))

        fig = go.Figure(data)

    annotations=additional_annotations

    additional_annotations.append(dict(xref='paper', yref='paper', x=1.0, y=1.4,

                              xanchor='right', yanchor='top',

                              text= '<i>'+comment+'</i>',

                              font=dict(family=font_family,

                                        size=18,

                                        color='#E6E6E6'

                                        ),

                              showarrow=False))

    # Title

    annotations.append(dict(xref='paper', yref='paper', x=title_position['x'], y=title_position['y'],

                                  xanchor='left', yanchor='bottom',

                                  text=title,

                                  font=dict(family=font_family,

                                            size=30,

                                            color='lightgrey'

                                            ),

                                  showarrow=False))

    annotations.append(dict(xref='paper', yref='paper', x=title_position['x'], y=title_position['y'],

                                  xanchor='left', yanchor='top',

                                  text=subtitle,

                                  font=dict(family=font_family,

                                            size=20,

                                            color='lightgrey'

                                            ),

                                  showarrow=False))

    fig.update_layout(



        xaxis=dict(

            title = x_title,

            showline=show_x_axis,

            showgrid=False,

            showticklabels=show_x_axis,

            linecolor='rgb(204, 204, 204)',

            linewidth=2,

            ticks='outside' if show_x_axis else None,

            titlefont=dict(

                    family=font_family,

                    size=14,

                    color='lightgrey'

                ),

            tickfont=dict(

                family=font_family,

                size=14,

                color='lightgrey',

            ),

            autorange=x_autorange,

            side=x_side

        ),

        yaxis=dict(

            title = y_title,

            showline=show_y_axis,

            showgrid=False,

            showticklabels=show_y_axis,

            linecolor='rgb(204, 204, 204)',

            linewidth=2,

            ticks='outside' if show_y_axis else None,

            titlefont=dict(

                    family=font_family,

                    size=14,

                    color='lightgrey'

                ),

            tickfont=dict(

                family=font_family,

                size=14,

                color='lightgrey',

            

            ),

            range=y_range

            #autorange=x_autorange,

            #side=x_side

        ),

        autosize=True,

        annotations=annotations,

        legend=legend,

        margin=margin,

        height=height,

        width=width,

        #legend_orientation="h",

        plot_bgcolor='white'

    )

    fig.show()

    
def cumsum_diff(x):

    x.values[1:] -= x.values[:-1].copy()

    return x

def last_n_days_new_cases(x,n=4):

    y=[sum(x.values[i-(n-1):i+1]) if i>(n-1) else sum(x.values[:i+1])  for i in range(len(x))]

    return pd.DataFrame({'cases':y},index=x.index).cases/n

def last_10_days_new_cases(x,n=10):

    y=[sum(x.values[i-(n-1):i+1]) if i>(n-1) else sum(x.values[:i+1])  for i in range(len(x))]

    return pd.DataFrame({'cases':y},index=x.index).cases/n

def format_string(string):

    return  string.replace('_',' ').title()

def get_country_name(alpha_2):

    name = pycountry.countries.get(alpha_2=alpha_2).name

    if name.find(',')>=0:

        k=name.find(',')

        name=name[0:k]

    return name
covid_19_raw = (pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

               )

covid_19_raw=covid_19_raw.loc[~((covid_19_raw['Country/Region']=='France')&(covid_19_raw.Date==datetime.datetime(2020,4,4,0,0,0)))]

mobility= (pd.read_csv('../input/gmobility/gmobility_2020-03-29.csv', parse_dates=['date'])

           .assign(country=lambda x: x.alpha_2.apply(lambda y:get_country_name(y)))

           .assign(country = lambda x: np.where(x.country=='Korea','South Korea',x.country))

          )

population= (pd.read_csv('../input/population/population_2020.csv')

             .assign(country = lambda x: np.where(x.country=='US','United States',x.country))

            )

covid_19=(covid_19_raw

          #.assign(Date=lambda x: x.Date.apply(lambda y: datetime.datetime.strptime(y, '%m/%d/%y').date()))

          .rename(columns={'Country/Region':'country'})

        .assign(country = lambda x: np.where(x.country=='US','United States',x.country))

        .assign(province=lambda x: np.where(x['Province/State'].isna()|(x['Province/State']=='UK'), x.country,x['Province/State']))

        .assign(new_cases=lambda x: x.sort_values(by='Date',ascending=True).groupby(['province','country']).Confirmed.apply(cumsum_diff))

      .assign(new_deaths=lambda x: x.sort_values(by='Date',ascending=True).groupby(['province','country']).Deaths.apply(cumsum_diff))

      .assign(last_3_days_new_cases=lambda x: x.sort_values(by='Date',ascending=True).groupby(['province','country']).new_cases.apply(last_n_days_new_cases))

      .assign(last_10_days_new_cases=lambda x: x.sort_values(by='Date',ascending=True).groupby(['province','country']).new_cases.apply(last_10_days_new_cases))

      .assign(growth = lambda x: x.new_cases/(x.Confirmed-x.new_cases+2))

      .assign(growth_3 = lambda x: x.last_3_days_new_cases/(x.Confirmed-x.last_3_days_new_cases+2))

      #.query('Date<datetime.date(2020,3,23)')

     )
first_day = covid_19.query('Confirmed>10').groupby('country')[['Date']].min().reset_index().rename(columns={'Date':'first_date'})

last_day = covid_19.query('Confirmed>10').groupby('country')[['Date']].max().reset_index()

covid_last = (covid_19.merge(last_day,on=['country','Date'])

              .groupby(['country','Date']).agg({'Confirmed':'sum','Deaths':'sum','last_3_days_new_cases':'sum','new_deaths':'sum','last_10_days_new_cases':'sum'})

               .reset_index()

               .merge(population[['country','population']],on='country')

             .merge(first_day,on='country')

             .assign(confirmed_rate=lambda x: x.Confirmed/x.population*1000000)

             .assign(mortality_rate=lambda x: x.Deaths/x.population*1000000)

              .assign(fatality_rate=lambda x: x.Deaths/x.Confirmed)

               .assign(new_cases_per_cap=lambda x: x.last_3_days_new_cases/x.population*1000000)

               .assign(new_cases_growth=lambda x: x.last_3_days_new_cases/x.last_10_days_new_cases-1)

             )
import plotly.express as px

for variable in ['retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','residential','workplaces']:

    var_min=mobility[variable].min()

    var_max=mobility[variable].max()

    fig = px.choropleth(mobility.rename(columns={variable:format_string(variable)}),

                        title = format_string(variable), 

                        locations="country", 

                              locationmode='country names', color=format_string(variable),

                               color_continuous_scale="Viridis",

                               hover_data=[format_string(variable)],

                               range_color=(var_min,var_max),



                                hover_name="country", 

                               #animation_frame="dt",

                              )

    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0},

                      #height=500,

                      #width=700,

                      #coloraxis_showscale=False

                     )

    fig.show()
df_joined = (covid_last

             .merge(mobility,on='country',how='left')

             .assign(mortality_rate=lambda x: x.Deaths/x.population)

             .assign(confirmed_rate=lambda x: x.Confirmed/x.population)

            )
covid_spread = (covid_19

 .query('Confirmed>10').groupby(['country','Date']).agg({'Confirmed':'sum','Deaths':'sum','last_3_days_new_cases':'sum','new_deaths':'sum','last_10_days_new_cases':'sum'})

 .reset_index()

 .merge(population[['country','population']],on='country')

 #.merge(mobility,on='country',how='left')

 .merge(first_day,on='country')

 .assign(confirmed_rate=lambda x: x.Confirmed/x.population*1000000)

 .assign(mortality_rate=lambda x: x.Deaths/x.population*1000000)

 .assign(fatality_rate=lambda x: x.Deaths/x.Confirmed)

 .assign(new_cases_per_cap=lambda x: x.last_3_days_new_cases/x.population*1000000)

 .assign(new_cases_growth=lambda x: x.last_3_days_new_cases/x.last_10_days_new_cases-1)

 .assign(new_deaths_per_cap=lambda x: x.new_deaths/x.population*1000000)

 .assign(day_nr = lambda x: (x.Date-x.first_date).apply(lambda y: y.days))

    

)


colors = sns.diverging_palette(145, 280, s=85, l=25, n=10).as_hex()

sns.palplot(sns.diverging_palette(145, 280, s=85, l=25, n=10))
colors_g = sns.cubehelix_palette(10, start=.5, rot=-.75).as_hex()

#sns.palplot(sns.cubehelix_palette(10, start=.5, rot=-.75))
top_countries  = (covid_last.query('population>1000000')[['confirmed_rate','country','population']]

                  .merge(mobility[['country']], on='country')

                  .assign(filter=lambda x: x.population*x.confirmed_rate)

                  .sort_values(by='filter').tail(40))

countries = list(top_countries.country)
metric='confirmed_rate'

feature = 'retail_and_recreation'

confirmes_spread = (covid_spread

                     .groupby(['country','day_nr']).agg({metric:'sum'})

                     .unstack('country')[metric]

                     .reset_index()

                    )



df_colors = (mobility[['country',feature]]

            .assign(x_bin = lambda x: pd.qcut(x.retail_and_recreation, 10, labels=False))

             .assign(color = lambda x: [colors[y] for y in x.x_bin])

            .drop('x_bin',axis=1)

             .merge(top_countries,how='outer',on='country')

             .fillna({'color':'#D8D8D8'})

             .set_index('country')

            )
plot_lines(confirmes_spread,

              title='Evolution of Confirmed cases per Capita',

              subtitle='Color by Mobility change in Retail & Recreation',

               x_title='Day since the first 10 cases confirmed',

               y_title='',

               hoverinfo='name',

                x='day_nr',

                y=countries,

                y_text=None,

                textposition=[None]*40,

                labelposition=['middle']*40,

                label_size=5,

                

                modes=['lines']*40,

                colors=list(df_colors.loc[countries,'color']),

                fill=[None]*40,

                line_width=[1]*40,

                line_dash=['solid']*40,

                height=None,

                width=None,

                legend=dict(x=1, y=1),

                title_position=dict(x=0, y=1.1),

                font_family='Montserrat',

                y_range=None,

                x_autorange=None,

                margin=dict(

                            autoexpand=False,

                            l=100,

                            r=50,

                            t=100,

                        ),

               additional_traces=[],

               additional_annotations=[],

               comment=''

                  )
font_family='Montserrat'

annotations=[]

annotations.append(dict(xref='x', yref='paper', x=0, y=0.3,

                              xanchor='left', yanchor='middle',

                              text="Mobility Decreased<br>MORE than Average",

                              font=dict(family=font_family,

                                        size=14,

                                        color=colors[2]

                                        ),

                              showarrow=False))

annotations.append(dict(xref='x', yref='paper', x=0, y=0.8,

                              xanchor='left', yanchor='middle',

                              text="Mobility Decreased<br>LESS than Average",

                              font=dict(family=font_family,

                                        size=14,

                                        color=colors[-3]

                                        ),

                              showarrow=False))
metric='confirmed_rate'

feature = 'retail_and_recreation'

confirmes_spread = (covid_spread

                     .merge(mobility[['country',feature]],on='country',how='left')

                     .fillna({feature:1})

                     .assign(metric=lambda x: x[metric]*np.sign(x[feature]-x[feature].median()))

                     .groupby(['country','day_nr']).agg({'metric':'sum'})

                     .unstack('country')['metric']

                     .reset_index()

                    )



df_colors = (mobility[['country',feature]]

            .assign(x_bin = lambda x: pd.qcut(x.retail_and_recreation, 10, labels=False))

             .assign(color = lambda x: [colors[y] for y in x.x_bin])

            .drop('x_bin',axis=1)

             .merge(top_countries,how='outer',on='country')

             .fillna({'color':'#D8D8D8'})

             .set_index('country')

            )
plot_lines(confirmes_spread,

              title='Evolution of Confirmed cases per Capita',

              subtitle='Color by Mobility change in Retail & Recreation',

               x_title='Day since the first 10 cases confirmed',

               y_title='',

               hoverinfo='name',

                x='day_nr',

                y=countries,

                y_text=None,

                textposition=[None]*40,

                labelposition=['middle']*40,

                label_size=5,

                

                modes=['lines']*40,

                colors=list(df_colors.loc[countries,'color']),

                fill=[None]*40,

                line_width=[1]*40,

                line_dash=['solid']*40,

                height=None,

                width=None,

                legend=dict(x=1, y=1),

                title_position=dict(x=0, y=1.1),

                font_family='Montserrat',

                y_range=[-2600,2000],

                x_autorange=None,

                margin=dict(

                            autoexpand=False,

                            l=100,

                            r=50,

                            t=100,

                        ),

               additional_traces=[],

               additional_annotations=annotations,

               comment=''

                  )
metric='new_cases_per_cap'

feature = 'retail_and_recreation'

confirmes_spread = (covid_spread

                     .merge(mobility[['country',feature]],on='country',how='left')

                     .fillna({feature:1})

                     .assign(metric=lambda x: x[metric]*np.sign(x[feature]-x[feature].median()))

                     .groupby(['country','day_nr']).agg({'metric':'sum'})

                     .unstack('country')['metric']

                     .reset_index()

                    )

df_colors = (mobility[['country',feature]]

            .assign(x_bin = lambda x: pd.qcut(x[feature], 10, labels=False))

             .assign(color = lambda x: [colors[int(y)] if y>=0 else '#D8D8D8'  for y in x.x_bin])

            .drop('x_bin',axis=1)

             .merge(top_countries,how='outer',on='country')

             .fillna({'color':'#D8D8D8'})

             .set_index('country')

            )
font_family='Montserrat'

annotations=[]

annotations.append(dict(xref='x', yref='paper', x=0, y=0.3,

                              xanchor='left', yanchor='middle',

                              text="Mobility Decreased<br>MORE than Average",

                              font=dict(family=font_family,

                                        size=14,

                                        color=colors[2]

                                        ),

                              showarrow=False))

annotations.append(dict(xref='x', yref='paper', x=0, y=0.8,

                              xanchor='left', yanchor='middle',

                              text="Mobility Decreased<br>LESS than Average",

                              font=dict(family=font_family,

                                        size=14,

                                        color=colors[-3]

                                        ),

                              showarrow=False))

plot_lines(confirmes_spread,

              title='Evolution of New Cases per Capita',

              subtitle='Color by Mobility change in Retail & Recreation',

               x_title='Day since the first 10 cases confirmed',

               y_title='',

               hoverinfo='name',

                x='day_nr',

                y=countries,

                y_text=None,

                textposition=[None]*40,

                labelposition=['middle']*40,

                label_size=5,

                

                modes=['lines']*40,

                colors=list(df_colors.loc[countries,'color']),

                fill=[None]*40,

                line_width=[1]*40,

                line_dash=['solid']*40,

                height=None,

                width=None,

                legend=dict(x=1, y=1),

                title_position=dict(x=0, y=1.1),

                font_family='Montserrat',

                y_range=[-180,150],

                x_autorange=None,

                margin=dict(

                            autoexpand=False,

                            l=100,

                            r=50,

                            t=100,

                        ),

               additional_traces=[],

               additional_annotations=annotations,

               comment=''

                  )
def plot_evolution(country_i, metric='new_cases_per_cap',feature = 'retail_and_recreation'):



    confirmes_spread = (covid_spread

                         .merge(mobility[['country',feature]],on='country',how='left')

                         .fillna({feature:1})

                         .assign(metric=lambda x: x[metric]*np.sign(x[feature]-x[feature].median()))

                         .groupby(['country','day_nr']).agg({'metric':'sum'})

                         .unstack('country')['metric']

                         .reset_index()

                        )

    df_colors = (mobility[['country',feature]]

                .assign(x_bin = lambda x: pd.qcut(x[feature], 10, labels=False))

                 .assign(color = lambda x: [colors[int(y)] if y>=0 else '#D8D8D8'  for y in x.x_bin])

                .drop('x_bin',axis=1)

                 .merge(top_countries[['country']].append(pd.DataFrame({'country':country_i},index=[0])).drop_duplicates(),how='outer',on='country')

                 .fillna({'color':'#D8D8D8'})

                 .assign(color=lambda x: np.where(x.country==country_i,'#FD0488',x.color))

                 .assign(line_width=lambda x: np.where(x.country==country_i,3,1))

                 .set_index('country')

                )



    countries_i=list(set(countries+[country_i]))



    font_family='Montserrat'

    annotations=[]

    annotations.append(dict(xref='x', yref='paper', x=0, y=0.3,

                                  xanchor='left', yanchor='middle',

                                  text="Mobility Decreased<br>MORE than Average",

                                  font=dict(family=font_family,

                                            size=14,

                                            color=colors[2]

                                            ),

                                  showarrow=False))

    annotations.append(dict(xref='x', yref='paper', x=0, y=0.8,

                                  xanchor='left', yanchor='middle',

                                  text="Mobility Decreased<br>LESS than Average",

                                  font=dict(family=font_family,

                                            size=14,

                                            color=colors[-3]

                                            ),

                                  showarrow=False))

    plot_lines(confirmes_spread,

                  title='Evolution of New Cases per Capita',

                  subtitle='Color by Mobility change in Retail & Recreation',

                   x_title='Day since the first 10 cases confirmed',

                   y_title='',

                   hoverinfo='name',

                    x='day_nr',

                    y=countries_i,

                    y_text=None,

                    textposition=[None]*41,

                    labelposition=['middle']*41,

                    label_size=5,



                    modes=['lines']*41,

                    opacity=[1]*41,

                    colors=df_colors.loc[countries_i,'color'],

                    fill=[None]*41,

                    line_width= df_colors.loc[countries_i,'line_width'],

                    line_dash=['solid']*41,

                    height=None,

                    width=None,

                    legend=dict(x=1, y=1),

                    title_position=dict(x=0, y=1.1),

                    font_family='Montserrat',

                    y_range=[-180,150],

                    x_autorange=None,

                    margin=dict(

                                autoexpand=False,

                                l=100,

                                r=50,

                                t=100,

                            ),

                   additional_traces=[],

                   additional_annotations=annotations,

                   comment=''

                      )


def plot_ribbon(country_i,p=0.05,metrics_of_interest=['new_cases_per_cap'], name_low='Lower Bound',name_high='Upper Bound',name_median='Median'):

    least_affected_countries  = (covid_last.query('population>1000000')[['confirmed_rate','country','population']]

                  .merge(mobility[['country']], on='country')

                  .sort_values(by='confirmed_rate').head(30))

    affected_countries  = (covid_last.query('population>1000000')[['confirmed_rate','country','population']]

                  .merge(mobility[['country']], on='country')

                  .sort_values(by='confirmed_rate').tail(50))

    

    location_names=(pd.DataFrame({

        'retail_and_recreation':'Retail &<br>Recreation', 

        'grocery_and_pharmacy': 'Grocery &<br>Pharmacy', 

        'parks':'Parks',

        'transit_stations': 'Transit<br>Stations', 

        'residential':'Residential', 

        'workplaces':'Workplaces'

        },

        index=[0]

        )

        .T.reset_index()

        .rename(columns={0:'location'})

        )

    mobility_i=mobility.loc[~mobility.country.isin(least_affected_countries.country)]

    p_low=(mobility_i

           [['retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','residential','workplaces']]

           .quantile(p).to_frame()

           .rename(columns={p:name_low})

          )

    p_median=(mobility_i

            [['retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','residential','workplaces']]

            .quantile((0.5)).to_frame()

            .rename(columns={0.5:name_median})

           )

    p_high=(mobility_i

            [['retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','residential','workplaces']]

            .quantile((1-p)).to_frame()

            .rename(columns={1-p:name_high})

           )

    mob_country=(mobility

                 .query(f'country=="{country_i}"')

                 .set_index('country')

                 [['retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','residential','workplaces']]

                 .T

                )

    df_locations = (p_low

                    .join(p_median)

                     .join(p_high)

                     .join(mob_country)

                     .reset_index()

                     .merge(location_names,on='index')

                    .assign(country_text=lambda x: ['{:.0%}'.format(i) for i in x[country_i]])

                )

    covid_last_pc = covid_last.set_index('country')[metrics_of_interest]

    covid_last_i=covid_last_pc.loc[set(list(affected_countries.country)+[country_i])]

    #covid_last_i = (covid_last_pc-covid_last_i.mean())/(covid_last_i.max()/2)

    covid_last_i = (np.log(covid_last_i+1)-np.log(covid_last_i+1).min())/(np.log(covid_last_i+1).max()-np.log(covid_last_i+1).min())

    case_names=(pd.DataFrame({

        'confirmed_rate':'Confirmed<br>Rate', 

        'mortality_rate': 'Mortality<br>Rate', 

        'new_cases_per_cap':'New Cases<br>per 1M Population',

        'fatality_rate': 'Fatality<br>Rate', 

        },

        index=[0]

        )

        .T.reset_index()

        .rename(columns={0:'location'})

        )

    c_low=(covid_last_i

           .quantile(p).to_frame()

           .rename(columns={p:name_low})

          )

    c_median=(covid_last_i

            .quantile((0.5)).to_frame()

            .rename(columns={0.5:name_median})

           )

    c_high=(covid_last_i

            .quantile((1-p)).to_frame()

            .rename(columns={1-p:name_high})

           )

    c_country=(covid_last_i.loc[country_i]

                 .T.to_frame()

                )

    c_text=(covid_last_pc.loc[country_i].to_frame()

     .assign(country_text=lambda x: ['{:.0f}'.format(x[country_i].iloc[0]),

                                     #'{:.0f}'.format(x[country_i].iloc[1]),

                                     #'{:.0f}'.format(x[country_i].iloc[2]),

                                     #'{:.1%}'.format(x[country_i].iloc[3]),

                                     ])

     .drop(country_i,axis=1)

     .reset_index()

    )

    df_covid_cases = (c_low

                    .join(c_median)

                     .join(c_high)

                     .join(c_country)

                     .reset_index()

                     .merge(case_names,on='index')

                     .merge(c_text,on='index')

                 )

    rib_color = colors[int(len(colors)/2)-1]

    rib_med_color = colors[0]

    df_all = (df_covid_cases.append(df_locations).assign(country_explanation = lambda x: 

                                    np.where(x[country_i]>x[name_high],'People visit<br>'+x.location+'<br> More then Elsewhere<br>',

                                     np.where(x[country_i]<x[name_low],'People visit<br>'+x.location+'<br> Less then Elsewhere<br>', ''

                                         ))+x.country_text

                                         )

                      .assign(label_position=lambda x: np.where(x[country_i]<x[name_low],'bottom center','top center'))

                     )

    plot_lines(df_all,

                  title='Change of Mobility in '+country_i,

                  subtitle='Robbon drown by Top 40 big and most Covid-19 affected coutries in the world',

                   x_title='',

                   y_title='',

                   hoverinfo='name',

                    x='location',

                    y=[name_low,name_high,name_median,country_i],

                    y_text=[name_low,'location','location','country_explanation'],

                    textposition=[None,None,'bottom center',list(df_all.label_position)],

                    labelposition=['top','middle','top','middle'],

                    label_size=12,

                    modes=['lines','lines','lines+text','lines+markers+text'],

                    colors=[rib_color,rib_color,rib_med_color,'#FD0488'],

                    fill=[None,'tonexty',None,None],

                    line_width=[0,0,1,0.3],

                    line_dash=['solid','solid','dash','dot'],

                    height=None,

                    width=None,

                    legend=dict(x=1, y=1),

                    title_position=dict(x=0, y=1.1),

                    font_family='Montserrat',

                    y_range=[-1.6,1],

                    x_autorange=None,

                    label_right=False,

                    margin=dict(

                                autoexpand=False,

                                #b=100,

                                l=100,

                                r=50,

                                t=120,

                            ),

                   additional_traces=[],

                   additional_annotations=[],

                   show_x_axis=False,

                   comment=''

                      )
def plot_rich_ribbon(country_i,metric_of_interest='new_cases_growth',countries=countries):

    country_color='#FD0488'

    #country_color='#FFA701'

    countries_i=countries.copy()

    if country_i in countries_i:

        countries_i.remove(country_i)

    countries_i=countries_i+[country_i] 

    location_names=(pd.DataFrame({

        'retail_and_recreation':'Retail &<br>Recreation', 

        'grocery_and_pharmacy': 'Grocery &<br>Pharmacy', 

        'parks':'Parks',

        'transit_stations': 'Transit<br>Stations', 

        'residential':'Residential', 

        'workplaces':'Workplaces'

        },

        index=[0]

        )

        .T.reset_index()

        .rename(columns={0:'location'})

        )

    mobility_i=mobility.loc[mobility.country.isin(countries_i)]

    mob_all = mobility_i.set_index('country')[['retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','residential','workplaces']].T

    mob_country=(mobility

                 .query(f'country=="{country_i}"')

                 .set_index('country')

                 [['retail_and_recreation','grocery_and_pharmacy','parks','transit_stations','residential','workplaces']]

                 .T

                )

    df_locations = (mob_all

                     #.join(mob_country)

                     .reset_index()

                     .merge(location_names,on='index')

                    .assign(country_text=lambda x: ['{:.0%}'.format(i) for i in x[country_i]])

                )



    covid_last_pc = covid_last.set_index('country')[[metric_of_interest]]

    covid_last_i=covid_last_pc.loc[countries_i]

    #covid_last_i = (covid_last_pc-covid_last_i.mean())/(covid_last_i.max()/2)

    #covid_last_i = ((np.log(covid_last_i+1)-np.log(covid_last_i+1).min())/(np.log(covid_last_i+1).max()-np.log(covid_last_i+1).min()))

    #covid_last_i = ((covid_last_i-covid_last_i.min())/(covid_last_i.max()-covid_last_i.min()))

    case_names=(pd.DataFrame({

        'confirmed_rate':'Confirmed<br>Rate', 

        'mortality_rate': 'Mortality<br>Rate', 

        'new_cases_per_cap':'New Cases<br>per 1M Population',

        'new_cases_growth':'New Cases<br>Growth',

        'fatality_rate': 'Fatality<br>Rate', 

        },

        index=[0]

        )

        .T.reset_index()

        .rename(columns={0:'location'})

        )

    c_country=(covid_last_i.loc[country_i]

                 .T.to_frame()

                )

    c_text=(covid_last_pc.loc[country_i].to_frame()

     .assign(country_text=lambda x: ['{:.0%}'.format(x[country_i].iloc[0]),

                                     #'{:.0f}'.format(x[country_i].iloc[1]),

                                     #'{:.0f}'.format(x[country_i].iloc[2]),

                                     #'{:.1%}'.format(x[country_i].iloc[3]),

                                     ])

     .drop(country_i,axis=1)

     .reset_index()

    )

    df_covid_cases = (covid_last_i.T

                         .reset_index()

                         .merge(case_names,on='index')

                         .merge(c_text,on='index')

                     )

    df_colors = (covid_last_i

         .assign(x_bin = lambda x: pd.qcut(x[metric_of_interest], 10, labels=False))

         .assign(color = lambda x: [colors_g[int(y)] if y>=0 else '#D8D8D8'  for y in x.x_bin])

                        .drop('x_bin',axis=1)

         .assign(color=lambda x: np.where(x.index==country_i,country_color,x.color))

         .assign(line_width=lambda x: np.where(x.index==country_i,3,1))

         .assign(opacity=lambda x: np.where(x.index==country_i,1,0.5))

         .assign(modes=lambda x: np.where(x.index==country_i,'lines+text','lines'))



        )

    df_all = (df_covid_cases[df_locations.columns]

              .append(df_locations)

              .assign(text=lambda x: x.location+'<br><br>'+x.country_text)

             )

 

    plot_lines(df_all,

                      title='Change of Mobility in '+country_i,

                      subtitle='Top 40 big and most Covid-19 affected coutries in the world',

                       x_title='',

                       y_title='',

                       hoverinfo='name',

                        x='location',

                        y=countries_i,

                        y_text=['text']*41,

                        textposition=[None]*42,

                        labelposition=['middle']*42,

                        label_size=12,

                        modes=df_colors.loc[countries_i,'modes'],

                        colors=df_colors.loc[countries_i,'color'],

                        fill=[None]*42,

                        line_width=df_colors.loc[countries_i,'line_width'],

                        line_dash=['solid']*42,

                        height=None,

                        width=None,

                        legend=dict(x=1, y=1),

                        title_position=dict(x=0, y=1.1),

                        font_family='Montserrat',

                        y_range=[-1.6,1],

                        x_autorange=None,

                        label_right=False,

                        opacity=df_colors.loc[countries_i,'opacity'],

                        margin=dict(

                                    autoexpand=False,

                                    #b=100,

                                    l=100,

                                    r=50,

                                    t=120,

                                ),

                       additional_traces=[],

                       additional_annotations=[],

                       show_x_axis=False,

                       comment=''

                          )
country_i='Lithuania'

plot_evolution(country_i, metric='new_cases_per_cap',feature = 'retail_and_recreation')

plot_rich_ribbon(country_i,metric_of_interest='new_cases_growth')
country_i='Italy'

plot_evolution(country_i, metric='new_cases_per_cap',feature = 'retail_and_recreation')

plot_rich_ribbon(country_i,metric_of_interest='new_cases_growth')
country_i='France'

plot_evolution(country_i, metric='new_cases_per_cap',feature = 'retail_and_recreation')

plot_rich_ribbon(country_i,metric_of_interest='new_cases_growth')
country_i='Sweden'

plot_evolution(country_i, metric='new_cases_per_cap',feature = 'retail_and_recreation')

plot_rich_ribbon(country_i,metric_of_interest='new_cases_growth')
country_i='United States'

plot_evolution(country_i, metric='new_cases_per_cap',feature = 'retail_and_recreation')

plot_rich_ribbon(country_i,metric_of_interest='new_cases_growth')
country_i='Austria'

plot_evolution(country_i, metric='new_cases_per_cap',feature = 'retail_and_recreation')

plot_rich_ribbon(country_i,metric_of_interest='new_cases_growth')