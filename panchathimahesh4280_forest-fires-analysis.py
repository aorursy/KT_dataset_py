#first let's import all necessery libraries for this analysis

import numpy as np

import pandas as pd

import plotly.graph_objects as go

import plotly.express as px
#using pandas library and 'read_csv' function to read amazon csv file as file already formated for us from Kaggle

amazon_df=pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

#examining head of the dataset

amazon_df.head(5)
amazon_df.shape
#checking if there are any nulls we are dealing with (missing data)

amazon_df.isna().sum()
#cheking unique values in the state column

amazon_df.state.unique()
#checking unique values in the month column

amazon_df.month.unique()
#creating a dictionary with translations of months

month_map={'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

          'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

          'Novembro': 'November', 'Dezembro': 'December'}

#mapping our translated months

amazon_df['month']=amazon_df['month'].map(month_map)

#checking the month column for the second time after the changes were made

amazon_df.month.unique()
#cheking the numeric percentile distribution for the fires reported

amazon_df.number.describe()
#chekcing how many fires were reported in 20 years 

amazon_df.number.sum()
#we are already given the year column, however for good practice we can also extract it from the date one

amazon_df['Year']=pd.DatetimeIndex(amazon_df['date']).year

#cheking unique years in new created column 

amazon_df.Year.unique()
#we are not going to be using old year column and date column as they serve no significant purpose anymore 

amazon_df.drop(columns=['date', 'year'], axis=1, inplace=True)

#changing order of columns for preffered format

amazon_df=amazon_df[['state','number','month','Year']]

#changing names of columns for preffered format

amazon_df.rename(columns={'state': 'State', 'number': 'Fire_Number', 'month': 'Month'}, inplace=True)

#checking changes made

amazon_df.head()
#creating a list of years we have 

years=list(amazon_df.Year.unique())

#creating an empty list, which will be populated later with amount of fires reported

sub_fires_per_year=[]

#using for loop to extract sum of fires reported for each year and append list above

for i in years:

    y=amazon_df.loc[amazon_df['Year']==i].Fire_Number.sum().round(0)

    sub_fires_per_year.append(y)

#creating a dictionary with results     

fire_year_dic={'Year':years,'Total_Fires':sub_fires_per_year}

#creating a new sub dataframe for later plot 

time_plot_1_df=pd.DataFrame(fire_year_dic)

#checking the dataframe

time_plot_1_df.head(5)
#using plotly Scatter 

time_plot_1=go.Figure(go.Scatter(x=time_plot_1_df.Year, y=time_plot_1_df.Total_Fires,

                                 mode='lines+markers', line={'color': 'red'}))

#layout changes

time_plot_1.update_layout(title='Brazil Fires per 1998-2017 Years',

                   xaxis_title='Year',

                   yaxis_title='Fires')

#showing the figure

time_plot_1.show()




#putting all available states in the list

states=list(amazon_df.State.unique())

#creating empty list for each state that will be later appended

acre_list=[]

alagoas_list=[] 

amapa_list=[] 

amazonas_list=[] 

bahia_list=[] 

ceara_list=[]

distrito_list=[] 

espirito_list=[] 

goias_list=[] 

maranhao_list=[] 

mato_list=[] 

minas_list=[]

para_list=[] 

paraiba_list=[] 

perna_list=[]

piau_list=[]

rio_list=[]

rondonia_list=[]

roraima_list=[]

santa_list=[]

sao_list=[]

sergipe_list=[]

tocantins_list=[]
#It get's interesting here



#breaking down fires reported for each state throughtout 20 years and appending empty lists

for x in states:

    st=x

    for i in years:

        ye=i

        if st=='Acre':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            acre_list.append(y)

        elif st=='Alagoas':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            alagoas_list.append(y)

        elif st=='Amazonas':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            amazonas_list.append(y)

        elif st=='Amapa':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            amapa_list.append(y)

        elif st=='Bahia':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            bahia_list.append(y)

        elif st=='Ceara':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            ceara_list.append(y)

        elif st=='Distrito Federal':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            distrito_list.append(y)

        elif st=='Espirito Santo':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            espirito_list.append(y)

        elif st=='Goias':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            goias_list.append(y)

        elif st=='Maranhao':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            maranhao_list.append(y)

        elif st=='Mato Grosso':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            mato_list.append(y)

        elif st=='Minas Gerais':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            minas_list.append(y)

        elif st=='Pará':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            para_list.append(y)

        elif st=='Paraiba':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            paraiba_list.append(y)

        elif st=='Pernambuco':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            perna_list.append(y)

        elif st=='Piau':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            piau_list.append(y)

        elif st=='Rio':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            rio_list.append(y)

        elif st=='Rondonia':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            rondonia_list.append(y)

        elif st=='Roraima':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            roraima_list.append(y)

        elif st=='Santa Catarina':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            santa_list.append(y)

        elif st=='Sao Paulo':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            sao_list.append(y)

        elif st=='Sergipe':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            sergipe_list.append(y)

        elif st=='Tocantins':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            tocantins_list.append(y)
#with those lists populated, now creating a powerful dataframe

time_plot_2_df=pd.DataFrame(list(zip(years, acre_list, alagoas_list, amapa_list, amazonas_list,

                                     bahia_list, ceara_list, distrito_list, espirito_list,

                                     goias_list, maranhao_list, mato_list, minas_list, para_list,

                                     paraiba_list, perna_list, piau_list, rio_list, rondonia_list,

                                     roraima_list, santa_list, sao_list, sergipe_list, tocantins_list)),

                            columns =['Year', 'Acre', 'Alagoas', 'Amapa', 'Amazonas', 'Bahia', 'Ceara',

                                      'Distrito Federal', 'Espirito Santo', 'Goias', 'Maranhao',

                                      'Mato Grosso', 'Minas Gerais', 'Pará', 'Paraiba', 'Pernambuco',

                                      'Piau', 'Rio', 'Rondonia', 'Roraima', 'Santa Catarina',

                                      'Sao Paulo', 'Sergipe', 'Tocantins'])

#checking the dataframe

time_plot_2_df.head(10)
#examining top 10 states with the most fires reported (please igone the year observation, will be removed later)

time_plot_2_df.sum().nlargest(11)
#creating a dataframe for bar plot visualization

bar_plot_df=pd.DataFrame(time_plot_2_df.sum().nlargest(11))

#reseting index for first column

bar_plot_df=bar_plot_df.reset_index()

#renaming

bar_plot_df.rename(columns={'index':'State', 0:'Reported_Fires'}, inplace=True)

#removing Year observation

bar_plot_df.drop(bar_plot_df[bar_plot_df.State == 'Year'].index, inplace=True)

#checking dataframe

bar_plot_df
#making barplot

bar_plot=px.bar(bar_plot_df, x='State', y='Reported_Fires', color='Reported_Fires',

           labels={'Reported_Fires':'Count of reported fires ', 'State':'States'}, color_continuous_scale='Reds')

#making layout changes

bar_plot.update_layout(xaxis_tickangle=-45, title_text='Top 10 States for Amount of Reported Fires per 1998-2017 Years')

#outputing plot

bar_plot.show()
#preparing a figure that will be populated 

time_plot_2 = go.Figure()

#adding individual graphs to the figure

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Mato Grosso'],

                                 mode='lines+markers', name='Mato Grosso', line={'color': 'red'}))

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Paraiba'],

                                 mode='lines+markers', name='Paraiba', line={'color': 'yellow'}))

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Sao Paulo'],

                                 mode='lines+markers', name='Sao Paulo', line={'color': 'green'}))

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Rio'],

                                 mode='lines+markers', name='Rio', line={'color': 'blue'}))

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Bahia'],

                                 mode='lines+markers', name='Bahia', line={'color': 'pink'}))

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Piau'],

                                 mode='lines+markers', name='Piau', line={'color': 'brown'}))

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Goias'],

                                 mode='lines+markers', name='Goias', line={'color': 'grey'}))

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Minas Gerais'],

                                 mode='lines+markers', name='Minas Gerais', line={'color': 'purple'}))

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Tocantins'],

                                 mode='lines+markers', name='Tocantins', line={'color': 'orange'}))

time_plot_2.add_trace(go.Scatter(x=time_plot_2_df.Year, y=time_plot_2_df['Amazonas'],

                                 mode='lines+markers', name='Amazonas', line={'color': 'gold'}))

#making changes to layout

time_plot_2.update_layout(title='Brazil Fires in Top-10 (frequent) regions per 1998-2017 Years',

                   xaxis_title='Year',

                   yaxis_title='Fires')

#outputing plot

time_plot_2.show()
#creating subdataframe for visualizing this states geographically

geo_plot_df=pd.DataFrame(time_plot_2_df.sum().nlargest(11))

#formatting new dataframe

geo_plot_df.rename(columns={0:'Count'}, inplace=True)

geo_plot_df.reset_index(inplace=True)

geo_plot_df.rename(columns={'index':'State'}, inplace=True)

geo_plot_df.drop(geo_plot_df.index[5], inplace=True)

#cheking new sub dataframe 

geo_plot_df
#taking my time and adding all coordinates (latitude and longitude) for this top 10 states

lat=[-16.350000, -22.15847, -23.533773, -22.908333, -11.409874, -21.5089, -16.328547,

     -19.841644, -21.175, -3.416843]

long=[-56.666668, -43.29321, -46.625290, -43.196388, -41.280857, -43.3228, -48.953403,

     -43.986511, -43.01778, -65.856064]

#adding new coordinates as columns to subdataframe above

geo_plot_df['Lat']=lat

geo_plot_df['Long']=long

#checking changes in subdataframe for geo visualization

geo_plot_df
#using scatter geo with above created subdataframe

fig = px.scatter_geo(data_frame=geo_plot_df, scope='south america',lat='Lat',lon='Long',

                     size='Count', color='State', projection='hammer')

fig.update_layout(

        title_text = '1998-2017 Top-10 States in Brazil with reported fires')

fig.show()
#according to different sources, months from June - November are the hottes in Brazil



#isolating the hottest months by season

month_array_summer=['June','July','August']

month_array_fall=['September','October','November']

#leaving data only for hottest months

box_plot_df_summer=amazon_df.loc[amazon_df['Month'].isin(month_array_summer)]

box_plot_df_fall=amazon_df.loc[amazon_df['Month'].isin(month_array_fall)]

#visualizing reports

box_plot=go.Figure()



box_plot.add_trace(go.Box(y=box_plot_df_summer.Fire_Number, x=box_plot_df_summer.Month,

                          name='Summer', marker_color='#3D9970',

                          boxpoints='all', jitter=0.5, whiskerwidth=0.2,

                          marker_size=2,line_width=2))

box_plot.add_trace(go.Box(y=box_plot_df_fall.Fire_Number, x=box_plot_df_fall.Month,

                         name='Fall', marker_color='#FF851B',

                         boxpoints='all', jitter=0.5, whiskerwidth=0.2,

                          marker_size=2,line_width=2))



box_plot.update_layout(

        title_text = 'Distribution of Fire Reports from 1998-2017 in the hottest months')

box_plot.show()