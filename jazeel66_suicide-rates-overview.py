import pandas as pd

import numpy as np
!pip install countrycode
import countrycode



from bokeh.plotting import figure, show, output_notebook

from bokeh.layouts import column, row, gridplot

from bokeh.models import Span, Label, ColumnDataSource, FixedTicker, Title

from bokeh.palettes import Pastel1



output_notebook()
df= pd.read_csv('../input/master.csv')



df.head()
# we will convert country name to continent and check the values

df['continent']= countrycode.countrycode.countrycode(codes= df['country'], origin= 'country_name', target= 'continent')



df['continent'].value_counts(dropna= False)
df[df.continent == 'Republic of Korea'].head() # this should be Asia
df.loc[df.continent == 'Republic of Korea', 'continent'] = 'Asia'

df['continent'].value_counts(dropna= False)
print(df.isna().sum()) # there are many missing values in HDI for year, we will drop this feature



df.drop('HDI for year', axis= 1, inplace= True)



'HDI for year' in df.columns # the column is removed
df.rename(index=str, columns= {' gdp_for_year ($) ': 'gdp_for_year',

         'gdp_per_capita ($)': 'gdp_per_capita',

         'country-year': 'country_year'}, inplace= True) # renaming the columns
df['age']= df.age.str.replace(' years','')



df['gdp_for_year']= df.gdp_for_year.str.replace(',','')
df.sample(3)
country_year_count= df.groupby('country')['year'].count()



country_year_count_idx= country_year_count[country_year_count <= 36].index
print(df.shape)

for values in country_year_count_idx:

    

#     print(values)

    df.drop(df[df.country == values].index, axis= 0, inplace= True)

    

print(df.shape)
df= df[df['year'] != 2016] # removing data from 2016

df.shape
df['sex']= np.where(df.sex == 'male', 'Male', 'Female')
df.head()
suicide_per100k= df.groupby('year')['suicides_no'].sum().div(df.groupby('year')['population'].sum()).mul(100000)
p= figure(title='Global Sucides (per 100k)', height= 300, width= 800, x_axis_label= 'Year', y_axis_label= 'Suicides per 100k')



p.circle(x= suicide_per100k.index, y= suicide_per100k)

p.line(x= suicide_per100k.index, y= suicide_per100k)

axhline=  Span(location= suicide_per100k.mean(), dimension= 'width', line_color= 'gray'

               , line_dash='dotdash')



citation = Label(x=2010, y=13.2, text='Global average', render_mode='canvas',

                 background_fill_color='white', background_fill_alpha=1.0)

p.add_layout(axhline)

p.add_layout(citation)



show(p)
continent_avg= df.groupby('continent')['suicides_no'].sum().div(df.groupby('continent')['population'].sum()).mul(100000)



continent_avg.sort_values(inplace= True)



continent_year_avg= df.groupby(['continent', 'year'])['suicides_no'].sum().div(df.groupby(['continent', 'year'])['population'].sum()).mul(100000)



continent_colors= ['#440154', '#3B528B', '#21908C', '#5DC863', '#FDE725']



source= ColumnDataSource(data= dict(continent= list(continent_avg.index)

                                    , suicide_nums= list(continent_avg)

                              , color= continent_colors))



bars_by_continent= figure(title= 'Global suicide(100k), by continent', toolbar_location= None, tools= ""

                          , height= 750, width= 400,x_range= list(continent_avg.index) 

                          ,y_axis_label= 'Suicide per 100k', x_axis_label= 'Continent')



bars_by_continent.vbar(x= 'continent', top= 'suicide_nums', source= source, width= 0.9, color= 'color')



bars_by_continent.yaxis.ticker= FixedTicker(ticks= list(np.arange(0, 19, 2))) # setting fixed ticks for the y-axis





# Creating the second part of the layout



africa_timeline= figure(title= 'Africa', width= 380, height= 150, tools= "", toolbar_location= None, title_location= 'right')

africa_timeline.title.align= 'center'



africa_timeline.line(x= continent_year_avg['Africa'].index, y= continent_year_avg['Africa']

                     , color= '#440154')

africa_timeline.circle(x= continent_year_avg['Africa'].index, y= continent_year_avg['Africa']

                       , color= '#440154')



americas_timeline= figure(title= 'Americas', width= 380, height= 150, tools= "", toolbar_location= None, title_location= 'right')

americas_timeline.title.align= 'center'



americas_timeline.line(x= continent_year_avg['Americas'].index, y= continent_year_avg['Americas']

                     , color= '#3B528B')

americas_timeline.circle(x= continent_year_avg['Americas'].index, y= continent_year_avg['Americas']

                       , color= '#3B528B')



oceania_timeline= figure(title= 'Oceania', width= 380, height= 150, tools= "", toolbar_location= None

                         ,y_axis_label= 'Suicide per 100k', title_location= 'right')

oceania_timeline.title.align= 'center'



oceania_timeline.line(x= continent_year_avg['Oceania'].index, y= continent_year_avg['Oceania']

                     , color= '#21908C')

oceania_timeline.circle(x= continent_year_avg['Oceania'].index, y= continent_year_avg['Oceania']

                       , color= '#21908C')



asia_timeline= figure(title= 'Asia', width= 380, height= 150, tools= "", toolbar_location= None, title_location= 'right')



asia_timeline.title.align= 'center'



asia_timeline.line(x= continent_year_avg['Asia'].index, y= continent_year_avg['Asia']

                     , color= '#5DC863')

asia_timeline.circle(x= continent_year_avg['Asia'].index, y= continent_year_avg['Asia']

                       , color= '#5DC863')



europe_timeline= figure(title= 'Europe', width= 380, height= 150, tools= "", toolbar_location= None,

                       x_axis_label= 'Year', title_location= 'right')

europe_timeline.title.align= 'center'



europe_timeline.line(x= continent_year_avg['Europe'].index, y= continent_year_avg['Europe']

                     , color= '#FDE725')

europe_timeline.circle(x= continent_year_avg['Europe'].index, y= continent_year_avg['Europe']

                       , color= '#FDE725')



cols= column(africa_timeline, americas_timeline, oceania_timeline, asia_timeline, europe_timeline)



africa_timeline.add_layout(Title(text= 'Trends over time, by continent'), 'above')



grid= gridplot([bars_by_continent, cols], ncols= 2)



# show(bars_by_continent)

show(grid)
suicide_count_by_sex= df.groupby('sex')['suicides_no'].sum().div(df.groupby('sex')['population'].sum()).mul(100000)



suicide_count_by_yearandsex= df.groupby(['sex', 'year'])['suicides_no'].sum().div(df.groupby(['sex', 'year'])['population'].sum()).mul(100000)



color= ['#F8766D', '#00BFC4']



source= ColumnDataSource(data= dict(sex= list(suicide_count_by_sex.index)

                                    ,suicide_nums= list(suicide_count_by_sex)

                                    ,color= color))

suicide_count_by_sex_figure= figure(title='Global suicides(per 100k), by Sex', x_axis_label= 'Sex', y_axis_label= 'Suicides per 100k' 

                                    ,width =400, height= 500, x_range= list(suicide_count_by_sex.index), toolbar_location= None, tools= "")



suicide_count_by_sex_figure.vbar(x= 'sex', top= 'suicide_nums', source= source, color= 'color', width= 0.8)



#femail line chart

suicide_female= figure(title= 'Female', width = 500, height= 250, title_location= 'right',tools="", y_axis_label= 'Suicides per 100k')

suicide_female.circle(x= suicide_count_by_yearandsex['Female'].index, y= suicide_count_by_yearandsex['Female'], color= '#F8766D')

suicide_female.line(x= suicide_count_by_yearandsex['Female'].index, y= suicide_count_by_yearandsex['Female'], color= '#F8766D')

suicide_female.title.align = 'center'



#male line chart

suicide_male= figure(title= 'Male', width = 500, height= 250, title_location= 'right', tools="", x_axis_label= 'Year'

                     , y_axis_label= 'Suicides per 100k')

suicide_male.circle(x= suicide_count_by_yearandsex['Male'].index, y= suicide_count_by_yearandsex['Male'], color= '#00BFC4')

suicide_male.line(x= suicide_count_by_yearandsex['Male'].index, y= suicide_count_by_yearandsex['Male'], color= '#00BFC4')

suicide_male.title.align = 'center'

suicide_female.add_layout(Title(text= 'Trends over Time, by Sex'), 'above')



cols= column(suicide_female, suicide_male)

grid= gridplot([suicide_count_by_sex_figure, cols], ncols= 2)



# show(suicide_count_by_sex_figure)

show(grid)
suicide_count_age= df.groupby('age')['suicides_no'].sum().div(df.groupby('age')['population'].sum()).mul(100000).sort_values()

suicide_count_ageandyear= df.groupby(['age', 'year'])['suicides_no'].sum().div(df.groupby(['age', 'year'])['population'].sum()).mul(100000)



suicide_count_age_figure= figure(title= 'Global suicides per 100k, by Age', width= 400, height= 700, x_range= list(suicide_count_age.index)

                                 ,x_axis_label= 'Year', y_axis_label= 'Suicide per 100k', tools= "")



source= ColumnDataSource(data= dict(age= list(suicide_count_age.index), counts= list(suicide_count_age)

                                    , colors= ['#440154', '#3B528B', '#2A788E','#21908C', '#5DC863', '#FDE725']))



suicide_count_age_figure.vbar(x= 'age', top= 'counts', source= source, color= 'colors', width= 0.9)



# show(suicide_count_age_figure)



# creating graphs for ages

#5-14

lessthan14_fig= figure(title= '5-14', title_location= 'right', width= 500, height= 125, tools= "")

lessthan14_fig.title.align= 'center'

lessthan14_fig.circle(x= suicide_count_ageandyear['5-14'].index,y= suicide_count_ageandyear['5-14'], color= '#440154')

lessthan14_fig.line(x= suicide_count_ageandyear['5-14'].index,y= suicide_count_ageandyear['5-14'], color= '#440154')



#15-24



lessthan24_fig= figure(title= '15-24', title_location= 'right', width= 500, height= 100, tools= "")

lessthan24_fig.title.align= 'center'

lessthan24_fig.circle(x= suicide_count_ageandyear['15-24'].index,y= suicide_count_ageandyear['15-24'], color= '#3B528B')

lessthan24_fig.line(x= suicide_count_ageandyear['15-24'].index,y= suicide_count_ageandyear['15-24'], color= '#3B528B')



#25-34



lessthan34_fig= figure(title= '25-34', title_location= 'right', width= 500, height= 100, tools= "")

lessthan34_fig.title.align= 'center'

lessthan34_fig.circle(x= suicide_count_ageandyear['25-34'].index,y= suicide_count_ageandyear['25-34'], color= '#2A788E')

lessthan34_fig.line(x= suicide_count_ageandyear['25-34'].index,y= suicide_count_ageandyear['25-34'], color= '#2A788E')



#35-54



lessthan54_fig= figure(title= '35-54', title_location= 'right', width= 500, height= 100, tools= "", y_axis_label= 'Suicides per 100k')

lessthan54_fig.title.align= 'center'

lessthan54_fig.circle(x= suicide_count_ageandyear['35-54'].index,y= suicide_count_ageandyear['35-54'], color= '#21908C')

lessthan54_fig.line(x= suicide_count_ageandyear['35-54'].index,y= suicide_count_ageandyear['35-54'], color= '#21908C')

lessthan54_fig.add_layout(Title(text='Suicide per 100k'), 'left')

#55-74



lessthan74_fig= figure(title= '55-74', title_location= 'right', width= 500, height= 100, tools= "")

lessthan74_fig.title.align= 'center'

lessthan74_fig.circle(x= suicide_count_ageandyear['55-74'].index,y= suicide_count_ageandyear['55-74'], color= '#5DC863')

lessthan74_fig.line(x= suicide_count_ageandyear['55-74'].index,y= suicide_count_ageandyear['55-74'], color= '#5DC863')



#75+



morethan74_fig= figure(title= '75+', title_location= 'right', width= 500, height= 125, tools= "", x_axis_label= 'Year')

morethan74_fig.title.align= 'center'

morethan74_fig.circle(x= suicide_count_ageandyear['75+'].index,y= suicide_count_ageandyear['75+'], color= '#FDE725')

morethan74_fig.line(x= suicide_count_ageandyear['75+'].index,y= suicide_count_ageandyear['75+'], color= '#FDE725')

lessthan14_fig.add_layout(Title(text= 'Trends over Time, by Age'), 'above')



cols= column(lessthan14_fig, lessthan24_fig, lessthan34_fig, lessthan54_fig, lessthan74_fig, morethan74_fig)



grid= gridplot([suicide_count_age_figure, cols], ncols=2)



show(grid)
