import pandas as pd

import numpy as np

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode()

import re

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')
def is_male(code):

    str_list = code.split('.')

    return ('MA' in str_list)

    

def is_female(code):

    str_list = code.split('.')

    return ('FE' in str_list)



#returns string with '.MA' or '.FE' deleted

def get_base_code(code):

    str_list = code.split('.')

    if 'MA' in str_list:

        str_list.remove('MA')

    if 'FE' in str_list:

        str_list.remove('FE')

    return '.'.join(str_list)



#uses regular expression operations to remove 'male' or 'males' from indicator names

def get_base_name(name):

    return re.sub('[, ]*male[, s]*', ' ', name,)
path = "../input/Indicators.csv"

indicators = pd.read_csv(path)
indicators.columns
#dataframe of male indicators

male_data = indicators[indicators.IndicatorCode.apply(is_male)]

male_data.columns = ['CountryName', 'CountryCode', 'MaleIndicatorName', 

                     'MaleIndicatorCode', 'Year', 'MaleValue']

male_data = male_data.assign(IndicatorBaseCode = male_data.MaleIndicatorCode.apply(get_base_code))



#dataframe of female indicators

female_data = indicators[indicators.IndicatorCode.apply(is_female)]

female_data.columns = ['CountryName', 'CountryCode', 'FemaleIndicatorName', 

                       'FemaleIndicatorCode', 'Year', 'FemaleValue']

female_data = female_data.assign(IndicatorBaseCode = female_data.FemaleIndicatorCode.apply(get_base_code))



#merge these two

gender_data = pd.merge(male_data, female_data, how='inner', 

                       on=['CountryName','CountryCode','Year','IndicatorBaseCode'])

gender_data['Inequality'] = (gender_data.FemaleValue - gender_data.MaleValue)/(gender_data.FemaleValue + gender_data.MaleValue)

gender_data = gender_data.assign(IndicatorBaseName = gender_data.MaleIndicatorName.apply(get_base_name))



#remove redundant information

gender_data.drop(['MaleIndicatorName', 'MaleIndicatorCode','FemaleIndicatorName',

                  'FemaleIndicatorCode'], axis=1, inplace=True)
indicator_groups = gender_data.groupby('IndicatorBaseName')

indicator_groups.size().sort_values(ascending=False)
def make_mf_year_plot(indicator_name, country):

    df = gender_data[(gender_data.IndicatorBaseName==indicator_name) & \

                     (gender_data.CountryName==country)]

    graph = df[['Year','MaleValue','FemaleValue']].plot(x='Year',

                                                        title = indicator_name + ' in ' + country, 

                                                        style = ['b','r'])

    plt.show()

    return

    

def make_ineq_year_plot(indicator_name, country):

    df = gender_data[(gender_data.IndicatorBaseName==indicator_name) & \

                     (gender_data.CountryName==country)]

    graph = df.plot(x='Year',y='Inequality')                                               

    plt.show()

    return



def make_map(indicator_name, year=None):

    

    #if a year is specified, plot data from that year, otherwise plot the average values over all years

    if year:

        filtered = gender_data[(gender_data.IndicatorBaseName==indicator_name) & \

                               (gender_data.Year==year)]

        

        #if there is no data from that year, exit

        if filtered.size == 0:

            print('No data for ' + indicator_name + ' in ' + str(year))

            return

        

        title = indicator_name + ' in ' + str(year)

        

        

    else:

        indicator_data = gender_data[(gender_data.IndicatorBaseName == indicator_name)].groupby(['CountryCode','CountryName'],

                                                                                                as_index=False)

        filtered=indicator_data.agg(np.mean)

        title = 'Average ' + indicator_name

    

    scl = [[-1.0, 'rgb(242,240,247)'],[-0.6, 'rgb(218,218,235)'],[-0.2, 'rgb(188,189,220)'],\

                [0.2, 'rgb(158,154,200)'],[0.6, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



    #max_val = abs(filtered.Inequality.values).max()



    data = [ dict(

            type='choropleth',

            colorscale = scl,

            autocolorscale = False,

            locations = filtered.CountryCode.values,

            z = filtered.Inequality.values,

            zmax = 1.0, #max_val,

            zmin = -1.0, #-1.0*max_val,

            text = filtered.CountryName + '<br>' + 

                    'Male Value: ' + filtered.MaleValue.apply('{:.2f}'.format) + '<br>' +

                    'Female Value: ' + filtered.FemaleValue.apply('{:.2f}'.format) +'<br>' +

                    'Relative Difference: ' + filtered.Inequality.apply('{:.2f}'.format) ,

            hoverinfo = 'text',

            marker = dict(

                line = dict(

                    color = 'rgb(255,255,255)',

                    width = 1

                )),

            colorbar = dict(

                title = "Inequality")

            ) ]



    layout = dict(

            title = title,

            geo = dict(

                scope='world',

                projection=dict(type='Mercator'),

                showlakes = True,

                lakecolor = 'rgb(255, 255, 255)'),

                 )



    fig = dict( data=data, layout=layout )

    iplot(fig, filename='d3-cloropleth-map')

    return
indicator_name = 'Prevalence of HIV (% ages 15-24)'

make_map(indicator_name)
make_mf_year_plot(indicator_name,'Mexico')

make_mf_year_plot(indicator_name,'South Africa')