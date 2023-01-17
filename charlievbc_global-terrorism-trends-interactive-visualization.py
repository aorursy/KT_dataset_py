import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import chardet

from collections import Counter

from bokeh.plotting import figure

from bokeh.io import show, output_notebook

from bokeh.layouts import column

from bokeh.models import Select, Slider, CustomJS, HoverTool, ColumnDataSource

from bokeh.palettes import Turbo11 as palette

import itertools



filevar = "../input/gtd/globalterrorismdb_0718dist.csv"

with open(filevar, 'rb') as f:

    encoding_info = chardet.detect(f.read(100000))  

try: 

    df_init = pd.read_csv(filevar, index_col=0, encoding=encoding_info.get('encoding'))

    print('File loading - Success!')

except:

    print('File loading - Failed!')
def createNewDf(df_init):

    """ Returns a dataframe with needed columns """

    df = df_init[['iyear','imonth','country_txt','region_txt',

                  'attacktype1_txt','targtype1_txt','gname',

                  'weaptype1_txt']]

    df['city'] = list(df_init['city'].fillna('Unknown'))

    df['natlty1_txt'] = list(df_init['natlty1_txt'].fillna('Unknown'))

    df['nperps'] = list(df_init['nperps'].fillna(0))

    df['nperpcap'] = list(df_init['nperpcap'].fillna(0))

    df['nkill'] = list(df_init['nkill'].fillna(0))

    df['nwound'] = list(df_init['nwound'].fillna(0))

    return df
# Columns with NaN values

print('city null value:', df_init.city.isnull().sum())

print('nationality null value:', df_init.natlty1_txt.isnull().sum())

print('Perpetrator count null value:', df_init.nperps.isnull().sum(), 

      'Minimum value:', min(df_init.nperps), 

      'Maximum value:', max(df_init.nperps))

print('Perpetrator captured count null value:', df_init.nperpcap.isnull().sum(), 

      'Minimum value:', min(df_init.nperpcap), 

      'Maximum value:', max(df_init.nperpcap))

print('Number of killed null value:', df_init.nkill.isnull().sum(), 

      'Minimum value:', min(df_init.nkill), 

      'Maximum value:', max(df_init.nkill))

print('Number of wounded null value:', df_init.nwound.isnull().sum(), 

      'Minimum value:', min(df_init.nwound), 

      'Maximum value:', max(df_init.nwound))
# Create a new dataframe for this project

df = createNewDf(df_init)
# first five rows of the dataframe

df.head()
def selectFilter(cat):

    """ Input parameter selection: global/ country/ perpetrators

        Returns: selection list, selection initial value, selection title, plot title

    """

    if cat == 'global':   

        f_list = ['Worldwide', 'Region', 'Country']

        f_init = 'Worldwide'

        f_title = 'Selection'

        f_title_plot = ['Basic Attack Statistics - Global', 

                        'Basic Attack Statistics - Per Region', 

                        'Basic Attack Statistics - Per Country']

    elif cat == 'country':

        f_list = sorted(df.country_txt.unique())

        f_init = 'Philippines'

        f_title = 'Country List'  

        f_title_plot = ['Country Attack Trend', 

                        'Country Attack Trend - Frequency per Month (1970-2017)', 

                        'Country Attack Trend - Cities with most attacks']

    else: # cat == 'perpetrators'

        f_list = ['Most_frequent_to_attack',  'With_most_fatalities',

                  'With_most_damages',        'Largest_member_count',

                  'Longest_running_in_years', 'Present_in_most_continents']

        f_init = 'Most_frequent_to_attack'

        f_title = 'Statistics List (per top 10)'

        f_title_plot = 'Top 10 Perpetrator Statistics'

    

    return f_list, f_init, f_title, f_title_plot  
def createTempDf(cat, f_val):

    """ Input parameter: cat = global/ country/ perpetrators; f_val = <subcategory>

        Returns: dictionary with restructured data base on cat & f_val.

    """

    if cat == 'global': 

        if f_val == 'Worldwide':

            count = Counter(df.iyear)

            count_sort = dict(sorted(count.items()))

            temp_df = {

                'x' : list(count_sort.keys()),

                'total' : list(count_sort.values()),

            }

        else: 

            ind = df.iyear.unique()

            col = df.region_txt.unique()

            col_reg = 'region_txt'

            col_yr = 'iyear'

            if f_val == 'Region': 

                temp_df = pd.DataFrame(index=ind, columns=col)

                for i in ind:

                    count = Counter(list(df.loc[df[col_yr]==i][col_reg]))

                    count_sort = dict(sorted(count.items()))

                    for k,v in count_sort.items():

                        temp_df.at[i,k] = v

                temp_df = temp_df.fillna(0) 

            else: # if f_val == 'Country'

                year = []; region = []; country = []; count = []

                for i in ind:

                    for j in df.region_txt.unique():

                        temp = df[['iyear',

                                   'region_txt',

                                   'country_txt']][(df['iyear']==i)&(df['region_txt']==j)]

                        ctr = Counter(temp.country_txt)

                        for k,v in ctr.items():

                            year.append(str(i))

                            region.append(j)

                            country.append(k)

                            count.append(v)

                temp_df = {

                    'x'       : region,

                    'y'       : count,

                    'country' : country,

                    'year'    : year

                    }                            

    elif cat == 'country':

        if f_val == 'stats':

            init_df = pd.DataFrame()

            for ctry in sorted(df.country_txt.unique()):

                temp = df[['country_txt','iyear','nkill',

                           'nwound','nperpcap']][df['country_txt']==ctry]

                init_df[ctry] = {'x':[], 'count':[], 'nkill':[], 'nwound':[], 'nperpcap':[]}

                year = []; count = []; nkill = []; nwound = []; nperpcap = []

                yr_ctr = Counter(temp.iyear)

                yr_ctr_sort = dict(sorted(yr_ctr.items()))

                for k,v in yr_ctr_sort.items():

                    year.append(k)

                    count.append(v)

                    nkill.append(sum(temp['nkill'][temp['iyear']==k]))

                    nwound.append(sum(temp['nwound'][temp['iyear']==k]))

                    n = temp[['nperpcap']][(temp['nperpcap'] > 0) & (temp['iyear']==k)]

                    nperpcap.append(sum(n.nperpcap)) 

                init_df[ctry]['x'] = year

                init_df[ctry]['count'] = count

                init_df[ctry]['nkill'] = nkill

                init_df[ctry]['nwound'] = nwound

                init_df[ctry]['nperpcap'] = nperpcap

            temp_df = init_df.to_dict()

        elif f_val == 'month':

            init_df = pd.DataFrame()

            for ctry in sorted(df.country_txt.unique()):

                init_df[ctry] = {'x':[], 'y':[]}               

                count_counter = Counter((df['imonth'][df['country_txt']==ctry]))

                new_dict = {x:y for x,y in count_counter.items() if x!=0}

                month_txt = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 

                             7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

                new_counter = dict((month_txt[key],value) for (key,value) in new_dict.items())

                month_list = list(new_counter.keys())

                count_list = list(new_counter.values())

                init_df[ctry]['x'] = month_list

                init_df[ctry]['y'] = count_list

            temp_df = init_df.to_dict()

        else: # if f_val == 'city'

            temp_df = dict()

            for ctry in sorted(df.country_txt.unique()):

                per_ctry = dict()

                for yr in df.iyear.unique():

                    city = []; count = []; gname = []; targtype1 = []; attacktype1 = []

                    city_counter = dict(Counter(df['city'][(df['country_txt']==ctry)&

                                                           (df['iyear']==yr)]).most_common(10))

                    for k,v in city_counter.items():

                        city.append(k)

                        count.append(v)

                        targtype1.append(list(Counter(df['targtype1_txt'][(df['country_txt']==ctry)&(df['iyear']==yr)&(df['city']==k)]).most_common(3)))

                        attacktype1.append(list(Counter(df['attacktype1_txt'][(df['country_txt']==ctry)&(df['iyear']==yr)&(df['city']==k)]).most_common(3)))

                        gname.append(list(Counter(df['gname'][(df['country_txt']==ctry)&(df['iyear']==yr)&(df['city']==k)]).most_common(3)))

                    per_yr = {

                        'x': city,

                        'y': count,

                        'gname': gname,

                        'targtype1': targtype1,

                        'attacktype1': attacktype1

                    }

                    per_ctry.update([(str(yr), per_yr)])

                temp_df.update([(ctry, per_ctry)])                 

    else: # if cat = 'perpetrators'

        perp_df = pd.DataFrame(columns=['gname','nkill','nwound','nperps','iyear','region'])

        for perp in sorted(df.gname.unique()):

            perp_df = perp_df.append({'gname':perp, 

                                      'nkill':sum(df['nkill'][df['gname']==perp]),

                                      'nwound':sum(df['nwound'][df['gname']==perp]),

                                      'nperps':sum(df['nperps'][(df['gname']==perp)&

                                                                (df['nperps']>0)]),

                                      'iyear':len(Counter(df['iyear'][df['gname']==perp])),

                                      'region':len(Counter(df['region_txt'][df['gname']==perp]))

                                     }, ignore_index=True)

        init_df = pd.DataFrame()        

        for desc in f_val:

            init_df[desc] = {'x':[],'y':[],'Most_Attacked_Country':[],'Most_Used_Weapon':[],

                             'Usual_Attack_Type':[],'Frequent_Targets':[],'Most_Attacked_Nationals':[]}

            if desc=='Most_frequent_to_attack':

                gname_ctr = dict(Counter(df['gname']).most_common(10))

                gname=list(gname_ctr.keys())

                count=list(gname_ctr.values())

            elif desc=='With_most_fatalities':

                gname=sorted(list(perp_df.sort_values('nkill').tail(10)['gname']),reverse=True)

                count=sorted(list(perp_df.sort_values('nkill').tail(10)['nkill']),reverse=True)             

            elif desc=='With_most_damages':

                gname=sorted(list(perp_df.sort_values('nwound').tail(10)['gname']),reverse=True) 

                count=sorted(list(perp_df.sort_values('nwound').tail(10)['nwound']),reverse=True)    

            elif desc=='Largest_member_count':

                gname=sorted(list(perp_df.sort_values('nperps').tail(10)['gname']),reverse=True) 

                count=sorted(list(perp_df.sort_values('nperps').tail(10)['nperps']),reverse=True) 

            elif desc=='Longest_running_in_years':

                gname=sorted(list(perp_df.sort_values('iyear').tail(10)['gname']),reverse=True) 

                count=sorted(list(perp_df.sort_values('iyear').tail(10)['iyear']),reverse=True) 

            else: # if f_val == 'Present_in_most_continents'

                gname= sorted(list(perp_df.sort_values('region').tail(10)['gname']),reverse=True) 

                count= sorted(list(perp_df.sort_values('region').tail(10)['region']),reverse=True) 

            country = []; weaptype1 = []; attacktype1 = []; targtype1 = []; natlty1 = []

            for i in gname:

                country.append(list(Counter(df['country_txt'][df['gname']==i]).most_common(3)))

                weaptype1.append(list(Counter(df['weaptype1_txt'][df['gname']==i]).most_common(3)))

                attacktype1.append(list(Counter(df['attacktype1_txt'][df['gname']==i]).most_common(3)))

                targtype1.append(list(Counter(df['targtype1_txt'][df['gname']==i]).most_common(3)))

                natlty1.append(list(Counter(df['natlty1_txt'][df['gname']==i]).most_common(3)))

            init_df[desc]['x'] = gname

            init_df[desc]['y'] = count

            init_df[desc]['Most_Attacked_Country'] = country

            init_df[desc]['Most_Used_Weapon'] = weaptype1

            init_df[desc]['Usual_Attack_Type'] = attacktype1

            init_df[desc]['Frequent_Targets'] = targtype1

            init_df[desc]['Most_Attacked_Nationals'] = natlty1

        temp_df = init_df.to_dict()

    

    return temp_df
def createLinePlot(source, plot_title, iterval, hover, xlabel, ylabel):

    """ Generates a line plot with hover tool. """

    colors = itertools.cycle(palette)

    plot = figure(title=plot_title, plot_height=500, 

                 x_axis_label=xlabel, y_axis_label=ylabel)

    for i in iterval:

        plot.line(x='x', y=i, color=next(colors), legend_label=i, 

                  source=source, line_width=3, alpha=0.5)

        plot.circle(x='x', y=i, color=next(colors), source=source)

    plot.legend.location = 'top_left'

    plot.add_tools(hover)

    show(plot)  
def createTrianglePlot(source, plot_title, iterval, hover, select, callback, xlabel, ylabel):

    """ Generates a plot with triangles as glyphs. With hover and selection callback tool. """

    plot = figure(title=plot_title, x_range=iterval, 

                  x_axis_label=xlabel, y_axis_label=ylabel)

    plot.triangle(x='x',y='y', size=20,alpha=0.5,source=source)

    plot.xaxis.major_label_orientation = 0.7

    plot.add_tools(hover)

    select.callback = callback

    show(column(select,plot))
def createStatsPlot(source, plot_title, hover, select, callback, xlabel, ylabel):

    """ Generates multiple plots with different glyphs (diamond/ circle/ square/ triangle). 

        With hover and selection callback tool. """

    plot = figure(title=plot_title, plot_width=700, plot_height=500, 

                 x_axis_label=xlabel, y_axis_label=ylabel)

    

    plot.diamond(x='x', y='count', source=source, color='green', alpha=0.5, 

                 size=15, legend_label='Attack Count')

    plot.line(x='x', y='count', source=source, color='green', line_width=2, 

              line_dash=[4, 4])



    plot.circle(x='x', y='nkill', source=source, color='firebrick', alpha=0.5, 

                size=15, legend_label='Killed')

    plot.line(x='x', y='nkill', source=source, color='firebrick', line_width=2, 

              line_dash=[4, 4])



    plot.square(x='x', y='nwound', source=source, color='navy', alpha=0.5, 

                size=15, legend_label='Wounded')

    plot.line(x='x', y='nwound', source=source, color='navy', line_width=2, 

              line_dash=[4, 4])



    plot.triangle(x='x', y='nperpcap', source=source, color='olive', alpha=0.5, 

                  size=15, legend_label='Perpetrators Captured')

    plot.line(x='x', y='nperpcap', source=source, color='olive', line_width=2, 

              line_dash=[4, 4])



    plot.legend.location = 'top_left'

    plot.legend.background_fill_alpha = 0.2

    plot.xaxis.major_label_orientation = 0.7

    plot.add_tools(hover)

    select.callback = callback

    show(column(select,plot))
def createBarPlot(source, hover, select, callback, plot):

    """ Generates a bar plot with hover and selection callback tool. """

    plot.vbar(x='x', top='y', width=0.5, source=source, alpha=0.5, color='green')  

    plot.add_tools(hover)

    plot.xaxis.major_label_orientation = 0.7

    select.callback = callback

    show(column(select,plot))
def createBarPlotSelectSlider(source, hover, select, slider, callback, plot):

    """ Generates a bar plot with hover, selection callback, and slider callback tool. """

    plot.vbar(x='x', top='y', width=0.5, source=source, alpha=0.5, color='green')  

    plot.add_tools(hover)

    plot.xaxis.major_label_orientation = 0.7

    select.js_on_change('value', callback)

    slider.js_on_change('value', callback)

    show(column(select,plot,slider))
# trial and error custom javascript callback. kids, don't try this at home.



def createCallbackGbl(cds_3_1970, cds_3_1971, cds_3_1972, cds_3_1973, cds_3_1974, 

                      cds_3_1975, cds_3_1976, cds_3_1977, cds_3_1978, cds_3_1979, 

                      cds_3_1980, cds_3_1981, cds_3_1982, cds_3_1983, cds_3_1984, 

                      cds_3_1985, cds_3_1986, cds_3_1987, cds_3_1988, cds_3_1989, 

                      cds_3_1990, cds_3_1991, cds_3_1992, cds_3_1993, cds_3_1994, 

                      cds_3_1995, cds_3_1996, cds_3_1997, cds_3_1998, cds_3_1999, 

                      cds_3_2000, cds_3_2001, cds_3_2002, cds_3_2003, cds_3_2004, 

                      cds_3_2005, cds_3_2006, cds_3_2007, cds_3_2008, cds_3_2009, 

                      cds_3_2010, cds_3_2011, cds_3_2012, cds_3_2013, cds_3_2014, 

                      cds_3_2015, cds_3_2016, cds_3_2017):

    callback = CustomJS(args={'source':cds_3_1970, 's_1970':cds_3_1970, 's_1971':cds_3_1971, 

                              's_1972':cds_3_1972, 's_1973':cds_3_1973, 's_1974':cds_3_1974, 

                              's_1975':cds_3_1975, 's_1976':cds_3_1976, 's_1977':cds_3_1977, 

                              's_1978':cds_3_1978, 's_1979':cds_3_1979, 's_1980':cds_3_1980, 

                              's_1981':cds_3_1981, 's_1982':cds_3_1982, 's_1983':cds_3_1983, 

                              's_1984':cds_3_1984, 's_1985':cds_3_1985, 's_1986':cds_3_1986, 

                              's_1987':cds_3_1987, 's_1988':cds_3_1988, 's_1989':cds_3_1989, 

                              's_1990':cds_3_1990, 's_1991':cds_3_1991, 's_1992':cds_3_1992, 

                              's_1993':cds_3_1993, 's_1994':cds_3_1994, 's_1995':cds_3_1995, 

                              's_1996':cds_3_1996, 's_1997':cds_3_1997, 's_1998':cds_3_1998, 

                              's_1999':cds_3_1999, 's_2000':cds_3_2000, 's_2001':cds_3_2001, 

                              's_2002':cds_3_2002, 's_2003':cds_3_2003, 's_2004':cds_3_2004, 

                              's_2005':cds_3_2005, 's_2006':cds_3_2006, 's_2007':cds_3_2007, 

                              's_2008':cds_3_2008, 's_2009':cds_3_2009, 's_2010':cds_3_2010, 

                              's_2011':cds_3_2011, 's_2012':cds_3_2012, 's_2013':cds_3_2013, 

                              's_2014':cds_3_2014, 's_2015':cds_3_2015, 's_2016':cds_3_2016, 

                              's_2017':cds_3_2017}, code="""

            console.log(' changed selected option', cb_obj.value);

            if (cb_obj.value == '1970'){

                source.data = s_1970.data

            }

            if (cb_obj.value == '1971'){

                source.data = s_1971.data

            }

            if (cb_obj.value == '1972'){

                source.data = s_1972.data

            }

            if (cb_obj.value == '1973'){

                source.data = s_1973.data

            }

            if (cb_obj.value == '1974'){

                source.data = s_1974.data

            }        

            if (cb_obj.value == '1975'){

                source.data = s_1975.data

            }        

            if (cb_obj.value == '1976'){

                source.data = s_1976.data

            }        

            if (cb_obj.value == '1977'){

                source.data = s_1977.data

            }        

            if (cb_obj.value == '1978'){

                source.data = s_1978.data

            }

            if (cb_obj.value == '1979'){

                source.data = s_1979data

            }        

            if (cb_obj.value == '1980'){

                source.data = s_1980.data

            }     

            if (cb_obj.value == '1981'){

                source.data = s_1981.data

            }

            if (cb_obj.value == '1982'){

                source.data = s_1982.data

            }

            if (cb_obj.value == '1983'){

                source.data = s_1983.data

            }

            if (cb_obj.value == '1984'){

                source.data = s_1984.data

            }

            if (cb_obj.value == '1985'){

                source.data = s_1985.data

            }

            if (cb_obj.value == '1986'){

                source.data = s_1986.data

            }

            if (cb_obj.value == '1987'){

                source.data = s_1987.data

            }

            if (cb_obj.value == '1988'){

                source.data = s_1988.data

            }

            if (cb_obj.value == '1989'){

                source.data = s_1989.data

            }

            if (cb_obj.value == '1990'){

                source.data = s_1990.data

            }

            if (cb_obj.value == '1991'){

                source.data = s_1991.data

            }

            if (cb_obj.value == '1992'){

                source.data = s_1992.data

            }

            if (cb_obj.value == '1993'){

                source.data = s_1993.data

            }

            if (cb_obj.value == '1994'){

                source.data = s_1994.data

            }

            if (cb_obj.value == '1995'){

                source.data = s_1995.data

            }

            if (cb_obj.value == '1996'){

                source.data = s_1996.data

            }

            if (cb_obj.value == '1997'){

                source.data = s_1997.data

            }

            if (cb_obj.value == '1998'){

                source.data = s_1998.data

            }

            if (cb_obj.value == '1999'){

                source.data = s_1999.data

            }

            if (cb_obj.value == '2000'){

                source.data = s_2000.data

            }

            if (cb_obj.value == '2001'){

                source.data = s_2001.data

            }

            if (cb_obj.value == '2002'){

                source.data = s_2002.data

            }

            if (cb_obj.value == '2003'){

                source.data = s_2003.data

            }

            if (cb_obj.value == '2004'){

                source.data = s_2004.data

            }

            if (cb_obj.value == '2005'){

                source.data = s_2005.data

            }

            if (cb_obj.value == '2006'){

                source.data = s_2006.data

            }

            if (cb_obj.value == '2007'){

                source.data = s_2007.data

            }

            if (cb_obj.value == '2008'){

                source.data = s_2008.data

            }

            if (cb_obj.value == '2009'){

                source.data = s_2009.data

            }

            if (cb_obj.value == '2010'){

                source.data = s_2010.data

            }

            if (cb_obj.value == '2011'){

                source.data = s_2011.data

            }

            if (cb_obj.value == '2012'){

                source.data = s_2012.data

            }

            if (cb_obj.value == '2013'){

                source.data = s_2013.data

            }

            if (cb_obj.value == '2014'){

                source.data = s_2014.data

            }

            if (cb_obj.value == '2015'){

                source.data = s_2015.data

            }

            if (cb_obj.value == '2016'){

                source.data = s_2016.data

            }       

            if (cb_obj.value == '2017'){

                source.data = s_2017.data

            }

            source.change.emit();

    """)

    return callback
def createCallback(val_cds, val_dict):

    """ Input parameter: column data source, data dictionary

        Returns: a callback customjs item 

        This function updates the source.data with new data base on the changed selection value.

    """

    callback = CustomJS(args={'source':val_cds, 'temp_dict': val_dict}, code="""

                console.log(' changed selected option', cb_obj.value);

                var new_data = temp_dict[cb_obj.value]

                source.data = new_data

                source.change.emit();

        """) 

    return callback
def createCallbackWithRange(val_cds, val_dict, plot):

    """ Input parameter: column data source, data dictionary, plot figure

        Returns: a callback customjs item 

        This function updates the source.data and x-range plot factor with new data base on 

        the changed selection value.

    """

    callback = CustomJS(args={'source':val_cds, 'temp_dict': val_dict, 'plot':plot}, code="""

                console.log(' changed selected option', cb_obj.value);

                var new_data = temp_dict[cb_obj.value]

                plot.x_range.factors = new_data['x']

                plot.change.emit();

                source.data = new_data

                source.change.emit();

        """) 

    return callback
def createCallbackSelectSlider(val_cds, val_dict, plot, select, slider):

    """ Input parameter: column data source, data dictionary, plot figure, select item, 

        slider item

        Returns: a callback customjs item 

        This function updates the source.data and x-range plot factor with the new data base on 

        the changed selection value + changed slider value.

    """

    callback = CustomJS(args={'source':val_cds, 'temp_dict':val_dict, 'plot':plot, 

                              'select':select, 'slider':slider}, code="""

                var new_data = temp_dict[select.value][slider.value]

                plot.x_range.factors = new_data['x']

                plot.change.emit();

                source.data = new_data

                source.change.emit();

        """) 

    return callback
def createHover(desc_var):

    """ Input parameter: a list containing the description and its variable. 

                         ie. createHover([('Total Count','@total'), ('Year','@year')])

        Returns: hover item

        This function displays the description and value upon hovering on the graph where the 

        hover item is added.

    """

    hover = HoverTool(tooltips=desc_var, mode='mouse')

    return hover
# This will display bokeh plots inline

output_notebook()
cat = 'global'

f_list, f_init, f_title, f_title_plot= selectFilter(cat)

temp_df = createTempDf(cat, f_init)

cds_1 = ColumnDataSource(temp_df)



hover = createHover([('Total Attack Count', '@total'),('Year', '@x')])

createLinePlot(cds_1, f_title_plot[0], ['total'], hover, 'Year', 'Count')
print('>> Top 5 terrorist groups 1971:', dict(Counter(df['gname'][df['iyear']==1971]).most_common(5)))

print('\n>> Top 5 terrorized countries 1971:', dict(Counter(df['country_txt'][df['iyear']==1971]).most_common(5)))

print('\n>> Top 5 motives 1971:', dict(Counter(df_init['motive'][df_init['iyear']==1971]).most_common(5)))
print('>> Top 5 terrorist groups 2014:', dict(Counter(df['gname'][df['iyear']==2014]).most_common(5)))

print('\n>> Top 5 terrorized countries 2014:', dict(Counter(df['country_txt'][df['iyear']==2014]).most_common(5)))

print('\n>> Top 5 targets 2014:', dict(Counter(df['targtype1_txt'][df['iyear']==2014]).most_common(5)))
my_dict = {'gname':'terrorist groups', 'country_txt':'terrorized countries', 'targtype1_txt':'targets'}

for i in [2012, 2013, 2014]:

    for k,v in my_dict.items():

        print('>> Top 5',v,i,':', dict(Counter(df[k][df['iyear']==i]).most_common(5)))

        print('\n')

    print('\n')
temp_df = createTempDf(cat, f_list[1])

cds_2 = ColumnDataSource(temp_df)



x=cds_2.data['index']; del cds_2.data['index']; cds_2.data['x'] = x

hover = createHover([('Year', '@x')])

createLinePlot(cds_2, f_title_plot[1], df.region_txt.unique(), hover, 'Year', 'Count')
print('>> Top 5 regions with the highest attack frequency (all time):', dict(Counter(df.region_txt).most_common(5)))

print('\n>> The most attacked region of all time:', dict(Counter(df.region_txt).most_common(1)))

print('\n>> Region that saw the highest terrorism frequency on 2014:', dict(Counter(df['region_txt'][df['iyear']==2014]).most_common(1)))

print("\n>> During 70's - 80's, the regions that experienced the most attacks were:", dict(Counter(df['region_txt'][df['iyear']<1980]).most_common(2)))

print("\n>> During 80's - 90's, the regions that experienced the most attacks were:", dict(Counter(df['region_txt'][(df['iyear']>1979)&(df['iyear']<1991)]).most_common(2)))

print('\n>> Few years before and after y2k, the incident attacks were lower.')

print("\n>> Regions that experienced the most attacks after 2010:", dict(Counter(df['region_txt'][df['iyear']>2010]).most_common(5)))

temp_df = pd.DataFrame(createTempDf(cat, f_list[2]))



for i in range(1970,2018,1):

    name = 'cds_3_%s' % i

    globals()[name] = ColumnDataSource(temp_df[temp_df['year']==str(i)])



hover = createHover([('Total Attack Count', '@y'),('Region', '@x'),

                     ('Country','@country'),('Year','@year')])

select = Select(title='Year', value='1970', options=list(map(str, df.iyear.unique())))

callback = createCallbackGbl(cds_3_1970, cds_3_1971, cds_3_1972, cds_3_1973, 

                             cds_3_1974, cds_3_1975, cds_3_1976, cds_3_1977, 

                             cds_3_1978, cds_3_1979, cds_3_1980, cds_3_1981, 

                             cds_3_1982, cds_3_1983, cds_3_1984, cds_3_1985, 

                             cds_3_1986, cds_3_1987, cds_3_1988, cds_3_1989, 

                             cds_3_1990, cds_3_1991, cds_3_1992, cds_3_1993, 

                             cds_3_1994, cds_3_1995, cds_3_1996, cds_3_1997, 

                             cds_3_1998, cds_3_1999, cds_3_2000, cds_3_2001, 

                             cds_3_2002, cds_3_2003, cds_3_2004, cds_3_2005, 

                             cds_3_2006, cds_3_2007, cds_3_2008, cds_3_2009, 

                             cds_3_2010, cds_3_2011, cds_3_2012, cds_3_2013, 

                             cds_3_2014, cds_3_2015, cds_3_2016, cds_3_2017)



createTrianglePlot(cds_3_1970, f_title_plot[2], df.region_txt.unique(), hover, select, callback, 'Region', 'Count')
from operator import itemgetter



c = Counter(df.country_txt)

min_key, min_count = min(c.items(), key=itemgetter(1))

max_key, max_count = max(c.items(), key=itemgetter(1))

print('Country that has the highest attack frequency of all time : ', max_key, max_count)

print('Country that has the lowest attack frequency of all time  : ', min_key, min_count)

cat = 'country'

f_list, f_init, f_title, f_title_plot = selectFilter(cat)

temp_dict = createTempDf(cat, 'stats')

cds_4 = ColumnDataSource(temp_dict[f_init])



hover = createHover([('Year','@x'),('Total Attack Count', '@count'),

                     ('Total Killed','@nkill'), ('Total Wounded','@nwound'),

                     ('Terrorists Captured','@nperpcap')])

select = Select(title=f_title, value=f_init, options=f_list)



callback = createCallback(cds_4, temp_dict)

createStatsPlot(cds_4, f_title_plot[0], hover, select, callback, 'Year', 'Count')
temp = df[['gname','targtype1_txt','weaptype1_txt']][(df['iyear']==1988)&(df['country_txt']=='Philippines')]

print('1988 top terrorists:',Counter(temp.gname).most_common(3))

print('\n1988 top targets:',Counter(temp.targtype1_txt).most_common(3))

print('\n1988 top weapons used:',Counter(temp.weaptype1_txt).most_common(3))
temp = df[['gname','targtype1_txt','weaptype1_txt']][(df['iyear']==2015)&(df['country_txt']=='Philippines')]

print('2015 top terrorists:',Counter(temp.gname).most_common(3))

print('\n2015 top targets:',Counter(temp.targtype1_txt).most_common(3))

print('\n2015 top weapons used:',Counter(temp.weaptype1_txt).most_common(3))
temp_dict = createTempDf(cat, 'month')

cds_5 = ColumnDataSource(temp_dict[f_init])



x_range = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

hover = createHover([('Month','@x'),('Count', '@y')])

select = Select(title=f_title, value=f_init, options=f_list)



plot = figure(title=f_title_plot[1], x_range=x_range, plot_height=300, 

             x_axis_label='Month', y_axis_label='Count')

callback = createCallback(cds_5, temp_dict)

createBarPlot(cds_5, hover, select, callback, plot)
temp_dict = createTempDf(cat, 'city')

cds_6 = ColumnDataSource(temp_dict[f_init]['2017'])



x_range = cds_6.data['x']

hover = createHover([('Total Count','@y'),

                     ('Frequent Terrorist Group','@gname'),

                     ('Frequent Target','@targtype1'),

                     ('Frequent Attack Type','@attacktype1')])

select_ctry = Select(title=f_title, value=f_init, options=f_list)

slider_yr = Slider(start=1970, end=2017, step=1, value=2017, 

                   title='Year (1970 till 2017, except 1993)')



plot = figure(title=f_title_plot[2], x_range=x_range,

             x_axis_label='City', y_axis_label='Count')

callback = createCallbackSelectSlider(cds_6, temp_dict, plot, select_ctry, slider_yr)

createBarPlotSelectSlider(cds_6, hover, select_ctry, slider_yr, callback, plot)
cat = 'perpetrators'

f_list, f_init, f_title, f_title_plot = selectFilter(cat)

temp_dict = createTempDf(cat, f_list)

cds_7 = ColumnDataSource(temp_dict[f_init])



x_range = cds_7.data['x']

hover = createHover([('Total Count','@y'), 

                     ('Most Attacked Country','@Most_Attacked_Country'), 

                     ('Most Used Weapon','@Most_Used_Weapon'), 

                     ('Usual Attack Type','@Usual_Attack_Type'), 

                     ('Frequent Targets','@Frequent_Targets'), 

                     ('Most Attacked Nationals','@Most_Attacked_Nationals')])

select = Select(title=f_title, value=f_init, options=f_list)



plot = figure(title=f_title_plot, x_range=x_range,

             x_axis_label='Group Name', y_axis_label='Count')

callback = createCallbackWithRange(cds_7, temp_dict, plot)

createBarPlot(cds_7, hover, select, callback, plot)