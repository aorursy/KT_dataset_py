import pandas as pd

import glob

import datetime

import json

import re

import numpy as np

from bokeh.io import output_notebook, push_notebook

from bokeh.io import show, save, output_file

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter, NumeralTickFormatter

from bokeh.palettes import Set1_9 as palette

from ipywidgets import interact, IntSlider

import ipywidgets as widget

output_notebook()
root_path = '/kaggle/input/CORD-19-research-challenge'

df_metadata = pd.read_csv('%s/metadata.csv' % root_path)
def publish_time_to_datetime(publish_time):

    if(str(publish_time) == 'nan'):

        return_date = None

        

    else:

        list_publish_time = re.split('[ -]',publish_time)

        if len(list_publish_time) >2 :

            try:

                #'2020 Jan 27'

                #'2017 Apr 7 May-Jun'

                return_date = datetime.datetime.strptime('-'.join(list_publish_time[:3]), '%Y-%b-%d')



            except :                

                try :

                    #'2020 03 16'

                    return_date = datetime.datetime.strptime('-'.join(list_publish_time[:3]), '%Y-%m-%d')

                    

                except:

                    #'2015 Jul-Aug'

                    return_date = datetime.datetime.strptime('-'.join(list_publish_time[:2]), '%Y-%b')



        elif len(list_publish_time) == 2:

            #'2015 Winter' -> 1 fev            

            if(list_publish_time[1] == 'Winter'):

                return_date = datetime.datetime(int(list_publish_time[0]), 2, 1)



            #'2015 Spring' -> 1 may            

            elif(list_publish_time[1] == 'Spring'):

                return_date = datetime.datetime(int(list_publish_time[0]), 5, 1)

                

            #'2015 Autumn' -> 1 nov

            elif(list_publish_time[1] in ['Autumn','Fall']):

                return_date = datetime.datetime(int(list_publish_time[0]), 11, 1)            

            else:

                #"2015 Oct"

                return_date = datetime.datetime.strptime('-'.join(list_publish_time), '%Y-%b')



        elif len(list_publish_time) == 1:

            #'2020'

            return_date = datetime.datetime.strptime('-'.join(list_publish_time), '%Y')



    return return_date
%%time

# thanks to Frank Mitchell

json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)

df_data = pd.DataFrame()



# set a break_limit for quick test (-1 for off)

break_limit = -1

print_debug = False



for i,file_name in enumerate(json_filenames):

    if(print_debug):print(file_name)

    

    # get the sha

    sha = file_name.split('/')[6][:-5]

    if(print_debug):print(sha)

    

    # get the all_sources information

    df_metadata_sha = df_metadata[df_metadata['sha'] == sha]

   

    if(df_metadata_sha.shape[0] > 0):

        s_metadata_sha = df_metadata_sha.iloc[0]

    

        # treat only if full text

        if(s_metadata_sha['has_full_text']):

            dict_to_append = {}

            dict_to_append['sha'] = sha

            dict_to_append['dir'] = file_name.split('/')[4]



            # publish time into datetime format        

            datetime_publish_time = publish_time_to_datetime(s_metadata_sha['publish_time'])



            if(datetime_publish_time is not None):

                dict_to_append['publish_time'] = datetime_publish_time

                dict_to_append['title'] = s_metadata_sha['title']



                # thanks to Frank Mitchell

                with open(file_name) as json_data:

                    data = json.load(json_data)



                    # get abstract

                    abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']))]            

                    abstract = "\n ".join(abstract_list)

                    dict_to_append['abstract'] = abstract





                    # get body

                    body_list = [data['body_text'][x]['text'] for x in range(len(data['body_text']))]            

                    body = "\n ".join(body_list)

                    dict_to_append['body'] = body





                df_data = df_data.append(dict_to_append, ignore_index=True)



    else:

        if(print_debug):print('not found')

                

    if (break_limit != -1):

        if (i>break_limit):

            break
df_publish_month = df_data.sha.groupby(df_data['publish_time'].dt.to_period("M")).count()



source = ColumnDataSource(data=dict(

    month = df_publish_month.index,

    month_tooltips = df_publish_month.index.strftime('%Y/%m'),

    publication_count = df_publish_month.values

))



tooltips = [('month','@month_tooltips'),('publication_count','@publication_count')]

tools = ['pan', 'box_zoom', 'wheel_zoom', 'reset', HoverTool(tooltips=tooltips, names=['hover_tool'])]

p = figure(plot_height=600,  plot_width=800,tooltips=tooltips, active_drag="pan", active_scroll='wheel_zoom')

p.line('month','publication_count',source=source)

p.xaxis.formatter=DatetimeTickFormatter(months=["%Y/%m"])

p.title.text = 'Publication count per Month'

p.xaxis[0].axis_label = 'Months'

p.yaxis[0].axis_label = 'Publication count'

show(p)