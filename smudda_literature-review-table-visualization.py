import numpy as np 

import pandas as pd 

import os

import ipywidgets as widgets



from IPython.display import display

from IPython.display import HTML

input_dir = '/kaggle/input/aipowered-literature-review-csvs/kaggle/working'

dataframe_collection = {}



for root, dirs, files in os.walk(input_dir):

    for table_dir in dirs:

        df_tables = {}

        table_dir_path = os.path.join(input_dir, table_dir)

        for filename in os.listdir(table_dir_path):

            table_file = os.path.join(table_dir_path, filename)

            table = pd.read_csv(table_file)

            df_tables[filename] = pd.read_csv(table_file)

        

        dataframe_collection[table_dir] = df_tables

def make_clickable(url, title):

    return '<a href="{}">{}</a>'.format(url,title)



def selectCol(x, df):

    if('Study Link' in df.columns):

        return x['Study Link']

    elif ('Link' in df.columns):

        return x['Link']

    elif ('URL' in df.columns):

        return x['URL']

        



def link_study_to_url(df):

    if('Study' in df.columns):

        df['Study'] = df.apply(lambda x: make_clickable(selectCol(x, df), x['Study']), axis=1) 

    elif('Title' in df.columns):

        df['Title'] = df.apply(lambda x: make_clickable(selectCol(x, df), x['Title']), axis=1)

def unique_sorted_values(array):

    unique = list(map(lambda x: x.replace(".csv", ""),array))

    unique.sort()

    return unique
redundant_fields = ['Study Link', 'Link', 'Journal', 'URL']

dropdowns = {}

for name, cat_dict in dataframe_collection.items():

    dropdowns[name] = widgets.Dropdown(options = unique_sorted_values(set(cat_dict.keys())), description=name, value=None)

    for key, df in cat_dict.items():

        link_study_to_url(df)

        drop_list = [col for col in df.columns if col in redundant_fields]

        df.drop(drop_list, axis=1, inplace=True) 

        df.drop('Unnamed: 0', axis=1, inplace=True)

output = widgets.Output() 



def key_dropdown_event_handler(change):

    output.clear_output(wait=True)

    with output:

        df = dataframe_collection['Key Scientific Questions'][change.new + '.csv']

        print("Table for {} - {}".format(change.new, 'Key Scientific Questions'))

        display(HTML(df.to_html(escape=False)))



def risk_dropdown_event_handler(change):

    output.clear_output(wait=True)

    with output:

        df = dataframe_collection['Risk Factors'][change.new + '.csv']

        print("Table for {} - {}".format(change.new, 'Risk Factors'))

        display(HTML(df.to_html(escape=False)))

item_layout = widgets.Layout(display='flex',

                    flex_flow='row',

                    align_items='stretch',

                    width='70%')



input_widgets = widgets.HBox([dropdowns['Key Scientific Questions'], dropdowns['Risk Factors']],layout=item_layout)

dropdowns['Key Scientific Questions'].observe(key_dropdown_event_handler, names="value")

dropdowns['Risk Factors'].observe(risk_dropdown_event_handler, names="value")
display(input_widgets)

display(output)