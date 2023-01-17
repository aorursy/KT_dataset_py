import pandas as pd

import numpy as np

import ipywidgets as wid
df=pd.read_excel('../input/data-bq/BQ-Assignment-Data-Analytics.xlsx')

df['Date'] = df['Date'].map(lambda x: x.strftime('%b %y'))

df = pd.pivot_table(df, values='Sales', index=['Item Type', 'Item','Item Sort Order'],

                    columns=['Date']).sort_values(by='Item Sort Order').reset_index().rename_axis(None, axis=1)

itemType= df['Item Type'].unique().tolist()
def multi_checkbox_widget(options_dict):

    output_widget = wid.Output()

    options = [x for x in options_dict.values()]

    layout = wid.Layout(border='2px solid black',width='100px',height='100px')

    options_widget = wid.VBox(options, layout=layout)

    return options_widget



options_dict = {

    x: wid.Checkbox(

        description=x, 

        value=False,

        indent=False

    ) for x in itemType

}

z= pd.DataFrame()

def f(**args):

    results = [key for key, value in args.items() if value]

    tmp=df[df['Item Type'].isin(results)].loc[:,df.columns != 'Item Type'].reset_index(drop=True)

    tmp.to_csv('export.csv')

    display(tmp)

    

inp = multi_checkbox_widget(options_dict)

out = wid.interactive_output(f, options_dict)

display(wid.VBox([inp, out]))