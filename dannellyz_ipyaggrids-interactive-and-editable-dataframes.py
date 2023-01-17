!pip install -q ipyaggrid

!jupyter nbextension enable --py --sys-prefix ipyaggrid
#Libraries Needed

import pandas as pd



#Settings to display pandas

pd.set_option('display.max_columns', None)



#Some basic set up

base_path = "/kaggle/input/uncover/"



#Read in sample dataframe

CAC_path = "coders_against_covid/"

file_name = "crowd-sourced-covid-19-testing-locations.csv"

coders_against_covid = pd.read_csv(base_path+CAC_path+file_name)

coders_against_covid.head(1)
from ipyaggrid import Grid



def simple_grid(df):



    column_defs = [{'headername':c,'field': c} for c in df.columns]



    grid_options = {

        'columnDefs' : column_defs,

        'enableSorting': True,

        'enableFilter': True,

        'enableColResize': True,

        'enableRangeSelection': True,

        'rowSelection': 'multiple',

    }



    g = Grid(grid_data=df,

             grid_options=grid_options,

             quick_filter=True,

             show_toggle_edit=True,

             sync_on_edit=True,

             export_csv=True,

             export_excel=True,

             theme='ag-theme-balham',

             show_toggle_delete=True,

             columns_fit='auto',

             index=False)

    return g

simple_grid(coders_against_covid)