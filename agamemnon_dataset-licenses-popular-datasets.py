import numpy as np

import pandas as pd



import os



import colorlover as cl



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import sys

!{sys.executable} -m pip install csvvalidator
# Read data and display small sample

dataset_versions = pd.read_csv('../input/DatasetVersions.csv')

dataset_versions.sample(5)
lcns = pd.DataFrame(dataset_versions.LicenseName.value_counts(normalize=True, ascending=False, dropna=True)).reset_index()

lcns.rename(columns={'index': 'License', 'LicenseName': 'Frequency'}, inplace=True)

lcns
n_elements = len(np.unique(lcns.License))

c_palette = cl.scales[str(n_elements)]['qual']['Set3'][::-1]



data = [go.Bar(

    y=lcns.License[::-1],

    x=lcns.Frequency[::-1],

    orientation='h',

    marker=dict(

        color=c_palette,

        line=dict(

            color='rgb(8,48,107)',

            width=1.5)

    ),

    opacity=0.8

    )]



layout = go.Layout(

    autosize=False,

    width=800,

    height=500,

    margin=go.layout.Margin(

        l=400,

        r=50,

        b=100,

        t=100,

        pad=4

    ),

    xaxis=dict(title='Normalised frequency')

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)
ds = pd.read_csv('../input/Datasets.csv')

ds.sample(10)
# Validate date

from csvvalidator import CSVValidator

field_names = ['TotalViews', 'TotalDownloads', 'TotalVotes', 'TotalKernels']

validator = CSVValidator(field_names=field_names)

for field_name in field_names:

    validator.add_value_check(field_name=field_name,

                             value_check=int,

                             code='ValueError',

                             message="{} must be integer".format(field_name))

validator.validate(ds[field_names].values)
# Main processing function

def get_popular_datasets(df, n_datasets, popularity_field):

    ds_popular = df.sort_values(by=popularity_field, ascending=False)[0:n_datasets].reset_index(drop=True)

    return ds_popular



# Test function

def test_get_popular_datasets():

    from io import StringIO

    test_data = StringIO("""

    Id,CreatorUserId,OwnerUserId,TotalViews,TotalDownloads,TotalVotes,TotalKernels

    1,1,1,100,50,30,20

    2,2,2,80,60,40,50

    3,3,3,180,20,65,32

    4,4,4,40,500,23,52

    """)

    test_csv = pd.read_csv(test_data)



    output_data_correct = StringIO("""

    Id,CreatorUserId,OwnerUserId,TotalViews,TotalDownloads,TotalVotes,TotalKernels

    3,3,3,180,20,65,32

    1,1,1,100,50,30,20

    """)

    output_csv_correct = pd.read_csv(output_data_correct)

    pd.testing.assert_frame_equal(output_csv_correct, get_popular_datasets(test_csv, n_datasets=2, popularity_field='TotalViews'))



test_get_popular_datasets()
n_popular = 10

ds_popular = get_popular_datasets(ds, n_datasets=n_popular, popularity_field='TotalViews')
c_palette = cl.scales[str(n_popular)]['qual']['Set3']

x_data = ds_popular[:n_popular]['Id'].astype(str)



trace1 = go.Bar(

    x=np.arange(len(x_data)),

    y=ds_popular[:n_popular]['TotalViews'],

    marker=dict(

        line=dict(

            color='rgb(8,48,107)',

            width=1.5)

    ),

    opacity=0.8,

    name = 'Total views'

)



trace2 = go.Bar(

    x=np.arange(len(x_data)),

    y=ds_popular[:n_popular]['TotalDownloads'],

    marker=dict(

        line=dict(

            color='rgb(8,48,107)',

            width=1.5)

    ),

    opacity=0.8,

    name = 'Total downloads'

)



trace3 = go.Bar(

    x=np.arange(len(x_data)),

    y=ds_popular[:n_popular]['TotalVotes'],

    marker=dict(

        line=dict(

            color='rgb(8,48,107)',

            width=1.5)

    ),

    opacity=0.8,

    name = 'Total votes'

)



trace4 = go.Bar(

    x=np.arange(len(x_data)),

    y=ds_popular[:n_popular]['TotalKernels'],

    marker=dict(

        line=dict(

            color='rgb(8,48,107)',

            width=1.5)

    ),

    opacity=0.8,

    name = 'Total kernels'

)



data = [trace1, trace2, trace3, trace4]



layout = go.Layout(

    autosize=False,

    width=700,

    height=500,

    xaxis=dict(title='Dataset',

               tickvals=np.arange(n_popular),

               ticktext=x_data),

    barmode='stack'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)