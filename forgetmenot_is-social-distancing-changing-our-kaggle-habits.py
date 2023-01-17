from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://www.aljazeera.com/mritems/imagecache/mbdxxlarge/mritems/Images/2020/3/19/d42a4e22ee344ff5a9f5061d4869cee1_18.jpg")
# importing libraries

import pandas as pd

import numpy as np



from datetime import datetime



import plotly.graph_objs as go

import plotly.io as pio

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



import statsmodels.api as sm

from scipy import stats
colors_array = ['#34495E', '#F1C40F', '#138D75', '#CB4335', '#808B96']



layout_custom = go.layout.Template(

    layout=go.Layout(titlefont=dict(size=24, color=colors_array[0]),

                    legend=dict(orientation='h',y=1.1)))
# importing needed files



dateparse = lambda x: pd.to_datetime(x, errors='coerce')



kernels = pd.read_csv('../input/meta-kaggle/KernelVersions.csv', parse_dates=[5, 8], date_parser=dateparse)

kernel_votes = pd.read_csv('../input/meta-kaggle/KernelVotes.csv', parse_dates=[3], date_parser=dateparse)

datasets = pd.read_csv('../input/meta-kaggle/DatasetVersions.csv', parse_dates=[5], date_parser=dateparse)

dataset_votes = pd.read_csv('../input/meta-kaggle/DatasetVotes.csv', parse_dates=[3], date_parser=dateparse)
print("\nKernels data set sample:\n\n")

display(kernels.tail())



print("\nKernels votes data set sample:\n\n")

display(kernel_votes.tail())



print("\nDatasets data set sample:\n\n")

display(datasets.tail())



print("\nDatasets votes data set sample:\n\n")

display(dataset_votes.tail())

print('Minimum date in kernels: ', kernels['CreationDate'].min())

print('Minimum date in kernel_votes: ', kernel_votes['VoteDate'].min())

print('Minimum date in datasets: ', datasets['CreationDate'].min())

print('Minimum date in dataset_votes: ', dataset_votes['VoteDate'].min())
kernels = kernels[(kernels['CreationDate'] >= '2018-01-01') & (kernels['CreationDate'] < '2020-03-29')]

datasets = datasets[(datasets['CreationDate'] >= '2018-01-01') & (datasets['CreationDate'] < '2020-03-29')]

kernel_votes = kernel_votes[(kernel_votes['VoteDate'] >= '2018-01-01') & (kernel_votes['VoteDate'] < '2020-03-29')]

dataset_votes = dataset_votes[(dataset_votes['VoteDate'] >= '2018-01-01') & (dataset_votes['VoteDate'] < '2020-03-29')]
daily_kernels_created = kernels.groupby([pd.Grouper(key='CreationDate', freq='D')])['Id'].count()

daily_datasets_created = datasets.groupby([pd.Grouper(key='CreationDate', freq='D')])['Id'].count()

daily_votes_kernels = kernel_votes.groupby([pd.Grouper(key='VoteDate', freq='D')])['Id'].count()

daily_votes_datasets = dataset_votes.groupby([pd.Grouper(key='VoteDate', freq='D')])['Id'].count()



daily_amounts = pd.concat([daily_kernels_created, daily_datasets_created, daily_votes_kernels, daily_votes_datasets], axis=1)

daily_amounts.columns = ['kernels', 'datasets', 'kernel_votes', 'datasets_votes']

daily_amounts = daily_amounts.reset_index()



weekly_amounts = daily_amounts.groupby([pd.Grouper(key='CreationDate', freq='W-MON', closed='left', label='left')]).sum().reset_index()



monthly_amounts = daily_amounts.groupby([pd.Grouper(key='CreationDate', freq='MS')]).sum().reset_index()
def make_plotly(fig, name, dataset, visualize_list, visible=True):

    #making traces for my graphs

    fig.add_trace(go.Scatter(x=dataset['CreationDate'], y=dataset[visualize_list[0]], name="Created", 

                             line=dict(color=colors_array[3]), visible=visible), row=1, col=1)

    fig.add_trace(go.Scatter(x=dataset['CreationDate'], y=dataset[visualize_list[1]], name="Votes", 

                             line=dict(color=colors_array[-1]), visible=visible), row=2, col=1)



# creating a figure with 2 subplots

fig = make_subplots(rows=2, cols=1, subplot_titles=("Created", "Votes"))



# daily part

make_plotly(fig, "Kernels", daily_amounts, ['kernels', 'kernel_votes'])

make_plotly(fig, "Datasets", daily_amounts, ['datasets', 'datasets_votes'], visible=False)



# weekly part

make_plotly(fig, "Kernels", weekly_amounts, ['kernels', 'kernel_votes'], visible=False)

make_plotly(fig, "Datasets", weekly_amounts, ['datasets', 'datasets_votes'], visible=False)



# monthly part

make_plotly(fig, "Kernels", monthly_amounts, ['kernels', 'kernel_votes'], visible=False)

make_plotly(fig, "Datasets", monthly_amounts, ['datasets', 'datasets_votes'], visible=False)



# the code below creates list for the filter

list_buttons = []

                

traces, dicts = (2, 6)

vector_size = traces*dicts

true_start = 0



for date_type in ['daily', 'weekly', 'monthly']:

    for kernel_or_dataset in ["Kernels", "Datasets"]:

        visibility_list = [False for i in range(vector_size)]

        visibility_list[true_start:(true_start+traces)] = [True]*traces

        dict_button = dict(label=kernel_or_dataset+" - "+date_type, method="update",

                           args=[{"visible": visibility_list},

                                 {"title": kernel_or_dataset+" - "+date_type}])

        list_buttons.append(dict_button)

        true_start+=traces



fig.layout.update(

    template=layout_custom,

    showlegend=False,

    updatemenus=[

        go.layout.Updatemenu(

            active=0,

            x=1.05, y=1.10,

            buttons=list_buttons)

    ])



fig.layout.update(title="Kernels - daily")



fig['layout']['yaxis1'].update(hoverformat =",d", tickformat=",d")

fig['layout']['yaxis2'].update(hoverformat =",d", tickformat=",d")



iplot(fig)
kernels_2020 = kernels[kernels['CreationDate'] >= '2020-01-01']

datasets_2020 = datasets[datasets['CreationDate'] >= '2020-01-01']



covid_kernels = kernels_2020.loc[[('covid' in str(string).lower()) or ('corona' in str(string).lower()) 

                             for string in kernels_2020['Title']]]

other_kernels = kernels_2020.loc[[('covid' not in str(string).lower()) and ('corona' not in str(string).lower()) 

                             for string in kernels_2020['Title']]]

covid_datasets = datasets_2020.loc[[('covid' in str(string).lower()) or ('corona' in str(string).lower()) 

                               for string in datasets_2020['Title']]]

other_datasets = datasets_2020.loc[[('covid' not in str(string).lower()) and ('corona' not in str(string).lower()) 

                               for string in datasets_2020['Title']]]
daily_covid_kernels = covid_kernels.groupby([pd.Grouper(key='CreationDate', freq='D')])['Id'].count()

daily_other_kernels = other_kernels.groupby([pd.Grouper(key='CreationDate', freq='D')])['Id'].count()

daily_covid_datasets = covid_datasets.groupby([pd.Grouper(key='CreationDate', freq='D')])['Id'].count()

daily_other_datasets = other_datasets.groupby([pd.Grouper(key='CreationDate', freq='D')])['Id'].count()



daily_amounts_covid = pd.concat([daily_covid_kernels, daily_other_kernels, daily_covid_datasets, daily_other_datasets], axis=1)

daily_amounts_covid.columns = ['covid_kernels', 'other_kernels', 'covid_datasets', 'other_datasets']

daily_amounts_covid = daily_amounts_covid.reset_index()



weekly_amounts_covid = daily_amounts_covid.groupby([pd.Grouper(key='CreationDate', freq='W-MON', closed='left', label='left')]).sum().reset_index()



monthly_amounts_covid = daily_amounts_covid.groupby([pd.Grouper(key='CreationDate', freq='MS')]).sum().reset_index()
def make_plotly_covid(fig, dataset, visible=True):

    # making traces for my graphs

    fig.add_trace(go.Scatter(x=dataset['CreationDate'], y=dataset['covid_kernels'], name="COVID", 

                             line=dict(color=colors_array[3]), visible=visible), 

                  row=1, col=1)

    fig.add_trace(go.Bar(x=dataset['CreationDate'], y=dataset['other_kernels'], name="non COVID", 

                         marker=dict(color=colors_array[-1]), opacity=0.4, visible=visible), 

                  row=1, col=1)

    

    fig.add_trace(go.Scatter(x=dataset['CreationDate'], y=dataset['covid_datasets'], name="COVID",

                             line=dict(color=colors_array[3]), visible=visible, showlegend=False), 

                  row=2, col=1)

    fig.add_trace(go.Bar(x=dataset['CreationDate'], y=dataset['other_datasets'], name="non COVID", 

                         marker=dict(color=colors_array[-1]), opacity=0.4, visible=visible, showlegend=False), 

                  row=2, col=1)



# creating figure with 2 subplots

fig = make_subplots(rows=2, cols=1, subplot_titles=('Kernels', 'Datasets'))



# daily, weekly and monthly parts

make_plotly_covid(fig, daily_amounts_covid)

make_plotly_covid(fig, weekly_amounts_covid, visible=False)

make_plotly_covid(fig, monthly_amounts_covid, visible=False)



# the code below creates list for the filter

list_buttons = []

                

traces, dicts = (4, 3)

vector_size = traces*dicts

true_start = 0



for date_type in ['Daily', 'Weekly', 'Monthly']:

    visibility_list = [False for i in range(vector_size)]

    visibility_list[true_start:(true_start+traces)] = [True]*traces

    dict_button = dict(label=date_type, method="update",

                       args=[{"visible": visibility_list},

                             {"title": date_type+' amounts by COVID/non COVID'}])

    list_buttons.append(dict_button)

    true_start+=traces



fig.layout.update(

    template=layout_custom,

    updatemenus=[

        go.layout.Updatemenu(

            active=0,

            x=1.05, y=1.10,

            buttons=list_buttons)

    ])



fig.layout.update(title='Daily amounts by COVID/non COVID')



fig['layout']['yaxis1'].update(hoverformat =',d', tickformat=',d')

fig['layout']['yaxis2'].update(hoverformat =',d', tickformat=',d')



iplot(fig)
kernels_votes = kernels_2020.loc[kernels_2020['CreationDate'] >= '2020-03-01',['Title', 'TotalVotes']]

kernels_votes['if_covid'] = [('covid' in str(string).lower()) or ('corona' in str(string).lower()) 

                             for string in kernels_votes['Title']]

kernels_votes.drop(['Title'], axis=1, inplace=True)
def quantile25(x): 

    return np.quantile(x, q=0.25)



def quantile75(x): 

    return np.quantile(x, q=0.75)    

    

kernels_votes.groupby(['if_covid']).agg([quantile25, np.mean, np.median, quantile75, np.max])
X = kernels_votes['if_covid']

y = kernels_votes['TotalVotes']



slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)



print('Average votes on kernel if no COVID is {}'.format(intercept))

print('Average votes on kernel if COVID is {} (or an increase of {})'.format(slope+intercept, slope))

print('p_value for this difference is way lower than 0.05 ({}), so it is significant'.format(p_value))