import pandas as pd



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
!pip3 install bar_chart_race

import bar_chart_race as bcr
%time population = pd.read_csv('../input/countries-population-from-1955-to-2020/Countries Population from 1995 to 2020.csv')
pop = population.drop(['Yearly % Change', 'Yearly Change', 'Migrants (net)', 'Median Age', 'Fertility Rate', 'Density (P/Km²)',

                      'Urban Pop %', 'Urban Population','Country\'s Share of World Pop %', 'World Population', 'Country Global Rank' ], axis = 1)
df = pop.pivot_table('Population',['Year'], 'Country')
df.sort_values(list(df.columns),inplace=True)

df = df.sort_index()
df
bcr.bar_chart_race(

    df=df,

    #filename='Population_Bar_Chart_Race.mp4',

    filename=None,

    orientation='h',

    sort='desc',

    n_bars=10,

    fixed_order=False,

    fixed_max=True,

    steps_per_period=10,

    interpolate_period=False,

    label_bars=True,

    bar_size=.90,

    period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},

    period_summary_func=lambda v, r: {'x': .99, 'y': .18,

                                      's': f'Population: {v.nlargest(39).sum():,.0f}',

                                      'ha': 'right', 'size': 8, 'family': 'Courier New'},

    period_length=500,

    figsize=(6.5,5),

    dpi=144,

    cmap='dark12',

    title='Population by Country.',

    title_size='',

    bar_label_size=7,

    tick_label_size=5,

    shared_fontdict={'family' : 'Helvetica','color' : '.1'},

    scale='linear',

    writer=None,

    fig=None,

    bar_kwargs={'alpha': .7},

    filter_column_colors=True)