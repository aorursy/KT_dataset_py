# extremely helpful resources (many huge thanks!)

# https://www.npr.org/2017/11/02/561639579/chart-how-the-tax-overhaul-would-affect-you

# https://www.kaggle.com/ievgenvp/plotly-vs-matplotlib-for-choropleth-maps



# data is a subset from https://www.kaggle.com/census/2013-american-community-survey

# It's huge, so I cut this down to just the columns needed in order to not bog down my kernel

# see https://www.kaggle.com/rickvenadata/extract-acs-personal-income-data



# also check out the blog article at: https://venadata.com/2017/11/13/putting-us-tax-reform-on-the-map/



# assuming single filers earning only wage income, no dependents, and rent (not own) their home

# this is super simple because:

# 1) None of these median incomes result in high enough state income taxes to trigger itemizing

# 2) no mortgage interest or property tax

# 3) single and no dependents earning only W-2 wages is the least complicated, many filers use a short form



# for further reading toward performing more complex analysis accounting for itemized deductions

# this would apply to high earners and/or homeowners where itemizing comes into play

# https://taxfoundation.org/state-individual-income-tax-rates-brackets-2017/

# https://www.usatoday.com/story/money/personalfinance/2017/04/16/comparing-average-property-taxes-all-50-states-and-dc/100314754/

# also deeper analysis is needed to cover families with children at various income levels

# even the single filer example gets more complex in higher income ranges

# The US tax system is highly complex!
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.graph_objs as go



from IPython.display import HTML

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
person_wages = pd.read_csv("../input/acs2013_person_wages.csv", dtype = {'ST': str, 'PUMA': str})

person_wages.head()
# Singles only! :-)

person_wages = person_wages.loc[person_wages['MAR'].isin([5])]

# works a "normal" job (not self-employed, unemployed, etc.)

person_wages = person_wages.loc[person_wages['COW'].isin(["1","2","3","4","5"])]

# set minimum wage income above standard deduction and personal exemption amount

person_wages = person_wages[(person_wages['WAGP'] >= 10650.0)]

# at least 18 years old and under 65

person_wages = person_wages[(person_wages['AGEP'] >= 18) & (person_wages['AGEP'] < 65)]

# exclude DC because it is an outlier

person_wages = person_wages.loc[~person_wages['ST'].isin(["11"])]

person_wages.head(10)
person_wages.shape
wages_st = person_wages[['ST','WAGP']]

wages_median = wages_st.groupby('ST', as_index=False).median()

wages_median.head()
states_dict = {'1': "AL", '2': "AK", "4": "AZ", "5": "AR",

              "6": "CA", "8": "CO", "9": "CT", "10": "DE",

              "11": "DC", "12": "FL", "13": "GA", "15": "HI",

              "16": "ID", "17": "IL", "18": "IN", "19": "IA",

              "20": "KS", "21": "KY", "22": "LA", "23": "ME",

              "24": "MD", "25": "MA", "26": "MI", "27": "MN",

              "28": "MS", "29": "MO", "30": "MT", "31": "NE",

              "32": "NV", "33": "NH", "34": "NJ", "35": "NM",

              "36": "NY", "37": "NC", "38": "ND", "39": "OH",

              "40": "OK", "41": "OR", "42": "PA", "44": "RI",

              "45": "SC", "46": "SD", "47": "TN", "48": "TX",

              "49": "UT", "50": "VT", "51": "VA", "53": "WA",

              "54": "WV", "55": "WI", "56": "WY"}
wages_median['ST'].replace(states_dict, inplace = True)

wages_median.head()
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\

            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = wages_median['ST'],

        z = wages_median['WAGP'].astype(float),

        locationmode = 'USA-states',

        text = wages_median['ST'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Wage (Median)")

        ) ]



layout = dict(

        title = 'Median Wages',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

iplot( fig, filename='d3-cloropleth-map')

tax_curr = wages_median.copy()

tax_curr['WAGP'] = tax_curr['WAGP'].apply(lambda x: x-10650)

tax_curr['WAGP'] = tax_curr['WAGP'].apply(lambda x: x*0.1 if x <= 9325 else (x-9325)*0.15+932.5)

tax_curr.head()
tax_new = wages_median.copy()

tax_new['WAGP'] = tax_new['WAGP'].apply(lambda x: x-12000)

tax_new['WAGP'] = tax_new['WAGP'].apply(lambda x: x*0.1 if x <= 9525 else (x-9525)*0.12+952.5)

tax_new.head()
tax_diff = wages_median.copy()



tax_diff['WAGPNEW'] = tax_diff['WAGP']

tax_diff['WAGP'] = tax_diff['WAGP'].apply(lambda x: x-10650)

tax_diff['WAGP'] = tax_diff['WAGP'].apply(lambda x: x*0.1 if x <= 9325 else (x-9325)*0.15+932.5)



tax_diff['WAGPNEW'] = tax_diff['WAGPNEW'].apply(lambda x: x-12000)

tax_diff['WAGPNEW'] = tax_diff['WAGPNEW'].apply(lambda x: x*0.1 if x <= 9525 else (x-9525)*0.12+952.5)



tax_diff['DIFF'] = tax_diff['WAGP'] - tax_diff['WAGPNEW']

tax_diff['PCT'] = tax_diff['DIFF'] / tax_diff['WAGP'] * 100



tax_diff.head()
scl = [[0, 'rgb(247,252,245)'], [0.125, 'rgb(229,245,224)'],

        [0.25, 'rgb(199,233,192)'], [0.375, 'rgb(161,217,155)'],

        [0.5, 'rgb(116,196,118)'], [0.625, 'rgb(65,171,93)'],

        [0.75, 'rgb(35,139,69)'], [0.875, 'rgb(0,109,44)'],

        [1, 'rgb(0,68,27)']]



data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = tax_diff['ST'],

        z = tax_diff['DIFF'].astype(float),

        locationmode = 'USA-states',

        text = tax_diff['ST'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Tax Savings")

        ) ]



layout = dict(

        title = 'Tax Savings Under New Policy',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

iplot( fig, filename='tax1-cloropleth-map')