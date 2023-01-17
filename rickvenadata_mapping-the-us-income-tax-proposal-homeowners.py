# extremely helpful resources (many huge thanks!)

# https://www.npr.org/2017/11/02/561639579/chart-how-the-tax-overhaul-would-affect-you

# https://www.kaggle.com/ievgenvp/plotly-vs-matplotlib-for-choropleth-maps



# data is a subset from https://www.kaggle.com/census/2013-american-community-survey

# It's huge, so I cut this down to just the columns needed in order to not bog down my kernel

# see https://www.kaggle.com/rickvenadata/extract-acs-personal-income-data



# also check out the blog article at: https://venadata.com/2017/11/13/putting-us-tax-reform-on-the-map/



# assuming single filers earning only wage income, no dependents, but do own their home

# this is only slightly more complicted than the most basic tax filing

# there is a 30% deduction allocated to mortgage and property tax in this example

# and of course the state income or sales tax deductions as applicable



# for further reading:

# https://taxfoundation.org/state-individual-income-tax-rates-brackets-2017/

# https://www.usatoday.com/story/money/personalfinance/2017/04/16/comparing-average-property-taxes-all-50-states-and-dc/100314754/

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

wages_median["DEDUCTED"] = wages_median["WAGP"] * 0.7

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
state_tax = {"AL": 1031, 'AK': 0, "AZ": 587, "AR": 723,

              "CA": 570, "CO": 1389, "CT": 745, "DE": 1092,

              "FL": 324, "GA": 1190, "HI": 1555,

              "ID": 830, "IL": 1076, "IN": 820, "IA": 1165,

              "KS": 788, "KY": 1122, "LA": 670, "ME": 658,

              "MD": 1478, "MA": 1581, "MI": 978, "MN": 1049,

              "MS": 635, "MO": 886, "MT": 688, "NE": 720,

              "NV": 472, "NH": 1380, "NJ": 525, "NM": 368,

              "NY": 1403, "NC": 1004, "ND": 450, "OH": 452,

              "OK": 694, "OR": 1968, "PA": 921, "RI": 665,

              "SC": 524, "SD": 225, "TN": 488, "TX": 462,

              "UT": 656, "VT": 678, "VA": 1357, "WA": 600,

              "WV": 900, "WI": 787, "WY": 280}
wages_median['ST_TAX'] = wages_median['ST']

wages_median['ST_TAX'].replace(state_tax, inplace = True)

wages_median['DEDUCTED'] = wages_median['DEDUCTED'] - wages_median['ST_TAX']

wages_median.head()
tax_curr = wages_median.copy()

tax_curr['DEDUCTED'] = tax_curr['DEDUCTED'].apply(lambda x: x-4150)

tax_curr['DEDUCTED'] = tax_curr['DEDUCTED'].apply(lambda x: x*0.1 if x <= 9325 else (x-9325)*0.15+932.5)

tax_curr.head()
tax_new = wages_median.copy()

tax_new['WAGP'] = tax_new['WAGP'].apply(lambda x: x-12000)

tax_new['WAGP'] = tax_new['WAGP'].apply(lambda x: x*0.1 if x <= 9525 else (x-9325)*0.12+952.5)

tax_new.head()
tax_diff = wages_median.copy()



tax_diff['WAGPNEW'] = tax_diff['WAGP']

tax_diff['WAGP'] = tax_diff['DEDUCTED'].apply(lambda x: x-4150)

tax_diff['WAGP'] = tax_diff['WAGP'].apply(lambda x: x*0.1 if x <= 9325 else (x-9325)*0.15+932.5)



tax_diff['WAGPNEW'] = tax_diff['WAGPNEW'].apply(lambda x: x-12000)

tax_diff['WAGPNEW'] = tax_diff['WAGPNEW'].apply(lambda x: x*0.1 if x <= 9525 else (x-9525)*0.12+952.5)



tax_diff['DIFF'] = tax_diff['WAGP'] - tax_diff['WAGPNEW']

tax_diff['PCT'] = tax_diff['DIFF'] / tax_diff['WAGP'] * 100



tax_diff.head()
scl = [[0, 'rgb(178,10,28)'], [0.35, 'rgb(230,145,90)'],

        [0.5, 'rgb(220,170,132)'], [0.6, 'rgb(190,190,190)'],

        [0.7, 'rgb(106,137,247)'], [1, 'rgb(5,10,172)']]



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

            title = "Tax Savings ($)")

        ) ]



layout = dict(

        title = 'Homeowner Tax Savings',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

iplot( fig, filename='tax1-cloropleth-map')
scl = [[0, 'rgb(178,10,28)'], [0.35, 'rgb(230,145,90)'],

        [0.5, 'rgb(220,170,132)'], [0.6, 'rgb(190,190,190)'],

        [0.7, 'rgb(106,137,247)'], [1, 'rgb(5,10,172)']]

        

data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = False,

        locations = tax_diff['ST'],

        z = tax_diff['PCT'].astype(int),

        locationmode = 'USA-states',

        text = tax_diff['ST'],

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Tax Savings (%)")

        ) ]



layout = dict(

        title = 'Percentage Tax Savings Under New Policy',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )

    

fig = dict( data=data, layout=layout )

iplot( fig, filename='tax2-cloropleth-map')