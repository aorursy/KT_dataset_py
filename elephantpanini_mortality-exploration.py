# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib as mp

from __future__ import division

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
mortality = pd.read_csv('../input/mort.csv')
mortality_drop = mortality.drop([

        'Mortality Rate, 1980* (Min)',

        'Mortality Rate, 1980* (Max)',

        'Mortality Rate, 1985* (Min)',

        'Mortality Rate, 1985* (Max)',

        'Mortality Rate, 1990* (Min)',

        'Mortality Rate, 1990* (Max)',

        'Mortality Rate, 1995* (Min)',

        'Mortality Rate, 1995* (Max)',

        'Mortality Rate, 2000* (Min)',

        'Mortality Rate, 2000* (Max)',

        'Mortality Rate, 2005* (Min)',

        'Mortality Rate, 2005* (Max)',

        'Mortality Rate, 2010* (Min)',

        'Mortality Rate, 2010* (Max)',

        'Mortality Rate, 2014* (Max)',

        'Mortality Rate, 2014* (Min)',

        '% Change in Mortality Rate, 1980-2014',

        '% Change in Mortality Rate, 1980-2014 (Min)',

        '% Change in Mortality Rate, 1980-2014 (Max)'

    ],axis=1)
# Review aggregation at a state-level



mortality_state_overall = (

    mortality_drop.loc[

        mortality_drop[

            'Location'

        ].isin(

            [

                'Alabama',

                'Alaska', 

                'Arizona', 

                'Arkansas', 

                'California', 

                'Colorado', 

                'Connecticut', 

                'Delaware', 

                'Florida', 

                'Georgia', 

                'Hawaii', 

                'Idaho', 

                'Illinois', 

                'Indiana',

                'Iowa', 

                'Kansas', 

                'Kentucky', 

                'Louisiana',

                'Maine', 

                'Maryland', 

                'Massachusetts', 

                'Michigan', 

                'Minnesota', 

                'Mississippi', 

                'Missouri', 

                'Montana Nebraska',

                'Nevada', 

                'New Hampshire', 

                'New Jersey', 

                'New Mexico', 

                'New York', 

                'North Carolina', 

                'North Dakota', 

                'Ohio', 

                'Oklahoma', 

                'Oregon', 

                'Pennsylvania',

                'Rhode Island', 

                'South Carolina', 

                'South Dakota', 

                'Tennessee', 

                'Texas', 

                'Utah', 

                'Vermont', 

                'Virginia', 

                'Washington', 

                'West Virginia', 

                'Wisconsin', 

                'Wyomin',

            ]

        )

    ]

)
# Change names of columns



mortality_state_overall.columns = [

    'state',

    'fips',

    'category',

    '1980',

    '1985',

    '1990',

    '1995',

    '2000',

    '2005',

    '2010',

    '2014'

]
mort_st_cat_yr = mortality_state_overall.drop('fips',axis=1)
mort_st_cat_yr[mort_st_cat_yr['category']=='Neonatal disorders'].state
mort_st_cat_yr_t = mort_st_cat_yr[mort_st_cat_yr['category']=='Neonatal disorders'].T
mort_neonat = mort_st_cat_yr_t.drop('category').reset_index()
mort_neonat_drop = mort_neonat.drop(0)
mort_neonat_drop
mort_neonat_drop.columns = [

    'year',

    'Alabama',

    'Alaska',

    'Arizona',

    'Arkansas',

    'California',

    'Colorado',

    'Connecticut',

    'Delaware',

    'Florida',

    'Georgia',

    'Hawaii',

    'Idaho',

    'Illinois',

    'Indiana',

    'Iowa',

    'Kansas',

    'Kentucky',

    'Louisiana',

    'Maine',

    'Maryland',

    'Massachusetts',

    'Michigan',

    'Minnesota',

    'Mississippi',

    'Missouri',

    'Nevada',

    'New Hampshire',

    'New Jersey',

    'New Mexico',

    'New York',

    'North Carolina',

    'North Dakota',

    'Ohio',

    'Oklahoma',

    'Oregon',

    'Pennsylvania',

    'Rhode Island',

    'South Carolina',

    'South Dakota',

    'Tennessee',

    'Texas',

    'Utah',

    'Vermont',

    'Virginia',

    'Washington',

    'West Virginia',

    'Wisconsin'

]
mort_neonat_drop
mort_neonat_drop.plot(

    kind='bar',

    x='year',

    figsize=(20,20),

    title='Deaths due to Neonatal disorders by State'

)