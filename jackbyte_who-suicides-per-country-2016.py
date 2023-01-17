# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))
df = pd.read_csv('../input/who_suicide_statistics.csv')

# Any results you write to the current directory are saved as output.
x = df[df.year == 2016]
s_bc_2016 = x[['country','suicides_no']].groupby(x.country).suicides_no.sum()
s_bc_2016 = pd.DataFrame(s_bc_2016)
s_bc_2016['c']=s_bc_2016.index
a = s_bc_2016.c.values
# there is probably a clean way to get the 3digit iso codes... but yeah this works, too
b = ['AIA','ARM','AUT','HRV','CYP','CZE','GRD','HUN','ISL','ISR','LTE','MUS',
 'MNG','MSR','NLD', 'PSE', 'QAT', 'MDA', '', 'ROU', 'SWE', 'TJK', 'THA', 'GBR', 'USA' ]
s_bc_2016['code'] = b
#s_bc_2016
# Couldn't figure out how to show all the countries...
# so i found a messy way to include all the countries in my df. Yeah I know
cs = ['AFG','ALB','DZA','ASM','AND','AGO','AIA','ATG','ARG','ARM','ABW','AUS','AUT','AZE','BHM','BHR','BGD','BRB','BLR','BEL','BLZ','BEN','BMU','BTN','BOL','BIH','BWA','BRA','VGB','BRN','BGR','BFA','MMR','BDI','CPV','KHM','CMR','CAN','CYM','CAF','TCD','CHL','CHN','COL','COM','COD','COG','COK','CRI','CIV','HRV','CUB','CUW','CYP','CZE','DNK','DJI','DMA','DOM','ECU','EGY','SLV','GNQ','ERI','EST','ETH','FLK','FRO','FJI','FIN','FRA','PYF','GAB','GMB','GEO','DEU','GHA','GIB','GRC','GRL','GRD','GUM','GTM','GGY','GNB','GIN','GUY','HTI','HND','HKG','HUN','ISL','IND','IDN','IRN','IRQ','IRL','IMN','ISR','ITA','JAM','JPN','JEY','JOR','KAZ','KEN','KIR','PRK','KOR','KSV','KWT','KGZ','LAO','LVA','LBN','LSO','LBR','LBY','LIE','LTU','LUX','MAC','MKD','MDG','MWI','MYS','MDV','MLI','MLT','MHL','MRT','MUS','MEX','FSM','MDA','MCO','MNG','MNE','MAR','MOZ','NAM','NPL','NLD','NCL','NZL','NIC','NGA','NER','NIU','MNP','NOR','OMN','PAK','PLW','PAN','PNG','PRY','PER','PHL','POL','PRT','PRI','QAT','ROU','RUS','RWA','KNA','LCA','MAF','SPM','VCT','WSM','SMR','STP','SAU','SEN','SRB','SYC','SLE','SGP','SXM','SVK','SVN','SLB','SOM','ZAF','SSD','ESP','LKA','SDN','SUR','SWZ','SWE','CHE','SYR','TWN','TJK','TZA','THA','TLS','TGO','TON','TTO','TUN','TUR','TKM','TUV','UGA','UKR','ARE','GBR','USA','URY','UZB','VUT','VEN','VNM','VGB','WBG','YEM','ZMB','ZWE']
csf = []
for c in cs:
    if (s_bc_2016['code']==c).any():
        pass
    else:
        csf.append(c)

z = pd.DataFrame(csf, columns=['code'])
z['suicide_no'] = 0
z['c'] = ''

x = pd.concat((s_bc_2016,z),sort=False)
#x.shape
# Combination of Rachels code from here https://www.kaggle.com/rtatman/dashboarding-with-notebooks-day-2-python
# And stuff from here https://plot.ly/python/choropleth-maps/

import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd

# these two lines are what allow your code to show up in a notebook!
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# specify what we want our map to look like
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = x['code'],
        z = x['suicides_no'],
        text = x['suicides_no']
        #locationmode = 'USA-states'
       ) ]

# chart information
layout = dict(
    title = '2016 Global Suicides',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'mercator'
        )
    )
)   
# actually show our figure
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-world-map' )