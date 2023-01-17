import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dataset =  pd.read_csv("../input/nyc-jobs.csv")

dataset.head(3)
#Proportion of External/Internal Jobs by Agency

postingtype_rate = dataset.pivot_table(values = 'Job ID', index='Agency', columns ='Posting Type', aggfunc='count')

postingtype_rate['Racional'] = postingtype_rate['External']/(postingtype_rate['External'] + postingtype_rate['Internal'])

postingtype_rate.sort_values(by='Racional', ascending = False).head(5)
#Mean of minimum wage by business title (top 10)

pd.options.display.float_format = '{:,.2f}'.format

wages_mean = dataset.pivot_table(values = 'Salary Range From', index='Business Title', aggfunc='mean')

wages_mean.rename(columns={'Salary Range From': 'Mean salary'}, inplace=True)

wages_mean['Mean salary monthly'] = wages_mean['Mean salary']/12

wages_mean.sort_values(by = 'Mean salary', ascending=False).head(10)
#Division Units jobs by agencies

dataset.pivot_table(values='Job ID', index = 'Agency', columns = 'Division/Work Unit', aggfunc='count', fill_value=0, margins=True).sort_values(by='All', ascending=False)