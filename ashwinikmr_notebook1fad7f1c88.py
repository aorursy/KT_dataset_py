# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#create datasets for each subset and then see and visualize in silos



pd_air_reserve = pd.read_csv('../input/air_reserve.csv')

pd_air_store_info = pd.read_csv('../input/air_store_info.csv')

pd_air_visit_data = pd.read_csv('../input/air_visit_data.csv')

pd_date_info = pd.read_csv('../input/date_info.csv')



print (pd_air_reserve.tail())

print (pd_air_store_info.tail())

print (pd_air_visit_data.tail())

print (pd_date_info.head())

#

pd_hpg_reserve = pd.read_csv('../input/hpg_reserve.csv')

pd_hpg_store_info = pd.read_csv('../input/hpg_store_info.csv')

pd_store_id_relation = pd.read_csv('../input/store_id_relation.csv')

pd_sample_submission = pd.read_csv('../input/sample_submission.csv')





print (pd_hpg_reserve.head())

print (pd_hpg_store_info.head())

print (pd_store_id_relation.head())

print (pd_sample_submission.head())







# code to split data in columns to their components - need to check the number of distinct restaurants and for each restaurant the number of records present 



pd_air_reserve.describe().transpose()



#pd_air_reserve = pd_air_reserve.add(pd_air_reserve['visit_datetime'].date)



#pd_air_visit_data = 







# code to visualize





# code to merge all data 



print (pd_air_reserve.describe(include='all').transpose())



from datetime import datetime



pd_air_reserve['new_visit_date'] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date() for d in pd_air_reserve['visit_datetime']]

pd_air_reserve['new_visit_time'] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S').time() for d in pd_air_reserve['visit_datetime']]



pd_air_reserve['new_reserve_date'] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date() for d in pd_air_reserve['reserve_datetime']]

pd_air_reserve['new_reserve_time'] = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S').time() for d in pd_air_reserve['reserve_datetime']]



print (pd_air_reserve.describe(include='all').transpose())



#summarize the information

pd_air_reserve.groupby(['air_store_id']['new_visit_date']).agg({'reserve_visitors':sum,

                                                               'new_visit_time':[min,max,mean]})



pd_air_reserve_summ = pd_air_reserve.groupby(['air_store_id','new_visit_date']).agg({'reserve_visitors':sum,

                                                               'new_visit_time':[min,max]})



print (pd_air_reserve_summ.describe(include='all').transpose())



pd_air_reserve_summ.head()

print(pd_air_reserve_summ.query('air_store_id=="air_00a91d42b08b08d9"'))
temp = pd_air_reserve_summ.query('air_store_id=="air_00a91d42b08b08d9"')

x = temp['new_visit_date']
import plotly.plotly as py

import plotly.graph_objs as go



pd_air_reserve_summ.index



x = pd_air_reserve_summ['air_00a91d42b08b08d9']

y = pd_air_reserve_summ['air_store_id'=='air_00a91d42b08b08d9','reserve_visitors','sum']



trace0 = go.Scatter(

    x ,

    y ,

    name = 'High 2014',

    line = dict(

        color = ('rgb(205, 12, 24)'),

        width = 4)

)



data= [trace0]

layout = dict(title = 'Average High and Low Temperatures in New York',

              xaxis = dict(title = 'Month'),

              yaxis = dict(title = 'Temperature (degrees F)'),

              )



fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-line')