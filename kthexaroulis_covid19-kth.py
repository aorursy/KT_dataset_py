# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Load csv files to variables 



def read_data():

    print('Reading files...')

    INPUT_DIR = '/kaggle/input/covid19'

    covid = pd.read_excel(f'{INPUT_DIR}/COVID-19-geographic-disbtribution-worldwide-2020-03-21.xlsx')

     



    print('covid has {} rows and {} columns'.format(covid.shape[0], covid.shape[1]))

    return covid



        

covid  = read_data()
covid
covid=covid.sort_values(by='DateRep')

covidtmp=covid[covid['Cases']>0].drop_duplicates(subset=['Countries and territories'], keep='first')

covidtmp['dayN']=1
temp2 =pd.merge(covid,covidtmp[['DateRep','Countries and territories','dayN']],how='left',on=['DateRep','Countries and territories'])  
for index, row in temp2[['Countries and territories']].drop_duplicates().iterrows():

    dayN=0

    dayD=0

    for index , row in temp2[temp2['Countries and territories']==row['Countries and territories']].iterrows():

        if (temp2.loc[index,'Cases']==0 and dayN==0 ) :

            temp2.set_value(index,'dayN',0)

        elif (dayN>0 ) :

            dayN=dayN+1

            temp2.set_value(index,'dayN',dayN)

        elif (temp2.loc[index,'Cases']>0 and dayN==0) :

            dayN=1

temp2[(temp2['dayN']>0) & (temp2['Countries and territories']=="Italy")  ]
import plotly.express as px



fig = px.line(temp2[(temp2['Countries and territories'].isin(["Italy","Greece","Spain"]))] , x="dayN", y="Cases", line_group="Countries and territories", color="Countries and territories", title='Virus', hover_name="Countries and territories")

fig.show()
import plotly.express as px



fig = px.line(temp2, x="DateRep", y="Cases", line_group="Countries and territories", color="Countries and territories", title='Virus', hover_name="Countries and territories")

fig.show()