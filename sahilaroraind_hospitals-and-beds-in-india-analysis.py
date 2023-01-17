import plotly.express as px
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
state_wise = pd.read_csv('../input/hospitals-and-beds-in-india/Hospitals_and_Beds_statewise.csv')

mod = pd.read_csv('../input/hospitals-and-beds-in-india/Hospitals and Beds maintained by Ministry of Defence.csv')

railways = pd.read_csv('../input/hospitals-and-beds-in-india/Hospitals and beds maintained by Railways.csv')

gov = pd.read_csv('../input/hospitals-and-beds-in-india/Number of Government Hospitals and Beds in Rural and Urban Areas .csv')

ayush = pd.read_csv('../input/hospitals-and-beds-in-india/AYUSHHospitals.csv')
state_wise.head(37).reset_index()
state_wise.rename(columns={'Unnamed: 0': 'states'}, inplace=True)
state_wise.shape
state_wise.isna().sum()
state_wise.fillna(0,axis =0, inplace=True)
state_wise.info(memory_usage='deep')
all_india_beds_PHC = state_wise.loc[36,["PHC", 'CHC', 'SDH', 'DH']]

print(all_india_beds_PHC)

# state_wise.PHC = state_wise.PHC.astype(int)
state_wise.drop([36], inplace=True)
data_type_list = {

    "PHC"   : int,

    "CHC"   : int, 

    "SDH"   : int,

    "DH"    : int,

    "Total" : int

}



state_wise = state_wise.astype(data_type_list)
state_wise.sort_values(by='PHC', ascending=True, inplace=True)
fig = px.bar(state_wise, x=state_wise['PHC'], y=state_wise['states'], orientation='h')

fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide', autosize=False,

    xaxis_title="Primary Health Centers",

    yaxis_title="states",

    width=800,

    height=1200,

    title_text = 'Total Primary Health Centers in india '+all_india_beds_PHC[0])



fig.show()
fig = px.bar(state_wise.sort_values(by="CHC", ascending=True,inplace=True), x=state_wise['CHC'], y=state_wise['states'], orientation='h')

fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide', autosize=False,

    xaxis_title="Community Health Centers(CHCs)",

    yaxis_title="states",

    width=800,

    height=1200,

    title_text = 'Total Community Health Centers(CHCs) in india '+all_india_beds_PHC[1])

fig.show()
fig = px.bar(state_wise.sort_values(by="SDH", ascending=True,inplace=True), x=state_wise['SDH'], y=state_wise['states'], orientation='h')

fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide', autosize=False,

    xaxis_title="Sub-District/Divisional Hospitals(SDHs)",

    yaxis_title="states",

    width=800,

    height=1200,

    title_text = 'Total Sub-District/Divisional Hospitals(SDHs) in india '+all_india_beds_PHC[2])

fig.show()
fig = px.bar(state_wise.sort_values(by="DH", ascending=True,inplace=True), x=state_wise['DH'], y=state_wise['states'], orientation='h')

fig.update_layout(uniformtext_minsize=5, uniformtext_mode='hide', autosize=False,

    width=800,

    height=1200,

    xaxis_title="District Hospitals(DHs)",

    yaxis_title="states",

    title_text = 'Total District Hospitals(DHs) in india '+all_india_beds_PHC[3])

fig.show()
fig = px.bar(state_wise.sort_values(by="Total", ascending=True,inplace=True), 

             x=state_wise['Total'], 

             y=state_wise['states'], 

             orientation='h')

fig.update_layout(uniformtext_minsize=4, uniformtext_mode='hide', autosize=False,

    xaxis_title="Total PHC,CHC,SDH,DH",

    yaxis_title="states",

    width=800,

    height=1200,

    title_text = 'Total PHC,CHC,SDH,DH in india ')

fig.show()
