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
%%time

import pandas as pd

import glob





path = r'/kaggle/input/hira-inoutpatient-statistics' # use your path

all_files = glob.glob(path + "/*.csv")



li = []



for filename in all_files:

    frame = pd.read_csv(filename, index_col=None, header=0, encoding='euc-kr')

    li.append(frame)



df = pd.concat(li, axis=0, ignore_index=True)

df.drop(['number_of_payments', 'sum_of_payments', 'paid_by_patients'], axis=1, inplace=True)
df.head()
%%time



import ipywidgets as widgets

from ipywidgets import interact, interact_manual

from ipywidgets import Button, HBox, VBox

import matplotlib.pyplot as plt

from matplotlib import font_manager as fm, rcParams





def pick_data(KCD, gender, year):

    

    global df

    

    # to set the initial conditions

    if len(KCD)==0:

        KCD = ['Z99']

    if len(gender)==0:

        gender = ['male', 'female']

    if len(year)==0:

        year = ['2018']



    df1 = df[(df['KCD3'].isin(KCD)) & (df['year'].isin(year)) & (df['gender'].isin(gender))].copy()



    #agegroup order customized

    df1['agegroup'] = pd.Categorical(df1['agegroup'], ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29',

                                                      '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',

                                                      '60-64', '65-69', '70-74', '75-79', '80-84', '85+'])



    df1 = df1.groupby(['gender', 'agegroup', 'year'])['number_of_patients'].sum().unstack(['gender', 'year'])



    df1.fillna(0, inplace=True)

    #df1 = df1.stack()





    df3 = df1



    #customized font setting for Korean language

    font = fm.FontProperties(fname="/kaggle/input/custom-font/NanumBarunGothic.ttf")

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))



    #x label rotation

    ax.tick_params(axis='x', labelrotation=45)



    #plotting

    ax.plot(df3.index.tolist(), df3.values)



    #legend outsize 

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels=df3.columns.tolist(), prop=font)



    #axes label font customization

    for tick in ax.xaxis.get_major_ticks():

        tick.label.set_fontproperties(font)



    for tick in ax.yaxis.get_major_ticks():

        tick.label.set_fontproperties(font)



    df3.head()





#widget settings

KCD = widgets.SelectMultiple(options=df['KCD3'].unique().tolist(), rows=10, description='KCD code')

gender = widgets.SelectMultiple(options=['male', 'female'], rows=3, description='gender')

year = widgets.SelectMultiple(options=sorted(df['year'].unique().tolist()), rows=6, description='year')

ui = HBox([KCD, gender, year])

out = widgets.interactive_output(pick_data, {'KCD':KCD, 'gender':gender, 'year':year})

display(ui, out)