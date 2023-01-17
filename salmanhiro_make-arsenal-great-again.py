# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/fifa19/data.csv')
arsenal_data = df.loc[df['Club'] == 'Arsenal']
arsenal_data
arsenal_data.columns
def extract_value_from(Value):

    out = Value.replace('€', '')

    if 'M' in out:

        out = float(out.replace('M', ''))*1000000

    elif 'K' in Value:

        out = float(out.replace('K', ''))*1000

    return float(out)
arsenal_data['Value'] = arsenal_data['Value'].apply(lambda x: extract_value_from(x))
plt.figure(figsize = (10,10))

sns.distplot(arsenal_data['Value'], kde = False, bins = 10)
print('Squad value: ', sum(arsenal_data['Value']),' €')
arsenal_data.rename(columns={'Club Logo':'Club_Logo'}, inplace=True)
import matplotlib.pyplot as plt
from IPython.display import Image, HTML



def path_to_image_html(path):

    return '<img src="'+ path + '" style=max-height:124px;"/>'
HTML(arsenal_data.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))
GK = arsenal_data[arsenal_data['Position'] == 'GK']
HTML(GK.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))
arsenal_data['Position'].value_counts()
CB = arsenal_data[(arsenal_data['Position'] == 'CB') | (arsenal_data['Position'] == 'RCB')]



HTML(CB.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))
RB = arsenal_data[(arsenal_data['Position'] == 'RB') | (arsenal_data['Position'] == 'RWB')]



HTML(RB.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))
RB = arsenal_data[(arsenal_data['Position'] == 'LB') | (arsenal_data['Position'] == 'LWB')]



HTML(RB.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))
DM = arsenal_data[(arsenal_data['Position'] == 'CDM') | (arsenal_data['Position'] == 'LDM') | (arsenal_data['Position'] == 'RDM')]



HTML(DM.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))
CM = arsenal_data[(arsenal_data['Position'] == 'CM') | (arsenal_data['Position'] == 'LCM') | (arsenal_data['Position'] == 'RCM')]



HTML(CM.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))
AM = arsenal_data[(arsenal_data['Position'] == 'CAM') | (arsenal_data['Position'] == 'LAM') | (arsenal_data['Position'] == 'RAM')]



HTML(AM.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))
WF = arsenal_data[(arsenal_data['Position'] == 'LM') | (arsenal_data['Position'] == 'RM')| (arsenal_data['Position'] == 'RW')| (arsenal_data['Position'] == 'LW')]



HTML(WF.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))
CF = arsenal_data[(arsenal_data['Position'] == 'CF') | (arsenal_data['Position'] == 'ST')]



HTML(CF.sort_values('Overall', ascending = False).to_html(escape=False ,

                                                          formatters=dict(Photo=path_to_image_html,Flag=path_to_image_html,

                                                                         Club_Logo=path_to_image_html)))