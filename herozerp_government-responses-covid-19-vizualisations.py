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



!pip install chart_studio
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")

plt.style.use('ggplot')



import cufflinks as cf

import plotly.express as px

import plotly.offline as py

from plotly.offline import plot

import plotly.graph_objs as go
url = "../input/government-responses-in-covid19-coronanet/coronanet_release.csv"

df = pd.read_csv(url)



df.head()
#Dropping some useless columns

df = df.drop(["record_id", "policy_id", "ISO_A3", "target_who_what", "link", "recorded_date", "domestic_policy"], axis=1)

df.info()
#How many null values ?

df.isnull().sum()
#Dropping columns with more than 10% of null values

for col in df.columns:

    if df[col].isnull().mean()*100 > 10:

        df = df.drop([col], axis=1)

        

df.isnull().sum()
df.head()
#Graph : Correct types by count

fig = px.bar(df["correct_type"].value_counts(), orientation="v", color=df["correct_type"].value_counts(), color_continuous_scale=px.colors.sequential.Plasma, 

             log_x=False, labels={'value':'Count', 

                                'index':'Correcty type',

                                 'color':'None'

                                })



fig.update_layout(

    font_color="black",

    title_font_color="red",

    legend_title_font_color="green",

    title_text="Correcty type by count"

)



fig.show()
#Graph : Entry types by count

fig = px.bar(df["entry_type"].value_counts(), orientation="v", color=df["entry_type"].value_counts(), color_continuous_scale=px.colors.sequential.Plasma, 

             log_x=False, labels={'value':'Count', 

                                'index':'Entry type',

                                 'color':'None'

                                })



fig.update_layout(

    font_color="black",

    title_font_color="red",

    legend_title_font_color="green",

    title_text="Entry type by count"

)



fig.show()
#Graph : Ten first countries by count

fig = px.bar(df["country"].value_counts()[0:20], orientation="v", color=df["country"].value_counts()[0:20], color_continuous_scale=px.colors.sequential.Plasma, 

             log_x=False, labels={'value':'Count', 

                                'index':'Country',

                                 'color':'None'

                                })



fig.update_layout(

    font_color="black",

    title_font_color="red",

    legend_title_font_color="green",

    title_text="Twenty first country by count"

)



fig.show()
#Graph : Twenty last countries by count

fig = px.bar(df["country"].value_counts()[-21:-1], orientation="v", color=df["country"].value_counts()[-21:-1], color_continuous_scale=px.colors.sequential.Plasma, 

             log_x=False, labels={'value':'Count', 

                                'index':'Country',

                                 'color':'None'

                                })



fig.update_layout(

    font_color="black",

    title_font_color="red",

    legend_title_font_color="green",

    title_text="Twenty last country by count"

)



fig.show()
#Graph : Init country levels by count

fig = px.bar(df["init_country_level"].value_counts(), orientation="v", color=df["init_country_level"].value_counts(), color_continuous_scale=px.colors.sequential.Plasma, 

             log_y=True, labels={'value':'Count', 

                                'index':'Country',

                                 'color':'None'

                                })



fig.update_layout(

    font_color="black",

    title_font_color="red",

    legend_title_font_color="green",

    title_text="Init country levels by count by count"

)



fig.show()
#Graph : Ten first enforcers by count

fig = px.bar(df["enforcer"].value_counts()[0:10], orientation="v", color=df["enforcer"].value_counts()[0:10], color_continuous_scale=px.colors.sequential.Plasma, 

             log_x=False, labels={'value':'Count', 

                                'index':'Country',

                                 'color':'None'

                                })



fig.update_layout(

    font_color="black",

    title_font_color="red",

    legend_title_font_color="green",

    title_text="Enforcer by count"

)



fig.show()