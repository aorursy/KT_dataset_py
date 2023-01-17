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



import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')



import seaborn as sns

import matplotlib as p

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.graph_objs as gobj

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)



import plotly.express as px       

import plotly.offline as py       

import plotly.graph_objects as go 

from plotly.subplots import make_subplots
pd.set_option('display.max_columns', 200)

df = pd.read_csv('/kaggle/input/uncover/UNCOVER/einstein/diagnosis-of-covid-19-and-its-clinical-spectrum.csv')

df.head()
df.describe()
df_hospitalized = df[(df['patient_addmited_to_regular_ward_1_yes_0_no']=='t')|

                     (df['patient_addmited_to_semi_intensive_unit_1_yes_0_no']=='t')|

                     (df['patient_addmited_to_intensive_care_unit_1_yes_0_no']=='t')].reset_index(drop=True)

df_hospitalized
df_not_hospitalized = df[(df['patient_addmited_to_regular_ward_1_yes_0_no']=='f')&

                         (df['patient_addmited_to_semi_intensive_unit_1_yes_0_no']=='f')&

                         (df['patient_addmited_to_intensive_care_unit_1_yes_0_no']=='f')].reset_index(drop=True)

df_not_hospitalized
ward = df_hospitalized[df_hospitalized['patient_addmited_to_regular_ward_1_yes_0_no']=='t'].groupby(

    ['patient_age_quantile', 'patient_addmited_to_regular_ward_1_yes_0_no'])['patient_addmited_to_regular_ward_1_yes_0_no'].count().to_frame(name = 'count')

ward = ward.sort_values('patient_age_quantile').reset_index()

ward = ward.drop(['patient_addmited_to_regular_ward_1_yes_0_no'], axis=1)

ward
siu = df_hospitalized[df_hospitalized['patient_addmited_to_semi_intensive_unit_1_yes_0_no']=='t'].groupby(

    ['patient_age_quantile', 'patient_addmited_to_semi_intensive_unit_1_yes_0_no'])['patient_addmited_to_semi_intensive_unit_1_yes_0_no'].count().to_frame(name = 'count')

siu = siu.sort_values('patient_age_quantile').reset_index()

siu = siu.drop(['patient_addmited_to_semi_intensive_unit_1_yes_0_no'], axis=1)

siu
icu = df_hospitalized[df_hospitalized['patient_addmited_to_intensive_care_unit_1_yes_0_no']=='t'].groupby(

    ['patient_age_quantile', 'patient_addmited_to_intensive_care_unit_1_yes_0_no'])['patient_addmited_to_intensive_care_unit_1_yes_0_no'].count().to_frame(name = 'count')

icu = icu.sort_values('patient_age_quantile').reset_index()

icu = icu.drop(['patient_addmited_to_intensive_care_unit_1_yes_0_no'], axis=1)

icu
fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=ward['patient_age_quantile'], y=ward['count'], name='Admitted to Ward', marker_color='rgb(162, 205, 90)'))

fig.add_trace(go.Bar(x=siu['patient_age_quantile'],y=siu['count'],name='Admitted to SIU',marker_color='rgb(255, 185, 15)'))

fig.add_trace(go.Bar(x=icu['patient_age_quantile'], y=icu['count'], name='Admitted to ICU', marker_color='rgb(255, 64, 64)'))



fig.update_layout(title='Analysis of Hospitalized Patients (-ve & +ve)',xaxis_tickfont_size=14,

                  yaxis=dict(title='Count',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0.5,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)





fig.update_xaxes(title_text="Patient Quantile", range=[-1, 20])



fig.show()
df_hospitalized_positive = df_hospitalized[df_hospitalized['sars_cov_2_exam_result']=='positive']

df_hospitalized_positive.head()
ward_pos = df_hospitalized_positive[df_hospitalized_positive['patient_addmited_to_regular_ward_1_yes_0_no']=='t'].groupby(

    ['patient_age_quantile', 'patient_addmited_to_regular_ward_1_yes_0_no'])['patient_addmited_to_regular_ward_1_yes_0_no'].count().to_frame(name = 'count')

ward_pos = ward_pos.sort_values('patient_age_quantile').reset_index()

ward_pos = ward_pos.drop(['patient_addmited_to_regular_ward_1_yes_0_no'], axis=1)



siu_pos = df_hospitalized_positive[df_hospitalized_positive['patient_addmited_to_semi_intensive_unit_1_yes_0_no']=='t'].groupby(

    ['patient_age_quantile', 'patient_addmited_to_semi_intensive_unit_1_yes_0_no'])['patient_addmited_to_semi_intensive_unit_1_yes_0_no'].count().to_frame(name = 'count')

siu_pos = siu_pos.sort_values('patient_age_quantile').reset_index()

siu_pos = siu_pos.drop(['patient_addmited_to_semi_intensive_unit_1_yes_0_no'], axis=1)



icu_pos = df_hospitalized_positive[df_hospitalized_positive['patient_addmited_to_intensive_care_unit_1_yes_0_no']=='t'].groupby(

    ['patient_age_quantile', 'patient_addmited_to_intensive_care_unit_1_yes_0_no'])['patient_addmited_to_intensive_care_unit_1_yes_0_no'].count().to_frame(name = 'count')

icu_pos = icu_pos.sort_values('patient_age_quantile').reset_index()

icu_pos = icu_pos.drop(['patient_addmited_to_intensive_care_unit_1_yes_0_no'], axis=1)
fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=ward_pos['patient_age_quantile'], y=ward_pos['count'], name='Admitted to Ward', marker_color='rgb(162, 205, 90)'))

fig.add_trace(go.Bar(x=siu_pos['patient_age_quantile'],y=siu_pos['count'],name='Admitted to SIU',marker_color='rgb(255, 185, 15)'))

fig.add_trace(go.Bar(x=icu_pos['patient_age_quantile'], y=icu_pos['count'], name='Admitted to ICU', marker_color='rgb(255, 64, 64)'))



fig.update_layout(title='Analysis of Hospitalized Patients (+ve only)',xaxis_tickfont_size=14,

                  yaxis=dict(title='Count',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0.5,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)





fig.update_xaxes(title_text="Patient Quantile", range=[-1, 20])



fig.show()
icu_pos
sns.catplot('sars_cov_2_exam_result', data= df_not_hospitalized, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = df_not_hospitalized['sars_cov_2_exam_result'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of COVID Exam Results for Not Hospitalized Patients', fontsize = 20, color = 'black')

plt.show()
nothosp_grouped = df_not_hospitalized.groupby(['patient_age_quantile', 'sars_cov_2_exam_result'])['sars_cov_2_exam_result'].count().to_frame(name = 'count')

nothosp_grouped = nothosp_grouped.sort_values('patient_age_quantile').reset_index()

nothosp_grouped_neg = nothosp_grouped[nothosp_grouped['sars_cov_2_exam_result'] == 'negative'].sort_values('patient_age_quantile').reset_index()

nothosp_grouped_pos = nothosp_grouped[nothosp_grouped['sars_cov_2_exam_result'] == 'positive'].sort_values('patient_age_quantile').reset_index()

display(nothosp_grouped_neg)

display(nothosp_grouped_pos)
fig = go.Figure()

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=nothosp_grouped_neg['patient_age_quantile'], y=nothosp_grouped_neg['count'], name='Negative', marker_color='rgb(162, 205, 90)'))

fig.add_trace(go.Bar(x=nothosp_grouped_pos['patient_age_quantile'],y=nothosp_grouped_pos['count'],name='Positive',marker_color='rgb(255, 185, 15)'))



fig.update_layout(title='Analysis of Not Hospitalized Patients (-ve & +ve)',xaxis_tickfont_size=14,

                  yaxis=dict(title='Count',titlefont_size=16,tickfont_size=14,),

    legend=dict(x=0.5,y=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'),

    barmode='group',bargap=0.15, bargroupgap=0.1)





fig.update_xaxes(title_text="Patient Quantile", range=[-1, 20])



fig.show()