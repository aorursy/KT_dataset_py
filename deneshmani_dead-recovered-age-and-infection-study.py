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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

from pandas_profiling import ProfileReport
# Credits - https://towardsdatascience.com/python-plotting-basics-simple-charts-with-matplotlib-seaborn-and-plotly-e36346952a3a



color_palette_list = ['#009ACD', '#ADD8E6', '#63D1F4', '#0EBFE9',   

                      '#C1F0F6', '#0099CC']



def plot_piechart(labels, percentages, title):

    fig, ax = plt.subplots()

    plt.rcParams['font.sans-serif'] = 'Arial'

    plt.rcParams['font.family'] = 'Arial'

    plt.rcParams['text.color'] = '#0a1df0'

    plt.rcParams['axes.labelcolor']= '#0a1df0'

    plt.rcParams['xtick.color'] = '#909090'

    plt.rcParams['ytick.color'] = '#909090'

    plt.rcParams['font.size']=12

    

    explode=(0.1,0)

    ax.pie(percentages, explode=explode, labels=labels, 

           autopct='%1.0f%%', colors=color_palette_list[0:2],

           shadow=False, startangle=0,   

           pctdistance=1.2,labeldistance=1.4)

    ax.axis('equal')

    ax.set_title(title, fontsize=16)

    ax.legend(frameon=False, bbox_to_anchor=(1.5,0.8))
df_covid_list_list = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

df_covid_open_line_list = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")
df_covid_list_list.info()
unnamed_columns = [c for c in df_covid_list_list.columns if c.startswith('Unnamed')]

df_covid_list_list.drop(unnamed_columns, axis=1, inplace=True)

df_covid_list_list.head()
df_covid_list_list['death'].unique()
df_covid_list_list['recovered'].unique()
df_covid_list_list['death_flag'] = ['1' if v not in ['0', '1'] else v for v in df_covid_list_list['death'].values]

df_covid_list_list['recovered_flag'] = ['1' if v not in ['0', '1'] else v for v in df_covid_list_list['recovered'].values]
def count_to_pie(input_series, title, alias=[]):

    """

    function to plot pie chart for distinct value counts

    :param input_series - value_counts() of a column

    :param title - title of the output plot

    :param alias - alias for the unique values to be mapped based on index

    """

    _labels = []

    _percentages = []

    total = input_series.sum()

    for i, r in input_series.items():

        _labels.append(i)

        _percentages.append(r/total)

    if alias and len(alias) == len(_labels):

        _labels = alias

    plot_piechart(_labels, _percentages, title)
count_to_pie(df_covid_list_list['death_flag'].value_counts(), title="Overall Death Percentage", alias=["Alive", "Dead"])
count_to_pie(df_covid_list_list['recovered_flag'].value_counts(), title="Overall Recovery Percentage",

             alias=["Not recovered", "Recovered"])
df_death = df_covid_list_list[df_covid_list_list['death_flag'] == '1']

df_recovered = df_covid_list_list[df_covid_list_list['recovered_flag'] == '1']
from warnings import filterwarnings

filterwarnings('ignore')



df_death.age.fillna(int(df_death.age.mean()), inplace=True)

df_recovered.age.fillna(int(df_recovered.age.mean()), inplace=True)
fig, ax = plt.subplots()

sns.kdeplot(df_death.age, ax=ax, color="r", label="Dead", shade=True)

sns.kdeplot(df_recovered.age, ax=ax, color="g", label="Recovered", shade=True)

plt.show()
df_covid_list_list['symptom_onset'] = pd.to_datetime(df_covid_list_list['symptom_onset'], errors='ignore')

df_covid_list_list['alive_period'] = [

    pd.to_datetime('02/28/2020') 

        if (r['death_flag'] == "0" and r['recovered_flag'] == "0") # when the patient is not recovered yet

        else pd.to_datetime(r['death'], errors='ignore')  # when the patient is dead

            if r['death_flag'] == "1"

            else pd.to_datetime(r['recovered'], errors='ignore')

    for i,r in df_covid_list_list.iterrows()

]
df_covid_list_list['alive_period'] = [ 

    v if isinstance(v, pd._libs.tslibs.timestamps.Timestamp) else pd.to_datetime('01/31/2020') # dates recorded since 1st Feb

    for v in df_covid_list_list['alive_period'].values

]

df_covid_list_list['symptom_onset'] = [

    v if isinstance(v, np.datetime64) else None for v in df_covid_list_list['symptom_onset'].values

]
df_covid_list_list['infection_period'] = df_covid_list_list['alive_period'] - df_covid_list_list['symptom_onset']

df_covid_list_list['infection_period'] = df_covid_list_list['infection_period'].dt.days

df_covid_list_list.infection_period.fillna(df_covid_list_list['infection_period'].mean(), inplace=True)
df_death_infc = df_covid_list_list[(df_covid_list_list['death_flag'] == '1') & (df_covid_list_list['infection_period'] > 0) ]

df_recovered_infc = df_covid_list_list[(df_covid_list_list['recovered_flag'] == '1') & (df_covid_list_list['infection_period'] > 0)]
from warnings import filterwarnings

filterwarnings('ignore')



df_death_infc.age.fillna(int(df_death_infc.age.mean()), inplace=True)

df_recovered_infc.age.fillna(int(df_recovered_infc.age.mean()), inplace=True)
# code credits - https://becominghuman.ai/introduction-to-timeseries-analysis-using-python-numpy-only-3a7c980231af



def moving_average(signal, period):

    buffer = [np.nan] * period

    for i in range(period,len(signal)):

        buffer.append(signal[i-period:i].mean())

    return buffer
# Scaling the values of infection period and age in common range



from sklearn.preprocessing import MinMaxScaler



age_scaled = MinMaxScaler().fit_transform(df_death_infc.age.values.reshape(-1, 1))

age_scaled = age_scaled.reshape(1, -1)[0]

df_death_infc['age_scaled'] = age_scaled



infection_period_scaled = MinMaxScaler().fit_transform(df_death_infc.infection_period.values.reshape(-1, 1))

infection_period_scaled = infection_period_scaled.reshape(1, -1)[0]

df_death_infc['infection_period_scaled'] = infection_period_scaled
fig, ax = plt.subplots()



ax.set_title("Comparison of age and infection period of dead patients")



df_death_infc['smoothen_infc_period'] = moving_average(df_death_infc.infection_period, 5)



sns.lineplot(x="age", y="infection_period", data=df_death_infc, ax=ax)

sns.lineplot(x="age", y="smoothen_infc_period", data=df_death_infc, ax=ax)



plt.show()
fig, ax = plt.subplots()



ax.set_title("Comparison of age and infection period of recovered patients")



df_recovered_infc['smoothen_infc_period'] = moving_average(df_recovered_infc.infection_period, 5)



sns.lineplot(x="age", y="infection_period", data=df_recovered_infc, ax=ax)

sns.lineplot(x="age", y="smoothen_infc_period", data=df_recovered_infc, ax=ax)





plt.show()
import math
list_of_symptoms = df_covid_list_list['symptom'].unique()

# before split and strip

list_of_symptoms[:5]
list_of_symptoms = [list(map(str.strip, v.split(","))) for v in list_of_symptoms if isinstance(v, str)]
# after split and strip

unique_symptoms = np.unique([i for v in list_of_symptoms for i in v])



common_name_dict = {

    'abdominal pain': 'abdominal pain',

    'breathlessness': 'breathlessness',

    'difficult in breathing': 'breathlessness',

    'difficulty breathing': 'breathlessness',

    'dyspnea': 'breathlessness',

    'respiratory distress': 'breathlessness',

    'shortness of breath': 'breathlessness',

    'chest discomfort': 'chest pain',

    'chest pain': 'chest pain',

    'chill': 'chillness',

    'chills': 'chillness',

    'cold': 'cold',

    'cough': 'cough',

    'coughing': 'cough',

    'dry cough': 'cough',

    'mild cough': 'cough',

    'cough with sputum': 'cough with sputum',

    'sputum': 'cough with sputum',

    'diarrhea': 'diarrhea',

    'aching muscles': 'fatigue',

    'fatigue': 'fatigue',

    'malaise': 'fatigue',

    'muscle aches': 'fatigue',

    'muscle cramps': 'fatigue',

    'muscle pain': 'fatigue',

    'myalgia': 'fatigue',

    'myalgias': 'fatigue',

    'physical discomfort': 'fatigue',

    'sore body': 'fatigue',

    'tired': 'fatigue',

    'feaver': 'fever',

    'feve\\': 'fever',

    'fever': 'fever',

    'flu': 'fever',

    'flu symptoms': 'fever',

    'high fever': 'fever',

    'mild fever': 'fever',

    'headache': 'headache',

    'heavy head': 'headache',

    'joint pain': 'joint pain',

    'loss of appetite': 'loss of appetite',

    'nausea': 'nausea',

    'vomiting': 'nausea',

    'pneumonia': 'pneumonia',

    'reflux': 'reflux',

    'nasal discharge': 'running nose',

    'runny nose': 'running nose',

    'sneeze': 'sneeze',

    'thirst': 'thirst',

    'itchy throat': 'throat pain',

    'sore throat': 'throat pain',

    'throat discomfort': 'throat pain',

    'throat pain': 'throat pain'

}



# replace the list of symptoms by the common name

cn_list_of_symptoms = [

    [common_name_dict.get(s, s) for s in l] # return same name if not found in common names

    for l in list_of_symptoms

]



print(f"before: {list_of_symptoms[:5]}")

print("-----------------------------------")

print(f"after: {cn_list_of_symptoms[:5]}")