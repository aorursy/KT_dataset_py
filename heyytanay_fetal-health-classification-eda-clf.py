! pip install -q dabl
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff



import dabl

from pandas_profiling import ProfileReport



from colorama import Fore, Style
def cout(string: str, color=Fore.RED):

    """

    Saves some work 😅

    """

    print(color+string+Style.RESET_ALL)

    

def statistics(dataframe, column):

    cout(f"The Average value in {column} is: {dataframe[column].mean():.4f}", Fore.RED)

    cout(f"The Maximum value in {column} is: {dataframe[column].max()}", Fore.BLUE)

    cout(f"The Minimum value in {column} is: {dataframe[column].min()}", Fore.YELLOW)

    cout(f"The 25th Quantile of {column} is: {dataframe[column].quantile(0.25):.4f}", Fore.GREEN)

    cout(f"The 50th Quantile of {column} is: {dataframe[column].quantile(0.50):.4f}", Fore.CYAN)

    cout(f"The 75th Quantile of {column} is: {dataframe[column].quantile(0.75):.4f}", Fore.MAGENTA)
data = pd.read_csv("../input/fetal-health-classification/fetal_health.csv")

data.head()
pr = ProfileReport(data)
pr.to_notebook_iframe()
dabl.plot(data, target_col='fetal_health')
statistics(data, column='baseline value')
plt.style.use("classic")

sns.distplot(data['baseline value'])

plt.title(f"Baseline Fetal Heartrate [\u03BC : {data['baseline value'].mean():.2f} bpm | \u03C3 : {data['baseline value'].std():.2f} bpm]")

plt.xlabel("Heart Rate (in bpm)")

plt.ylabel("Count")

plt.show()



# Also the plotly figure

fig = ff.create_distplot(

    hist_data=[data['baseline value'].tolist()],

    group_labels=['baseline value'],

    colors=['#0B43EA'],

    show_hist=False,

    show_rug=False,

)



fig.layout.update({'title':"Baseline Fetal Heartrate"})



fig.show()
statistics(data, column='accelerations')
plt.style.use("classic")

sns.distplot(data['accelerations'], color='magenta')

plt.title(f"Accelerations Per Second")

plt.xlabel("Accelerations")

plt.ylabel("Count")

plt.show()



# Also the plotly figure

fig = ff.create_distplot(

    hist_data=[data['accelerations'].tolist()],

    group_labels=['accelerations'],

    colors=['#E00DAB'],

    show_hist=False,

    show_rug=False,

)



fig.layout.update({'title':"Accelerations Per Second"})



fig.show()
statistics(data, column='fetal_movement')
plt.style.use("classic")

sns.distplot(data['fetal_movement'], color='green')

plt.title(f"Fetal Movement")

plt.xlabel("Movements per second")

plt.ylabel("Count")

plt.show()



# Also the plotly figure

fig = ff.create_distplot(

    hist_data=[data['fetal_movement'].tolist()],

    group_labels=['fetal_movement'],

    colors=['#0BE047'],

    show_hist=False,

    show_rug=False,

)



fig.layout.update({'title':"Movements Per Second"})



fig.show()
statistics(data, column='uterine_contractions')
plt.style.use("classic")

sns.distplot(data['uterine_contractions'], color='red')

plt.title(f"Uterine Contractions")

plt.xlabel("Contractions per second")

plt.ylabel("Count")

plt.show()



# Also the plotly figure

fig = ff.create_distplot(

    hist_data=[data['uterine_contractions'].tolist()],

    group_labels=['uterine_contractions'],

    colors=['#FF001D'],

    show_hist=False,

    show_rug=False,

)



fig.layout.update({'title':"Uterine Contractions Per Second"})



fig.show()
statistics(data, column='abnormal_short_term_variability')
plt.style.use("classic")

sns.distplot(data['abnormal_short_term_variability'], color='orange')

plt.title(f"Percentage of Time with Abnormal Short Term Variability")

plt.xlabel("Percentage of Time")

plt.ylabel("Count")

plt.show()



# Also the plotly figure

fig = ff.create_distplot(

    hist_data=[data['abnormal_short_term_variability'].tolist()],

    group_labels=['abnormal_short_term_variability'],

    colors=['#FFB600'],

    show_hist=False,

    show_rug=False,

)



fig.layout.update({'title':"Percentage of Time with Abnormal Short Term Variability"})



fig.show()
statistics(data, column="fetal_health")
names = list(dict(data['fetal_health'].value_counts()).keys())

values = data['fetal_health'].value_counts().tolist()



fig = go.Bar(x = names,

            y = values,

            marker = dict(color = 'rgba(0, 255, 0, 0.5)',

                         line=dict(color='rgb(0,0,50)',width=1.5)),

            text = names)



layout = go.Layout()

fig = go.Figure(data = fig, layout = layout)

fig.update_layout(title_text='Fetal Health (Target Variable)')

fig.show()
vals = [len(data[data['fetal_health']==1.0]['fetal_health']), len(data[data['fetal_health']==2.0]['fetal_health']), len(data[data['fetal_health']==3.0]['fetal_health'])]

idx = ['Normal', 'Suspect', 'Pathological']

fig = px.pie(

    values=vals,

    names=idx,

    title='Fetal Health Pie Chart (Target Variable)',

    color_discrete_sequence=px.colors.sequential.Agsunset

)

fig.show()