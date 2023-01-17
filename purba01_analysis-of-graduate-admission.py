!pip install chart_studio



import chart_studio

import chart_studio.plotly as py

username = 'username'

api_key = 'api_key'



chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
import pandas as pd

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

from plotly.subplots import make_subplots



df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv', sep=r'\s*,\s*')

df.info()
df.describe().T[1:7]
df.head(10)
fig = make_subplots(rows=1,

                    cols=3,

                    subplot_titles=('GRE vs Chance of Admit', 'TOEFL vs Chance of Admit', 'CGPA vs Chance of Admit'))





trace_gre = go.Scatter(x = df['GRE Score'],

                       y = df['Chance of Admit'],

                       mode = 'markers',

                       name = 'GRE',

                       marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                       text = df['University Rating'])



trace_toefl = go.Scatter(x = df['TOEFL Score'],

                        y = df['Chance of Admit'],

                       mode = 'markers',

                       name = 'Toefl',

                       marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                       text = df['University Rating'])



trace_cgpa = go.Scatter(x = df['CGPA'],

                       y = df['Chance of Admit'],

                       mode = 'markers',

                       name = 'CGPA',

                       marker = dict(color = 'rgba(56, 126, 80, 0.8)'),

                       text = df['University Rating'])





fig.add_trace(trace_gre, row=1, col=1)

fig.add_trace(trace_toefl, row=1, col=2)

fig.add_trace(trace_cgpa, row=1, col=3)



fig.update_layout(height=400, width=1000,

                  title_text="GRE, TOEFL, CGPA vs Chance of Admit")



iplot(fig, filename='graph_one.html')





fig = px.density_heatmap(df, x='Chance of Admit', y='Research')

fig.show()
threshold = 0.80

without_research = df[(df['Research'] == 0) & (df['Chance of Admit'] >= threshold)]

with_research = df[(df['Research'] == 1) & (df['Chance of Admit'] >= threshold)]


fig = make_subplots(rows=2,

                    cols=2)



trace_gre_vs_toefl = go.Scatter(x = without_research['GRE Score'],

                       y = without_research['TOEFL Score'],

                       mode = 'markers',

                       name = 'GRE vs Toefl',

                       marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                       text = without_research['University Rating'])



trace_cgpa_vs_gre = go.Scatter(x = without_research['CGPA'],

                        y = without_research['GRE Score'],

                       mode = 'markers',

                       name = 'CGPA vs GRE',

                       marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                       text = without_research['University Rating'])



trace_lor_vs_sop = go.Scatter(x = without_research['LOR'],

                       y =  without_research['SOP'],

                       mode = 'markers',

                       name = 'LOR vs SOP',

                       marker = dict(color = 'rgba(56, 126, 80, 0.8)'),

                       text = without_research['University Rating'])



trace_rating_vs_gre = go.Scatter(x = without_research['University Rating'],

                       y =  without_research['GRE Score'],

                       mode = 'markers',

                       name = 'University Rating vs GRE',

                       marker = dict(color = 'rgba(255, 0, 0, 0.8)'),

                       text = without_research['University Rating'])





fig.add_trace(trace_gre_vs_toefl, row=1, col=1)

fig.add_trace(trace_cgpa_vs_gre, row=1, col=2)

fig.add_trace(trace_lor_vs_sop, row=2, col=1)

fig.add_trace(trace_rating_vs_gre, row=2, col=2)



# Update xaxis properties

fig.update_xaxes(title_text="GRE", row=1, col=1)

fig.update_xaxes(title_text="CGPA", row=1, col=2)

fig.update_xaxes(title_text="LOR", row=2, col=1)

fig.update_xaxes(title_text="University Rating", row=2, col=2)



# Update yaxis properties

fig.update_yaxes(title_text="TOEFL", row=1, col=1)

fig.update_yaxes(title_text="GRE", row=1, col=2)

fig.update_yaxes(title_text="SOP", row=2, col=1)

fig.update_yaxes(title_text="GRE", row=2, col=2)





fig.update_layout(height=1000, width=1000, title_text=f"Without research experience and chance of admit is more than {threshold * 100}%")



iplot(fig, filename='graph_two.html')

def histogram_comparison(feature, df=[without_research, with_research], names=['Without Research Experience', 

                                                                               'With Research Experience']):

    fig = go.Figure()

    

    for idx,name in enumerate(names):

        fig.add_trace(go.Histogram(x=df[idx][feature],name=name))



    fig.update_layout(barmode='stack')

    fig.update_traces(opacity=0.75)

    fig.update_xaxes(title_text=feature)

    fig.update_yaxes(title_text="Count")



    return fig
fig = histogram_comparison('Chance of Admit')

fig.show()

fig = histogram_comparison('GRE Score')

fig.show()
fig = histogram_comparison('CGPA')

fig.show()
fig = make_subplots(rows=1, 

                    cols=2,  

                    subplot_titles=('Without Research Experience', 'With Research Experience'))



trace_without_research = go.Bar(x=without_research['University Rating'], 

                                y=without_research['Chance of Admit'],

                                name='Without Research Experience',

                                marker=dict(color = 'rgba(56, 126, 80, 0.8)'))



trace_with_research = go.Bar(x=with_research['University Rating'], 

                             y=with_research['Chance of Admit'],

                             name='With Research Experience',

                             marker=dict(color = 'rgba(56, 126, 80, 0.8)'))



fig.add_trace(trace_without_research, row=1, col=1)

fig.add_trace(trace_with_research, row=1, col=2)



fig.update_layout(showlegend=False)



fig.update_xaxes(title_text="University Rating", row=1, col=1)

fig.update_xaxes(title_text="University Rating", row=1, col=2)



fig.update_yaxes(title_text="Chance of Admit", row=1, col=1)

fig.update_yaxes(title_text="Chance of Admit", row=1, col=2)







fig.show()
df = df.drop('Serial No.', axis=1)

df
from xgboost import XGBClassifier





model = XGBClassifier()

X = df.drop('Chance of Admit', axis=1)

y= df['Chance of Admit']

model.fit(X, y)



feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), columns=['value', 'feature'])



fig = px.bar(feature_importance.sort_values(by='value', ascending=True), x='value', y='feature', orientation='h')

fig.show()
