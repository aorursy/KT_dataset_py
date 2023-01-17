import pandas as pd

import plotly.express as px



import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')

%matplotlib inline



import plotly.graph_objs as go
df = pd.read_csv('../input/military-expenditure-of-countries-19602019/Military Expenditure.csv')
df.head()
japan = df[df["Name"] == "Japan"]
japan_ME = japan.drop(labels=["Name","Code","Type","Indicator Name"], axis=1).transpose()
japan_ME.rename(columns={117:"Military_expenses"}, inplace=True)



japan_ME.reset_index(inplace=True)
plt.figure(figsize = (25,5))

sns.barplot(x="index",y="Military_expenses",data=japan_ME.reset_index())
japan_ME["accumulate"] = japan_ME.Military_expenses.cumsum(axis = 0)

japan_ME.head()
trace0 = go.Bar(x=japan_ME["index"],y=japan_ME.Military_expenses,name="Military_expenses",yaxis="y1")

trace1 = go.Scatter(x=japan_ME["index"],y=japan_ME.accumulate,name="accumulate",yaxis="y2",mode='lines+markers')
layout = go.Layout(xaxis=dict(title="日本の軍事費（年次と累積）", range=[min(japan_ME["index"]),max(japan_ME["index"])]),

                  yaxis=dict(title="Military expenses",side="left",showgrid=False, range=[0,max(japan_ME.Military_expenses*1.2)]),

                   yaxis2=dict(title="accumulate",side="right",overlaying="y",range = [0, max(japan_ME.accumulate*1.2)], showgrid=False))
import plotly.offline as offline

offline.init_notebook_mode()



fig = dict(data=[trace0,trace1],layout=layout)

offline.iplot(fig)
df["sum"] = df.sum(axis=1, skipna=True, numeric_only=True)
df_country = df[df["Type"]=="Country"]



df_top3 = df_country.sort_values(by="sum", ascending=False).head(3)
df_top3.drop(labels=["Code","Type","Indicator Name","Name"], axis=1, inplace=True)
df_top3_transpose = df_top3.transpose()
top3 = df_top3_transpose.rename(columns={249:"United States",38:"China",75:"France"})
top3.drop(labels="sum",axis=0, inplace=True)
japan_ME.set_index("index", inplace=True)
top3_and_japan = japan_ME.join(top3).rename(columns={"Military_expenses":"Japan"})
top3_and_japan.head()
top3_and_japan.drop(columns="accumulate", axis=1, inplace=True)
data = pd.melt(top3_and_japan.reset_index(), id_vars="index")



data.head()
fig = px.line(data, x="index", y="value", color="variable", line_group="variable", hover_name="variable",line_shape="spline", render_mode="svg")



fig.show()
country = data.variable.unique()



fig_2 = go.Figure()



for i in country:

    fig_2.add_trace(go.Bar(

        x=data["index"],

        y=data["value"][data.variable == str(i)],

        name = str(i)))



fig_2.show()
top3["USA_accumulate"] = top3["United States"].cumsum(axis=0)

top3["China_accumulate"] = top3["China"].cumsum(axis=0)

top3["France_accumulate"] = top3["France"].cumsum(axis=0)
top3["USA_accumulate"] = top3["United States"].cumsum(axis=0)

top3["China_accumulate"] = top3["China"].cumsum(axis=0)

top3["France_accumulate"] = top3["France"].cumsum(axis=0)
top3_accumulate = top3.drop(columns=["United States","China","France"], axis=1)
top3_accumulate_reshape = pd.melt(top3_accumulate.reset_index(), id_vars="index")
fig_1 = px.bar(top3_accumulate_reshape, x="variable",y="value",color="variable", animation_frame="index",animation_group="variable", range_y=[0,max(top3_accumulate_reshape.value)*1.5])



fig_1.show()
top20 = df_country.sort_values(by="sum", ascending=False).head(20)



labels = top20["Name"]



top20.drop(labels=["Name","Code","Type","Indicator Name","sum"], axis=1, inplace=True)
top20_transpose = top20.transpose().rename(columns= labels)

top20_transpose.head()
top20 = pd.melt(top20_transpose.reset_index(), id_vars="index")
fig_2 = px.bar(top20, x="variable",y="value",color="variable", animation_frame="index",animation_group="variable")



fig_2.show()
top20_accumulate = top20_transpose.cumsum()
top20_accumulate_reshape = pd.melt(top20_accumulate.reset_index(), id_vars="index")
fig_3 = px.bar(top20_accumulate_reshape, x="variable",y="value",color="variable", animation_frame="index",animation_group="variable", range_y=[0,max(top20_accumulate_reshape.value)*1.2])



fig_3.show()