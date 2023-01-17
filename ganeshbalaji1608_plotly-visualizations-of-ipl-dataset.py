import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.set_option("display.max_columns", 999)
pd.set_option("display.max_rows", 999)
import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure()
df = pd.read_csv("../input/indian-premier-leagueipl-data-set/ipl.csv")
def automate(df):
    print(df.head(3))
    print("\n")
    print("---------------------------------------------")
    print("shape of the df is {}".format(df.shape))
    print("---------------------------------------------")
    print("\n")
    print("Info of the df: \n\n".format(df.info()))
    print("---------------------------------------------")
    print("\n")
    print("Null values in the df :")
    print(df.isnull().sum())
automate(df)
cata_features = df.select_dtypes(include = 'O')
num_features = df.select_dtypes(exclude = 'O')
print("Total Number of Catagorical Features, {}".format(cata_features.shape[1]))
print("Total Number of Numerical Features, {}".format(num_features.shape[1]))
for i in cata_features:
    print(i, df[i].value_counts()[0:10])
    print("---------------------")
plt.figure(figsize = (7,7))
plt.pie(df['batsman'].value_counts()[0:15],shadow = True,labels = df['batsman'].value_counts().index[0:15], autopct='%1.1f%%')
plt.title("Mostly played batsmans!!")
plt.figure(figsize = (8,8))
sns.scatterplot('runs', 'overs', hue = "bat_team", data = df)
plt.legend(loc = "best", bbox_to_anchor = [0,0.1,0,0.8])

fig = px.scatter_3d(df, 'runs', 'overs','wickets', color = "bat_team")
fig.show()
for i in cata_features:
    if i != "date" and i != "venue":
        fig = px.histogram(df, x= i, color = "wickets")
        fig.show()
fig = px.scatter(df, x="runs", y="wickets", color="bat_team", facet_col="bat_team",
                 title="Adding Traces To Subplots Witin A Plotly Express Figure")

reference_line = go.Scatter(x=[2, 4],
                            y=[4, 8],
                            mode="lines",
                            line=go.scatter.Line(color="gray"),
                            showlegend=False)

fig.show()
fig = px.scatter(df, x="striker", y="runs", size="wickets", color="bowl_team", hover_name="bat_team", log_x=True, size_max=40)
fig.show()

fig = px.scatter(df, x="runs", y="wickets", animation_frame="date", color = "bowl_team",
           size="total", animation_group="bat_team", hover_name="batsman",
           log_x=True, size_max=30,range_x=[1,40])

fig.show()