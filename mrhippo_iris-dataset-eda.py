import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #visualization
import seaborn as sns #visualization
import plotly.graph_objs as go #visualization
from plotly.offline import init_notebook_mode, iplot, plot
import warnings
init_notebook_mode(connected=True) 
# filter warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/iris/Iris.csv")
data.head(15)
data.info()
data.describe()
setosa = data[data["Species"]=="Iris-setosa"]
versicolor = data[data["Species"]=="Iris-versicolor"]
virginica = data[data["Species"]=="Iris-virginica"]
plt.style.use("ggplot")
plt.figure(figsize = (10,10))
sns.countplot(data["Species"],palette = "cubehelix") 
plt.text(2.55,0.8,str(data["Species"].value_counts()) ,fontsize = 18, color = "black") 
plt.title("")
plt.show()
sns.pairplot(data, hue ="Species", markers = "+")
plt.show()
desc = data.describe()
print(desc[:3])
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True,annot_kws = {"size": 12}, linewidths=0.5, fmt = '.3f', ax=ax)
plt.title("Correlation Between Features", fontsize = 20)
plt.show()
from plotly.subplots import make_subplots

fig = make_subplots(rows=4,cols=1,subplot_titles = ("SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"))

fig.append_trace(go.Scatter(x = data.Id,
                  y = data.SepalLengthCm,
                  mode = "lines",
                  name = "SepalLengthCm",
                  marker = dict(color = 'rgba(1,102,94, 0.8)')),row = 1, col = 1)

fig.append_trace(go.Scatter(x = data.Id,
                  y = data.SepalWidthCm,
                  mode = "lines",
                  name = "SepalWidthCm",
                  marker = dict(color = 'rgba(191,129,45, 0.8)')),row = 2, col = 1)
          
fig.append_trace(go.Scatter(x = data.Id,
                  y = data.PetalLengthCm,
                  mode = "lines",
                  name = "PetalLengthCm",
                  marker = dict(color = 'rgba(16, 112, 2, 0.8)')),row = 3, col = 1)
          
fig.append_trace(go.Scatter(x = data.Id,
                  y = data.PetalWidthCm,
                  mode = "lines",
                  name = "PetalWidthCm",
                  marker = dict(color = 'rgba(118,42,131, 0.8)')),row = 4, col = 1)
          
fig.update_layout(height = 1800, width = 900, title = "Fatures-Ids",template="plotly_white")

fig.update_xaxes(title_text="Id", row=1, col=1)
fig.update_xaxes(title_text="Id", row=2, col=1)
fig.update_xaxes(title_text="Id", row=3, col=1)
fig.update_xaxes(title_text="Id", row=4, col=1)

fig.update_yaxes(title_text="SepalLengthCm", row=1, col=1)
fig.update_yaxes(title_text="SepalWidthCm", row=2, col=1)
fig.update_yaxes(title_text="PetalLengthCm", row=3, col=1)
fig.update_yaxes(title_text="PetalWidthCm", row=4, col=1)

iplot(fig)
data_swrm_plt = data.drop(["Id"],axis = 1)
sns.set(style="whitegrid",palette = "muted")

data_swrm = pd.melt(data_swrm_plt,id_vars="Species",
                    var_name="Features",
                    value_name='Values')
plt.figure(figsize = (13,8))
sns.swarmplot(x="Features", y="Values",hue="Species", data=data_swrm)
plt.title("Swarmplot")
plt.show()
fig = go.Figure(data = [go.Scatter(
    
        y= setosa.PetalLengthCm,
        x= setosa.PetalWidthCm,
        mode= 'markers',
        marker= dict(
            color= setosa.SepalLengthCm,
            size= setosa.SepalWidthCm*10,
            showscale= True),
        text = setosa.Id
)])

fig.update_xaxes(title_text="PetalLengthCm")
fig.update_yaxes(title_text="PetalWidthCm")

fig.update_layout(title = "Setosa Features",template="plotly_white")

iplot(fig)

fig2 = go.Figure(data = [go.Scatter(
    
        y= versicolor.PetalLengthCm,
        x= versicolor.PetalWidthCm,
        mode= 'markers',
        marker= dict(
            color= versicolor.SepalLengthCm,
            size= versicolor.SepalWidthCm*10,
            showscale= True),
        text = versicolor.Id
)])

fig2.update_xaxes(title_text="PetalLengthCm")
fig2.update_yaxes(title_text="PetalWidthCm")

fig2.update_layout(title = "Versicolor Features",template="plotly_white")

iplot(fig2)

fig3 = go.Figure(data = [go.Scatter(
    
        y= virginica.PetalLengthCm,
        x= virginica.PetalWidthCm,
        mode= 'markers',
        marker= dict(
            color= virginica.SepalLengthCm,
            size= virginica.SepalWidthCm*10,
            showscale= True),
        text = virginica.Id
)])

fig3.update_xaxes(title_text="PetalLengthCm")
fig3.update_yaxes(title_text="PetalWidthCm")

fig3.update_layout(title = "Virginica Features",template="plotly_white")

iplot(fig3)
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2,subplot_titles = ("SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"))

fig.append_trace(go.Box(y = setosa.SepalLengthCm, name = "Setosa",marker_color = 'rgb(77,136,255)'),row = 1, col = 1)
fig.append_trace(go.Box(y = versicolor.SepalLengthCm,name = "Versicolor",marker_color = 'rgb(230,0,230)'),row = 1, col = 1)
fig.append_trace(go.Box(y = virginica.SepalLengthCm,name = "Vriginica",marker_color = 'rgb(0,128,106)'),row = 1, col = 1)

fig.append_trace(go.Box(y = setosa.SepalWidthCm,name = "Setosa",showlegend = True,marker_color = 'rgb(77,136,255)'),row = 1, col = 2)
fig.append_trace(go.Box(y = versicolor.SepalWidthCm,name = "Versicolor",showlegend = True,marker_color = 'rgb(230,0,230)'),row = 1, col = 2)
fig.append_trace(go.Box(y = virginica.SepalWidthCm,name = "Virginica",marker_color = 'rgb(0,128,106)'),row = 1, col = 2)

fig.append_trace(go.Box(y = setosa.PetalLengthCm,name = "Setosa",showlegend = True,marker_color = 'rgb(77,136,255)'),row = 2, col = 1)
fig.append_trace(go.Box(y = versicolor.PetalLengthCm,name = "Versicolor",showlegend = True,marker_color = 'rgb(230,0,230)'),row = 2, col = 1)
fig.append_trace(go.Box(y = virginica.PetalLengthCm,name = "Virginica",marker_color = 'rgb(0,128,106)'),row = 2, col = 1)

fig.append_trace(go.Box(y = setosa.PetalWidthCm,name = "Setosa",showlegend = True,marker_color = 'rgb(77,136,255)'),row = 2, col = 2)
fig.append_trace(go.Box(y = versicolor.PetalWidthCm,name = "Versicolor",showlegend = True,marker_color = 'rgb(230,0,230)'),row = 2, col = 2)
fig.append_trace(go.Box(y = virginica.PetalWidthCm,name = "Virginica",marker_color = 'rgb(10,128,106)'),row = 2, col = 2)

fig.update_xaxes(title_text="Specie", row=1, col=1)
fig.update_xaxes(title_text="Specie", row=1, col=2)
fig.update_xaxes(title_text="Specie", row=2, col=1)
fig.update_xaxes(title_text="Specie", row=2, col=2)

fig.update_yaxes(title_text="SepalLengthCm", row=1, col=1)
fig.update_yaxes(title_text="SepalWidthCm", row=1, col=2)
fig.update_yaxes(title_text="PetalLengthCm", row=2, col=1)
fig.update_yaxes(title_text="PetalWidthCm", row=2, col=2)

fig.update_layout(height=1000, width=800, title_text="Boxplot Features",template = "plotly_white")
fig.show()
# 3D scatterplot1
trace1 = go.Scatter3d(
                      x = setosa.SepalWidthCm,
                      y = setosa.PetalLengthCm,
                      z = setosa.SepalLengthCm,
                      text = setosa.Id,
                      mode = "markers",
                      name = "Setosa",
                      marker = dict(
                           size = 6,
                           color = setosa.PetalWidthCm,
                           symbol = "circle",line = dict(color = "rgb(255,255,255)", width = 0.5)
                      )
)
trace2 = go.Scatter3d(
                      x = versicolor.SepalWidthCm,
                      y = versicolor.PetalLengthCm,
                      z = versicolor.SepalLengthCm,
                      text = versicolor.Id,
                      mode = "markers",
                      name = "Versicolor",
                      marker = dict(
                           size = 6,
                           color = versicolor.PetalWidthCm,
                           symbol = "square",line = dict(color = "rgb(255,255,255)", width = 0.5)
                      )
)
trace3 = go.Scatter3d(
                      x = virginica.SepalWidthCm,
                      y = virginica.PetalLengthCm,
                      z = virginica.SepalLengthCm,
                      text = virginica.Id,
                      mode = "markers",
                      name = "Virginica",
                      marker = dict(
                           size = 6,
                           color = virginica.PetalWidthCm, 
                           symbol = "cross",line = dict(color = "rgb(255,255,255)", width = 0.5)
                      )
)

combine = [trace1,trace2,trace3]
layout = go.Layout(
    scene = dict(
    xaxis =dict(
        title = "SepalWidthCm(x)"),
    yaxis =dict(
        title ="PetalLengthCm(y)"),
    zaxis =dict(
        title = "SepalLengthCm(z)"),),
    width = 760,
    margin = dict(l = 10,r = 10,b = 10,t = 10 ),
template="plotly_white")
    
fig = go.Figure(data = combine, layout = layout)

iplot(fig)
import plotly.graph_objs as go

specie_list = list(data.Species.unique())

sepallengthcm = []
sepalwidthcm = []
petallengthcm = []
petalwidthcm = []

for i in specie_list:
    x = data[data["Species"] == i]
    sepallengthcm.append(sum(x.SepalLengthCm)/len(x)) 
    sepalwidthcm.append(sum(x.SepalWidthCm)/len(x))   
    petallengthcm.append(sum(x.PetalLengthCm)/len(x)) 
    petalwidthcm.append(sum(x.PetalWidthCm)/len(x))
    
trace1 = go.Bar(    
    x = specie_list,
    y = sepallengthcm,
    text = np.around(sepallengthcm,2),
    textposition = "outside",
    textfont = dict(size = 15),
    name = "SepalLengthCm",
    marker = dict(color = "rgba(73,0,106,0.6)",
                 line = dict(color = "rgb(0,0,0)", width = 1.5))

)

trace2 = go.Bar(    
    x = specie_list,
    y = sepalwidthcm,
    text = np.around(sepalwidthcm,2),
    textposition = "outside",
    textfont = dict(size = 15),
    name = "SepalWidthCm",
    marker = dict(color = "rgba(122,1,119,0.6)",
                 line = dict(color = "rgb(0,0,0)", width = 1.5))

)

trace3 = go.Bar(    
    x = specie_list,
    y = petallengthcm, 
    text = np.around(petallengthcm,2),
    textposition = "outside",
    textfont = dict(size = 15),
    name = "PetalLengthCm",
    marker = dict(color = "rgba(174,1,126,0.6)",
                 line = dict(color = "rgb(0,0,0)", width = 1.5))

)

trace4= go.Bar(    
    x = specie_list,
    y = petalwidthcm,
    text = np.around(petalwidthcm,2),
    textposition = "outside",
    textfont = dict(size = 15),
    name = "PetalWidthCm",
    marker = dict(color = "rgba(221,52,151,0.6)",
                 line = dict(color = "rgb(0,0,0)", width = 1.5))

)

data = [trace1,trace2,trace3,trace4] 
layout = go.Layout(template="plotly_white",title = "Fature Value Means")
fig = go.Figure(data = data,layout = layout)

fig.update_xaxes(title_text = "Specie")
fig.update_yaxes(title_text = "Max Value")

iplot(fig)