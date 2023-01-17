import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
import matplotlib.pyplot as plt 
import plotly.offline as py
from plotly import tools
import plotly.figure_factory as ff
import seaborn as sns
import warnings
py.init_notebook_mode(connected=True)
warnings.simplefilter('ignore', FutureWarning)
training_set = pd.read_csv('../input/training_set.csv')
# training_set = pd.read_csv('training_set.csv')
meta_training_set = pd.read_csv("../input/training_set_metadata.csv")
# meta_training_set = pd.read_csv("training_set_metadata.csv")
test_set_meta = pd.read_csv("../input/test_set_metadata.csv")
# test_set_meta = pd.read_csv("test_set_metadata.csv")
full_meta_data = pd.concat([meta_training_set,test_set_meta],sort=True)

print("sample of meta_training data")
meta_training_set.sample(5)
print("training_set_metadata info")
meta_training_set.info()
print ("sample of training data")
training_set.sample(5)
print("training_set info")
training_set.info()
fig,ax = plt.subplots(1,2,figsize=(12,5))
ax[0].barh(meta_training_set.drop(['target'], axis=1).isnull().sum().index,meta_training_set.drop(['target'], axis=1).isnull().sum().values)
ax[0].set_xlabel("NAN count")
ax[0].set_title("NAN count per feature\n(traing_meta data)")
ax[1].barh(test_set_meta.isnull().sum().index,test_set_meta.isnull().sum().values)
ax[1].set_xlabel("NAN count")
ax[1].set_title("NAN count per feature\n(test_meata data)")
targets_classes = meta_training_set.target.unique()
print ("There are {} unique classes.".format(len(targets_classes)))
for i in range(len(targets_classes)):
    print("class_{}".format(targets_classes[i]))
objects_per_target = pd.DataFrame(meta_training_set.groupby("target", as_index = False)["object_id"].count()).sort_values(by="object_id",ascending=False)
objects_per_target['target_list']=list(map(lambda x: "class_{}".format(x),objects_per_target.target))
objects_per_target = objects_per_target.rename(columns = {"object_id": "objects_count"})

pie_plot= go.Pie(values = objects_per_target['objects_count'],
                 labels = objects_per_target['target_list'],
                 hoverinfo="label+percent",
                 hole= .3)

layout = go.Layout(title = "Classes distribution ")

fig = go.Figure(data=[pie_plot], layout=layout)

py.iplot(fig)
bar_trace  = go.Bar(x= objects_per_target.target_list,
                 y= objects_per_target.objects_count,
                 marker=dict(color='#f0000a',
                             line=dict(color='rgb(8,48,107)',width=1.5,)),
                 name = "objects count",
                 opacity=0.7,
                 hoverinfo="name + y")

layout = go.Layout(title='Count of objects per class',
                   xaxis=dict(tickangle=-45),
                   yaxis = dict(title = " Number of objects"))

fig = go.Figure(data=[bar_trace], layout=layout)

py.iplot(fig)
colors = ['blue','gray','red','green','pink',
          'steelblue','yellow','magenta','brown',
          'orange','tan','seagreen','mintcream',
          'yellowgreen','chocolate','rosybrown',
          'dodgerblue','heather']
for i in range (0,len(objects_per_target.target)):
    class_ = meta_training_set[meta_training_set.target == objects_per_target.target.values[i]]
    trace=go.Scatter(
        x=class_['gal_l'],
        y=class_['gal_b'],
        mode = 'markers',
        marker=dict(color=colors[i]),
        text= "Longitude = {} °".format(class_['gal_l'].values[i])+"<br>"+ "Latitude = {} °".format(class_['gal_b'].values[i]),
        hoverinfo="text",
        connectgaps=True,
        name = objects_per_target.target_list.values[i],
        textfont=dict(family='Arial', size=12),
    )
    layout = go.Layout(
        title = objects_per_target.target_list.values[i]+" distrbution in space",
       xaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(150, 150, 150)',
                linewidth=2,
                gridcolor='rgb(90, 90, 90)',
                ticks='outside',
                tickcolor='rgb(80, 80, 80)',
                tickwidth=2,
                ticklen=5,
                tickfont=dict(
                family='Arial',
                size=13,
                color='rgb(180, 180, 180)',
            ),
        ),
        yaxis=dict(
                showgrid=True,
                zeroline=True,
                showline=False,
                gridcolor='rgb(80, 80, 80)',
                showticklabels=True,
                tickcolor='rgb(150, 150, 150)',
                tickwidth=2,
                ticklen=5,
                tickfont=dict(
                family='Arial',
                size=13,
                color='rgb(180, 180, 180)')
        ),
       font=dict(family='Arial', size=12,
                color='rgb(180, 180, 180)'),
                showlegend=True, 
                width = 600,
                height = 300,
                paper_bgcolor='rgba(0, 0, 0,.9)',
                plot_bgcolor='rgba(0, 0, 0,0)')
    
    fig = go.Figure(data=[trace], layout= layout)
    py.iplot(fig)
    
traces = []
for i in range (len(targets_classes)):
    class_ = meta_training_set[meta_training_set.target == targets_classes[i]]
    traces.append(go.Scatter(
        x=class_['gal_l'],
        y=class_['gal_b'],
        mode = 'markers',
        marker=dict(color=colors[i]),
        text= "class_{}".format(targets_classes[i])+"<br>"+"Longitude = {} °".format(class_['gal_l'].values[i])+"<br>"+ "Latitude = {} °".format(class_['gal_b'].values[i]),
        hoverinfo="text",
        connectgaps=True,
        name = "class_{}".format(targets_classes[i]),
        textfont=dict(family='Arial', size=12),
    ))
layout = go.Layout(
    title = "Classes distrbution in space",
   xaxis=dict(
            title = "Galactical Longitude (°)",
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(150, 150, 150)',
            linewidth=2,
            gridcolor='rgb(90, 90, 90)',
            ticks='outside',
            tickcolor='rgb(80, 80, 80)',
            tickwidth=2,
            ticklen=5,
            tickfont=dict(
            color='rgb(180, 180, 180)',
        ),
    ),
    yaxis=dict(
            title = "Galactical Latitude (°)",
            showgrid=True,
            zeroline=True,
            showline=False,
            gridcolor='rgb(80, 80, 80)',
            showticklabels=True,
            tickcolor='rgb(150, 150, 150)',
            tickwidth=2,
            ticklen=5,
            tickfont=dict(
            color='rgb(180, 180, 180)')
    ),
   font=dict(family='Arial', size=12,
            color='rgb(200, 200, 200)'),
            showlegend=True, 
            width = 750,
            height = 550,
            paper_bgcolor='rgba(0, 0, 0,.9)',
            plot_bgcolor='rgba(0, 0, 0,0)'
)
fig = go.Figure(data=traces, layout= layout)
py.iplot(fig)

for trace in traces :
    trace.marker.opacity = 0.1
layout.title = "Classes distrbution in space (low opacity) "
fig = go.Figure(data=traces, layout= layout)
py.iplot(fig)
# the galactic feature determines whether the object inside the milky way galactic==1 or outside milky way galactic==0
meta_training_set['galactic'] = list(map(lambda x: 1 if x==0 else 0,meta_training_set['hostgal_photoz']))
def is_class (class_number):
    meta_training_set['_{}'.format(class_number)]=list(map(lambda x: 1 if x==class_number else 0,meta_training_set.target))
for class_num in objects_per_target.target:
    is_class(class_num)
corr_meta = meta_training_set.drop(['object_id','target'],axis=1).corr()
plt.subplots(figsize=(12,10))
sns.heatmap(corr_meta[11:].drop(corr_meta.columns[11:],axis=1),annot=True)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 objects_per_target.target,
                                                 meta_training_set['target'])
print("Classes weights:"+"\n\n",list(zip(objects_per_target.target,class_weights)))
bar_trace  = go.Bar(x= objects_per_target.target_list,
                 y= class_weights,
                 marker=dict(color='#f0000a',
                             line=dict(color='rgb(8,48,107)',width=1.5,)),
                 name = "class weight",
                 opacity=0.7,
                 hoverinfo="name + y")

layout = go.Layout(title='Classes weights',
                   xaxis=dict(tickangle=-45),
                   yaxis = dict(title = " class weight "))

fig = go.Figure(data=[bar_trace], layout=layout)

py.iplot(fig)
group_labels = ['distmod distribution']
dist_plot = ff.create_distplot([meta_training_set['distmod'].dropna().values],
                               group_labels ,
                               bin_size=.3,
                               colors = ['rgba(5,20,100,.8)'])
dist_plot['layout'].update(title="distmod distribution plot",
                           width = 700,
                           height = 450)

py.iplot(dist_plot)

full_meta_data['galactic'] = list(map(lambda x: 1 if x==0 else 0,full_meta_data['hostgal_photoz']))
test_set_meta['galactic'] = list(map(lambda x: 1 if x==0 else 0,test_set_meta['hostgal_photoz']))

pie_plot_test= go.Pie(values = test_set_meta["galactic"].value_counts(),
                 labels = ["Extragalactic","Galactic"],
                 hoverinfo="label+percent",
                 domain = dict(x=[0, .5],
                               y=[0.5,1]),
                 hole = 0.4
                    )

pie_plot_train= go.Pie(values = meta_training_set["galactic"].value_counts(),
                 labels = ["Extragalactic","Galactic"],
                 hoverinfo="label+percent",
                 domain = dict(x=[0.5, 1],
                               y=[0.5,1]),
                 hole = 0.4)

pie_plot_all= go.Pie(values = full_meta_data["galactic"].value_counts(),
                 labels = ["Extragalactic","Galactic"],
                 hoverinfo="label+percent",
                 domain = dict(x=[0, 1],
                               y=[0,0.5]),
                 hole = 0.4)

layout = go.Layout(title = "Galactic vs Extragalactic",
                  annotations = [dict(text = "test",
                                      font = dict(size=15),
                                      x=0.225,
                                      y=0.775,
                                      showarrow= False),
                                 dict(text = "train",
                                      font = dict(size=15),
                                      x=0.775,
                                      y=0.775,
                                     showarrow= False),
                                 dict(text = "all",
                                      font = dict(size=15),
                                      x=0.50,
                                      y=0.225,
                                      showarrow= False)]
                                    )

fig = go.Figure(data=[pie_plot_test,pie_plot_train,pie_plot_all], layout=layout)
py.iplot(fig)
galactic_classes = meta_training_set.groupby(['galactic']).get_group(1)['target'].value_counts()
extragalactic_classes = meta_training_set.groupby(['galactic']).get_group(0)['target'].value_counts()
galactic_classes_list = list(map(lambda x: "class_{}".format(x),galactic_classes.index))
extragalactic_classes_list = list(map(lambda x: "class_{}".format(x),extragalactic_classes.index))

bar_trace_1  = go.Bar(x=galactic_classes_list,
                 y= galactic_classes.values,
                 marker=dict(color='#f0000a',
                             line=dict(color='rgb(8,48,107)',width=1.5,)),
                 name = "galactic",
                 opacity=0.7,
                 hoverinfo="name + y")
bar_trace_2  = go.Bar(x= extragalactic_classes_list,
                 y= extragalactic_classes.values,
                 marker=dict(color='#fff00a',
                             line=dict(color='rgb(8,48,107)',width=1.5,)),
                 name = "extragalactic",
                 opacity=0.7,
                 hoverinfo="name + y")

layout = go.Layout(title='Galactic vs Extragalactic per class',
                   xaxis=dict(tickangle=-45),
                   yaxis = dict(title = " Number of objects"))

fig = go.Figure(data=[bar_trace_1,bar_trace_2], layout=layout)
py.iplot(fig)
group_labels = ['MWEBV distribution']
dist_plot = ff.create_distplot([meta_training_set['mwebv'].dropna().values],
                               group_labels ,
                               bin_size=.1,
                               colors = ['rgba(5,20,100,.8)'])
dist_plot['layout'].update(title="MWEBV distribution plot",
                           width = 700,
                           height = 450)

py.iplot(dist_plot)

pie_plot_test= go.Pie(values = test_set_meta["ddf"].value_counts(),
                 labels = ["Outside DDF","Inside DDF"],
                 hoverinfo="label+percent",
                 domain = dict(x=[0, .5],
                               y=[0.5,1]),
                 hole = 0.4
                    )

pie_plot_train= go.Pie(values = meta_training_set["ddf"].value_counts(),
                 labels = ["Outside DDF","Inside DDF"],
                 hoverinfo="label+percent",
                 domain = dict(x=[0.5, 1],
                               y=[0.5,1]),
                 hole = 0.4)

pie_plot_all= go.Pie(values = full_meta_data["ddf"].value_counts(),
                 labels = ["Outside DDF","Inside DDF"],
                 hoverinfo="label+percent",
                 domain = dict(x=[0, 1],
                               y=[0,0.5]),
                 hole = 0.4)

layout = go.Layout(title = "Object distribution according to DDF survey",
                  annotations = [dict(text = "test",
                                      font = dict(size=15),
                                      x=0.225,
                                      y=0.775,
                                      showarrow= False),
                                 dict(text = "train",
                                      font = dict(size=15),
                                      x=0.775,
                                      y=0.775,
                                     showarrow= False),
                                 dict(text = "all",
                                      font = dict(size=15),
                                      x=0.50,
                                      y=0.225,
                                      showarrow= False)]
                                    )

fig = go.Figure(data=[pie_plot_test,pie_plot_train,pie_plot_all], layout=layout)
py.iplot(fig)
inside_ddf_target = meta_training_set.groupby(['ddf']).get_group(1)['target'].value_counts()
inside_ddf_target_list = list(map(lambda x: "class_{}".format(x),inside_ddf_target.index))
bar_trace  = go.Bar(x= inside_ddf_target_list,
                 y= inside_ddf_target.values,
                 marker=dict(color='#f0000a',
                             line=dict(color='rgb(8,48,107)',width=1.5,)),
                 name = "objects count",
                 opacity=0.7,
                 hoverinfo="name + y")

layout = go.Layout(title='Count of objects per class (inside DDF survey area)',
                   xaxis=dict(tickangle=-45),
                   yaxis = dict(title = " Number of objects"))

fig = go.Figure(data=[bar_trace], layout=layout)

py.iplot(fig)
training_passband_0 = training_set[training_set['passband'] == 0]
training_passband_1 = training_set[training_set['passband'] == 1]
training_passband_2 = training_set[training_set['passband'] == 2]
training_passband_3 = training_set[training_set['passband'] == 3]
training_passband_4 = training_set[training_set['passband'] == 4]
training_passband_5 = training_set[training_set['passband'] == 5]
def plot_class_time_series (class_,training_passband):
    f, ax = plt.subplots(6,figsize=(8, 12))
    f.suptitle('class_{}'.format(class_))
    class_len = len(meta_training_set[meta_training_set['target']== class_])
    for i in range (0,6):
                       object_id = meta_training_set[meta_training_set['target'] == class_]['object_id'].values[i+np.random.randint(class_len-i)]
                       ax[i].scatter(training_passband[training_passband['object_id'] == object_id]['mjd'],
                                    training_passband[training_passband['object_id'] == object_id]['flux'])
                       ax[i].plot(training_passband[training_passband['object_id'] == object_id]['mjd'],
                                    training_passband[training_passband['object_id'] == object_id]['flux'])
                       ax[i].set_xlabel('')
                       ax[5].set_xlabel('mjd time')
    f.tight_layout()
    f.subplots_adjust(top=.95)

print ("sample of light curves of u passband (passband_0) ")
for i in range(0,5):
    plot_class_time_series(targets_classes[i],training_passband_0)