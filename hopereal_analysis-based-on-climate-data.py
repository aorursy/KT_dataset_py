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
import plotly.graph_objs as go

import plotly.express as px

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
file= pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv",

              encoding='cp936',parse_dates=['dt'])
file.info()
file.head(3)
def changeview(fig):

    fig.update_layout(

    yaxis=dict(

            showline=False,

            showgrid=False,

            showticklabels=True,

            linecolor='rgb(204, 204, 204)',

            linewidth=2,

            title='Average temp',

            title_font_color='rgb(204, 204, 204)',

            ticks='outside',

            tickfont=dict(

                family='Arial',

                size=15,

                color='#75878a',

            )),

    xaxis=dict(

            showline=False,

            showgrid=False,

            showticklabels=True,

            linecolor='rgb(204, 204, 204)',

            linewidth=2,

            title='Year',

            title_font_color='rgb(204, 204, 204)',

            ticks='outside',

            tickfont=dict(

                family='Arial',

                size=15,

                color='#75878a',

            )),

    plot_bgcolor='#333333',

        paper_bgcolor='#333333'

    )



    annotations = []

    # Title

    annotations.append(dict(xref='paper', yref='paper', x=0.25, y=1.005,

                                  xanchor='left', yanchor='bottom',

                                  text='Average temp for each year',

                                  font=dict(family='Arial',

                                            size=30,

                                            color='#bacac6'),

                                  showarrow=False))

# Source

    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=-0.11,

                                  xanchor='center', yanchor='top',

                                  text='Source:GlobalTemperatures.csv ',

                                  font=dict(family='Arial',

                                            size=12,

                                            color='rgb(150,150,150)'),

                                  showarrow=False))

    fig.update_layout(annotations=annotations)

    fig.update_traces(marker_color='#478384')
#Regularized polynomial regression

def ax_bfit_reg(list_x,list_y,predict_year,key_n,lamda,label_x,label_y,title,line_color,line_name):

    x_test = np.linspace(min(list_x)[0],max(list_x)[0],100)[:,None]

    X = np.ones_like(list_x)

    X_test = np.ones_like(x_test)

    for i in range(1,key_n+1):

        X = np.hstack((X,list_x**i))

        X_test = np.hstack((X_test,x_test**i))

    w = np.linalg.solve(np.dot(X.T,X) + list_x.size*lamda*

                        np.identity(key_n+1),np.dot(X.T,list_y))

    coef_=w

    

    #new_line

    attach_list=np.array([])

    for i in range(key_n+1):

        attach_list=np.append(attach_list,predict_year**i)

    X_NEW=np.append(X_test,attach_list)

    X_NEW=X_NEW.reshape(int(len(X_NEW)/(key_n+1)),key_n+1)

    #y_value

    f_test = np.dot(X_NEW,w)

    x_pro=np.append(x_test,[predict_year])

    y_pro=[i[0] for i in f_test]

    x_ax=[i[0]for i in list_x]

    y_ax=[i[0] for i in list_y]

    dic={label_x:x_ax,label_y:y_ax}

    answer_sourse=pd.DataFrame(dic)

    reference_line=go.Scatter(x=x_pro,y=y_pro,mode='lines', line_shape='spline',line_color=line_color,name=line_name,

                              showlegend=False)

    print ('When the condition：%s is %s ，we predict %s will be %.6f '%(label_x,predict_year,label_y,f_test[-1]))

    return reference_line,coef_

#The function has seven inputs

#list_x         

#list_y         

#predict_year   

#key_n           

#lamda           

#label_x        

#label_y         

#title           

#line_color

#line_name
average_temp=file.groupby(file.dt.dt.year).mean()

fig=px.scatter(average_temp,x=average_temp.index,

            y=average_temp.LandAverageTemperature.values)

changeview(fig)
from sklearn.linear_model import Ridge

rid=Ridge(alpha=.01)

rid.fit([[i] for i in average_temp.index],average_temp.LandAverageTemperature)
ydata_adjust=np.array([])

for i in average_temp.index:

    ydata_adjust=np.append(ydata_adjust,i*rid.coef_)
reference_line=go.Scatter(x=average_temp.index,

                          y=ydata_adjust,mode='lines',line=go.scatter.Line(color='grey'),

                          line_shape='spline',showlegend=False,line_color='#339966',legendgroup='Ridge Regression',name='Ridge Regression')
fig.add_trace(reference_line)
average_temp=file.groupby(file.dt.dt.year).mean()

fig=px.scatter(average_temp,x=average_temp.index,

            y=average_temp.LandAverageTemperature.values)

changeview(fig)

xlist=np.array([[float(i)]for i in average_temp.index ])

ylist=np.array([[float(i)]for i in average_temp.LandAverageTemperature.values])
trace_pro,coef=ax_bfit_reg(list_x=xlist,list_y=ylist,predict_year=2016,

            key_n=3,lamda=.02,label_x='x',label_y='y',title='title',line_color='grey',line_name='average temp')
changeview(fig)

fig.add_trace(trace_pro)
max_temp_file=average_temp['LandMaxTemperature']

min_temp_file=average_temp['LandMinTemperature']

max_temp_file.dropna(how='any',inplace=True)

min_temp_file.dropna(how='any',inplace=True)

xlist_max=np.array([[float(i)]for i in max_temp_file.index ])

xlist_min=np.array([[float(i)]for i in min_temp_file.index ])

ylist_max=np.array([[float(i)]for i in max_temp_file.values])

ylist_min=np.array([[float(i)]for i in min_temp_file.values])

trace_max,coef_max=ax_bfit_reg(list_x=xlist_max,list_y=ylist_max,predict_year=2016,

            key_n=3,lamda=.02,label_x='x',label_y='y',title='title',line_color='#990033',line_name='max temp')



trace_min,coef_min=ax_bfit_reg(list_x=xlist_min,list_y=ylist_min,predict_year=2016,

            key_n=3,lamda=.02,label_x='x',label_y='y',title='title',line_color='#6699CC',line_name='mix temp')

fig.add_trace(trace_max)

fig.add_trace(trace_min)
def guess_previous(list_x,w,key_n,line_color,line_name):

    x_test = np.linspace(min(list_x)[0],max(list_x)[0],100)[:,None]

    X = np.ones_like(list_x)

    X_test = np.ones_like(x_test)

    for i in range(1,key_n+1):

        X = np.hstack((X,list_x**i))

        X_test = np.hstack((X_test,x_test**i))

    f_test = np.dot(X_test,w)

    x_pro=[i[0] for i in x_test]

    y_pro=[i[0] for i in f_test]

    reference_line=go.Scatter(x=x_pro,y=y_pro,mode='lines', line_shape='spline',

                              line_color=line_color,name=line_name,showlegend=False)

    return reference_line
previous_max=guess_previous(np.array([[float(i)]for i in range(int(xlist[0][0]),int(xlist_max[0][0]))]),coef_max,3,'#ddada6','previous max temp')

previous_min=guess_previous(np.array([[float(i)]for i in range(int(xlist[0][0]),int(xlist_min[0][0]))]),coef_min,3,'#aacdbf','previous min temp')
fig.add_trace(previous_max)

fig.add_trace(previous_min)