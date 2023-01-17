# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Add plotly library
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
credits=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
movies=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
movies.columns
movies.head(5)
# Way 1: Find the index of top 10 acc. to runTime
#movieInd=[]
#for i in runTime:
#    movieInd.append(list(duration).index(i))
    
# Way 2: Find the index of top 10 acc. to runTime
new_index = (movies['runtime'].sort_values(ascending=False)).index.values
sorted_data = movies.reindex(new_index)[:11]

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['title'], y=sorted_data['runtime'])
plt.xticks(rotation= -15)
plt.xlabel('Name of the Movie')
plt.ylabel('Duration of the Movie (minute)')
plt.title('Top 10 Movie with the Longest Duration')
plt.show()
# Draw the barplot with Plotly Library

# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = sorted_data.title,
                y = sorted_data.runtime,
                name = "Duration of the Movie",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = sorted_data.title)
data = trace1
layout = go.Layout(barmode = "relative")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
new_index = (movies['revenue'].sort_values(ascending=False)).index.values
sorted_data = movies.reindex(new_index)[:11]
profit = sorted_data[["budget", "revenue"]]

Ind=[]    
for i in range(0,11):
    Ind.append(list(profit["revenue"])[i]-list(profit["budget"])[i])  
print("Calculation Result: {}".format(Ind))

sorted_data['Profit']=pd.DataFrame({"Profit" : Ind}) 
print("Data Result: {}".format(np.array(sorted_data['Profit'])))
sorted_data.columns
data = sorted_data

# visualize
f,ax1 = plt.subplots(figsize =(16,8))
sns.pointplot(x='title',y='budget',data=data,color='lime',alpha=0.8)
sns.pointplot(x='title',y='revenue',data=data,color='red',alpha=0.8)
sns.pointplot(x='title',y='Profit',data=data,color='black',alpha=0.8)
plt.text(40,0.55,'budget',color='lime',fontsize = 18,style = 'italic')
plt.text(40,0.6,'revenue',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'Profit',color='black',fontsize = 18,style = 'italic')
plt.xlabel('Name',fontsize = 15,color='blue')
plt.ylabel('Values (billion)',fontsize = 15,color='blue')
plt.title('Analysis of the Movies for Budget, Revenue, Profit ',fontsize = 20,color='blue')
plt.xticks(rotation= -18)
plt.grid()
# prepare data frames
R = sorted_data["revenue"]
B = sorted_data["budget"]
P = sorted_data["Profit"]
N = sorted_data["title"]

# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = N,
                    y = R,
                    mode = "markers",
                    name = "revenue",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'), # 255 değeri red-green-blue arası bir değer ile rengi verir. 0.8 ise saydamlığı ifade eder
                    text= N)
# creating trace2
trace2 =go.Scatter(
                    x = N,
                    y = B,
                    mode = "markers",
                    name = "budget",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= N)
# creating trace3
trace3 =go.Scatter(
                    x = N,
                    y = P,
                    mode = "markers",
                    name = "profit",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= N)
data = [trace1, trace2, trace3]
layout = dict(title = 'Analysis of the Movies for Budget, Revenue, Profit',
              xaxis= dict(title= 'Name of the Movies',ticklen= 5,zeroline= False), # kalınlığı ifade eder
              yaxis= dict(title= 'Values',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
movies.head(2)
new_index = (movies['vote_average'].sort_values(ascending=False)).index.values
sorted_data = movies.reindex(new_index)
Count_T=np.mean(sorted_data["vote_count"])
sorted_data=sorted_data[sorted_data["vote_count"]>Count_T][:10] # Choose the vote number bigger than Count_T
# vote numbers per movies
labels = sorted_data.title
#colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0,0,0,0,0]
sizes = sorted_data.vote_count.values

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')
plt.title('Vote Counts According to the Movie Names',color = 'blue',fontsize = 15)
plt.show()
# vote numbers per movies

pie1 = sorted_data.vote_count
#pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4
labels = sorted_data.title
# figure
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Number of Vote",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Vote Numbers per Movies",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": True,
              "text": "Percentage of Vote",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)
# Success rate is going to be calculated with this equation: (assume 10000=vote_count, 10=vote_average)
av=10
cnt=10000
i=0
Success_Rate=[]
for i in range(0,10):
    Success_Rate.append(((list(sorted_data["vote_average"])[i]/av)*100*0.85)+((list(sorted_data["vote_count"])[i]/cnt)*100*0.15))
sorted_data['Success_Rate'] = Success_Rate
sorted_data.columns
# Here we see the most successful three movies

# create trace1 
trace1 = go.Scatter(
                x = sorted_data.title,
                y = ((sorted_data.vote_count)/100),
                mode = "lines",
                name = "Vote Numbers",
                marker = dict(color = 'rgba(16, 112, 255, 0.8)'),
                text = sorted_data.title)
# create trace2 
trace2 = go.Scatter(
                x = sorted_data.title,
                y = ((sorted_data.vote_average)*10),
                name = "Vote Rate",
                mode = "lines",
                marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                text = sorted_data.title)
# create trace3 
trace3 = go.Scatter(
                x = sorted_data.title,
                y = sorted_data.Success_Rate,
                name = "Success Rate",
                mode = "lines+markers",
                marker = dict(color = 'rgba(255, 0, 0, 1)',
                          line=dict(color='rgb(0,0,0)',width=1.5)),
                text = sorted_data.title)

data = [trace1, trace2, trace3]
layout = dict(title = 'Movie Success Rate of Top 10 Universities ',
              xaxis= dict(title= 'Most Successful 10 Movie',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Value in Percentage (%) ',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


