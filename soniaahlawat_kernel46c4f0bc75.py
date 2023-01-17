# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        naukri_data = os.path.join(dirname, filename)

data = pd.read_csv(naukri_data)
data.head(5)
data.head(10)
data.describe()
print("Dataset has {} rows and {} columns in the dataset.".format(data.shape[0], data.shape[1]))
data.columns  #the attributes 
data.isnull().sum() # the column-wise sum of missing values
print("Missing Data")

plt.figure(figsize=(20,5))
sns.heatmap(data.isnull())
            
plt.show()


missing_percent= (data.isnull().sum()/len(data))[(data.isnull().sum()/len(data))>0].sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percentage':missing_percent*100})
print(missing_data)


mis_data= (data.isnull().sum() / len(data)) * 100
print(mis_data)
mis_data= mis_data.drop(mis_data[mis_data == 0].index).sort_values(ascending=True)
plt.figure(figsize=(20,5), facecolor = 'yellow')
plt.title('MISSING DATA')
plt.xlabel('mis_data.index')
plt.ylabel('mis_data')
sns.barplot(x=mis_data.index,y=mis_data)

(data.info())
print("Job Title")

data['Job Title'].value_counts().head(15)
print("Role")

data['Role'].value_counts().head(15)
print("Location")

data['Location'].value_counts().head(15)
data[data['Location']=='Bengaluru']['Job Title'].value_counts().head(10)
industry = data["Industry"].value_counts()[:10]

industry.plot(kind = "barh",color = "maroon") 
plt.title("Industry")
plt.show()
data_role = data['Role Category'].value_counts().nlargest(n=10)
figure = px.pie(data_role, 
       values = data_role.values, 
       names = data_role.index, 
       title = "Role Categories (Top 10)", 
       color = data_role.values,
       color_discrete_sequence = px.colors.qualitative.Dark24)
figure.update_traces(opacity = 1,
                  marker_line_color = 'rgb(28,88,107)',
                  marker_line_width = 2)
figure.update_layout(title_x = 0.5)
figure.show()
data_salary = data['Job Salary'].value_counts().nlargest(n=10)
figure = px.pie(data_salary, 
       values = data_salary.values, 
       names = data_salary.index, 
       title = "Job Salary (Top 10)", 
       color = data_salary.values,
       color_discrete_sequence = px.colors.qualitative.Dark2)
figure.update_traces(opacity = 1,
                  marker_line_color = 'rgb(28,88,107)',
                  marker_line_width = 2)
figure.update_layout(title_x = 0.5)
figure.show()
data_location = data['Location'].value_counts().nlargest(n=10)
figure = px.pie(data_location, 
       values = data_location.values, 
       names = data_location.index, 
       title = "Locations (Top 10)", 
       color = data_location.values,
       color_discrete_sequence = px.colors.qualitative.D3)
figure.update_traces(opacity = 1,
                  marker_line_color = 'rgb(58,100,157)',
                  marker_line_width = 2)
figure.update_layout(title_x = 0.5)
figure.show()
print("EXPERIENCE")

experience = data["Job Experience Required"].value_counts().nlargest(n=10)
job_experience = data["Job Experience Required"].value_counts()
figure = make_subplots(1,2,
                    subplot_titles = ["Top 10 experience ranges", 
                                      "All experience ranges"])
figure.append_trace(go.Bar(y = experience.index,
                          x = experience, 
                          orientation='h',
                          marker = dict(color = experience.values, coloraxis="coloraxis", showscale=False),
                          texttemplate = "%{value:,s}",
                          textposition = "outside",
                          name="Top 10 experience ranges",
                          showlegend = False),
                
                 row = 1,
                 col = 1)
figure.update_traces(opacity = 1)
figure.update_layout(coloraxis = dict(colorscale='rainbow'))
figure.append_trace(go.Scatter(x = job_experience.index,
                          y = job_experience,
                        line=dict(color="#008B8B",
                                    width=2),
                          showlegend = True),
                 row = 1,
                 col = 2)
figure.update_layout(showlegend = False)
figure.show()
experience = data["Job Experience Required"].value_counts().nlargest(n=20)
salary = data["Job Salary"].value_counts()
figure = make_subplots(2,1,
                    subplot_titles = ["Top 10 experience ranges", 
                                      "Job Salary ranges"])
figure.append_trace(go.Bar(y = experience.index,
                          x = experience, 
                          orientation='h',
                          marker = dict(color = experience.values, coloraxis="coloraxis", showscale=False),
                          texttemplate = "%{value:,s}",
                          textposition = "outside",
                          name="Top 10 experience ranges",
                          showlegend = False),
                
                 row = 1,
                 col = 1)
figure.update_traces(opacity = 1)
figure.update_layout(coloraxis = dict(colorscale='rainbow'))
figure.append_trace(go.Scatter(x = experience.index,
                          y = salary.index,
                        line=dict(color="#008B8B",
                                    width=2),
                          showlegend = True),
                 row = 2,
                 col = 1)
figure.update_layout(showlegend = False)
figure.show()
title = data['Job Title'].value_counts().nlargest(n=10)
figure = make_subplots(2, 1,
                    subplot_titles=["Job Titles", 
                                    "Job Titles in Mumbai, Bengaluru and Delhi NCR (Top 10)"])
figure.append_trace(go.Bar(y = title.index,
                        x = title.values,
                        orientation = 'h',
                        marker = dict(color = title.values, coloraxis = "coloraxis"),
                        texttemplate = "%{value:,s}",
                        textposition = "inside",
                        showlegend = False),
                  row = 1,
                  col = 1)
figure.update_layout(coloraxis_showscale = False)
figure.update_layout(height = 1000, 
                  width = 800,
                  yaxis = dict(autorange = "reversed"),
                  coloraxis = dict(colorscale = 'portland'),
                  coloraxis_colorbar = dict(yanchor ="top", y=1, x=0)
)

figure.append_trace(go.Bar(y = data[data['Location'].isin(['Mumbai'])]['Job Title'].value_counts().nlargest(n=10).index,
                        x = data[data['Location'].isin(['Mumbai'])]['Job Title'].value_counts().
                        nlargest(n = 10).values,
                        marker_color = 'darkturquoise',
                        orientation ='h',
                        showlegend = True,
                        name = "Mumbai"),
                row = 2,
                col = 1)

figure.append_trace(go.Bar(y = data[data['Location'].isin(['Bengaluru'])]['Job Title'].value_counts().nlargest(n=10).index,
                        x = data[data['Location'].isin(['Bengaluru'])]['Job Title'].value_counts().nlargest(n=10).values,
                        marker_color = 'mediumorchid',
                        orientation = 'h',
                        showlegend = True,
                        name = "Bengaluru"),
                row = 2,
                col = 1,
                )

figure.append_trace(go.Bar(y = data[data['Location'].isin(['Delhi NCR'])]['Job Title'].value_counts().nlargest(n=10).index,
                        x = data[data['Location'].isin(['Delhi NCR'])]['Job Title'].value_counts().nlargest(n=10).values,
                        marker_color = 'seagreen',
                        orientation = 'h',
                        showlegend = True,
                        name = "Delhi NCR"),
                row = 2,
                col = 1,
                )
figure.update_layout(legend=dict(x = 1,
                              y = 0.2))
figure.show()
experience = data["Job Experience Required"].value_counts().nlargest(n=10)
job_experience = data["Job Experience Required"].value_counts()
figure = make_subplots(2, 1,
                    subplot_titles =["Job Experience Required (Top 10)", 
                                    "Experience Required in Mumbai, Bengaluru and Delhi NCR (Top 10)"])
figure.append_trace(go.Bar(y = experience.index,
                        x = experience.values,
                        orientation = 'h',
                        marker = dict(color = experience.values, coloraxis = "coloraxis"),
                        texttemplate = "%{value:,s}",
                        textposition = "inside",
                        showlegend = False),
                  row = 1,
                  col = 1)
figure.update_layout(coloraxis_showscale = False)
figure.update_layout(height = 1000, 
                  width = 800,
                  yaxis = dict(autorange = "reversed"),
                  coloraxis = dict(colorscale = 'portland'),
                  coloraxis_colorbar = dict(yanchor ="top", y=1, x=0)
)

figure.append_trace(go.Bar(y = data[data['Location'].isin(['Mumbai'])]['Job Experience Required'].value_counts().nlargest(n=10).index,
                        x = data[data['Location'].isin(['Mumbai'])]['Job Experience Required'].value_counts().
                        nlargest(n = 10).values,
                        marker_color = 'darkturquoise',
                        orientation ='h',
                        showlegend = True,
                        name = "Mumbai"),
                row = 2,
                col = 1)

figure.append_trace(go.Bar(y = data[data['Location'].isin(['Bengaluru'])]['Job Experience Required'].value_counts().nlargest(n=10).index,
                        x = data[data['Location'].isin(['Bengaluru'])]['Job Experience Required'].value_counts().nlargest(n=10).values,
                        marker_color = 'mediumorchid',
                        orientation = 'h',
                        showlegend = True,
                        name = "Bengaluru"),
                row = 2,
                col = 1,
                )

figure.append_trace(go.Bar(y = data[data['Location'].isin(['Delhi NCR'])]['Job Experience Required'].value_counts().nlargest(n=10).index,
                        x = data[data['Location'].isin(['Delhi NCR'])]['Job Experience Required'].value_counts().nlargest(n=10).values,
                        marker_color = 'seagreen',
                        orientation = 'h',
                        showlegend = True,
                        name = "Delhi NCR"),
                row = 2,
                col = 1,
                )
figure.update_layout(legend=dict(x = 1,
                              y = 0.2))
figure.show()
salary = data["Job Salary"].value_counts().nlargest(n=10)
figure = make_subplots(2, 1,
                    subplot_titles =["Job Salary(Top 10)", 
                                    "Job Salary in Mumbai, Bengaluru and Delhi NCR (Top 10)"])
figure.append_trace(go.Bar(y = salary.index,
                        x = salary.values,
                        orientation = 'h',
                        marker = dict(color = salary.values, coloraxis = "coloraxis"),
                        texttemplate = "%{value:,s}",
                        textposition = "inside",
                        showlegend = False),
                  row = 1,
                  col = 1)
figure.update_layout(coloraxis_showscale = False)
figure.update_layout(height = 1000, 
                  width = 800,
                  yaxis = dict(autorange = "reversed"),
                  coloraxis = dict(colorscale = 'portland'),
                  coloraxis_colorbar = dict(yanchor ="top", y=1, x=0)
)

figure.append_trace(go.Bar(y = data[data['Location'].isin(['Mumbai'])]['Job Salary'].value_counts().nlargest(n=10).index,
                        x = data[data['Location'].isin(['Mumbai'])]['Job Salary'].value_counts().
                        nlargest(n = 10).values,
                        marker_color = 'darkturquoise',
                        orientation ='h',
                        showlegend = True,
                        name = "Mumbai"),
                row = 2,
                col = 1)

figure.append_trace(go.Bar(y = data[data['Location'].isin(['Bengaluru'])]['Job Salary'].value_counts().nlargest(n=10).index,
                        x = data[data['Location'].isin(['Bengaluru'])]['Job Salary'].value_counts().nlargest(n=10).values,
                        marker_color = 'mediumorchid',
                        orientation = 'h',
                        showlegend = True,
                        name = "Bengaluru"),
                row = 2,
                col = 1,
                )

figure.append_trace(go.Bar(y = data[data['Location'].isin(['Delhi NCR'])]['Job Salary'].value_counts().nlargest(n=10).index,
                        x = data[data['Location'].isin(['Delhi NCR'])]['Job Salary'].value_counts().nlargest(n=10).values,
                        marker_color = 'seagreen',
                        orientation = 'h',
                        showlegend = True,
                        name = "Delhi NCR"),
                row = 2,
                col = 1,
                )
figure.update_layout(legend=dict(x = 1,
                              y = 0.2))
figure.show()
function = data['Functional Area'].value_counts().nlargest(n=10)
figure = px.pie(function, 
       values = function.values, 
       names = function.index, 
       title = "Functional Area (Top 10)", 
       color = function.values,
       color_discrete_sequence = px.colors.qualitative.D3)
figure.update_traces(opacity = 1,
                  marker_line_color = 'rgb(58,100,157)',
                  marker_line_width = 2)
figure.update_layout(title_x = 0.8)
figure.show()

skill = data['Key Skills'].str.split("|", n = 10, expand = True) 

data['skill_1'] = skill[0]
data['skill_2'] = skill[1]
data['skill_3'] = skill[2]
data['skill_4'] = skill[3]


skill
from wordcloud import WordCloud, STOPWORDS
cloud_1 = data['skill_1'].values 

wordcloud = WordCloud(max_font_size=100, max_words=30000, background_color="white").generate(str(cloud_1))

plt.imshow(wordcloud)
plt.axis("off")
plt.show()
from wordcloud import WordCloud, STOPWORDS
cloud_2 = data['skill_2'].values 

wordcloud = WordCloud(max_font_size=100, max_words=30000, background_color="white").generate(str(cloud_2))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()