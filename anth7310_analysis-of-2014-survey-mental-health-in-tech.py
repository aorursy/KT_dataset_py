import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
!pip install chart_studio
import chart_studio.plotly as py

from plotly.subplots import make_subplots
df = pd.read_csv('survey.csv')
df.head()
df.columns
df.describe(include='all')
x = df.Gender.value_counts().index.tolist()[::-1]
y = df.Gender.value_counts().tolist()[::-1]
data = pd.DataFrame({'name': x, 'count': y})

fig = px.bar(data, x="count", y="name", orientation='h')

# Update Figure Visuals
fig.update_layout(
    height=1000, width=800, 
    template="plotly_dark",
    title_text="Gender Count" # Name of Plot
    )

fig.show()
# Interactive tool to relabel Gender Pronouns
# If Male pronoun enter 1
# If Female pronoun enter 2
# If Other, press any other button

male_pronouns = []
female_pronouns =[]
other = []

for p in df.Gender.unique():
  print(p)
  x = input()
  if x == '1': # male
    male_pronouns.append(p)
  elif x == '2': # female
    female_pronouns.append(p)
  else: # other
    other.append(p)
# After Relabeling, these are the final listing of pronouns
male_pronouns = ['M', 'Male', 'male', 'm', 'Male-ish', 'maile', 'Cis Male', 
                 'Mal', 'Male (CIS)', 'Make', 'Male ', 'Man', 'msle', 'Mail', 
                 'cis male', 'Malr', 'Cis Man']

female_pronouns = ['Female', 'female', 'Cis Female', 'F', 'Woman', 'f', 
                   'Femake', 'woman', 'Female ', 'cis-female/femme', 
                   'Female (cis)', 'femail']

other = ['Trans-female', 'something kinda male?', 'queer/she/they', 
         'non-binary', 'Nah', 'All', 'Enby', 'fluid', 'Genderqueer', 
         'Androgyne', 'Agender', 'Guy (-ish) ^_^', 'male leaning androgynous', 
         'Trans woman', 'Neuter', 'Female (trans)', 'queer', 
         'A little about you', 'p', 
         'ostensibly male, unsure what that really means']

# Change the names of some genders
reduced_pronouns = [{p: 'Male' for p in male_pronouns},
                    {p: 'Female' for p in female_pronouns},
                    {p: 'Other' for p in other}]

for p in reduced_pronouns:
  df.Gender = df.Gender.replace(p)
df.Gender.value_counts()
# check value count
x = df.Gender.value_counts().index.tolist()
y = df.Gender.value_counts().tolist()

fig = make_subplots(rows=1, cols=2)

# Initialize figure with subplots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "bar"}, {"type": "pie"}]])


DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

colors = DEFAULT_PLOTLY_COLORS[:len(x)]

# Add bar graph of Gender Count
fig.add_trace(
    go.Bar(x=x, y=y,
           text=y,
           textposition='auto',
           marker_color=colors,
           showlegend=False), 
    row=1, col=1
)

# Add pie chart of Gender Count
fig.add_trace(
    go.Pie(labels=x, values=y,
           marker_colors=colors
           ),
    row=1, col=2
)

# Update Figure Visuals
fig.update_layout(
    height=600, width=800, 
    template="plotly_dark",
    title_text="Gender Count" # Name of Plot
    )

fig.show()
x = df.state.value_counts().index.tolist()
y = df.state.value_counts().tolist()

fig = go.Figure(go.Bar(x=x,y=y))

fig.show()
import plotly.express as px
# survey sample population in USA
fig = px.choropleth(locations=x, locationmode="USA-states", color=y, 
                    scope="usa",
                    color_continuous_scale = 'Reds')

fig.update_layout(
    title_text = '2014 Mental Health Survey',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
# oldest person ever lived is 122
# assume only employing adults (>18)
idx = (df.Age >= 18) & (df.Age < 122)

x = df.Age[idx].value_counts().index.tolist()
y = df.Age[idx].value_counts().tolist()

fig = go.Figure(
    go.Bar(x=x,y=y)
    )

fig.update_layout(
    title_text = 'Age distribution',
    
)

fig.show()
# Add histogram data
idx = (df.Age >= 18) & (df.Age < 122)
x1 = df.Age[idx][df.Gender=='Male']
x2 = df.Age[idx][df.Gender=='Female']

# Group data together
hist_data = [x1, x2]

group_labels = ['Male', 'Female']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels)
fig.show()
# Add histogram data
idx = (df.Age >= 18) & (df.Age < 122) & (df.tech_company=='Yes')
x1 = df.Age[idx][df.Gender=='Male']
x2 = df.Age[idx][df.Gender=='Female']

# Group data together
hist_data = [x1, x2]

group_labels = ['Male', 'Female']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels)
fig.show()
column_names = ['Timestamp', 'Age', 'Gender', 'Country', 'state', 'self_employed',
                'family_history', 'treatment', 'work_interfere', 'no_employees',
                'remote_work', 'tech_company', 'benefits', 'care_options',
                'wellness_program', 'seek_help', 'anonymity', 'leave',
                'mental_health_consequence', 'phys_health_consequence', 'coworkers',
                'supervisor', 'mental_health_interview', 'phys_health_interview',
                'mental_vs_physical', 'obs_consequence', 'comments']
ordered_labels = {
    'Gender': ['Male', 'Female', 'Other'],
    'anonymity': ['No', 'Yes', "Don't know"],
    'benefits': ['No', 'Yes', "Don't know"],
    'care_options': ['No', 'Yes', 'Not sure'],
    'coworkers': ['No', 'Some of them', 'Yes'],
    'family_history': ['No', 'Yes'],
    'leave': ['Very difficult',
      'Somewhat difficult',
      'Somewhat easy',
      'Very easy',
      "Don't know"],
    'mental_health_consequence': ['No', 'Maybe', 'Yes'],
    'mental_health_interview': ['No', 'Maybe', 'Yes'],
    'mental_vs_physical': ['No', 'Yes', "Don't know"],
    'no_employees': ['1-5',
      '6-25',
      '26-100',
      '100-500',
      '500-1000',
      'More than 1000'],
    'obs_consequence': ['No', 'Yes'],
    'phys_health_consequence': ['No', 'Maybe', 'Yes'],
    'phys_health_interview': ['No', 'Maybe', 'Yes'],
    'remote_work': ['No', 'Yes'],
    'seek_help': ['No', 'Yes', "Don't know"],
    'self_employed': ['No', 'Yes'],
    'supervisor': ['No', 'Some of them', 'Yes'],
    'tech_company': ['No', 'Yes'],
    'treatment': ['No', 'Yes'],
    'wellness_program': ['No', 'Yes', "Don't know"],
    'work_interfere': ['Never', 'Rarely', 'Sometimes', 'Often']
    }
# find all columns with at most 5 unique entries for bar graph
bars = []
for c in df.columns:
  if len(df[c].unique()) <=6:
    bars.append(c)

size = round(len(bars)**(1/2))

fig = make_subplots(
    rows=size, cols=size,
    subplot_titles=bars)

j = 1
for i, c in enumerate(bars):
  x_temp = df[c].value_counts().index.tolist()
  y_temp = df[c].value_counts().tolist()

  # resort x labels
  x = []
  y = []
  for label in ordered_labels[c]:
    try:
      index = x_temp.index(label)
      x.append(x_temp[index])
      y.append(y_temp[index])
    except Exception as e:
      x.append(label)
      y.append(0)

  if i % size == 0 and i > 0:
      j += 1

  # add figure to subplot
  fig.add_trace(go.Bar(x=x, y=y, 
                       text=y,
                       textposition='auto',
                       showlegend=False),
                row=j, col=(i % size)+1)
  
fig.update_layout(title_text="2014 Health Survey In Tech",
                  height=1000, 
                  # width=800, 
                  )

fig.show()
import plotly.graph_objects as go
def stacked_horizontal_bar(x_data, y_data, top_labels):
  # function to display stacked horizontal bar chart

  colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
          'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
          'rgba(190, 192, 213, 1)']
  
  colors = colors[:len(top_labels)]

  fig = go.Figure()

  for i in range(0, len(x_data[0])):
      for xd, yd in zip(x_data, y_data):
          fig.add_trace(go.Bar(
              x=[xd[i]], y=[yd],
              width=0.75,
              orientation='h',
              marker=dict(
                  color=colors[i],
                  line=dict(color='rgb(248, 248, 249)', width=1)
              )
          ))

  fig.update_layout(
      xaxis=dict(
          showgrid=False,
          showline=False,
          showticklabels=False,
          zeroline=False,
          domain=[0.15, 1]
      ),
      yaxis=dict(
          showgrid=False,
          showline=False,
          showticklabels=False,
          zeroline=False,
      ),
      barmode='stack',
      paper_bgcolor='rgb(248, 248, 255)',
      plot_bgcolor='rgb(248, 248, 255)',
      margin=dict(l=120, r=10, t=100, b=80),
      showlegend=False,
  )

  annotations = []

  for yd, xd in zip(y_data, x_data):
      # labeling the y-axis
      annotations.append(dict(xref='paper', yref='y',
                              x=0.14, y=yd,
                              xanchor='right',
                              text=str(yd),
                              font=dict(family='Arial', size=14,
                                        color='rgb(67, 67, 67)'),
                              showarrow=False, align='right'))
      # labeling the first percentage of each bar (x_axis)
      annotations.append(dict(xref='x', yref='y',
                              x=xd[0] / 2, y=yd,
                              text=str(xd[0]) + '%',
                              font=dict(family='Arial', size=14,
                                        color='rgb(248, 248, 255)'),
                              showarrow=False))
      # labeling the first Likert scale (on the top)
      if yd == y_data[-1]:
          annotations.append(dict(xref='x', yref='paper',
                                  x=xd[0] / 2, y=1.1,
                                  text=top_labels[0],
                                  font=dict(family='Arial', size=14,
                                            color='rgb(67, 67, 67)'),
                                  showarrow=False))
      space = xd[0]
      for i in range(1, len(xd)):
              # labeling the rest of percentages for each bar (x_axis)
              annotations.append(dict(xref='x', yref='y',
                                      x=space + (xd[i]/2), y=yd,
                                      text=str(xd[i]) + '%',
                                      font=dict(family='Arial', size=14,
                                                color='rgb(248, 248, 255)'),
                                      showarrow=False))
              # labeling the Likert scale
              if yd == y_data[-1]:
                  annotations.append(dict(xref='x', yref='paper',
                                          x=space + (xd[i]/2), y=1.1,
                                          text=top_labels[i],
                                          font=dict(family='Arial', size=14,
                                                    color='rgb(67, 67, 67)'),
                                          showarrow=False))
              space += xd[i]

  fig.update_layout(annotations=annotations)

  fig.show()
top_labels = ['No', 'Yes']

y_data = ['tech_company',
          'self_employed',
          'family_history',
          'treatment',
          'remote_work',
          'obs_consequence'
]
# reorder because labeling is inverted
y_data = list(reversed(y_data))

# return percentages
x_data = []
for label in y_data:
  d = df[label].value_counts() / sum(df[label].value_counts())
  d = round(d*100, 0).apply(int)
  x_data.append(list(d))

y_data_description = {
    'self_employed' : 'Are you self-employed?',

    'family_history' : 'Do you have a family <br>'+ 
                      'history of mental illness?',

    'treatment' : 'Have you sought treatment <br>' +
                  'for a mental health condition?',   

    'remote_work' : 'Do you work remotely<br>'+ 
                  'at least 50% of the time?',

    'tech_company' : 'Is your employer primarily<br>'+
                    'a tech company/organization?',

    'obs_consequence' : 'Have you heard of or observed<br>'+
                        'negative consequences for<br>'+
                        'coworkers with mental health<br>' + 
                        'conditions in your workplace?'
}


y_data = [y_data_description[y] for y in y_data]

stacked_horizontal_bar(x_data, y_data, top_labels)
top_labels = ['No', 'Yes', 'Don\'t Know']

y_data = ['benefits',
          'care_options',
          'wellness_program',
          'mental_vs_physical',
          'seek_help',
          'anonymity'
]

# reorder because labeling is inverted
y_data = list(reversed(y_data))

# return percentages
x_data = []
for label in y_data:
  d = df[label].value_counts() / sum(df[label].value_counts())
  d = round(d*100, 0).apply(int)
  x_data.append(list(d))

y_data_description = {
    'benefits' : 
      'Does your employer provide<br>' +
      'mental health benefits?',

    'care_options' : 
      'Do you know the options for<br>' +
      'mental health care your employer provides?',

    'wellness_program' : 
      'Has your employer ever discussed<br>' + 
      'mental health as part of an employee<br>' +
      'wellness program?',   

    'mental_vs_physical': 
      'Do you feel that your employer takes<br>' + 
      'mental health as seriously as<br>' + 
      'physical health?',

    'seek_help' : 
      'Does your employer provide resources<br>' +
      'to learn more about mental health issues<br>' + 
      'and how to seek help?',

    'anonymity' : 
      'Is your anonymity protected if you choose<br>' + 
      'to take advantage of mental health or<br>' +
      'substance abuse treatment resources?',
}


y_data = [y_data_description[y] for y in y_data]

stacked_horizontal_bar(x_data, y_data, top_labels)
top_labels = ['No', 'Maybe', 'Yes']

y_data = ['mental_health_consequence',
          'mental_health_interview',
          'phys_health_consequence',
          'phys_health_interview'
]

# reorder because labeling is inverted
y_data = list(reversed(y_data))

# return percentages
x_data = []
for label in y_data:
  d = df[label].value_counts() / sum(df[label].value_counts())
  d = round(d*100, 0).apply(int)
  x_data.append(list(d))

y_data_description = {
    'mental_health_consequence' : 
      'Do you think that discussing a<br>' +
      'mental health issue with your<br>' +
      'employer would have<br>' + 
      'negative consequences?',


    'phys_health_consequence' : 
      'Do you think that discussing a<br>' + 
      'physical health issue with your<br>' + 
      'employer would have<br>' + 
      'negative consequences?',

    'mental_health_interview' : 
      'Would you bring up a mental health<br>' + 
      'issue with a potential<br>' + 
      'employer in an interview?', 

    'phys_health_interview': 
      'Would you bring up a physical<br>' + 
      'health issue with a potential<br>' + 
      'employer in an interview?'
}


y_data = [y_data_description[y] for y in y_data]

stacked_horizontal_bar(x_data, y_data, top_labels)
top_labels = ['No', 'Some of them', 'Yes']

y_data = ['coworkers',
          'supervisor',
]

# reorder because labeling is inverted
y_data = list(reversed(y_data))

# return percentages
x_data = []
for label in y_data:
  d = df[label].value_counts() / sum(df[label].value_counts())
  d = round(d*100, 0).apply(int)
  x_data.append(list(d))

y_data_description = {
    'coworkers' : 
      'Would you be willing to discuss<br>'+
      'a mental health issue<br>' +
      'with your coworkers?',

    'supervisor' : 
      'Would you be willing to discuss<br>' +
      'a mental health issue<br>' +
      'with your direct supervisor(s)?'
}


y_data = [y_data_description[y] for y in y_data]

stacked_horizontal_bar(x_data, y_data, top_labels)
# MALE VS FEMALE
def cat_plot(df, CAT_x, CAT_y, title=""):
  """
  Categorical plot to display between 2 variables
  Displays the plot

  CAT_x is x-axis
  CAT_y is y-axis

  title is name of entire plot
  """

  CAT_x, CAT_y = CAT_y, CAT_x # to lazy to change code below, easier to swap

  fig = make_subplots(
      rows=1, cols=1,
      subplot_titles=[f'{CAT_y} vs. {CAT_x}'])

  for label_x in ordered_labels[CAT_x]:

    data = df[CAT_y][df[CAT_x] == label_x].value_counts()

    
    x_temp = data.index.tolist()
    y_temp = data.tolist()

    # resort x labels
    x = []
    y = []
    for label_y in ordered_labels[CAT_y]:
      try: # element not in list
        index = x_temp.index(label_y)
        x.append(x_temp[index])
        y.append(y_temp[index])
      except Exception as e:
        x.append(label_y)
        y.append(0)


    fig.add_trace(go.Bar(x=x,
                        y=y,
                        text=y,
                        textposition='auto',
                        showlegend=True,
                        name=label_x))
  
  fig.update_layout(title_text=f"2014 Health Survey In Tech-{title}",
                  height=300, 
                  width=800, 
                  )

  fig.show()
column_names = ['Timestamp', 'Age', 'Gender', 'Country', 'state', 'self_employed',
                'family_history', 'treatment', 'work_interfere', 'no_employees',
                'remote_work', 'tech_company', 'benefits', 'care_options',
                'wellness_program', 'seek_help', 'anonymity', 'leave',
                'mental_health_consequence', 'phys_health_consequence', 'coworkers',
                'supervisor', 'mental_health_interview', 'phys_health_interview',
                'mental_vs_physical', 'obs_consequence', 'comments']
CAT_x = 'family_history' # category on x label
CAT_y = 'obs_consequence' # category on y label
cat_plot(df,
    CAT_x, CAT_y)
def cat_plot_3(df, x, y, z):
  """
  3 dim categorical plot
  x is x-axis
  y is y-axis
  z is different plot
  """
  for label in ordered_labels[z]:
    cat_plot(df[df[z] == label],
             CAT_x, CAT_y,
             title=f'{z}={label}')

CAT_x = 'family_history' # category on x label
CAT_y = 'obs_consequence' # category on y label
CAT_z = 'tech_company'

cat_plot_3(df, CAT_x, CAT_y, CAT_z)
# Python program to generate WordCloud 
  
# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from tqdm.auto import tqdm # loading bar
  
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in tqdm(df.comments[~df.comments.isnull()]):
      
  # typecaste each val to string 
  val = str(val) 

  # split the value 
  tokens = val.split() 
    
  # Converts each token into lowercase 
  for i in range(len(tokens)): 
      tokens[i] = tokens[i].lower() 
    
  comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
