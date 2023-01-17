import pandas as pd
import numpy as np # package for linear algebra
from keras.preprocessing import text, sequence

import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import calendar
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm


# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

import os
print(os.listdir("../input"))
from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
%matplotlib inline
path='../input/io/'
dtypes_donations = {'Project ID' : 'object','Donation ID':'object','device': 'object','Donor ID' : 'object','Donation Included Optional Donation': 'object','Donation Amount': 'float64','Donor Cart Sequence' : 'int64'}
donations = pd.read_csv(path+'Donations.csv',dtype=dtypes_donations)

dtypes_donors = {'Donor ID' : 'object','Donor City': 'object','Donor State': 'object','Donor Is Teacher' : 'object','Donor Zip':'object'}
donors = pd.read_csv(path+'Donors.csv', low_memory=False,dtype=dtypes_donors)

dtypes_schools = {'School ID':'object','School Name':'object','School Metro Type':'object','School Percentage Free Lunch':'float64','School State':'object','School Zip':'int64','School City':'object','School County':'object','School District':'object'}
schools = pd.read_csv(path+'Schools.csv', dtype=dtypes_schools)#error_bad_lines=False

dtypes_teachers = {'Teacher ID':'object','Teacher Prefix':'object','Teacher First Project Posted Date':'object'}
teachers = pd.read_csv(path+'Teachers.csv', dtype=dtypes_teachers)#error_bad_lines=False,
                   
dtypes_projects = {'Project ID' : 'object','School ID' : 'object','Teacher ID': 'object','Teacher Project Posted Sequence':'int64','Project Type': 'object','Project Title':'object','Project Essay':'object','Project Subject Category Tree':'object','Project Subject Subcategory Tree':'object','Project Grade Level Category':'object','Project Resource Category':'object','Project Cost':'object','Project Posted Date':'object','Project Current Status':'object','Project Fully Funded Date':'object'}
projects = pd.read_csv(path+'Projects.csv',parse_dates=['Project Posted Date','Project Fully Funded Date'], dtype=dtypes_projects)#error_bad_lines=False, warn_bad_lines=False,

dtypes_resources = {'Project ID' : 'object','Resource Item Name' : 'object','Resource Quantity': 'float64','Resource Unit Price' : 'float64','Resource Vendor Name': 'object'}
resources = pd.read_csv(path+'Resources.csv', dtype=dtypes_resources)#
donations.head()[:3]
donors.head()[:3]
schools.head()[:3]
teachers.head()[:3]
projects.head()[:3]
resources.head()[:3]
donors_donations = donations.merge(donors, on='Donor ID', how='inner')
donors_donations.head()[:4]
donations["Donation Amount"].describe().apply(lambda x: format(x, 'f'))
schools['School Percentage Free Lunch'].describe()
df = donors.groupby("Donor State")['Donor City'].nunique().to_frame().reset_index()
X = df['Donor State'].tolist()
Y = df['Donor City'].apply(float).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Donor State' : Z[-5:][::-1], 'Count of Donor Cities' : sorted(Y)[-5:][::-1] })
sns.barplot(x="Donor State",y="Count of Donor Cities",data=data)
plt.rcParams["figure.figsize"] = [12,6]
sns.countplot(x='School Metro Type',data = schools)
donors_state_amount=donors_donations.groupby('Donor State')['Donation Amount'].sum().reset_index()
donors_state_amount['Donation Amount']=donors_state_amount['Donation Amount'].apply(lambda x: format(x, 'f'))

df = donors_state_amount[['Donor State','Donation Amount']]
X = df['Donor State'].tolist()
Y = df['Donation Amount'].apply(float).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Donor State' : Z[-5:][::-1], 'Total Donation Amount' : sorted(Y)[-5:][::-1] })
sns.barplot(x="Donor State",y="Total Donation Amount",data=data)
temp = donors_donations["Donor State"].value_counts().head(25)
temp.iplot(kind='bar', xTitle = 'State name', yTitle = "Count", title = 'Top Donor States')
state_count = temp.to_frame(name="number_of_projects").reset_index()
state_count = state_count.rename(columns= {'index': 'Donor State'})
# merging states with projects and amount funded
donor_state_amount_project = state_count.merge(donors_state_amount, on='Donor State', how='inner')

val = [x/y for x, y in zip(donor_state_amount_project['Donation Amount'].apply(float).tolist(),donor_state_amount_project['number_of_projects'].tolist())]
state_average_funding = pd.DataFrame({'Donor State':donor_state_amount_project['Donor State'][-5:][::-1],'Average Funding':val[-5:][::-1]})
sns.barplot(x="Donor State",y="Average Funding",data=state_average_funding)
per_teacher_as_donor = donors['Donor Is Teacher'].value_counts().to_frame().reset_index()
per_teacher_as_donor = per_teacher_as_donor.rename(columns= {'index': 'Types'})
labels = ['Donor is a Teacher','Donor is not a Teacher']
values = per_teacher_as_donor['Donor Is Teacher'].tolist()
colors = ['#96D38C', '#E1396C']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
py.iplot([trace], filename='styled_pie_chart')
temp = donors_donations['Donor Cart Sequence'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,'values': temp.values})
labels = df['labels'].tolist()
values = df['values'].tolist()
colors = ['#96D38C', '#E1396C','#C0C0C0','#FF0000','#F08080']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))

py.iplot([trace], filename='styled_pie_charimport calendart')
schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe()
projects_resources = projects.merge(resources, on='Project ID', how='inner')
projects_resources.head()[:4]
resource_vendor_name=projects_resources['Resource Vendor Name'].value_counts()
resource_vendor_name.iplot(kind='bar', xTitle = 'Vendor Name', yTitle = "Count", title = 'Resource Vendor',color='green')
projects_resources['Project Title'].fillna('Blank',inplace=True)
project_title = projects_resources['Project Title'].value_counts().to_frame().reset_index()[:5]
project_title = project_title.rename(columns= {'index': 'Project Title','Project Title':'Count'})
sns.barplot(x="Count",y="Project Title",data=project_title).set_title('Unique project title')
school_project = schools.merge(projects, on='School ID', how='inner')
school_project[:4]
school_project_count = school_project.groupby('School Metro Type')['Project ID'].count().reset_index()
school_project_count = pd.DataFrame({'School Metro Type':school_project_count['School Metro Type'],'Project Count':school_project_count['Project ID']})
sns.barplot(x="School Metro Type",y="Project Count",data=school_project_count)
temp = school_project['Project Current Status'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,'values': temp.values})
labels = df['labels'].tolist()
values = df['values'].tolist()
colors = ['#96D38C', '#E1396C','#C0C0C0','#FF0000']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value',
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))

py.iplot([trace], filename='styled_pie_chart')
project_open_close=school_project[['Project Resource Category','Project Posted Date','Project Fully Funded Date']]
project_open_close['Project Posted Date'] = pd.to_datetime(project_open_close['Project Posted Date'])
project_open_close['Project Fully Funded Date'] = pd.to_datetime(project_open_close['Project Fully Funded Date'])

time_gap = []
for i in range(school_project['School ID'].count()):
    if school_project['Project Current Status'][i] =='Fully Funded':
        time_gap.append(abs(project_open_close['Project Fully Funded Date'][i]-project_open_close['Project Posted Date'][i]).days)
    else:
        time_gap.append(-1)

project_open_close['Time Duration(days)'] = time_gap
project_open_close.head()
project_open_close_resource=project_open_close.groupby('Project Resource Category')['Time Duration(days)'].mean().reset_index()
df = project_open_close_resource[['Project Resource Category','Time Duration(days)']]
X = df['Project Resource Category'].tolist()
Y = df['Time Duration(days)'].apply(int).tolist()
Z = [x for _,x in sorted(zip(Y,X))]

data = pd.DataFrame({'Project Resource Category' : Z[0:5], 'Total Time Duration(days)' : sorted(Y)[0:5] })
sns.barplot(x="Total Time Duration(days)",y="Project Resource Category",data=data)
project_open_close.head()
school_project["Project Posted Date"] = pd.to_datetime(school_project["Project Posted Date"])
school_project['Project Posted year']=school_project['Project Posted Date'].dt.year
# school_project['Project Posted year']
# school_project.head()
school_project["Project Fully Funded Date"] = pd.to_datetime(school_project["Project Fully Funded Date"])
school_project['Project Funded year']=school_project['Project Fully Funded Date'].dt.year
# school_project.head()
temp=school_project['Project Posted year'].value_counts()
temp1=school_project['Project Funded year'].value_counts()
temp.iplot(kind='bar', xTitle = 'Year', yTitle = "Count", title = 'Project Posted In a Year')
temp1.iplot(kind='bar', xTitle = 'Year', yTitle = "Count", title = 'Project Funded In a Year',color='blue')
school_project['Project Posted Month']=school_project['Project Posted Date'].dt.month
school_project['Project Posted Month'] = school_project['Project Posted Month'].apply(lambda x: calendar.month_abbr[x])
month_count=school_project['Project Posted Month'].value_counts()
month_count.iplot(kind='bar', xTitle = 'Month', yTitle = "Count", title = 'Project Posted Month Wise')
# school_project["Project Fully Funded Date"] = pd.to_datetime(school_project["Project Fully Funded Date"])
# school_project['Project Funded Month']=school_project['Project Funded Date'].dt.month
# school_project['Project Funded  Month'] = school_project['Project Posted Month'].apply(lambda x: calendar.month_abbr[x])
donations = donations.merge(donors, on="Donor ID", how="left")
df = donations.merge(projects,on="Project ID", how="left")
donor_amount_df = df.groupby(['Donor ID', 'Project ID'])['Donation Amount'].sum().reset_index()
donor_amount_df.head()
# data preprocessing
features = ['Project Subject Category Tree','Project Title','Project Essay']
for col in features:
    projects[col] = projects[col].astype(str).fillna('fillna')
    projects[col] = projects[col].str.lower()

# tokenizing text
final_projects = projects[features]    
tok=text.Tokenizer(num_words=1000,lower=True)
tok.fit_on_texts(list(final_projects))
final_projects=tok.texts_to_sequences(final_projects)
final_projects_train=sequence.pad_sequences(final_projects,maxlen=150)
EMBEDDING_FILE = '../input/glove840b300dtxt/glove.840B.300d.txt'
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
max_features = 100000
embed_size=300

word_index = tok.word_index
#prepare embedding matrix

num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
