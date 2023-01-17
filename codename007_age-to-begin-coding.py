import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
codebook_data = pd.read_csv("../input/HackerRank-Developer-Survey-2018-Codebook.csv")
numeric_mapping_data = pd.read_csv("../input/HackerRank-Developer-Survey-2018-Numeric-Mapping.csv")
country_code_mapping_data = pd.read_csv("../input/Country-Code-Mapping.csv")
numeric_data = pd.read_csv("../input/HackerRank-Developer-Survey-2018-Numeric.csv")
values_data = pd.read_csv("../input/HackerRank-Developer-Survey-2018-Values.csv")
print("Shape of codebook data",codebook_data.shape)
print("Shape of numeric_mapping data",numeric_mapping_data.shape)
print("Shape of country_code_mapping data",country_code_mapping_data.shape)
print("Shape of numeric data",numeric_data.shape)
print("Shape of values data",values_data.shape)
print("Column names of codebook data\n",codebook_data.columns)
print("Column names of numeric_mapping data\n",numeric_mapping_data.columns)
print("Column names of country_code_mapping data\n",country_code_mapping_data.columns)
print("Column names of numeric data\n",numeric_data.columns)
print("Column names of values data\n",values_data.columns)
codebook_data.head()
numeric_mapping_data.head()
country_code_mapping_data.head()
numeric_data.head()
values_data.head()
codebook_data.describe()
numeric_mapping_data.describe()
numeric_mapping_data.describe(include=["O"])
country_code_mapping_data.describe()
country_code_mapping_data.describe(include=["O"])
numeric_data.describe()
numeric_data.describe(include=["O"])
values_data.describe()
values_data.describe(include=["O"])
# checking missing data in codebook_data 
total = codebook_data.isnull().sum().sort_values(ascending = False)
percent = (codebook_data.isnull().sum()/codebook_data.isnull().count()*100).sort_values(ascending = False)
missing_codebook_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_codebook_data
# checking missing data in numeric_mapping_data 
total = numeric_mapping_data.isnull().sum().sort_values(ascending = False)
percent = (numeric_mapping_data.isnull().sum()/numeric_mapping_data.isnull().count()*100).sort_values(ascending = False)
missing_numeric_mapping_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_numeric_mapping_data
# checking missing data in country_code_mapping_data 
total = country_code_mapping_data.isnull().sum().sort_values(ascending = False)
percent = (country_code_mapping_data.isnull().sum()/country_code_mapping_data.isnull().count()*100).sort_values(ascending = False)
missing_country_code_mapping_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_country_code_mapping_data
# checking missing data in numeric_data 
total = numeric_data.isnull().sum().sort_values(ascending = False)
percent = (numeric_data.isnull().sum()/numeric_data.isnull().count()*100).sort_values(ascending = False)
missing_numeric_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_numeric_data.head(25)
# checking missing data in values_data 
total = values_data.isnull().sum().sort_values(ascending = False)
percent = (values_data.isnull().sum()/values_data.isnull().count()*100).sort_values(ascending = False)
missing_values_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_values_data.head(40)
# df.drop(df.index[[1,3]])
temp = values_data.drop(values_data.index[list(values_data[values_data["q3Gender"]=='Non-Binary'].index) + list(values_data[values_data["q3Gender"]=='#NULL!'].index)])["q3Gender"].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Male V.S. Female')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
#Distribution of Age to begin coding
temp = values_data['q1AgeBeginCoding'].value_counts()
plt.figure(figsize=(15,8))
sns.barplot(temp.index, temp.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=20)
plt.xlabel('Age to begin coding', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Count of Ages to begin coding", fontsize=16)
plt.show()
#Distribution of Age 
temp = values_data['q2Age'].value_counts()
plt.figure(figsize=(15,8))
sns.barplot(temp.index, temp.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=20)
plt.xlabel('People Age', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Count of People Ages who partcipated in Survey", fontsize=16)
plt.show()
#Distribution of countries
temp = values_data['CountryNumeric2'].value_counts().head(20)
plt.figure(figsize=(15,8))
sns.barplot(temp.index, temp.values, alpha=0.9, color=color[0])
plt.xticks(rotation='vertical', fontsize=20)
plt.xlabel('Country Name', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title("Number of people from different countries participated in survey", fontsize=16)
plt.show()
plt.figure(figsize=(15,8))
count = values_data['CountryNumeric2'].value_counts()
squarify.plot(sizes=count.values,label=count.index, value=count.values)
plt.title('Distribution of sectors')
codebook_data = codebook_data.set_index('Data Field')
columns = values_data.columns[values_data.columns.str.startswith('q25')]
length = len(columns)
columns = columns.drop('q25LangOther')
f,ax= plt.subplots(4,6,figsize=(16,12))
axs = ax.ravel()
for i,c in enumerate(columns):
    sns.countplot(values_data[c],ax=axs[i],palette='cool')
    axs[i].set_ylabel('')
    axs[i].set_xlabel('')
    axs[i].set_title(codebook_data.loc[c]['Survey Question'])
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.suptitle(' Programming language known or willing to know',fontsize=14);
columns = values_data.columns[values_data.columns.str.startswith('q26')]
length = len(columns)
f,ax= plt.subplots(4,5,figsize=(16,12))
axs = ax.ravel()
for i,c in enumerate(columns):
    if values_data[c].nunique()>1: 
        sns.countplot(values_data[c],ax=axs[i],palette='cool')
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_title(codebook_data.loc[c]['Survey Question'])
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.suptitle('Frame work known or willing to know',fontsize=14);
columns = values_data.columns[values_data.columns.str.startswith('q28')]
length = len(columns)
columns = columns.drop('q28LoveOther')
f,ax= plt.subplots(4,6,figsize=(16,12))
axs = ax.ravel()
for i,c in enumerate(columns):
    if values_data[c].nunique()>1: 
        sns.countplot(values_data[c],ax=axs[i],palette='cool')
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_title(codebook_data.loc[c]['Survey Question'])
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.suptitle('Love or Hate programming language',fontsize=14);
columns = values_data.columns[values_data.columns.str.startswith('q29')]
length = len(columns)
f,ax= plt.subplots(4,5,figsize=(16,12))
axs = ax.ravel()
for i,c in enumerate(columns):
    if values_data[c].nunique()>1: 
        sns.countplot(values_data[c],ax=axs[i],palette='cool')
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_title(codebook_data.loc[c]['Survey Question'])
plt.subplots_adjust(hspace=0.4,wspace=0.4)
plt.suptitle('Love or Hate Frame work',fontsize=14);