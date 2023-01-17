Image("../input/picture/Data_aanalyst.jpeg")
from IPython.display import Image
import os
!ls ../input/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from wordcloud import WordCloud
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import seaborn as sns
from pandas_profiling import ProfileReport

import plotly.express as px
data=pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')

data.head()
data.drop(['Unnamed: 0'], axis=1,inplace=True)
data.isnull().sum()
data['Easy Apply'].value_counts()
data['Competitors'].value_counts()
data['Rating'].value_counts()[:5]
data['Job Title'].value_counts()
!pip install dexplot
!pip install chart_studio
!pip install pandas-profiling

import dexplot as dxp
import chart_studio.plotly as py
report = ProfileReport(data)
report
data.replace(to_replace =-1 , value=np.nan,inplace=True)
data.replace(to_replace ='-1' , value=np.nan,inplace=True)
data.replace(to_replace =-1.0 , value=np.nan,inplace=True)
data.info()
def FindingMissingValues(dataFrame):
    for col in dataFrame.columns:
        print('{0:.2f}% or {1} values are Missing in {2} Column'.format(dataFrame[col].isna().sum()/len(dataFrame)*100,dataFrame[col].isna().sum(),col),end='\n\n')

FindingMissingValues(data)
data.drop(['Competitors'],1,inplace = True)
data.drop('Job Description',1,inplace=True)
dxp.count('Rating',data)
fig = px.scatter(data, x="Rating", y="Company Name", 
                 color="Rating", 
                 hover_data=['Headquarters','Location', 'Type of ownership', 'Industry', 'Sector'], 
                 title = "Data Analyst jobs")
fig.show()
rat_jobs = data["Rating"].value_counts()
fig, ax = plt.subplots(figsize=(14,9))
rect1 = sns.barplot(x=rat_jobs.index, y=rat_jobs.values, palette="deep")
ax.set_title("Total count Of all rated jobs", fontweight="bold")

for p in rect1.patches:
    rect1.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.5, p.get_height()+1.3), ha='center', va='bottom', color= 'black', rotation=90)
plt.rcParams["figure.figsize"] = (12,9)
#plt.style.use("classic")
color = plt.cm.PuRd(np.linspace(0,1,20))
data["Company Name"].value_counts().sort_values(ascending=False).head(20).plot.bar(color=color)
plt.title("Top 20 Company with Highest number of Jobs ",fontsize=20)
plt.xlabel("Company Name",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()
com =data['Company Name'].value_counts()
company = pd.DataFrame({'Company': com.index,'Number of Jobs':com.values})
company.head()
#Number of Jobs according to Different Company