# Import necessary tools 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot
import cufflinks as cf
init_notebook_mode(connected = True)
cf.go_offline()
import seaborn as sns
df = pd.read_csv("../input/data.csv")
df.head()
Disease = df['disease'].value_counts()
Disease.iplot(kind = 'bar', theme = 'solar',colors = 'Blue', xTitle = 'Disease Names', yTitle = 'No of patients', title = 'Diseases Frequency'
     )
Year_of_birth = [ ]
for str in list(df['dob']):
    year = int(str.split('-')[0])
    Year_of_birth.append(year)
df['YOB'] = Year_of_birth
df.head()
df['AGE'] = 2017 - df['YOB']
disease = list(df['disease'].unique())

for x in disease:
    trace = df[df['disease'] == x].groupby('gender').count()['AGE']
    trace.iplot(kind = 'bar', title = x, theme = 'solar')
    

df.groupby('gender').count()['id'].iplot(kind = 'bar', theme = 'solar')
df.groupby('ancestry').count()['id'].iplot(kind = 'bar', theme = 'solar')
df['AGE'].value_counts().iplot(kind = 'bar', theme = 'solar')
