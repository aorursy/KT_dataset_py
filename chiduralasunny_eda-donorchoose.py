# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization library
import seaborn as sns # visualization library
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
#plt.style.use('sns') # style of plots. 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',100)
%matplotlib inline
# Any results you write to the current directory are saved as output.
resources = pd.read_csv('../input/Resources.csv',error_bad_lines = False,warn_bad_lines = False)
schools = pd.read_csv('../input/Schools.csv',error_bad_lines = False, warn_bad_lines = False)
donors = pd.read_csv('../input/Donors.csv', low_memory= False)
donations = pd.read_csv('../input/Donations.csv')
teachers = pd.read_csv('../input/Teachers.csv')
projects =  pd.read_csv('../input/Projects.csv',error_bad_lines=False,warn_bad_lines = False)
resources.head()
schools.head()
donors.head()
donations.head()
teachers.head()
projects.head()
donors_donations = donations.merge(donors, on='Donor ID', how='inner')
donors_donations.head()
donors_donations.groupby('Donation Included Optional Donation')['Donation Included Optional Donation'].value_counts().head().index
#Donation Included Optional Donation EDA
labels = donors_donations.groupby('Donation Included Optional Donation')['Donation Included Optional Donation'].value_counts().head().index
values = donors_donations.groupby('Donation Included Optional Donation')['Donation Included Optional Donation'].value_counts().head().values

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='Donation Included Optional Donation')
#Donation Amount EDA
labels = donors_donations['Donation Amount'].value_counts().head().index
values = donors_donations['Donation Amount'].value_counts().head().values

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='Donation Amount')
donors_donations['Donor Is Teacher'].value_counts().head()
#Donor Is Teacher EDA
labels = donors_donations['Donor Is Teacher'].value_counts().head().index
values = donors_donations['Donor Is Teacher'].value_counts().head().values

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='Donor Is Teacher')
#Donor City EDA
data = [go.Bar(
            x=donors_donations["Donor City"].value_counts().head().index,
            y=donors_donations["Donor City"].value_counts().head()
    )]

py.iplot(data, filename='Donor City')
#"Donor City PIE EDA
labels = donors_donations['Donor City'].value_counts().head().index
values = donors_donations['Donor City'].value_counts().head().values

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='Donor City')
#Donor State PIE EDA
labels = donors_donations['Donor State'].value_counts().head().index
values = donors_donations['Donor State'].value_counts().head().values

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='Donor State')
projects_schools = projects.merge(schools, on='School ID', how='inner')
projects_schools.head(10)
#Project Type EDA
labels = projects_schools['Project Type'].value_counts().head().index
values = projects_schools['Project Type'].value_counts().head().values

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='Project Type')
#School Metro Type EDA
data = [go.Bar(
            x=projects_schools["School Metro Type"].value_counts().head().index,
            y=projects_schools["School Metro Type"].value_counts().head()
    )]

py.iplot(data, filename='School Metro Type')
#School State Type EDA
data = [go.Bar(
            x=projects_schools["School State"].value_counts().head().index,
            y=projects_schools["School State"].value_counts().head()
    )]

py.iplot(data, filename='School State')
#School County Type EDA
data = [go.Bar(
            x=projects_schools["School County"].value_counts().head().index,
            y=projects_schools["School County"].value_counts().head()
    )]

py.iplot(data, filename='School County')

data = [go.Bar(
            x=projects_schools["Project Essay"].value_counts().head(15).index,
            y=projects_schools["Project Essay"].value_counts().head(15)
    )]

py.iplot(data, filename='Project Essay')
data = [go.Bar(
            x=projects_schools["Project Resource Category"].value_counts().head(15).index,
            y=projects_schools["Project Resource Category"].value_counts().head(15)
    )]

py.iplot(data, filename='Project Resource Category')
