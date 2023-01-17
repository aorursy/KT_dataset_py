# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

myComplaints=pd.read_csv("../input/comcast_consumeraffairs_complaints.csv",skiprows=1,names=['author','posted_on','rating','text'])

#### Function to split ####

extract = lambda x: pd.Series([i for i in (x.split(','))])
#### This is to pick only complaints which has rating less than 3 ####

badComplaints_loc=myComplaints['author'][myComplaints.rating < 3 ].apply(extract)
### Remove the records which doesnt have a proper location ######

badComplaints= badComplaints_loc[badComplaints_loc[1].str.len() ==3 ]
#### Pick the top 10 locations #####

myBar=(badComplaints.groupby([1]).count().sort_values(0,ascending=False).head(10))
##### Plotting the below bar chart #####

myPlot=myBar.plot(kind='bar',legend=False,title="Top 10 Complaint Locations",colormap="copper_r")

myPlot.set_xlabel("States")

myPlot.set_ylabel("#Complaints")
#### This is to pick only complaints which has rating less than 3 ####

badComplaints_yr=myComplaints['posted_on'][myComplaints.rating < 3 ].apply(extract)
### Remove the records which are older than 2010 ######

badComplaints=badComplaints_yr[badComplaints_yr[1].isin ([' 2016',' 2015',' 2014',' 2013',' 2012',' 2011',' 2010'])]
#### Group by year #####

myLine=badComplaints.groupby([1]).count()
##### Plotting the below bar chart #####



myPlot=myLine.plot(kind='line',legend=False,title="Yearly Complaint Trend",colormap="copper_r",linewidth=3)

myPlot.set_xlabel("Year")

myPlot.set_ylabel("#Complaints")