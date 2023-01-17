import pandas as pd

import matplotlib 

import matplotlib.pyplot as plt

from IPython.display import HTML,SVG

import seaborn as sns

import numpy as np 
h1b = pd.read_csv("../input/h1b_kaggle.csv")
h1b.head()
# Similar output as above, but the data is fetched from the bottom i.e. last rows.

h1b.tail()
h1b = h1b.iloc[:,1:]

h1b = h1b.iloc[:, :8]
# With Nan 

h1b["CASE_STATUS"].value_counts()
# Analyse all the case status except for the NAN

h1b = h1b.dropna()



# Taking only one column of certified status

h1b["CASE_STATUS"].value_counts()
# Cleaning the data for Analysing Approvals and rejections

year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)

year = year[year["CASE_STATUS"] == "CERTIFIED"]

Certified_per_year = year["YEAR"].value_counts()
year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)

year = year[year["CASE_STATUS"] == "DENIED"]

Denied_per_year = year["YEAR"].value_counts()
year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)

year = year[year["CASE_STATUS"] == "CERTIFIED-WITHDRAWN"]

CW_per_year = year["YEAR"].value_counts()
year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)

year = year[year["CASE_STATUS"] == "WITHDRAWN"]

W_per_year = year["YEAR"].value_counts()
year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)

year = year[year["CASE_STATUS"] == "PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED"]

P_per_year = year["YEAR"].value_counts()
year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)

year = year[year["CASE_STATUS"] == "REJECTED"]

R_per_year = year["YEAR"].value_counts()
year = h1b.drop(h1b.columns[[1,2,3,4,5,7]], axis=1)

year = year[year["CASE_STATUS"] == "INVALIDATED"]

I_per_year = year["YEAR"].value_counts()
labels1 = 'INVALIDATED', 'PENDING', 'WITHDRAWN','CERTIFIED-WITHDRAWN', 'REJECTED','DENIED','CERTIFIED'



plt.plot(I_per_year,'kh', P_per_year,'bo', W_per_year,'y*', CW_per_year,'g^',R_per_year,'cs',Denied_per_year,'m+',Certified_per_year,'rp')

plt.legend(('INVALIDATED', 'PENDING', 'WITHDRAWN','CERTIFIED-WITHDRAWN', 'REJECTED','DENIED','CERTIFIED'))

plt.show()
h1b1 = []

h1 = []

for i in h1b["WORKSITE"]:

    h1 = i.split(',')

    h1b1.append(h1[1])

df = pd.DataFrame({'col':h1b1})

df['col'].value_counts()
data_frame = [h1b,df]

data_frame = pd.concat(data_frame, axis=1)
# Certified by states

data_frame = data_frame.dropna()

states = data_frame.drop(data_frame.columns[[1,2,3,4,5,6,7]], axis=1)
certified_states = states[states["CASE_STATUS"] == "CERTIFIED"]

denied_states = states[states["CASE_STATUS"] == "DENIED"]

cw_states = states[states["CASE_STATUS"] == "CERTIFIED-WITHDRAWN"]

w_states = states[states["CASE_STATUS"] == "WITHDRAWN"]
h = sorted(set(h1b1))
objects = h



y_pos = np.arange(len(objects))



performance0 = denied_states['col'].value_counts().sort_index()

performance1 = certified_states['col'].value_counts().sort_index()

performance2 = cw_states['col'].value_counts().sort_index()

performance3 = w_states['col'].value_counts().sort_index()





plt.plot(y_pos, performance0, label = 'Denied')

plt.plot(y_pos, performance1, label = 'Certified')

plt.plot(y_pos, performance2, label = 'Certified-Withdrawn')

plt.plot(y_pos, performance3, label = 'Withdrawn')



plt.xticks(y_pos, objects,rotation=90)

plt.xlabel('States')

plt.ylabel('Applications')

plt.title('No. of Applicants status of H1B Visa based on states')

plt.legend()

plt.show()
