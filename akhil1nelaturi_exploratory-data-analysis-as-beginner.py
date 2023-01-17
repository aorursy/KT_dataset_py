cd /kaggle/input/habermans-survival-data-set
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import warnings

from prettytable import PrettyTable

warnings.filterwarnings('ignore')



data=pd.read_csv('haberman.csv',names=["age","year", "nodes","status"])
data.head()
data.status.value_counts()
# How many data-points and features ?

print(data.size,data.shape[1]-1)
# What are the features in given data set?

print([i for i in data.columns[:-1]])
# What need to be classified ?

print ('Wether the patient survived lessthan(status=2) or more than 5 years (status= 1) after operation')
# The data was collected between ?

print('19{} and 19{}'.format(data.year.min(),data.year.max()))
# The age group of patients?

print(data.age.min(),data.age.max())
##https://matplotlib.org/gallery/pie_and_polar_charts/pie_features.html#sphx-glr-gallery-pie-and-polar-charts-pie-features-py



# How many patients survived More than 5 years(status=1) and Less than 5 years (status=2)?

print('Patients survived More than 5 years are',data.status.value_counts()[1],'and Less than 5 years are ',data.status.value_counts()[2])

status_counts=[data.status.value_counts()[1],data.status.value_counts()[2]]

labels='Status 1','Status 2'



fig1,ax1=plt.subplots(figsize=(6,6));

ax1.pie(status_counts,explode = (0, 0.1),labels=labels,shadow=1,startangle=90,autopct='%1.1f%%');

ax1.axis('equal');

plt.show();
# How many values are unknown or not available?

data.info()
data.describe()
g=data.groupby('status')
g['age','nodes'].describe()
# Out of patients not survied, percentage of them don't even have a single node

data[(data.nodes==0) & (data.status==2)].shape[0]/data[data.status==2].shape[0]*100
stts1=g['nodes'].quantile([i*0.1 for i in range(11)])[1].values

stts2=g['nodes'].quantile([i*0.1 for i in range(11)])[2].values

x=[i*10 for i in range(11)]

table=PrettyTable()

table.field_names=['Percentile','No of Nodes (Status 1)','No of Nodes (Status 2)']

for i in range(11):

  table.add_row(np.around([x[i],stts1[i],stts2[i]],2))

print(table)
plt.figure(figsize=(10,5))

plt.plot(x,stts1);

plt.scatter(x,stts1,c='g',label='status 1');

plt.plot(x,stts2);

plt.scatter(x,stts2,c='r',label='status 2');

plt.legend()

plt.xlabel('Percentile')

plt.ylabel('No of Nodes')

plt.title('Percentile plot of no of Nodes for status 1 and 2')

plt.grid()
stts1=g['age'].quantile([i*0.1 for i in range(11)])[1].values

stts2=g['age'].quantile([i*0.1 for i in range(11)])[2].values

x=[i*10 for i in range(11)]

table=PrettyTable()

table.field_names=['Percentile','Age (Status 1)','Age (Status 2)']

for i in range(11):

  table.add_row(np.around([x[i],stts1[i],stts2[i]],2))

print(table)
plt.figure(figsize=(10,5))

plt.plot(x,stts1);

plt.scatter(x,stts1,c='g',label='status 1');

plt.plot(x,stts2);

plt.scatter(x,stts2,c='r',label='status 2');

plt.legend()

plt.xlabel('Percentile')

plt.ylabel('No of Nodes')

plt.title('Percentile plot of no of Nodes for status 1 and 2')

plt.grid()
plt.figure(figsize=(20,5))

plt.subplot(121)

sns.boxplot(x='status',y='nodes', data=data)

#plt.show()

plt.subplot(122)

sns.boxplot(x='status',y='age', data=data)

plt.suptitle('Box plots of No of Nodes present and Age of patients')

plt.show()
d=np.array(data.age.loc[data.status==1])

s=np.array(data.age.loc[data.status==2])



fig,axes=plt.subplots(1,3,sharey=True,sharex=True,figsize=(17,5));



sns.distplot(d,ax=axes[0],label='status 1');

sns.distplot(s,ax=axes[1],label='status 2',color='r');

sns.distplot(d,ax=axes[2],label='status 1' );

sns.distplot(s,ax=axes[2],label='status 2',color='r');

axes[0].legend();

axes[0].set_xlabel('Age of people with status 1')

axes[1].set_xlabel('Age of people with status 2')

axes[1].legend();

axes[2].legend();

plt.suptitle('Histograms of age');

plt.xlabel('Age');
d=np.array(data.nodes.loc[data.status==1])

s=np.array(data.nodes.loc[data.status==2])



fig,axes=plt.subplots(1,3,sharey=True,sharex=True,figsize=(17,5));



sns.distplot(d,ax=axes[0],label='status 1');

sns.distplot(s,ax=axes[1],label='status 2',color='r');

sns.distplot(d,ax=axes[2],label='status 1' );

sns.distplot(s,ax=axes[2],label='status 2',color='r');

axes[0].legend();

axes[0].set_xlabel('No of Nodes')

axes[1].set_xlabel('No of Nodes')

axes[1].legend();

axes[2].legend();

plt.suptitle('Histograms of No of Nodes');

plt.xlabel('No of Nodes');
d=np.array(data.year.loc[data.status==1])

s=np.array(data.year.loc[data.status==2])



fig,axes=plt.subplots(1,3,sharey=True,sharex=False,figsize=(17,5));



sns.distplot(d,ax=axes[0],label='status 1');

sns.distplot(s,ax=axes[1],label='status 2',color='r');

#plt.xticks(d,['19'+str(i) for  i in d],rotation=90)

sns.distplot(d,ax=axes[2],label='status 1' );

axes[0].grid()

axes[1].grid()

sns.distplot(s,ax=axes[2],label='status 2',color='r');

axes[0].legend();

axes[0].set_xlabel('Year')

axes[1].set_xlabel('Year')

axes[1].legend();

axes[2].legend();

plt.xticks(d,['19'+str(i) for  i in d],rotation=90)

plt.suptitle('Histograms of Year');

plt.xlabel('Year');

plt.grid()
plt.figure(figsize=(20,5))

plt.subplot(121)

sns.violinplot(x='status',y='nodes', data=data)

#plt.show()

plt.subplot(122)

sns.violinplot(x='status',y='age', data=data)

plt.suptitle('Violin plots of No of Nodes present and Age of patients')

plt.show()
counts, bin_edges = np.histogram(data.nodes[data.status==1], bins=10,density = True)

plt.figure(figsize=(20,6))

plt.subplot(121)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='PDF status 1',c='b')

plt.plot(bin_edges[1:], cdf,label='CDF status 1',c='r')

counts, bin_edges = np.histogram(data.nodes[data.status==2], bins=10,density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='PDF status 2',c='g')

plt.plot(bin_edges[1:], cdf,label='CDF status 2',c='200')

plt.legend()

plt.xlabel('Number of Nodes')

plt.title('CDF and PDF of Number of nodes present for stauts 1 and 2')

#plt.show();



counts, bin_edges = np.histogram(data.age[data.status==1], bins=10,density = True)

plt.subplot(122)

#plt.figure(figsize=(15,7))

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='PDF status 1',c='b')

plt.plot(bin_edges[1:], cdf,label='CDF status 1',c='r')

counts, bin_edges = np.histogram(data.age[data.status==2], bins=10,density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label='PDF status 2',c='g')

plt.plot(bin_edges[1:], cdf,label='CDF status 2',c='200')

plt.legend()

plt.xlabel('Age of patients')

plt.title('CDF and PDF of Age of patients for stauts 1 and 2')

plt.show();
plt.close();

sns.set_style("whitegrid");

sns.pairplot(data, hue="status",markers=["o", "s"],vars=data[['age','nodes','year']], size=3,diag_kind="kde");

plt.show()