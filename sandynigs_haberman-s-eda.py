import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
df = pd.read_csv('../input/habermans-survival-data-set/haberman.csv', header=None, names=['age', 'year', 'nodes', 'status'])

df.head()
df.shape
df.columns
df.describe()
df['status'].value_counts()/df.shape[0]
df.info()
plt.figure()

sns.distplot(df['age']).set_title("Histogram for age") #Creates a histogram pdf, for count of values, use countplot.

plt.show()
#It will make more sense if we combine age histogram with status.

plt.figure()



ax1 = plt.subplot(1,2,1)

sns.distplot(df[df['status']==1]['age']) #Age of all patients who lived for less than 5 years.

ax1.set_title(">5 yrs")



ax2 = plt.subplot(1,2,2)

sns.distplot(df[df['status']==2]['age']) #Age of all patients who lived for more than 5 years.

ax2.set_title("<5 yrs")



plt.tight_layout() #Avoids overlapping

plt.show()

plt.figure()

sns.countplot(df['year']).set_title("Histogram for year")

plt.show()
#It will make more sense if we combine age histogram with status.

plt.figure()



ax1 = plt.subplot(1,2,1)

sns.countplot(df[df['status']==1]['year']) #Age of all patients who lived for more than 5 years.

ax1.set_title(">5 yrs")



ax2 = plt.subplot(1,2,2)

sns.countplot(df[df['status']==2]['year']) #Age of all patients who lived for less than 5 years.

ax2.set_title("<5 yrs")



plt.tight_layout() #Avoids overlapping

plt.show()
print("patients with age<50: ",len(df[(df['year']==65) & (df['age']<50)])) #While using 'and' inside dataframe it is 

                                                                            #important to use '('.

print("patients with age>50: ",len(df[(df['year']==65) & (df['age']>50)]))
#It will make more sense if we combine age histogram with status.

plt.figure()



ax1 = plt.subplot(1,2,1)

sns.distplot(df[df['year']==65]['age'])

ax1.set_title("Age of patients in 1965")



ax2 = plt.subplot(1,2,2)

sns.countplot(df[df['year']==65]['nodes']) 

ax2.set_title("Nodes of patients in 1965")



plt.tight_layout() #Avoids overlapping

plt.show()

#It will make more sense if we combine age histogram with status.

plt.figure()



ax1 = plt.subplot(1,2,1)

sns.distplot(df[df['year']==60]['age'])

ax1.set_title("Age of patients in 1965")



ax2 = plt.subplot(1,2,2)

sns.countplot(df[df['year']==60]['nodes']) 

ax2.set_title("Nodes of patients in 1965")



plt.tight_layout() #Avoids overlapping

plt.show()

plt.figure()

sns.countplot(df['nodes']).set_title("Histogram for count of nodes")

plt.show()
#Maximum number of patients have shown 0 nodes. Let's check their status.

labels = df[df['nodes']==0]['status'].value_counts().index

values = df[df['nodes']==0]['status'].value_counts().values



explode = [0.1, 0]

fig1, ax1 = plt.subplots()

ax1.pie(values, labels=labels, explode=explode,

            autopct='%1.1f%%', startangle=90)

    # autopct is used for labelling inside pie wedges, startangle rotates pie counterclockwise

ax1.axis('equal')

ax1.set_title("Pie graph for node")

plt.tight_layout()

plt.show()
labels = df['status'].value_counts().index

values = df['status'].value_counts().values



explode = [0.1, 0]

fig1, ax1 = plt.subplots()

ax1.pie(values, labels=labels, explode=explode,

            autopct='%1.1f%%', startangle=90)

    # autopct is used for labelling inside pie wedges, startangle rotates pie counterclockwise

ax1.axis('equal')

ax1.set_title("Pie graph for status")

plt.tight_layout()

plt.show()
#Pair plot by default will consider every numeric feature to plot. Use vars to specify which columns to use.

plt.figure()

sns.set_style("whitegrid")

sns.pairplot(df, hue="status", vars=['age', 'year', 'nodes'], size=3)

plt.show()
for col in list(df.columns):

    if col!="status":

        plt.figure()

        sns.FacetGrid(df, hue="status").map(sns.distplot, col).add_legend()

        plt.title("PDF for "+col)

        plt.show()
df_status_one = df[df['status']==1]

df_status_two = df[df['status']==2]
#Observation for 'age'

label = ["pdf of status 1","cdf of status 1","pdf of status 2","cdf of status 2"] #This is takken in order



counts, bin_edges = np.histogram(df_status_one['age'], bins=10, density=True)

print("counts list is=>",counts) #Density of count of element in each bin

print("bin edges are=>",bin_edges) #bins



pdf = counts/sum(counts) #density at each bin. 

cdf = np.cumsum(pdf)

print("pdf is=>",pdf)

print("cdf is=>",cdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(df_status_two['age'], bins=10, density=True)

print("counts list is=>",counts) #Density of count of element in each bin

print("bin edges are=>",bin_edges) #bins



pdf = counts/sum(counts) #density at each bin. 

cdf = np.cumsum(pdf)

print("pdf is=>",pdf)

print("cdf is=>",cdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



plt.xlabel("age")

plt.title("PDF and CDF for age column when status is 1 and 2")

plt.legend(label)



plt.show()
#Observation for 'year'

label = ["pdf of status 1","cdf of status 1","pdf of status 2","cdf of status 2"] #This is takken in order



counts, bin_edges = np.histogram(df_status_one['year'], bins=10, density=True)

print("counts list is=>",counts) #Density of count of element in each bin

print("bin edges are=>",bin_edges) #bins



pdf = counts/sum(counts) #density at each bin. 

cdf = np.cumsum(pdf)

print("pdf is=>",pdf)

print("cdf is=>",cdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(df_status_two['year'], bins=10, density=True)

print("counts list is=>",counts) #Density of count of element in each bin

print("bin edges are=>",bin_edges) #bins



pdf = counts/sum(counts) #density at each bin. 

cdf = np.cumsum(pdf)

print("pdf is=>",pdf)

print("cdf is=>",cdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



plt.xlabel("year")

plt.title("PDF and CDF for year column when status is 1 and 2")

plt.legend(label)



plt.show()
#Observation for 'nodes'

label = ["pdf of status 1","cdf of status 1","pdf of status 2","cdf of status 2"] #This is takken in order



counts, bin_edges = np.histogram(df_status_one['nodes'], bins=10, density=True)

print("counts list is=>",counts) #Density of count of element in each bin

print("bin edges are=>",bin_edges) #bins



pdf = counts/sum(counts) #density at each bin. 

cdf = np.cumsum(pdf)

print("pdf is=>",pdf)

print("cdf is=>",cdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(df_status_two['nodes'], bins=10, density=True)

print("counts list is=>",counts) #Density of count of element in each bin

print("bin edges are=>",bin_edges) #bins



pdf = counts/sum(counts) #density at each bin. 

cdf = np.cumsum(pdf)

print("pdf is=>",pdf)

print("cdf is=>",cdf)



plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)



plt.xlabel("nodes")

plt.title("PDF and CDF for nodes column when status is 1 and 2")

plt.legend(label)



plt.show()
print("Median:")

print(np.median(df_status_one['age']), "for status one and", np.median(df_status_two['age']), "for status two")

print(np.median(df_status_one['year']), "for status one and", np.median(df_status_two['year']), "for status two")

print(np.median(df_status_one['nodes']), "for status one and", np.median(df_status_two['nodes']), "for status two")





print("\nQuantile:")

print("For age")

print(np.percentile(df_status_one['age'], np.arange(0,100,25)), "for status one and", np.percentile(df_status_two['age'], np.arange(0,100,25)), "for status two")

print("For year")

print(np.percentile(df_status_one['year'], np.arange(0,100,25)), "for status one and", np.percentile(df_status_two['year'], np.arange(0,100,25)), "for status two")

print("For nodes")

print(np.percentile(df_status_one['nodes'], np.arange(0,100,25)), "for status one and", np.percentile(df_status_two['nodes'], np.arange(0,100,25)), "for status two")



print("\n90th percentile:")

print("For age")

print(np.percentile(df_status_one['age'], 90), "for status one and", np.percentile(df_status_two['age'], 90), "for status two")

print("For year")

print(np.percentile(df_status_one['year'], 90), "for status one and", np.percentile(df_status_two['year'], 90), "for status two")

print("For nodes")

print(np.percentile(df_status_one['nodes'], 90), "for status one and", np.percentile(df_status_two['nodes'], 90), "for status two")



from statsmodels import robust

print("\nMedian Absolute Deviation:")

print("For age")

print(robust.mad(df_status_one['age']), "for status one and", robust.mad(df_status_two['age']), "for status two")

print("For year")

print(robust.mad(df_status_one['year']), "for status one and", robust.mad(df_status_two['year']), "for status two")

print("For nodes")

print(robust.mad(df_status_one['nodes']), "for status one and", robust.mad(df_status_two['nodes']), "for status two")



print("\nIQR:")

print("For age")

print( (np.percentile(df_status_one['age'], 75)-np.percentile(df_status_one['age'], 25)), "for status one and", (np.percentile(df_status_one['age'], 75)-np.percentile(df_status_one['age'], 25)), "for status two")

print("For year")

print( (np.percentile(df_status_one['year'], 75)-np.percentile(df_status_one['year'], 25)), "for status one and", (np.percentile(df_status_one['year'], 75)-np.percentile(df_status_one['year'], 25)), "for status two")

print("For nodes")

print( (np.percentile(df_status_one['nodes'], 75)-np.percentile(df_status_one['nodes'], 25)), "for status one and", (np.percentile(df_status_one['nodes'], 75)-np.percentile(df_status_one['nodes'], 25)), "for status two")
plt.figure()



ax1 = plt.subplot(1,3,1)

sns.boxplot(x='status', y='age', data=df)

ax1.set_title("Box plot for age")



ax2 = plt.subplot(1,3,2)

sns.boxplot(x='status', y='year', data=df)

ax2.set_title("Box plot for year")



ax3 = plt.subplot(1,3,3)

sns.boxplot(x='status', y='nodes', data=df)

ax3.set_title("Box plot for nodes")





plt.tight_layout()

plt.show()
#Nodes for status 1 have too many outliers.
plt.figure()



ax1 = plt.subplot(1,3,1)

sns.boxplot(x='status', y='age', data=df)

ax1.set_title("Box plot for age")



ax2 = plt.subplot(1,3,2)

sns.boxplot(x='status', y='year', data=df,)

ax2.set_title("Box plot for year")



ax3 = plt.subplot(1,3,3)

sns.boxplot(x='status', y='nodes', data=df,)

ax3.set_title("Box plot for nodes")





plt.tight_layout()

plt.show()
plt.figure()



ax1 = plt.subplot(1,3,1)

sns.violinplot(x='status', y='age', data=df)

ax1.set_title("Box plot for age")



ax2 = plt.subplot(1,3,2)

sns.violinplot(x='status', y='year', data=df)

ax2.set_title("Box plot for year")



ax3 = plt.subplot(1,3,3)

sns.violinplot(x='status', y='nodes', data=df)

ax3.set_title("Box plot for nodes")





plt.tight_layout()

plt.show()