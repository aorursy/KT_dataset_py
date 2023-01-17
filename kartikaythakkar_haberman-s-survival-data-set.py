import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
df=pd.read_csv("../input/haberman.csv")
df
df.shape  #so the no. of points are 306 and no. of features are 3
df.columns  #
# Attribute Information:



# column[0]: Age of patient at time of operation (numerical)

# column[1]: Patient's year of operation (year - 1900, numerical)

# column[2]: Number of positive auxillary nodes detected (numerical)

# column[3]: Survival status (class attribute) 1 = the patient survived 5 years or longer , 2 = the patient died within 5 year
df['1.1'].value_counts()  #Imbalanced dataset 

                #There are two classes '1','2'.

                # No. of datapoints belonging to '1' are 224 i.e 224 patients survived 5 years or longer

                # No. of datapoints belonging to '2' are 81 i.e 81 patients died within 5 year
# OBJECTIVE



# Our objective is to find either the patient belong to the "patient survived 5 years or longer" or "he died within 5 year"

# We are to predict column[3] by the given dataset

# and column[2] will be the play the most important role in finding out our conclusion
#Univariate Analysis 
import numpy as np

survived=df.loc[df["1.1"] == 1]

died=df.loc[df["1.1"] == 2]

plt.plot(survived['30'],np.zeros_like(survived['30']),'ro')

plt.plot(died['30'],np.zeros_like(died['30']),'b')

plt.show()
#if age<33:

#    patient survived

#if age>78:

#    patient died

#and due to overlapping we cannot depict for age>33 and age<78
import numpy as np

survived=df.loc[df["1.1"] == 1]

died=df.loc[df["1.1"] == 2]

plt.plot(survived['64'],np.zeros_like(survived['30']),'ro')

plt.plot(died['64'],np.zeros_like(died['30']),'b')

plt.show()
#patient's year of operation can not be a good option to select for univariate analysis
import numpy as np

survived=df.loc[df["1.1"] == 1]

died=df.loc[df["1.1"] == 2]

plt.plot(survived['1'],np.zeros_like(survived['30']),'ro')

plt.plot(died['1'],np.zeros_like(died['30']),'b')

plt.show()
#if Number of positive auxillary nodes detected > 30:

#    then the patient died
#So to choose "1" i.e 'Number of positive auxillary nodes detected' is the perfect univariate.
#PDF
sns.FacetGrid(df,hue='1.1',size=5).map(sns.distplot,'30').add_legend()

plt.show()  #There's a lot of overlapping in it
sns.FacetGrid(df,hue='1.1',size=5).map(sns.distplot,'64').add_legend()

plt.show()  #There's a lot of overlapping in it
sns.FacetGrid(df,hue="1.1",size=5).map(sns.distplot,'1').add_legend()

plt.show()
# PDF,CDF
counts,bin_edges=np.histogram(survived['1'],bins=10,density=True)

pdf=counts/sum(counts)

print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,"ro-")

plt.plot(bin_edges[1:],cdf,"r*-")



counts,bin_edges=np.histogram(died['1'],bins=10,density=True)

pdf=counts/sum(counts)

print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,"go-")

plt.plot(bin_edges[1:],cdf,"g*-")

sns.set_style("darkgrid")

plt.show()
#if Number of positive auxillary nodes detected > 46:

#    Patient died
# boxplot
sns.boxplot(x='1.1',y='1',data=df)

plt.xlabel('Survival status ')

plt.ylabel('Number of positive auxillary nodes detected')

plt.show()
# There are 50% of the died patients who had Number of positive auxillary nodes detected < 4

# There are 75% of the survived patients who had Number of positive auxillary nodes detected < 4
#if we consider the blue box fully,then  the readings are as follows



#if auxiliary nodes< 4:

#    Patient belongs to category 1

#else:

#    Patient belongs to category 2

    

#if we consider the blue box fully,then there will be almost around 40% error
# violinplot
sns.violinplot(x='1.1',y='1',data=df,size=10)

plt.xlabel("Survival Status")

plt.ylabel('Number of positive auxillary nodes detected')

plt.show()
# Bell curve of survived patients is more as compared to died patients
#Bivariate Analysis
sns.set_style("whitegrid")

sns.FacetGrid(df,hue='1.1',size=4).map(plt.scatter,'30','1').add_legend()

plt.show()
#if age<40:

#    patient survived

#and we can not seperate for age>40
sns.set_style("whitegrid")

sns.FacetGrid(df,hue='1.1',size=4).map(plt.scatter,'1','64').add_legend()

plt.show()
sns.set_style("whitegrid")

sns.FacetGrid(df,hue='1.1',size=4).map(plt.scatter,'30','64').add_legend()

plt.show()
#if age<40 and age>70:

#    patient survived
plt.close()

sns.pairplot(df,hue="1.1",size=2)

plt.show()