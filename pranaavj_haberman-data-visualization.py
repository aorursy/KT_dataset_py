import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import os
print(os.listdir("../input"))

names = ['Age', 'Year operation', 'Axillary nodes detected', 'Survival status']
haberman = pd.read_csv('../input/haberman.csv',names=names )
print(haberman.head(5))
#Shape of the habermans data set 
print(haberman.shape)

#Columns of the dataset
print(haberman.columns)
print(haberman['Year operation'].value_counts())
print(haberman['Year operation'].max())
#print(haberman['Axillary nodes detected'].value_counts())
#print(haberman['Axillary nodes detected'].max())

print(haberman['Survival status'].value_counts())
print(haberman['Survival status'].value_counts(normalize = True))
print(haberman['Survival status'].max())
print(haberman.info())
print(haberman.describe())
# modifying the Survival status column values to be meaningful as well as categorical
haberman['Survival status'] = haberman['Survival status'].map({1:"yes", 2:"no"})
haberman['Survival status'] = haberman['Survival status'].astype('category')

print(haberman.head())
print(haberman['Survival status'].value_counts())

haberman.plot(kind='scatter',x='Age',y='Axillary nodes detected')
sns.set_style('whitegrid')
g = sns.FacetGrid(haberman,hue="Survival status",size=5)
g = g.map(plt.scatter,'Age','Axillary nodes detected')
g = g.add_legend()
plt.show()
plt.close()
sns.set_style("whitegrid")
sns.pairplot(haberman,hue='Survival status',size=3)

pat_sur = haberman.loc[haberman['Survival status']=='yes']
pat_nosur = haberman.loc[haberman['Survival status']=='no']
print("Patient Survived: ","\n")
print(pat_sur.describe())
print("Patient Not Survived: ","\n")
print(pat_nosur.describe())
# 1-D scatter plot for patient age



plt.plot(pat_sur['Age'],np.zeros_like(pat_sur['Age']),'o')
plt.plot(pat_nosur['Age'],np.zeros_like(pat_nosur['Age']),'o')

plt.xlabel('Age')
plt.legend(['Survived','Not Survived'])


g = sns.FacetGrid(haberman,hue='Survival status',size=4)
g = g.map(sns.distplot,'Age')
g.add_legend()

g = sns.FacetGrid(haberman,hue='Survival status',size=4)
g = g.map(sns.distplot,'Year operation')
g.add_legend()
g = sns.FacetGrid(haberman,hue="Survival status",size=7)
g = g.map(sns.distplot,'Axillary nodes detected')
g.add_legend()
count, bin_edges = np.histogram(pat_sur['Age'],bins=10,density=True)

print(count,"\n")

print(sum(count),"\n")

pdf = count/(sum(count))

print(pdf,"\n")

print(bin_edges,"\n")

cdf = np.cumsum(pdf)

print(cdf,"\n")

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel('Age')
plt.legend(['pdf','cdf'])

count, bin_edges = np.histogram(pat_nosur['Age'],bins=10,density=True)

print(count,"\n")

print(sum(count),"\n")

pdf = count/(sum(count))

print(pdf,"\n")

print(bin_edges,"\n")

cdf = np.cumsum(pdf)

print(cdf,"\n")

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel("Age")
plt.legend(['pdf','cdf'])
#Plots of CDF of Age for Survival and Non-survival.

count, bin_edges = np.histogram(pat_sur['Age'],bins=10,density=True)

pdf = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

count, bin_edges = np.histogram(pat_nosur['Age'],bins=10,density=True)

pdf = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel("Age")

plt.legend(['pdf Suvival','cdf Suvival','pdf Non Suvival','cdf Non Suvival'])
#Plots of CDF of Year Operation for Survival and Non-survival.

count, bin_edges = np.histogram(pat_sur['Year operation'],bins=10,density=True)

pdf = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


count, bin_edges = np.histogram(pat_nosur['Year operation'],bins=10,density=True)

pdf = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


plt.xlabel("Year operation")

plt.legend(['pdf Suvival','cdf Suvival','pdf Non Suvival','cdf Non Suvival'])

#Plots of CDF of Axillary nodes detected for Survival and Non-survival.

count, bin_edges = np.histogram(pat_sur['Axillary nodes detected'],bins=10,density=True)

pdf = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

count, bin_edges = np.histogram(pat_nosur['Axillary nodes detected'],bins=10,density=True)

pdf = count/(sum(count))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel("Axillary nodes detected")

plt.legend(['pdf Suvival','cdf Suvival','pdf Non Suvival','cdf Non Suvival'])
print('Mean:')
print("Mean Age of Patient Survived:",np.mean(pat_sur['Age']))
print("Mean Age of Patient not Survived:",np.mean(pat_nosur['Age']),"\n")

print("Mean Positive nodes of Patient Survived:",np.mean(pat_sur['Axillary nodes detected']))
print("Mean Positive nodes of Patient not Survived:",np.mean(pat_nosur['Axillary nodes detected']),"\n")

print("Mean year of operation of Patient Survived:",np.mean(pat_sur['Year operation']))
print("Mean year of operation of Patient not Survived:",np.mean(pat_nosur['Year operation']),"\n")

print("\nStd-dev:");
print("Standard deviation of the Age of Patient Survived:",np.std(pat_sur['Age']))
print("Standard deviation of the Age of Patient not Survived:",np.std(pat_nosur['Age']),"\n")

print("Standard deviation nodes of Patient Survived:",np.std(pat_sur['Axillary nodes detected']))
print("Standard deviation nodes of Patient not Survived:",np.std(pat_nosur['Axillary nodes detected']),"\n")

print("Standard deviation of operation of Patient Survived:",np.std(pat_sur['Year operation']))
print("Standard deviation of operation of Patient not Survived:",np.std(pat_nosur['Year operation']),"\n")

#Affect of one or small number of Outlier(an error) on both mean and standard-deviation

print('\nMean with Outliner')
print('For survived Patients :'+str(np.mean(np.append(pat_sur['Age'],300))))

print('\nStd-dev with Outliner')
print('For survived Patients :' + str(np.std(np.append(pat_sur['Age'],200))))

# Median with respect to age for survived patients

print("\nMedians:")
print("Median Age of suvived patient:",np.median(pat_sur["Age"]))
print("Median Age of not suvived patient:",np.median(pat_nosur["Age"]),"\n")

print("Median Axillary nodes detected of suvived patient:",np.median(pat_sur["Axillary nodes detected"]))
print("Median Axillary nodes detected of not suvived patient:",np.median(pat_nosur["Axillary nodes detected"]),"\n")


print("\n90th Percentiles:")
print("90th Percentiles Age of Survived patient",np.percentile(pat_sur["Age"],90))
print("90th Percentiles Age of patient not survived",np.percentile(pat_nosur["Age"],90),"\n")

print("90th Percentiles Axillary nodes detected of Survived patient",np.percentile(pat_sur["Axillary nodes detected"],90))
print("90th Percentiles Axillary nodes detected of patient not survived",np.percentile(pat_nosur["Axillary nodes detected"],90),"\n")

print("90th Percentiles Year operation of Survived patient",np.percentile(pat_sur["Year operation"],90))
print("90th Percentiles Year operation of patient not survived",np.percentile(pat_nosur["Year operation"],90),"\n")

from statsmodels import robust

print("\nMedian Absolute Deviation:")
print("MAD Age of suvived patient:",robust.mad(pat_sur["Age"]))
print("MAD Age of not suvived patient:",robust.mad(pat_nosur["Age"]),"\n")

print("MAD Axillary nodes detected of suvived patient:",robust.mad(pat_sur["Axillary nodes detected"]))
print("MAD Axillary nodes detected of not suvived patient:",robust.mad(pat_nosur["Axillary nodes detected"]),"\n")

sns.boxplot(x='Survival status',y='Axillary nodes detected', data=haberman)
plt.show()
sns.boxplot(x='Survival status',y='Age', data=haberman)
plt.show()
sns.boxplot(x='Survival status',y='Year operation', data=haberman)
plt.show()
sns.violinplot(x='Survival status',y='Axillary nodes detected', data=haberman)
plt.show()
sns.violinplot(x='Survival status',y='Age', data=haberman)
plt.show()
sns.violinplot(x='Survival status',y='Year operation', data=haberman)
plt.show()
