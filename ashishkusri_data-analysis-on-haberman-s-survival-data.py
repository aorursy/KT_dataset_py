#Reading Dataset into pandas dataframe

import pandas as p

hab = p.read_csv("haberman.csv")

hab

# Number of data Points and Features

print(hab.shape)
# Features/Attributes/Column name in our dataset

print(hab.columns)

# Data points for each class or How many patient survived and how many not

hab['Survival'].value_counts()  # 1 :=> the patient survived 5 years or longer 
                                # 2 :=> the patient died within 5 year
#2-D Scatter plot when we did not color the points by their class-label/Survival-type
#Analysing by taking two feature 'Patient_Age' on x-axis and '#PositiveAxiliaryNode' on y-axis

import seaborn as sb
import matplotlib.pyplot as mp

sb.set_style('whitegrid')

hab.plot(kind='scatter', x='Patient_Age',y='#PositiveAxiliaryNode')

mp.show()
#2-D scatter plot when we did color the points by their class-label/Survival-type
#Analysing by taking two feature 'Patient_Age' on x-axis and '#PositiveAxiliaryNode' on y-axis

sb.set_style('whitegrid')

sb.FacetGrid(hab,hue='Survival',size=4)\
  .map(mp.scatter,'Patient_Age','#PositiveAxiliaryNode')\
  .add_legend()

mp.show()

#Pairwise combination of each of features

mp.close() #by itself closes the current figure
sb.set_style('whitegrid')
sb.pairplot(hab,hue='Survival',size=3)
mp.show()
#1-D Scatter Plot for 'Patient_Age'

import numpy as np

hab_survive=hab.loc[hab["Survival"]==1]
hab_notsurv=hab.loc[hab["Survival"]==2]

mp.plot(hab_survive["Patient_Age"],np.zeros_like(hab_survive["Patient_Age"]),'o') # 'o' is there to show plot as dot/circle shape/o shape instead of just line

mp.plot(hab_notsurv["Patient_Age"],np.zeros_like(hab_notsurv["Patient_Age"]),'o')

mp.show()
#Histogram/PDF for 'Patient_Age'

mp.close()

sb.FacetGrid(hab,hue="Survival",size=5)\
  .map(sb.distplot,"Patient_Age")\
  .add_legend()

mp.show()
#Histogram/PDF for 'Year_of_operation'

mp.close()

sb.FacetGrid(hab,hue="Survival",size=5)\
  .map(sb.distplot,"Year_of_operation")\
  .add_legend()

mp.show()
#Histogram/PDF for '#PositiveAxiliaryNode'

mp.close()

sb.FacetGrid(hab,hue="Survival",size=5)\
  .map(sb.distplot,"#PositiveAxiliaryNode")\
  .add_legend()

mp.show()
#PDF calculated as counts/frequencies of data points in each window.

#Plot CDF of 'Patient_Age' for survived Patient

counts, bin_edges = np.histogram(hab_survive['Patient_Age'], bins=10, 
                                 density = True)

print(counts)
print(sum(counts))

pdf = counts/(sum(counts))

print('********************')
print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

mp.plot(bin_edges[1:],pdf);
mp.plot(bin_edges[1:], cdf)

mp.show()


#Plot CDF of 'patient_Age' for not-survived Patient

counts, bin_edges = np.histogram(hab_notsurv['Patient_Age'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))

print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

mp.plot(bin_edges[1:],pdf);
mp.plot(bin_edges[1:], cdf)


mp.show()
#plot of data points of both class as together


#Plot CDF of 'patient_Age' for survived Patient

counts, bin_edges = np.histogram(hab_survive['Patient_Age'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))

print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

mp.plot(bin_edges[1:],pdf);
mp.plot(bin_edges[1:], cdf)


#Plot CDF of 'patient_Age' for not-survived Patient

counts, bin_edges = np.histogram(hab_notsurv['Patient_Age'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))

print(pdf)
print(bin_edges)

cdf = np.cumsum(pdf)

mp.plot(bin_edges[1:],pdf);
mp.plot(bin_edges[1:], cdf)


mp.show()


#Means ,calculate the Central Tendency of the observation of a feature

print('Means:')
print(np.mean(hab_survive['Patient_Age']))
print(np.mean(hab_notsurv['Patient_Age']))

#Standard-Deviation will tell us about Spread i.e what are the average square distance of each point from the Mean

print("\nStd-dev:");
print(np.std(hab_survive['Patient_Age']))
print(np.std(hab_notsurv['Patient_Age']))

#But one or small number of Outlier(an error) can corrupt both mean and standard-deviation

print('\nFor mean')
print('For survived Patients :'+str(np.mean(np.append(hab_survive['Patient_Age'],300))))

print('\nStd-dev')
print('For survived Patients :' + str(np.std(np.append(hab_survive['Patient_Age'],200))))


#Median is equivalent to Mean, which calculate the middle value of an obervastion

print('\nMedian:')
print('Median of patient age for survived patient:'+str(np.median(hab_survive['Patient_Age'])))
print('Median of patient age for not survived patient:'+str(np.median(hab_notsurv['Patient_Age'])))

#Median with an outlier
print('\nMedian with an outlier for survived patients')
print(np.median(np.append(hab_survive['Patient_Age'],300))) #we can see that there is no corruption by an outlier

print('\nQuantiles')
print('0th,25th,50th,75th percentile value of patient age of survived patient:'+str(np.percentile(hab_survive['Patient_Age'],np.arange(0,100,25))))
print('0th,25th,50th,75th percentile value of patient age of not survived patient:'+str(np.percentile(hab_notsurv['Patient_Age'],np.arange(0,100,25))))

print('\n90th Percentile')
print('90th percentile value of patient age for survived patient:'+str(np.percentile(hab_survive['Patient_Age'],90)))
print('90th percentile value of patient age for not survived patient:'+str(np.percentile(hab_notsurv['Patient_Age'],90)))

#MAD-Median Absolute Deviation is equivalent to standard-deviation, which measure that how faraway our data points from central tendency(here is median)

print('\nMedian Abolute Deviation')
from statsmodels import robust
print('MAD value of patient age for survived patient:'+str(robust.mad(hab_survive['Patient_Age'])))
print('MAD value of patient age for not survived patient:'+str(robust.mad(hab_notsurv['Patient_Age'])))

print('\nMAD with an outlier for survived patient')
print(robust.mad(np.append(hab_survive['Patient_Age'],200))) #we can see that there is no corruption by an outlier

#Box-plot with whiskers: another method of visualizing the  1-D scatter plot more intuitivey.

sb.boxplot(x='Survival',y='Patient_Age',data=hab)
mp.show()
#A violin plot combines both Histogram/PDF and box-plot 

sb.violinplot(x='Survival', y='Patient_Age', data=hab,size=8)

mp.show()
