import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input")) #listing data files
import matplotlib.pyplot as plt
print(plt.style.available) # look at available plot styles
#emulates the aesthetics of ggplot
plt.style.use("ggplot")
from operator import attrgetter
#data list indicating preference (notes) in interval [1,6]
data1 = pd.DataFrame({"preferences":[4,6,2,2,1,2,3,2,4,4]})
fv1 = data1["preferences"].value_counts(sort=False)  #fv1 is a series
#1st column (indicating notes is the index of the series)
fv1
#making data look nice: create data frame from series above
notes = pd.DataFrame({"notes":fv1.index,"frequence":fv1.values},index=range(len(fv1)))
notes
plt = notes.frequence.plot.bar()
#Let's calculate frequency distributions of another data set using intervals of data
mylist = [20,18,6,24,33,9,10,19,27,33,22,17,19,31,25,21,28,13,21,12,33,23,18,13,7,16,7,26]

#create dataframe from the list above
data2 = pd.DataFrame( {'values':mylist} )
#defining explicit intervals of classification
fv2 = data2["values"].value_counts(sort=False,bins=[4,9,14,19,24,29,34])
print(fv2)
#or calculate width
max = np.max(data2['values'])
min = np.min(data2['values'])
print("min =",min,"max =",max)
fv2 = data2["values"].value_counts(sort=False,bins=range(4,35,5))
fv2
#convert series result to a dataframe
table1 = pd.DataFrame({"intervals":fv2.index,"f":fv2.values},index=range(len(fv2)))
table1
#cumulative frequence
table1 = (table1.assign(F=table1.f.cumsum()))
#total number or frequencies 
N = table1.f.sum()
#relative frequencies
table1['f%'] = (table1['f'] / N) * 100
table1['F%'] = (table1['F'] / N) * 100
table1
plt = table1.plot.bar(x='intervals', y="f")
mylist = [2,3,3,5,8,9,12]
#create data frame
data = pd.DataFrame( {'values':mylist} )
data
sum_of_data = data.values.sum()
n = len(data.index)
mean = sum_of_data / n
mean
mean = data.values.mean()
mean
data['dev'] = data['values'] - mean
data
data.dev.sum()
data['abs_dev'] = np.absolute(data['dev'])
data
Mean_deviation = data['abs_dev'].sum()
Mean_deviation
data['square_dev'] = data['dev']**2
data
variance = data.square_dev.sum()/len(data.index)
print("variance=",variance)
print("std_dev=",np.sqrt(variance))
#pandas calculations
#Delta Degrees of Freedom: denominator of fraction is (n - ddof)
#in this case, biased formula with n at denominator
data['values'].std(ddof = 0)
#having a sample, we use unbiased formula with n-1
data['values'].std(ddof = 1)
# read csv (comma separated value) into dataframe
data2 = pd.read_csv('../input/test2.csv')
data2.head()
data_mean = data2.weight.mean()
data_sd = data2.weight.std(ddof = 1)
print("mean =",data_mean,"standard deviation =",data_sd)
data2.describe()
#freq by intervals (interval limits analytic defined)
weight_f = data2["weight"].value_counts(sort=False,bins=[84,94,104,114,124,134,144,164,174])
weight_f
#interval limits defined using range
weight_f = data2["weight"].value_counts(sort=False,bins=range(84,175,10))
weight_f
#convert series result to a dataframe
table = pd.DataFrame({"intervals":weight_f.index, "f":weight_f.values},index=range(len(weight_f)))
table
#get middle of intervals (mean of margins) 
#m stands for x values as the "representative" value for the interval
#so for the 1st row, for example, we could say we have 3 values of 88.9995
table['m'] = table['intervals'].map(attrgetter('mid'))
table
table['fm'] = table['f'] * table['m']
table['fm2'] = table['f'] * table['m'] ** 2
table
sum_f = table.f.sum()
sum_fm = table.fm.sum()
sum_fm2 = table.fm2.sum()

s = np.sqrt((sum_f * sum_fm2 - sum_fm ** 2) / (sum_f * (sum_f-1)) ) 
print("std. dev for freq. distribution =", s)
#which is closed to
print("std. dev of data =", data_sd)
mean_f = sum_fm / sum_f
print("mean of freq. distrib. =",mean_f)
#which is closed to
print("data mean =", data_mean)
