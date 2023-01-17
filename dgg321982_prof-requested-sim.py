### first simulation, alphabet with hand code power distribution

from random import choices
import string
import matplotlib.pyplot as plt
import seaborn as sns

dna_fre = []

multiple = 1000

for l in string.ascii_lowercase:
    
    value = multiple
    
    dna_fre.append(value)
    

print ("dna_fre looks like this:", dna_fre)

##### assume 1 DNA molecule generates 1 RNA molecule
##### so dna_fre = rna_fre, no need to define rna_fre

##### and then we subsample the RNA population as rna_sub

dna_sub = choices(string.ascii_lowercase, k=1000, weights=dna_fre)
rna_sub = choices(string.ascii_lowercase, k=1000, weights=dna_fre)

### plot
x = []
y = []
a = []

for index, l in enumerate(string.ascii_lowercase):
    x.append(dna_sub.count(l))
    y.append(rna_sub.count(l))
    a.append(l)

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(a):
    ax.annotate(a[i], (x[i], y[i]))


#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
fig.show()

ax.set_xlabel('DNA')
ax.set_ylabel('RNA')
### first simulation, alphabet with hand code power distribution

from random import choices
import string
import matplotlib.pyplot as plt
import seaborn as sns

dna_fre = []

multiple = 1000

total_dna = 0

for l in string.ascii_lowercase:
    
    value = multiple
    
    dna_fre.append(value)
    
    total_dna += value
    

print ("dna_fre looks like this:", dna_fre)

##### assume 1 DNA molecule generates 1 RNA molecule
##### so dna_fre = rna_fre, no need to define rna_fre

##### and then we subsample the RNA population as rna_sub

#### Just for letter "t"

target = "t"
x = []
y = []
a = []

for i in range(200):

    dna_sub = choices(string.ascii_lowercase, k=1000, weights=dna_fre)
    rna_sub = choices(string.ascii_lowercase, k=1000, weights=dna_fre)

    x.append(dna_sub.count(target))
    y.append(rna_sub.count(target))
    a.append(str(i))
### plot



fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(a):
    ax.annotate(a[i], (x[i], y[i]))


#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
fig.show()

ax.set_xlabel('DNA')
ax.set_ylabel('RNA')


print (len(s))
####without replacement
import numpy as np
import matplotlib.pyplot as plt
import math
#from random import choices
#import numpy as np
import seaborn as sns
from scipy.stats import linregress


how_many_unique_otus = 1000


####first we generate the dna population
a = 5. # shape
mu = 3.
sigma = 1. # mean and standard deviation
#s = np.random.lognormal(mu, sigma, 1000)

#s = np.random.pareto(a, dna_sample_size)
s = np.random.lognormal(mu, sigma, how_many_unique_otus)

s = list(map(int, s))

#print (sum(s))

count, bins, ignored = plt.hist(s, 100, align='mid')

plt.xlabel('OTU Frequency')
plt.ylabel('Counts')


dna_fre = {}
for i in range(len(s)):
    dna_fre[i] = s[i]

#non_zero = 0
#weight = []
label = []

for key in dna_fre:
    if dna_fre[key] > 0:
        label += [key] * dna_fre[key]
        #non_zero += 1


#print ("sum_weight", sum(weight))
        
rna_sample_sizes = [int(sum(s) * 0.2), int(sum(s) * 0.5), int(sum(s) * 1)]

fig = plt.figure(1)
cutoff = 12
#print (len(rna_sample_sizes))

for index, rna_sample_size in enumerate(rna_sample_sizes):
#rna_sample_size = 10000
    #print (rna_sample_size)

    #### now do the rna sampling with replacement
    #rna_sub = choices(label, k=rna_sample_size, weights=weight)
    
    #### now do the rna sampling without replacement
    
    #print (sum(s), len(label))
    rna_sub = np.random.choice(label, rna_sample_size, replace=False).tolist()
    
    
    #print (dna_fre)
    #print (weight)
    #print (rna_sub)
    x0 = []
    y0 = []
    x0_y0 = []

    
    for l in label:

        if dna_fre[l] > 0:
            x0.append(dna_fre[l])
            y0.append(rna_sub.count(l))
            x0_y0.append(rna_sub.count(l)/dna_fre[l])


    #fig, ax = plt.subplots()
    
    #ax1 = fig.add_subplot(len(rna_sample_sizes),2, index + 1)
    #ax1.scatter(x, y)
    
    x1 = []
    y1 = []
    x1_y1 = []

    
    for l in label:

        if dna_fre[l] > cutoff:
            x1.append(dna_fre[l])
            y1.append(rna_sub.count(l))
            x1_y1.append(rna_sub.count(l)/dna_fre[l])


    #fig, ax = plt.subplots()
    slope0, intercept0, r_value0, p_value0, std_err0 = linregress(x0, y0)
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x1, y1)

    
    #ax2 = fig.add_subplot(len(rna_sample_sizes),2, index + 2)
    #ax2.scatter(x, y)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.scatter(x0, x0_y0)
    #ax2.scatter(x0, y0)
    sns.regplot(x=x0, y=y0, ax=ax3)
    ax2.scatter(x1, x1_y1)
    sns.regplot(x=x1, y=y1, ax=ax4)
    
    ax1.set_title(str(round(rna_sample_size/sum(s), 2)))
    #ax1.set_title('mu:' + str(mu) + ', DNA:' + str(sum(s)) + ", RNA: " + str(rna_sample_size))
    ax2.set_title("cutoff: " + str(cutoff))
    ax3.set_title("slope: " + str(round(slope0, 2)))
    
    ax4.set_title("slope: " + str(round(slope1, 2)))

    ax1.set_xlabel('DNA count')
    ax1.set_ylabel('RNA/DNA')
    ax3.set_ylabel('RNA count')
    #print (x)
    #print (y)
    #for i, txt in enumerate(label):
    #    ax.annotate(label[i], (x[i], y[i]))
    #best fit curve
    #ax = sns.regplot(x=x, y=y, order=4, scatter_kws={"color": "steelblue"}, line_kws={"color": "green"})
    
    #fig.suptitle('mu:' + str(3) + ', DNA:' + str(sum(s)) + ", RNA: " + str(rna_sample_size))
    #ax.set_xlabel('DNA count')
    #ax.set_ylabel('RNA/DNA')
####without replacement
import numpy as np
import matplotlib.pyplot as plt
import math
#from random import choices
#import numpy as np
import seaborn as sns
from scipy.stats import linregress


how_many_unique_otus = 1000


####first we generate the dna population
a = 5. # shape
mu = 3.
sigma = 1. # mean and standard deviation
#s = np.random.lognormal(mu, sigma, 1000)

#s = np.random.pareto(a, dna_sample_size)
s = np.random.lognormal(mu, sigma, how_many_unique_otus)

s = list(map(int, s))

#print (sum(s))

count, bins, ignored = plt.hist(s, 100, align='mid')

plt.xlabel('OTU Frequency')
plt.ylabel('Counts')


dna_fre = {}
for i in range(len(s)):
    dna_fre[i] = s[i]

#non_zero = 0
#weight = []
label = []

for key in dna_fre:
    if dna_fre[key] > 0:
        label += [key] * dna_fre[key]
        #non_zero += 1


#print ("sum_weight", sum(weight))
        
rna_sample_sizes = [int(sum(s) * 0.2), int(sum(s) * 0.5), int(sum(s) * 1)]

fig = plt.figure(1)
cutoff = 12
#print (len(rna_sample_sizes))

for index, rna_sample_size in enumerate(rna_sample_sizes):
#rna_sample_size = 10000
    #print (rna_sample_size)

    #### now do the rna sampling with replacement
    #rna_sub = choices(label, k=rna_sample_size, weights=weight)
    
    #### now do the rna sampling without replacement
    
    #print (sum(s), len(label))
    rna_sub = np.random.choice(label, rna_sample_size, replace=True).tolist()
    
    
    #print (dna_fre)
    #print (weight)
    #print (rna_sub)
    x0 = []
    y0 = []
    x0_y0 = []

    
    for l in label:

        if dna_fre[l] > 0:
            x0.append(dna_fre[l])
            y0.append(rna_sub.count(l))
            x0_y0.append(rna_sub.count(l)/dna_fre[l])


    #fig, ax = plt.subplots()
    
    #ax1 = fig.add_subplot(len(rna_sample_sizes),2, index + 1)
    #ax1.scatter(x, y)
    
    x1 = []
    y1 = []
    x1_y1 = []

    
    for l in label:

        if dna_fre[l] > cutoff:
            x1.append(dna_fre[l])
            y1.append(rna_sub.count(l))
            x1_y1.append(rna_sub.count(l)/dna_fre[l])


    #fig, ax = plt.subplots()
    slope0, intercept0, r_value0, p_value0, std_err0 = linregress(x0, y0)
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x1, y1)

    
    #ax2 = fig.add_subplot(len(rna_sample_sizes),2, index + 2)
    #ax2.scatter(x, y)
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.scatter(x0, x0_y0)
    #ax2.scatter(x0, y0)
    sns.regplot(x=x0, y=y0, ax=ax3)
    ax2.scatter(x1, x1_y1)
    sns.regplot(x=x1, y=y1, ax=ax4)
    
    ax1.set_title(str(round(rna_sample_size/sum(s), 2)))
    #ax1.set_title('mu:' + str(mu) + ', DNA:' + str(sum(s)) + ", RNA: " + str(rna_sample_size))
    ax2.set_title("cutoff: " + str(cutoff))
    ax3.set_title("slope: " + str(round(slope0, 2)))
    
    ax4.set_title("slope: " + str(round(slope1, 2)))

    ax1.set_xlabel('DNA count')
    ax1.set_ylabel('RNA/DNA')
    ax3.set_ylabel('RNA count')
    #print (x)
    #print (y)
    #for i, txt in enumerate(label):
    #    ax.annotate(label[i], (x[i], y[i]))
    #best fit curve
    #ax = sns.regplot(x=x, y=y, order=4, scatter_kws={"color": "steelblue"}, line_kws={"color": "green"})
    
    #fig.suptitle('mu:' + str(3) + ', DNA:' + str(sum(s)) + ", RNA: " + str(rna_sample_size))
    #ax.set_xlabel('DNA count')
    #ax.set_ylabel('RNA/DNA')
