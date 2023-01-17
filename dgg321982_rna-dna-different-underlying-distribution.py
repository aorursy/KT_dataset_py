### first simulation, alphabet with hand code power distribution

from random import choices
import string
import matplotlib.pyplot as plt
import seaborn as sns

dna_fre = []

multiple = 1000


total = 0
for l in string.ascii_lowercase:
    value = 1
    if l == "a":
        value = 40 * multiple
    elif l == "b":
        value = 23 * multiple
    elif l == "c":
        value = 12 * multiple
    elif l == "d":
        value = 0.3 * multiple
    else:
        value = 0.01 * multiple
    
    dna_fre.append(value)

    total += value
    
print ("total DNA reads:", total)

print ("dna_fre looks like this:", dna_fre)

##### assume 1 DNA molecule generates 1 RNA molecule
##### so dna_fre = rna_fre, no need to define rna_fre

##### and then we subsample the RNA population as rna_sub


rna_sub = choices(string.ascii_lowercase, k=5000, weights=dna_fre)

### plot
x = []
y = []
a = []

for index, l in enumerate(string.ascii_lowercase):
    x.append(dna_fre[index])
    y.append(rna_sub.count(l) / dna_fre[index])
    a.append(l)

fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(a):
    ax.annotate(a[i], (x[i], y[i]))


#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
fig.show()

ax.set_xlabel('DNA')
ax.set_ylabel('RNA/DNA')
import numpy as np
import matplotlib.pyplot as plt
import math
from random import choices



dna_sample_size = 30000
rna_sample_size = 10000

####first we generate the dna population
a = 5. # shape

s = np.random.pareto(a, dna_sample_size)

max_value = max(s)

###because pareto generate values of a certain density, hard to control. Therefore
### multiple it by a factor 10 to produce more otus
factor = 50

print (max_value)

dna_fre = {}

#### ini dna_fre
for i in range(0, int(max_value * factor) + 1):
    dna_fre[i] = 0

#### look into the generated pareto data set, count each data point into its otu bin
for i in s:
    bin_index = int(i * factor)
    dna_fre[bin_index] += 1

### uncomment the next line if you want details in dna_fre
#print ("here is the otu : read counts", dna_fre)






####IMPORTANT: later we assume, 1 molecule DNA generates 1 molecule RNA.
#### So if we have 40000 reads in otu "0" in DNA population, 
#### we also have 40000 RNA in otu "0" in RNA population.
#### BUT:
#### the RNA population is staying in the background. A subset is drawn from this RNA population
#### and it is this subset of RNA we are interested in.
#### we are going to plot this RNA subset against the DNA population.
#### I have found, it doesn't matter whether the subsampling is with or without replacement.


### define weight, label for RNA sampleing function "choices" later. 
### Weight holds the amount of reads, label holds 
### the otu names. So for example for otu "0": 2826, we put 2826 into weight, and "0" into label
### repeat it for otus.
### These two arrays hold the exact same info like dna_fre


#### just plot dna_fre to see whether it is what we want
weight = []
label = []
for key in dna_fre:
    weight.append(dna_fre[key])
    label.append(key)

fig, ax = plt.subplots()
ax.scatter(label, weight)

for i, txt in enumerate(label):
    ax.annotate(label[i], (label[i], weight[i]))

#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
fig.show()
ax.set_xlabel('OTU')
ax.set_ylabel('DNA')
#### now do the rna sampling

rna_sub = choices(label, k=rna_sample_size, weights=weight)
#print ("RNA sub:", rna_sub)
#print (sub)

### now just plot the the RNA against DNA for each otu, it should be a "rubber bulb pipette" shape

### the "rubber bulb" are the small guys, because they tend to have more extreme situations

### the "pipette" part is the big guys, because the law of large numbers, they are closing on the
### RNA / DNA sample ratio

x = []
y = []


for l in label:

    x.append(dna_fre[l])
    y.append(rna_sub.count(l))


fig, ax = plt.subplots()
ax.scatter(x, y)

for i, txt in enumerate(label):
    ax.annotate(label[i], (x[i], y[i]))
#best fit curve
#ax = sns.regplot(x=x, y=y, order=4, scatter_kws={"color": "steelblue"}, line_kws={"color": "green"})

ax.set_xlabel('DNA count')
ax.set_ylabel('RNA count')
#### Now generate the RNA/DNA vs DNA graph


### An horizontal "T" graph is expected. 
### The big guys form a follow around the RNA/DNA sampling ratio, in this case 2

x = []
y = []


for l in label:
    if dna_fre[l] != 0 and rna_sub.count(l) != 0:
        x.append(dna_fre[l])
        #y.append(sub.count(l) / dna_fre[l])
        
        y.append(rna_sub.count(l) / dna_fre[l])
    else:
        x.append(0)
        y.append(0)

fig, ax = plt.subplots()
ax.scatter(x, y)

#ax = sns.regplot(x=x, y=y, order=2, scatter_kws={"color": "steelblue"}, line_kws={"color": "green"})


ax.set_xlabel('DNA')
ax.set_ylabel('RNA/DNA')
############ plot both in rel. abundance

### in fact, this plot is the second graph in disguise. Just scale both axes to 0-1 range.
### still a straight line because all values are divided by a constant in both axes.

x = []
y = []


for l in label:
    if dna_fre[l] != 0 and rna_sub.count(l) != 0:
        x.append(dna_fre[l]/dna_sample_size)
        y.append(rna_sub.count(l) / rna_sample_size)
        
        #y.append(math.log(sub.count(l) / dna_fre[l]))
    else:
        x.append(0)
        y.append(0)

fig, ax = plt.subplots()
ax.scatter(x, y)

ax.set_xlabel('rel. DNA')
ax.set_ylabel('rel. RNA')
############ plot RNA/DNA and DNA in rel. abundance

### It still generates the same shape
###

x = []
y = []


for l in label:
    if dna_fre[l] != 0 and rna_sub.count(l) != 0:
        x.append(dna_fre[l]/dna_sample_size)
        y.append((rna_sub.count(l) / rna_sample_size) / (dna_fre[l]/dna_sample_size))
        
        #y.append(math.log(sub.count(l) / dna_fre[l]))
    else:
        x.append(0)
        y.append(0)

fig, ax = plt.subplots()
ax.scatter(x, y)
#ax = sns.regplot(x=x, y=y, order=2, scatter_kws={"color": "steelblue"}, line_kws={"color": "green"})

ax.set_xlabel('rel. DNA')
ax.set_ylabel('rel. RNA / rel. DNA')
##### plot y in log(RNA/DNA, 10)

### again, same info like the second to the last graph,
### this time the floor is around log(2, 10) = 0.3

x = []
y = []


for l in label:
    if dna_fre[l] != 0 and rna_sub.count(l) != 0:
        x.append(dna_fre[l])
        #y.append(sub.count(l) / dna_fre[l])
        
        y.append(math.log(rna_sub.count(l) / dna_fre[l], 10))
    else:
        x.append(0)
        y.append(0)

fig, ax = plt.subplots()
ax.scatter(x, y)
#ax = sns.regplot(x=x, y=y, order=1, scatter_kws={"color": "steelblue"}, line_kws={"color": "green"})


ax.set_xlabel('DNA')
ax.set_ylabel('log(RNA/DNA)')

##### plot with cut off of 10 in both DNA and RNA

x = []
y = []


for l in label:
    if dna_fre[l] >= 10 and rna_sub.count(l) >= 10:
        x.append(dna_fre[l])
        #y.append(sub.count(l) / dna_fre[l])
        
        y.append(math.log(rna_sub.count(l) / dna_fre[l], 10))
    else:
        x.append(0)
        y.append(0)

fig, ax = plt.subplots()
ax.scatter(x, y)
#ax = sns.regplot(x=x, y=y, order=2, scatter_kws={"color": "steelblue"}, line_kws={"color": "green"})

ax.set_xlabel('DNA (DNA >= 10)')
ax.set_ylabel('log(RNA/DNA) (RNA >= 10)')


fig.show()
### now try the whole thing again, this time the OTUs are "power" distributed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a = 4. # shape
samples = 15000
unique_otu = 300
factor = 10000


s = np.random.power(a, samples)

fre = {}

for i in range(unique_otu):
    fre[i] = 0

for i in s:
    #print (i * factor)
    bin_index = int(i * factor / (factor/unique_otu))
    fre[bin_index] += 1

#### uncomment the next line if you want to see the details of the otus
##print ("DNA fre:", fre)

### plot the DNA : otu 
weight = []
label = []
for key in fre:
    weight.append(fre[key])
    label.append(key)

fig, ax = plt.subplots()
ax.scatter(label, weight)

for i, txt in enumerate(label):
    ax.annotate(label[i], (label[i], weight[i]))

#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
fig.show()

ax.set_xlabel('OTU')
ax.set_ylabel('DNA')

#print (label)
#print (weight)

### subset the rna population
rna_sub = choices(label, k=3000, weights=weight)
#print (sub)




x = []
y = []


for l in label:
    
    x.append(fre[l])
    y.append(rna_sub.count(l))


fig, ax = plt.subplots()
ax.scatter(x, y)


for i, txt in enumerate(label):
    ax.annotate(label[i], (x[i], y[i]))
ax.set_xlabel('DNA')
ax.set_ylabel('RNA')

#### best fit curve
#ax = sns.regplot(x=x, y=y, order=4, scatter_kws={"color": "steelblue"}, line_kws={"color": "green"})
### uncomment the next line if you want to see the detail in rna_fre
#print ("RNA fre:", rna_fre)
    
### plot the RNA/DNA vs DNA
x = []
y = []


for l in label:
    if fre[l] != 0:
        x.append(fre[l])
        y.append(rna_sub.count(l) / fre[l])
    else:
        x.append(0)
        y.append(0)

fig, ax = plt.subplots()
ax.scatter(x, y)
#ax = sns.regplot(x=x, y=y, order=2, scatter_kws={"color": "steelblue"}, line_kws={"color": "green"})


ax.set_xlabel('DNA')
ax.set_ylabel('RNA/DNA')
#print ("99 was chosen:", sub.count(99), "while 30 was chosen", sub.count(30))
#print ("out of", len(sub), "RNAs")
print ("there are :", sum(weight), "otus")
#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
fig.show()
