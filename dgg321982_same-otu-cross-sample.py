from random import choices
import string
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import linregress
import statsmodels.api as sm


###### Idea:
###### run "number_of_trial" separate simulations, each of them consists of
###### 1 dummy DNA population and 1 identical 
###### RNA population (identical to the DNA population).
###### But the RNA population later will be subsampled and it is
###### the subsample that we are interested in.


number_of_trial = 50


##### otu_trial_dna_rna holds the simulation data
##### structure: {otu_name: [[DNA, RNA], [DNA, RNA] ... (number_of_trial pairs)]}
##### otu_trial_dna_rna = {"0": [[2632, 1652], [2312, 1577], [2432, 1422] ...], 
#####                      "1": [[1332, 1323], [1421, 1232]]}

otu_trial_dna_rna = {}

dna_sample_size = 30000
rna_sample_size = 30000




####first we generate the dna population, otu is named by indexing
####so the first otu in the list is called "0", second is "1" and so on

a = 5. # shape

for trial in range(number_of_trial):
    s = np.random.pareto(a, dna_sample_size)

    max_value = max(s)

    ###because pareto generate values of a certain density, hard to control. Therefore
    ### multiple it by a factor 10 to produce more otus
    factor = 80


    #### look into the generated pareto data set, count each data point into its otu bin
    for i in s:
        otu_name = int(i * factor)
        
        if otu_name not in otu_trial_dna_rna:
            otu_trial_dna_rna[otu_name] = []
            
            for i in range(number_of_trial):
                otu_trial_dna_rna[otu_name].append([0,0])
        
        otu_trial_dna_rna[otu_name][trial][0] += 1/dna_sample_size
#print (otu_trial_dna_rna)
    
    weight = []
    label = []

    for otu in otu_trial_dna_rna:
        
        weight.append(otu_trial_dna_rna[otu][trial][0])
        label.append(otu)
    
    #### now do the rna sampling
    rna_sub = choices(label, k=rna_sample_size, weights=weight)
    
    for otu in rna_sub:
        otu_trial_dna_rna[otu][trial][1] += 1/rna_sample_size
    
    #print (rna_sub_fre, dna_fre)
    #trial_container.append([dna_fre, rna_sub_fre])
    
##### Here I try to reproduce Yu's sorted otu vs slope plot with matplotlib's errorbar function

selected_otu = []
slopes = []
ci = []
for otu in otu_trial_dna_rna:
    x = []
    y = []
    for trial in range(number_of_trial):
        if otu_trial_dna_rna[otu][trial][0] > 0 and otu_trial_dna_rna[otu][trial][1] > 0:
        
            y.append(math.log(otu_trial_dna_rna[otu][trial][1]/otu_trial_dna_rna[otu][trial][0]))
            x.append(math.log(otu_trial_dna_rna[otu][trial][0]))
    
    if len(x) >= 30:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        otu_dna_count = [otu_trial_dna_rna[otu][trial][0] for x in range(number_of_trial)]
        average = round(sum(otu_dna_count)/len(otu_dna_count), 3)
        #print (otu, slope, 1.96 * std_err)
        selected_otu.append(str(otu) + " ave: (" + str(average) + ")")
        slopes.append(slope)
        ci.append(1.96 * std_err)

### to reproduce the sorting in Yu's graph
selected_otu = [x for _,x in sorted(zip(slopes,selected_otu), reverse=True)]
ci = [x for _,x in sorted(zip(slopes,ci), reverse=True)]
slopes = sorted(slopes, reverse=True)

plt.rcParams['figure.figsize'] = [30, 30]

fig, ax = plt.subplots()

ax.errorbar(slopes, selected_otu, xerr=ci)
ax.set_title('Slope +- 95% CI')


plt.show()

from random import choices
import string
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import linregress
import statsmodels.api as sm


###### Idea:
###### run "number_of_trial" separate simulations, each of them consists of
###### 1 dummy DNA population and 1 identical 
###### RNA population (identical to the DNA population).
###### But the RNA population later will be subsampled and it is
###### the subsample that we are interested in.


number_of_trial = 50


##### otu_trial_dna_rna holds the simulation data
##### structure: {otu_name: [[DNA, RNA], [DNA, RNA] ... (number_of_trial pairs)]}
##### otu_trial_dna_rna = {"0": [[2632, 1652], [2312, 1577], [2432, 1422] ...], 
#####                      "1": [[1332, 1323], [1421, 1232]]}

otu_trial_dna_rna = {}

dna_sample_size = 80000
rna_sample_size = 40000




####first we generate the dna population, otu is named by indexing
####so the first otu in the list is called "0", second is "1" and so on

a = 5. # shape

for trial in range(number_of_trial):
    s = np.random.pareto(a, dna_sample_size)

    max_value = max(s)

    ###because pareto generate values of a certain density, hard to control. Therefore
    ### multiple it by a factor 10 to produce more otus
    factor = 80


    #### look into the generated pareto data set, count each data point into its otu bin
    for i in s:
        otu_name = int(i * factor)
        
        if otu_name not in otu_trial_dna_rna:
            otu_trial_dna_rna[otu_name] = []
            
            for i in range(number_of_trial):
                otu_trial_dna_rna[otu_name].append([0,0])
        
        otu_trial_dna_rna[otu_name][trial][0] += 1/dna_sample_size
#print (otu_trial_dna_rna)
    
    weight = []
    label = []

    for otu in otu_trial_dna_rna:
        
        weight.append(otu_trial_dna_rna[otu][trial][0])
        label.append(otu)
    
    #### now do the rna sampling
    rna_sub = choices(label, k=rna_sample_size, weights=weight)
    
    for otu in rna_sub:
        otu_trial_dna_rna[otu][trial][1] += 1/rna_sample_size
    
    #print (rna_sub_fre, dna_fre)
    #trial_container.append([dna_fre, rna_sub_fre])
    
##### Here I try to reproduce Yu's sorted otu vs slope plot with matplotlib's errorbar function

selected_otu = []
slopes = []
ci = []
for otu in otu_trial_dna_rna:
    x = []
    y = []
    for trial in range(number_of_trial):
        if otu_trial_dna_rna[otu][trial][0] > 0 and otu_trial_dna_rna[otu][trial][1] > 0:
        
            y.append(math.log(otu_trial_dna_rna[otu][trial][1]/otu_trial_dna_rna[otu][trial][0]))
            x.append(math.log(otu_trial_dna_rna[otu][trial][0]))
    
    if len(x) >= 30:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        otu_dna_count = [otu_trial_dna_rna[otu][trial][0] for x in range(number_of_trial)]
        average = round(sum(otu_dna_count)/len(otu_dna_count), 3)
        #print (otu, slope, 1.96 * std_err)
        selected_otu.append(str(otu) + " ave: (" + str(average) + ")")
        slopes.append(slope)
        ci.append(1.96 * std_err)

### to reproduce the sorting in Yu's graph
selected_otu = [x for _,x in sorted(zip(slopes,selected_otu), reverse=True)]
ci = [x for _,x in sorted(zip(slopes,ci), reverse=True)]
slopes = sorted(slopes, reverse=True)

plt.rcParams['figure.figsize'] = [30, 30]

fig, ax = plt.subplots()

ax.errorbar(slopes, selected_otu, xerr=ci)
ax.set_title('Slope +- 95% CI')


plt.show()

from random import choices
import string
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import linregress
import statsmodels.api as sm


number_of_trial = 50


##### otu_trial_dna_rna holds the simulation data
##### structure: {otu_name: [[DNA, RNA], [DNA, RNA] ... (number_of_trial pairs)]}
##### otu_trial_dna_rna = {"0": [[2632, 1652], [2312, 1577], [2432, 1422] ...], 
#####                      "1": [[1332, 1323], [1421, 1232]]}

otu_trial_dna_rna = {}

dna_sample_size = 30000
rna_sample_size = 10000




####first we generate the dna population, otu is named by indexing
####so the first otu in the list is called "0", second is "1" and so on

a = 5. # shape

for trial in range(number_of_trial):
    s = np.random.pareto(a, dna_sample_size)

    max_value = max(s)

    ###because pareto generate values of a certain density, hard to control. Therefore
    ### multiple it by a factor 10 to produce more otus
    factor = 80


    #### look into the generated pareto data set, count each data point into its otu bin
    for i in s:
        otu_name = int(i * factor)
        
        if otu_name not in otu_trial_dna_rna:
            otu_trial_dna_rna[otu_name] = []
            
            for i in range(number_of_trial):
                otu_trial_dna_rna[otu_name].append([0,0])
        
        otu_trial_dna_rna[otu_name][trial][0] += 1
#print (otu_trial_dna_rna)
    
    weight = []
    label = []

    for otu in otu_trial_dna_rna:
        
        weight.append(otu_trial_dna_rna[otu][trial][0])
        label.append(otu)
    
    #### now do the rna sampling
    rna_sub = choices(label, k=rna_sample_size, weights=weight)
    
    for otu in rna_sub:
        otu_trial_dna_rna[otu][trial][1] += 1
    
    #print (rna_sub_fre, dna_fre)
    #trial_container.append([dna_fre, rna_sub_fre])

### plot the DNA count distribution of RNA-0 OTUs:
    
DNA_count_from_RNA_0_otu = []

for otu in otu_trial_dna_rna:
    x = []
    y = []
    for trial in range(number_of_trial):
        if otu_trial_dna_rna[otu][trial][1] == 0:
            if otu_trial_dna_rna[otu][trial][0] > 0:
                DNA_count_from_RNA_0_otu.append(otu_trial_dna_rna[otu][trial][0])

#print (DNA_count_from_RNA_0_otu)

sns.distplot(DNA_count_from_RNA_0_otu, kde=False).set_title('Small OTUs are more likely to have 0 RNA counts')
plt.xlabel("DNA counts")
plt.ylabel("OTU counts")
from random import choices
import string
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import linregress
import statsmodels.api as sm


number_of_trial = 1


##### otu_trial_dna_rna holds the simulation data
##### structure: {otu_name: [[DNA, RNA], [DNA, RNA] ... (number_of_trial pairs)]}
##### otu_trial_dna_rna = {"0": [[2632, 1652], [2312, 1577], [2432, 1422] ...], 
#####                      "1": [[1332, 1323], [1421, 1232]]}

otu_trial_dna_rna = {}

dna_sample_size = 30000
rna_sample_size = 10000




####first we generate the dna population, otu is named by indexing
####so the first otu in the list is called "0", second is "1" and so on

a = 5. # shape

x = []
y = []

for trial in range(number_of_trial):
    s = np.random.pareto(a, dna_sample_size)

    max_value = max(s)

    ###because pareto generate values of a certain density, hard to control. Therefore
    ### multiple it by a factor 10 to produce more otus
    factor = 80


    #### look into the generated pareto data set, count each data point into its otu bin
    for i in s:
        otu_name = int(i * factor)
        
        if otu_name not in otu_trial_dna_rna:
            otu_trial_dna_rna[otu_name] = []
            
            for i in range(number_of_trial):
                otu_trial_dna_rna[otu_name].append([0,0])
        
        otu_trial_dna_rna[otu_name][trial][0] += 1
#print (otu_trial_dna_rna)
    
    weight = []
    label = []

    for otu in otu_trial_dna_rna:
        
        weight.append(otu_trial_dna_rna[otu][trial][0])
        label.append(otu)
    
    #### now do the rna sampling
    rna_sub = choices(label, k=rna_sample_size, weights=weight)
    
    for otu in rna_sub:
        otu_trial_dna_rna[otu][trial][1] += 1
    
    for otu in otu_trial_dna_rna:
        
        if otu_trial_dna_rna[otu][trial][0] > 0 and otu_trial_dna_rna[otu][trial][1] > 0:
            log_x = math.log(otu_trial_dna_rna[otu][trial][0])
            log_y = math.log(otu_trial_dna_rna[otu][trial][1] / otu_trial_dna_rna[otu][trial][0])
        
            x.append(log_x)
            y.append(log_y)

### The vacuum plot
ax = sns.scatterplot(x=x, y=y, s=200).set_title('OTUs with DNA > 0 and RNA > 0')
plt.xlabel("log(DNA)")
plt.ylabel("log(RNA/DNA)")
    #print (rna_sub_fre, dna_fre)
    #trial_container.append([dna_fre, rna_sub_fre])

### create log(RNA/DNA) vs log(DNA) vacuum plot


from random import choices
import string
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import linregress
import statsmodels.api as sm


number_of_trial = 1


##### otu_trial_dna_rna holds the simulation data
##### structure: {otu_name: [[DNA, RNA], [DNA, RNA] ... (number_of_trial pairs)]}
##### otu_trial_dna_rna = {"0": [[2632, 1652], [2312, 1577], [2432, 1422] ...], 
#####                      "1": [[1332, 1323], [1421, 1232]]}

otu_trial_dna_rna = {}

dna_sample_size = 30000
rna_sample_size = 10000




####first we generate the dna population, otu is named by indexing
####so the first otu in the list is called "0", second is "1" and so on

a = 5. # shape

x = []
y = []

for trial in range(number_of_trial):
    s = np.random.pareto(a, dna_sample_size)

    max_value = max(s)

    ###because pareto generate values of a certain density, hard to control. Therefore
    ### multiple it by a factor 10 to produce more otus
    factor = 80


    #### look into the generated pareto data set, count each data point into its otu bin
    for i in s:
        otu_name = int(i * factor)
        
        if otu_name not in otu_trial_dna_rna:
            otu_trial_dna_rna[otu_name] = []
            
            for i in range(number_of_trial):
                otu_trial_dna_rna[otu_name].append([0,0])
        
        otu_trial_dna_rna[otu_name][trial][0] += 1
#print (otu_trial_dna_rna)
    
    weight = []
    label = []

    for otu in otu_trial_dna_rna:
        
        weight.append(otu_trial_dna_rna[otu][trial][0])
        label.append(otu)
    
    #### now do the rna sampling
    rna_sub = choices(label, k=rna_sample_size, weights=weight)
    
    for otu in rna_sub:
        otu_trial_dna_rna[otu][trial][1] += 1
    
    for otu in otu_trial_dna_rna:
        
        if otu_trial_dna_rna[otu][trial][0] > 0 and otu_trial_dna_rna[otu][trial][1] > 0:
            ox = otu_trial_dna_rna[otu][trial][0]
            oy =otu_trial_dna_rna[otu][trial][1] / otu_trial_dna_rna[otu][trial][0]
        
            x.append(ox)
            y.append(oy)

### The vacuum plot
ax = sns.scatterplot(x=x, y=y, s=200).set_title('OTUs with DNA > 0 and RNA > 0')
plt.xlabel("DNA")
plt.ylabel("RNA/DNA")
    #print (rna_sub_fre, dna_fre)
    #trial_container.append([dna_fre, rna_sub_fre])

### create log(RNA/DNA) vs log(DNA) vacuum plot

