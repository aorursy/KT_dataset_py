import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

df = pd.read_csv("../input/new_york_tree_census_1995.csv")
result = pd.DataFrame(index=[1,2,4,8,16,32,64,128,256,512,1024,2048], 
                      columns=[1,2,4,8,16,32,64,128,256,512,1024,2048])
# number of different species
len(df["spc_common"].unique())
# keep only the 10 most common species
df = df[df.spc_common.isin(df["spc_common"].value_counts()[:10].index)]
# check the current distribution for the 10 most common trees
print(df["spc_common"].value_counts()/len(df))
(df["spc_common"].value_counts()/len(df)).plot(kind="barh")
df_copy = df.copy()

df_copy["spc_common"] = 0
df_copy[df["spc_common"] == "LONDON PLANETREE"] = 1

plt.title('The distribution of "LONDON PLANETREE", 1 for positive examples and 0 otherwise')
plt.hist(df_copy["spc_common"],bins=np.arange(0,1.02,0.02))
plt.show()
for sample_size in [1,2,4,8,16,32,64,128,256,512,1024,2048]:
    for number_of_samples in [1,2,4,8,16,32,64,128,256,512,1024,2048]:
        result.loc[sample_size,number_of_samples] = []
        for i in range(number_of_samples):
            temp = df.sample(sample_size)
            mean_of_samples = len(temp[temp["spc_common"]=='LONDON PLANETREE'])/sample_size
            result.loc[sample_size,number_of_samples].append(mean_of_samples)
num_max = 12

f, ax = plt.subplots(num_max,num_max,figsize=(12,12))

for i in range(num_max):
    for j in range(num_max):
        plt.setp(ax[i][j].get_xticklabels(), visible=False)
        plt.setp(ax[i][j].get_yticklabels(), visible=False)
        ax[i][j].hist(result.iloc[i,j],bins=np.arange(0,1.02,0.02))

plt.subplots_adjust(wspace=0, hspace=0)   

f.suptitle('Random Sample Mean distribution, function of sample size and number of samples', 
           fontsize = 16)
plt.show()

temp = df_copy.sample(1024)
mean = temp["spc_common"].mean()
confidence_interval = 1.96*(temp["spc_common"].std())/np.sqrt(1024)

lower_bound = mean - confidence_interval
upper_bound = mean + confidence_interval

print("mean is:", mean, "with a 95% confidence between:", lower_bound, "and", upper_bound)
real_mean = df_copy["spc_common"].mean()
number_of_simulations = 10000

for sample_size in [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]:
    number_outside = 0
    intervals = []
    for i in range(number_of_simulations):
        temp = df_copy.sample(sample_size)
        mean = temp["spc_common"].mean()
        confidence_interval = 1.96*(temp["spc_common"].std())/np.sqrt(sample_size)
        intervals.append(confidence_interval)
        if abs(real_mean-mean) > confidence_interval:
            number_outside += 1
            
        
    print(sample_size, "\t", int(1000*(1 - number_outside/number_of_simulations)+0.5)/10, "% \tCI:", int(sum(intervals)*10000/len(intervals)+0.5)/10000)