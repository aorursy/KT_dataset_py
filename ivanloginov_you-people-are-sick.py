import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.DataFrame(pd.read_csv('../input/sick_people.csv'))
df.info()
df.head()
def draw_simple_hist(data_as_list, colors_as_list, labels_as_list, title):
    
    barlist=plt.bar(range(len(data_as_list)), data_as_list, alpha=0.5, color=colors_as_list)

    for rect in barlist:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%d' % int(height),
                    ha='center', va='bottom')
    axes = plt.gca()
    axes.set_ylim([0,1.2*max(data_as_list)])
    plt.xticks(range(len(data_as_list)),labels_as_list)
    plt.title(title)
m = sum(df.loc[(df.iloc[:,2]=='M') & (df.iloc[:,0]>2),df.columns[0]])
w = sum(df.loc[(df.iloc[:,2]=='F') & (df.iloc[:,0]>2),df.columns[0]])
title = 'Desease frequency by sex'

draw_simple_hist([m,w],['b','r'],['M','F'],title)
plt.show()
def permutation_sample(data_1, data_2):
    data = np.concatenate((data_1,data_2))
    permuted_data = np.random.permutation(data)
    perm_sample_1 = permuted_data[:len(data_1)]
    perm_sample_2 = permuted_data[len(data_1):]
    
    return perm_sample_1, perm_sample_2
def make_perm_reps(data_1, data_2, func, size=1):

    perm_replicates = np.empty(size)

    for i in range(size):

        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates
def diff_in_times(data_1, data_2):
    return sum(data_1)/sum(data_2)
def calc_p_value(data_1, data_2):
    initial_diff_times = diff_in_times(data_1, data_2)

    perm_replicates = make_perm_reps(data_1, data_2, diff_in_times, size=10000)

    p_value = sum(perm_replicates>=initial_diff_times) / len(perm_replicates)
    
    return initial_diff_times, perm_replicates, p_value
def draw_perm_distr(perm_replicates):
    plt.hist(perm_replicates, color='b', alpha=0.5, bins=100)
    plt.title("Permutations' measurements")
    plt.xlabel('value')
    plt.ylabel('frequency')
def draw_p_value(perm_replicates, initial_diff_times, p_value):
    perms = plt.hist(perm_replicates[perm_replicates<initial_diff_times], color='b', alpha=0.5, bins=100, label='permutations')
    init = plt.plot([initial_diff_times]*2,[0,500], color='r', alpha=1, linewidth=3, label='observed difference')
    pv = plt.hist(perm_replicates[perm_replicates>=initial_diff_times], color='g', alpha=0.5, bins=100, label='p-value')
    plt.title('The share of values at least as extreme as observed, p-value')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc=1)
    plt.xlabel('value')
    plt.ylabel('frequency')
men = df.loc[(df.iloc[:,2]=='M') & (df.iloc[:,0]>2),df.columns[0]]
women = df.loc[(df.iloc[:,2]=='F') & (df.iloc[:,0]>2),df.columns[0]]


initial_diff_times, perm_replicates, p_value = calc_p_value(men, women)
plt.figure(figsize=(15,5))
plt.subplot(121)
draw_perm_distr(perm_replicates)
plt.subplot(122)
draw_p_value(perm_replicates, initial_diff_times, p_value)
plt.show()
print('And the p-value equals... %1.3f' %p_value)
over_35 = df.loc[(df.iloc[:,1]>35) & (df.iloc[:,0]>2),df.columns[0]]
under_35 = df.loc[(df.iloc[:,1]<=35) & (df.iloc[:,0]>2),df.columns[0]]

initial_diff_times, perm_replicates, p_value = calc_p_value(over_35, under_35)
plt.figure(figsize=(15,5))
plt.subplot(121)
title = 'Desease frequency by age'
draw_simple_hist([sum(over_35),sum(under_35)],['b','r'],['Elders (over 35)','Babies (under 35)'],title)
plt.subplot(122)
draw_p_value(perm_replicates, initial_diff_times, p_value)
plt.show()
print('The p-value equals %1.3f' %p_value)
