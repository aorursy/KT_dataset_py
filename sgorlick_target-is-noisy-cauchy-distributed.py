import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
target = pd.read_csv('../input/train.csv',usecols=['target'],squeeze=True)
target.value_counts().head(10)
top_targets = list(target.value_counts().head(4).index)
def sample_targets(n):
    sample_plot_data=target[~target.isin(top_targets)].sample(n)
    sample_plot_data=sample_plot_data.groupby(np.round(sample_plot_data,2)).agg('count')
    sample_plot_data=sample_plot_data.rename('count').reset_index()
    return sample_plot_data
k=50
fig = plt.figure(figsize=(25,k*5))
for i in range(k) : 
    if i==0:
        ax = fig.add_subplot(k,1,i+1)
    else:
        ax = fig.add_subplot(k,1,i+1,sharex=ax,sharey=ax)
    sample_plot_data = sample_targets(10000)
    ax.plot(sample_plot_data['target'],sample_plot_data['count'],linewidth=.2,color='g')
    plt.plot(sample_plot_data['target'],cauchy.pdf(sample_plot_data['target'])*120,color='r')
    plt.xticks(np.linspace(-15,15,15+15+1),np.linspace(-15,15,15+15+1))
    plt.xlabel(f'Mean : %.2f, Std : %.2f' % (np.mean(sample_plot_data.target),np.std(sample_plot_data.target)) ) 
plt.show()