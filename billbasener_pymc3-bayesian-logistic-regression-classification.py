# This is a simple example of using pymc3 for Bayesian inference of the parameter distribution.

# written by William F Basener

# University of Virginia



import pymc3 as pm

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt
def plot_traces(traces, retain=0):

    '''

    Convenience function:

    Plot traces with overlaid means and values

    '''



    ax = pm.traceplot(traces[-retain:],

                      lines=tuple([(k, {}, v['mean'])

                                   for k, v in pm.summary(traces[-retain:]).iterrows()]))



    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):

        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'

                    ,xytext=(5,10), textcoords='offset points', rotation=90

                    ,va='bottom', fontsize='large', color='#AA0022')
data = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

data.head()
g = sns.pairplot(data, hue="species", palette="husl", markers=["o", "s", "D"])
with pm.Model() as model:

    pm.glm.GLM.from_formula(formula = 'species ~ sepal_length + sepal_width + petal_length + petal_width', 

                            data = data, 

                            family = pm.glm.families.Binomial())



    trace = pm.sample(1000)
trace.varnames
plot_traces(trace)
pm.plots.forestplot(trace, figsize=(12, 5))

# The creates a matplotlib plot, so we can modify with standard matplotlib commands

plt.grid()  # add a grid to the plot
plt.figure(figsize=(9,7))

sns.jointplot(trace['petal_length'], trace['petal_width'], kind="hex", color="#4CB391")

plt.xlabel("petal_length")

plt.ylabel("petal_width");

plt.show()



plt.figure(figsize=(9,7))

sns.jointplot(trace['sepal_length'], trace['sepal_width'], kind="hex", color="#4CB391")

plt.xlabel("sepal_length")

plt.ylabel("sepal_width");

plt.show()