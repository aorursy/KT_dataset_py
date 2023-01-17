import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(25,10))



sns.set(style="whitegrid", palette="muted")



dataset = pd.read_csv('../input/movehubcostofliving.csv')



cost = pd.melt(dataset, "City", var_name="Attributes")



swarm_plot = sns.swarmplot(x="Attributes", y="value", hue="City", data=cost)

box = swarm_plot.get_position()

swarm_plot.set_position([box.x0 - 0.09, box.y0, box.width * 0.8, box.height])

plt.legend(bbox_to_anchor=(1.05, 1.08), loc=2, borderaxespad=0., ncol=5)



plt.show()
plt.figure(figsize=(20,10))



dataset = pd.read_csv('../input/movehubqualityoflife.csv')



cost = pd.melt(dataset, "City", var_name="Attributes")



swarm_plot = sns.swarmplot(x="Attributes", y="value", hue="City", data=cost)

box = swarm_plot.get_position()

swarm_plot.set_position([box.x0 - 0.09, box.y0, box.width * 0.8, box.height])

plt.legend(bbox_to_anchor=(1.05, 1.08), loc=2, borderaxespad=0., ncol=5)



plt.show()