import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
gender = {'F': 0, 'M': 1}
board = {'Central': 0, 'Others': 1}
hsc_s = {'Arts': 0, 'Commerce': 1, 'Science': 2}
degree_t = {'Comm&Mgmt': 0, 'Sci&Tech': 1, 'Others': 2}
workexp = {'No': 0, 'Yes': 1}
specialization = {'Mkt&HR': 0, 'Mkt&Fin': 1}
placed = {'Not Placed': 0, 'Placed': 1}

data.gender = data.gender.map(gender)
data.ssc_b = data.ssc_b.map(board)
data.hsc_b = data.hsc_b.map(board)
data.hsc_s = data.hsc_s.map(hsc_s)
data.degree_t = data.degree_t.map(degree_t)
data.wprkex = data.workex.map(workexp)
data.specialisation = data.specialisation.map(specialization)
data.status = data.status.map(placed)
data.head()
data.info()
data.drop('sl_no', inplace=True, axis=1)
plt.rcParams['figure.figsize'] = (12, 14)
data.hist();
plt.rcParams['figure.figsize'] = (10, 6)
sns.heatmap(data.corr());
plt.rcParams['figure.figsize'] = (10, 5)

plt.subplot(1, 2, 1)
sns.boxplot(x='ssc_p', data=data)

plt.subplot(1, 2, 2)
sns.boxplot(x='hsc_p', data=data);
data.corrwith(data['status'])
plot_feat = ['ssc_b', 'hsc_b', 'degree_t', 'specialisation']

plt.rcParams['figure.figsize'] = (10, 8)

for i, x in enumerate(plot_feat):
    plt.subplot(2, 2, i+1)
    sns.countplot(x='status', hue=x, data=data)
plt.show()

