import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



from sklearn.svm import SVC
raw_data = pd.read_csv('../input/horse-colic/horse.csv')

raw_data.head()
raw_data.shape
raw_data.isnull().sum()
sns.countplot(data=raw_data, x='outcome');
print(raw_data.outcome.value_counts())



sns.countplot(data=raw_data, x='outcome', hue='surgery');

plt.show()



sns.countplot(data=raw_data, x='outcome', hue='pain');

plt.show()



g = sns.catplot(data=raw_data, x='outcome', col='surgery', hue='pain', kind='count');

g.fig.suptitle('Horse deaths by Pain & Surgery');

plt.subplots_adjust(top=0.9)
g = sns.catplot(data=raw_data, x='outcome', hue='pain', col='age', kind='count');

g.fig.suptitle('Horse deaths by Pain & Age');

plt.subplots_adjust(top=0.9)
g = sns.FacetGrid(data=raw_data, col='outcome', margin_titles=True, height=6)

g.map(plt.hist, 'pulse')

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Outcome by Pulse');
g = sns.catplot(data=raw_data, x='peripheral_pulse', col='outcome', kind='count');

g.fig.suptitle('Outcome by Peripheral Pulse');

plt.subplots_adjust(top=0.9)
reduced_absent_pulse = raw_data[raw_data.outcome.isin(('died','euthanized')) & raw_data.peripheral_pulse.isin(('reduced','absent'))]



g = sns.catplot(data=reduced_absent_pulse, x='capillary_refill_time', col='outcome', kind='count');

g.fig.suptitle('Outcome by Capillary refill time');

plt.subplots_adjust(top=0.9)
g = sns.catplot(data=raw_data, x='abdominal_distention', col='outcome', hue='surgery', kind='count');

g.fig.suptitle('Outcome by Abdominal Distention & Surgery');

plt.subplots_adjust(top=0.9)
severe_died = raw_data[raw_data.outcome.isin(('died','euthanized')) & (raw_data.abdominal_distention=='severe') & (raw_data.surgery=='no')]



g = sns.countplot(data=severe_died, x='peristalsis').set_title('Horses with severe abdominal distention, did not undergo surgery and died');