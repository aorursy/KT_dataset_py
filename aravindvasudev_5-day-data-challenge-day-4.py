import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# read the dataset

data = pd.read_csv('../input/20151001hundehalter.csv')



data.head()
# plot bar chart for `GESCHLECHT` (Gender)

plt.figure(figsize=(14, 7))

sns.countplot(x='GESCHLECHT', data=data).set_title('Bar plot for Gender')
# plot bar chart for `ALTER` (Age)

plt.figure(figsize=(14, 7))

sns.countplot(x='ALTER', data=data).set_title('Age')