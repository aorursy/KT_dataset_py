import pandas as pd;

import seaborn as sns;

import matplotlib.pyplot as plt;

sns.set(style="whitegrid");

%matplotlib inline
dtf = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1");
dtf.head()
dtf.columns
# Male Match Rate By Age

m = dtf[dtf.gender == 1];

sns.barplot(x="match", y="age", data=m, orient='h')

sns.despine(left=True, bottom=True)
# Female Match Rate By Age

f = dtf[dtf.gender == 0];

sns.barplot(x="match", y="age", data=f, orient='h')

sns.despine(left=True, bottom=True)
# Male Match Rate By Race

sns.barplot(x="match", y="race", data=m, orient='h')

sns.despine(left=True, bottom=True)
# Female Match Rate By Race

sns.barplot(x="match", y="race", data=m, orient='h')

sns.despine(left=True, bottom=True)
# Match Rate By Goal

sns.barplot(x="match", y="goal", data=dtf, orient='h')

sns.despine(left=True, bottom=True)