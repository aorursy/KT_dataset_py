import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import sqrt, arange
from scipy import stats
%matplotlib inline
student = pd.read_csv("../input/student.csv")
student.head()
sample_size = student.shape[0]
sample_size
student.shape

fig, axes = plt.subplots(1,2, figsize = (14,4))
sns.boxplot(x='Pstatus', y='G3', data=student, ax=axes[0])
sns.pointplot(x='Pstatus', y='G3', data=student, ax=axes[1]);
fig, axes = plt.subplots(1,2, figsize = (14,4))
sns.boxplot(x='failures', y='G3', data=student, ax=axes[0])
sns.pointplot(x='failures', y='G3', data=student, ax=axes[1]);
student.groupby('Pstatus')['G3'].var()
Pstatus_together = student['G3'][student['Pstatus']=='T']
Pstatus_apart = student['G3'][student['Pstatus']=='A']
stats.bartlett(Pstatus_together, Pstatus_apart)
statut_failure_table = pd.crosstab(student['failures'], student['Pstatus'])
statut_failure_table
fig, axes = plt.subplots(1,2, figsize = (14,4))
student['failures'].value_counts().plot(kind='bar', ax=axes[0], title='les échecs')
student['Pstatus'].value_counts().plot(kind='bar', ax=axes[1], title='statut  des parents');
100*(statut_failure_table.T/statut_failure_table.apply(sum, axis=1)).T
fig, axes = plt.subplots(1,2, figsize = (14,4))
statut_failure_table.plot(kind='bar', stacked=True, ax=axes[0]);
(100*(statut_failure_table.T/statut_failure_table.apply(sum, axis=1)).T).plot(kind='bar', stacked=True, ax=axes[1]);
fig, axes = plt.subplots(1,2, figsize = (14,4))
(statut_failure_table.T).plot(kind='bar', stacked=True, ax=axes[0]);
(100*(statut_failure_table/statut_failure_table.apply(sum, axis=0)).T).plot(kind='bar', stacked=True, ax=axes[1]);
chi_stat, p_value, dof, expected = stats.chi2_contingency(statut_failure_table)
chi_stat