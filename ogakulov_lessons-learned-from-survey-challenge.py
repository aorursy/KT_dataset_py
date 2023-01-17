# Import standard libraries
import pandas as pd 
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# READ IN THE DATA
# Survey data
surv = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv', low_memory=False)

# get the dataset id for the 2018 Kaggle Survey Kernel
datasets_v = pd.read_csv('../input/meta-kaggle/DatasetVersions.csv')
data_v = datasets_v.loc[datasets_v['Title'] == '2018 Kaggle ML & DS Survey Challenge', 'Id']

# get the kernels that used any version of the survey data
kernel_data = pd.read_csv('../input/meta-kaggle/KernelVersionDatasetSources.csv')
kernel_v = kernel_data.loc[kernel_data['SourceDatasetVersionId'].isin(data_v), 'KernelVersionId']

# get the kernel ids from kernel versions
kernels = pd.read_csv('../input/meta-kaggle/Kernels.csv', parse_dates=['MadePublicDate', 'MedalAwardDate', 'CreationDate'])
chall_kernels = kernels[kernels['CurrentKernelVersionId'].isin(kernel_v)]

# Set additional medal level, when no medal is awarded
pd.options.mode.chained_assignment = None
chall_kernels.loc[:, 'Medal_or_0'] = chall_kernels['Medal'].fillna(0)

# Count kernels by each date and medal level
t = pd.pivot_table(chall_kernels, values=['Id'], index='MadePublicDate', columns=['Medal_or_0'], aggfunc='count')
t = t.fillna(0)

# PLOT
# figure setting
plt.figure(figsize=(15,5))

# Draw bars
plt.bar(x=t.index, height=t['Id', 3], color='goldenrod', width=0.7, label='Bronze')
plt.bar(x=t.index, bottom=t['Id', 3], height=t['Id', 2], color='silver', width=0.7, label='Silver')
plt.bar(x=t.index, bottom=t['Id', 3] + t['Id', 2], height=t['Id', 1], color='gold', width=0.7, label='Gold')
plt.bar(x=t.index, bottom=t['Id', 3] + t['Id', 2] + t['Id', 1], height=t['Id', 0], color='beige', width=0.7, label='No Medal')

# Format x-axis
plt.xlabel('Date Kernel was Made Public')
ax = plt.gca()
plt.xticks(pd.date_range(start=min(t.index), periods=5, freq='W'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

# Format y-axis
plt.ylabel('Count of Kernels')

# Other formatting
plt.tight_layout()
plt.legend(loc='best', fontsize=12);

# Annotations:
plt.annotate('Challenge Launched', (mdates.date2num(t.index[7]), 10), (mdates.date2num(t.index[5]), 25), arrowprops=dict(arrowstyle='->'))
chall_end = mdates.date2num(t.index[31] + datetime.timedelta(hours=11.5)) # end of challenge 
plt.plot([chall_end, chall_end], [0, 26], color='black', linewidth=1, linestyle=':')
plt.text(chall_end - 3.85, 25, 'Challenge Deadline');
# Proportion of kernels that are awarded a medal
t['Id', 'Medal'] = t['Id', 1] + t['Id', 2] + t['Id', 3]
med_prop = t['Id', 'Medal'].cumsum() / (t['Id', 0].cumsum() + t['Id', 'Medal'].cumsum())

# PLOT
# Figure
plt.figure(figsize=(12,6))

# Draw lineplot
plt.plot(med_prop.index, med_prop)

# Format x-axis
plt.xlabel('Date Kernel was Made Public')
ax = plt.gca()
plt.xticks(pd.date_range(start=min(t.index), periods=5, freq='W'))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

# Format y-axis
plt.ylabel('Proportion of Kernels with Medals')

# Annotations:
plt.annotate('Challenge Launched', (mdates.date2num(t.index[7]), 0.75), (mdates.date2num(t.index[4]), 0.95), arrowprops=dict(arrowstyle='->'));
#chall_end = mdates.date2num(t.index[31] + datetime.timedelta(hours=11.5)) # end of challenge 
#plt.plot([chall_end, chall_end], [0, 0.95], color='black', linewidth=1, linestyle=':')
#plt.plot([mdates.date2num(t.index[0]), mdates.date2num(t.index[35])], [0.5, 0.5], color='black', linewidth=1, linestyle=':')
#plt.text(chall_end - 3.9, 25, 'Challenge Deadline');