import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
!ls ../input/
dataset = pd.read_csv('../input/environmental-remediation-sites.csv')

dataset.info()
dataset.columns
program_types = dataset.groupby('Program Type')['Program Number'].count()

plt.figure(figsize = (12, 8))

plt.title("Various Program Types in New York")

plt.ylabel("Count")

plt.xlabel("Program Type")

sns.barplot(x = program_types.index, y = program_types.values)
site_classes = dataset.groupby('Site Class')['Program Number'].count()

plt.figure(figsize = (12, 8))

plt.title("Status/Class of each remediation site in New York")

plt.ylabel("Count")

plt.xlabel("Site Class/Status")

sns.barplot(x = site_classes.index, y = site_classes.values)
completed_sites = dataset[dataset['Site Class'] == 'C ']

completed_sites['Project Completion Date'] = pd.to_datetime(completed_sites['Project Completion Date']).dt.strftime("%Y")

completed_sites = completed_sites.groupby('Project Completion Date')['Program Number'].count()

plt.figure(figsize = (12, 8))

plt.title("Completion dates for various remediation sites in New York")

plt.ylabel("Count")

plt.xlabel("Date")

plt.xticks(rotation = 90)

sns.lineplot(x = completed_sites.index, y = completed_sites.values)
len(dataset['Contaminants'].dropna().unique())
contaminants = dataset.groupby('Contaminants')['Program Number'].count().sort_values(ascending = False)

contaminants.head(10)
contaminants.tail(10)
control_types = dataset.groupby('Control Type')['Program Number'].count()

labels = [index + ": {:.2f}%".format(count/dataset['Control Type'].count()*100) for index, count in zip(control_types.index, control_types.values)]

plt.figure(figsize = (12, 8))

plt.title("Control types for various remediation sites in New York")

plt.pie(x = control_types.values, labels = labels)