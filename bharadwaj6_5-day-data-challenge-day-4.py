import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
results = pd.read_csv('../input/survey_results_public.csv')

results.info()
results.describe()
results.head()
list(results.columns)
fig, ax = plt.subplots(figsize=(12, 10))

bar_graph = sns.countplot(data=results, x='Country', order=results['Country'].value_counts().index)

plt.xlim(0, 10)

bar_graph.set_title('top 10 countries')

plt.show()