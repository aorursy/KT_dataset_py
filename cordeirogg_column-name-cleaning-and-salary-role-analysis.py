# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#cm = sns.light_palette("navy", as_cmap=True)
#cm = sns.diverging_palette(240, 10, n=40, as_cmap=True)
cm = "Greens"

#sns.palplot(sns.diverging_palette(240, 10, n=40))
df = pd.read_csv('/kaggle/input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv')
df.columns = [x.replace("(", "")
        .replace(")", "")
        .replace("'", "")
        .replace(", ", "_")
        .lower()
        for x in df.columns]
source_salary_list = [x for x in df["p16_salary_range"].unique() if str(x) != 'nan']
salary_list = [x.split("R$")[1]
               .split("/mês")[0]
               .replace(".", "")
               .strip() for x in source_salary_list]
replacement_map = {i1: i2 for i1, i2 in zip(source_salary_list, salary_list)}
df['p16_salary_range_lower_range'] = df['p16_salary_range'].map(replacement_map)
df['p16_salary_range_lower_range'] = df['p16_salary_range_lower_range'].apply(lambda x: x if pd.isnull(x) else int(x))
df["d6_anonymized_role"].value_counts()
df["p16_salary_range_lower_range"].value_counts()
pd.pivot_table(data=df, index="d6_anonymized_role", columns="p16_salary_range_lower_range", values="p0_id", aggfunc=len, margins=True)
pivot = pd.crosstab(index=df["d6_anonymized_role"], columns=df["p16_salary_range_lower_range"], values=df["p0_id"], aggfunc=len, normalize='index', margins=True)
pivot.style.background_gradient(cmap=cm, axis=None)
if "Economista" in pivot.index:
    pivot = pivot.drop(axis=0, labels="Economista")
    
#pivot.style.background_gradient(cmap=cm, axis=None)
pivot.style.background_gradient(cmap=cm, axis=1)
stacked_pivot = pivot.stack().reset_index().rename(columns={0:'value'})
stacked_pivot
plt.figure(figsize=(50, 10))
sns.barplot(x=stacked_pivot.d6_anonymized_role, y=stacked_pivot.value, hue=stacked_pivot.p16_salary_range_lower_range)

