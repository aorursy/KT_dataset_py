# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# loading data 
lifts = pd.read_csv('../input/openpowerlifting.csv')
# i'll take a quick look
lifts.head()
# Let's limit this to Men, for now
liftsM = lifts.query("Sex == 'M'")
# Let's apply a simple regression to see how age effects lifts
tmp = abs(liftsM[['Age', 'TotalKg']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['TotalKg'])+20)
sns.regplot(tmp['Age'], (tmp['TotalKg']), order=4, scatter_kws={"alpha": 0.05}, line_kws={"color": 'black'})
plt.title('Total KG lifted (10K samples)');
# Looking at bench press
tmp = abs(liftsM[['Age', 'BestBenchKg']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['BestBenchKg'])+20)
sns.regplot(tmp['Age'], (tmp['BestBenchKg']), order=4, scatter_kws={'color': "C2", "alpha": 0.05}, line_kws={"color": 'black'})
plt.title("Bench Press", size=16)
plt.ylabel("Measured in kg");
tmp = abs(liftsM[['Age', 'BestDeadliftKg']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['BestDeadliftKg'])+20)
sns.regplot(tmp['Age'], (tmp['BestDeadliftKg']), order=4, scatter_kws={"alpha": 0.05}, line_kws={"color": 'black'})
plt.title("Deadlifts", size=16)
plt.ylabel("Measured in kg");
tmp = abs(liftsM[['Age', 'BestSquatKg']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['BestSquatKg'])+20)
sns.regplot(tmp['Age'], (tmp['BestSquatKg']), order=4, scatter_kws={'color': "C1","alpha": 0.05}, line_kws={"color": 'black'})
plt.title("Squats", size=16)
plt.ylabel("Measured in kg");
tmp = abs(liftsM[['Age', 'Wilks']].dropna().sample(10000, random_state=42))
plt.figure(figsize=(12,6))
plt.ylim(0,max(tmp['Wilks'])+20)
sns.regplot(tmp['Age'], (tmp['Wilks']), order=4, scatter_kws={"alpha": 0.05}, line_kws={"color": 'black'})
plt.title("Wilks", size=16)
plt.ylabel("Wilks Score ");
plt.figure(figsize=(12,6)), plt.xlim(18,80), plt.title('Regression Line for Powerlifts by Age')
tmp = abs(liftsM[['Age', 'BestDeadliftKg']].dropna().sample(5000))
sns.regplot(tmp['Age'], tmp['BestDeadliftKg'], order=4, scatter=False, label='Deadlift')
tmp = abs(liftsM[['Age', 'BestSquatKg']].dropna().sample(5000))
sns.regplot(tmp['Age'], tmp['BestSquatKg'], order=4, scatter=False, label='Squat')
tmp = abs(liftsM[['Age', 'BestBenchKg']].dropna().sample(5000))
sns.regplot(tmp['Age'], tmp['BestBenchKg'], order=4, scatter=False, label='Bench')
plt.ylabel("Best fit regression for lift, in kg"), plt.legend()
# tmp = abs(liftsM[['Age', 'BestDeadliftKg']].dropna().sample(5000))
# sns.regplot(tmp['Age'], tmp['BestDeadliftKg'], order=4, scatter_kws={ "alpha": 0.1})
