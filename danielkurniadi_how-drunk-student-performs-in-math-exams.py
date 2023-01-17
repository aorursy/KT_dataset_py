# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualize
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#Core data manipulation
import numpy as np
import pandas as pd

#Matlplotlib & Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# others
import os
import warnings
warnings.filterwarnings('ignore')
#import data of student's math grade
path = "../input/student-mat.csv"
student_df = pd.read_csv(path)

student_df.head(7)
#statistical summary of floating point data
student_df.describe()
#statistical summary of categorical/ordinal data:
student_df.describe(include=['O'])
#Let's get the average grade of three exams:
student_df["G_avg"] = (student_df["G1"]+student_df["G2"]+student_df["G3"])/3

#Then combine workday and weekend alcohol consumption
student_df["Dalc"] = student_df["Dalc"]+student_df["Walc"]
# Pie plot & Bar plot

#setup pie plot
labels= student_df["Dalc"].unique()
sizes = student_df["Dalc"].value_counts()[labels]
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)  # explode 1st slice

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

#plotting pie plot
ax1.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True);
ax1.axis('equal')
ax1.set_title("Pie Plot")

#plotting bar plot
sns.countplot(x="Dalc", data=student_df, ax=ax2).set_title("Count/Bar plot");
#define bivariate bar plotting function
def bivariatte_barplot(df, x="Dalc", y="G_avg", hue=None, ax=None, color_set=1):
  pivtab_ser = df.groupby([x, hue])[y].mean().reset_index()
  #plotting
  sns.barplot(x=x, y=y, hue=hue,
               data=pivtab_ser, ax=ax, palette="Set%s"%(color_set+1)).set_ylabel("average grade")

# categorical features to plot
hues = ['sex', 'school', 'romantic']

#plotting
fig, axes = plt.subplots(1, len(hues), figsize=(17,6), sharey=True)
for idx, hue in enumerate(hues):
  bivariatte_barplot(student_df, hue=hue, ax=axes[idx], color_set=idx)
#calculate average students grade over the entire batch
#then mark if student performs below or above average 
avg_batch = student_df["G_avg"].mean()
student_df['is_abv_avg'] = student_df["G_avg"] > avg_batch
# swarm plot
fig, ax = plt.subplots(1,1,figsize=(10,7))
g = sns.swarmplot(x="Dalc", y="G_avg", 
                  hue="is_abv_avg", data=student_df, ax=ax, palette="Set1")
# make a new columns to mark
student_df["is_healthy"] = student_df["health"]>=3

student_df.head()
# define function for factor plotting
def multivariatte_factplot(x="Dalc", y="G_avg", hue="sex", col="is_healthy", df=pd.DataFrame(), cs=1):
  piv_tab = df.groupby([x, hue, col])[y].mean().reset_index()
  #plotting
  sns.factorplot(x=x, y=y, hue=hue, col=col, 
                     data=piv_tab, kind='bar', palette="Set%s"%cs, size=7);
  
# execute factor plotting
multivariatte_factplot(df=student_df);
# another example
multivariatte_factplot(hue="school", col="address", df=student_df, cs=2);
