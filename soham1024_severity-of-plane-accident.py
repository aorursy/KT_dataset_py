# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use(['ggplot'])

# read training file

df_train=pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

df_train.head(5)
#if any nan

df_train.isnull().sum()
# Replace All strings into integres

replacements = {

  1: 'Minor_Damage_And_Injuries',

  2: 'Significant_Damage_And_Fatalities',

  3: 'Significant_Damage_And_Serious_Injuries',

  4: 'Highly_Fatal_And_Damaging'

}

#df_train.Severity=df_train['Severity'].replace(replacements, inplace=True)
from scipy.stats import norm

sns.distplot(df_train.Safety_Score, fit=norm, kde=False)
g = sns.boxplot(x="Safety_Score", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Safety_Score", y="Severity", data=df_train, color=".2")
sns.distplot(df_train.Days_Since_Inspection,fit=norm, kde=False)
g = sns.boxplot(x="Days_Since_Inspection", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Days_Since_Inspection", y="Severity", data=df_train, color=".2")
sns.distplot(df_train.Total_Safety_Complaints, fit=norm, kde=False)
g = sns.boxplot(x="Total_Safety_Complaints", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Total_Safety_Complaints", y="Severity", data=df_train, color=".2")
sns.distplot(df_train.Control_Metric, fit=norm, kde=False)
g = sns.boxplot(x="Control_Metric", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Control_Metric", y="Severity", data=df_train, color=".2")
sns.distplot(df_train.Turbulence_In_gforces, fit=norm, kde=False)
g = sns.boxplot(x="Turbulence_In_gforces", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Turbulence_In_gforces", y="Severity", data=df_train, color=".2")
sns.distplot(df_train.Cabin_Temperature, fit=norm, kde=False)
g = sns.boxplot(x="Cabin_Temperature", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Cabin_Temperature", y="Severity", data=df_train, color=".2")
sns.distplot(df_train.Accident_Type_Code, fit=norm, kde=False)
g = sns.boxplot(x="Accident_Type_Code", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Accident_Type_Code", y="Severity", data=df_train, color=".2")
sns.distplot(df_train.Max_Elevation, fit=norm, kde=False)
g = sns.boxplot(x="Max_Elevation", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Max_Elevation", y="Severity", data=df_train, color=".2")
sns.distplot(df_train.Violations, fit=norm, kde=False)
g = sns.boxplot(x="Violations", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Violations", y="Severity", data=df_train, color=".2")
sns.distplot(df_train.Adverse_Weather_Metric, fit=norm, kde=False)
g = sns.boxplot(x="Adverse_Weather_Metric", y="Severity", data=df_train, whis=np.inf)

g = sns.swarmplot(x="Adverse_Weather_Metric", y="Severity", data=df_train, color=".2")
# dropping Accident Id

df_train.drop('Accident_ID',1,inplace=True)
g = sns.PairGrid(df_train)

g = g.map_diag(plt.hist, edgecolor="b")

g = g.map_offdiag(plt.scatter, edgecolor="w", s=40)
g = sns.PairGrid(df_train)

g = g.map_upper(plt.scatter)

g = g.map_lower(sns.kdeplot, cmap="Blues_d")

g = g.map_diag(sns.kdeplot, lw=3, legend=False)