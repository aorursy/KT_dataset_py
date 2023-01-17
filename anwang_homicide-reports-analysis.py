# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plots
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import math
# Set plotting style
sns.set_style('whitegrid')

# Rounding the integer to the next hundredth value plus an offset of 100
def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100 
df_data=pd.read_csv('../input/database.csv',low_memory=False)
df_data.columns
ax = sns.countplot(x=df_data["Agency Type"])
plt.xticks(rotation=30)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Agency Type'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(16,10)) 
ax = sns.countplot(x=df_data["State"])
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['State'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(16,10)) 
ax = sns.countplot(x=df_data["Month"], color="blue")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Month'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(8,4)) 
ax = sns.countplot(x=df_data["Crime Type"])
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Crime Type'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(8,4)) 
ax = sns.countplot(x=df_data["Victim Sex"])
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Victim Sex'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.distplot(df_data["Victim Age"][df_data["Victim Age"]<100])

#df_data["Victim Age"][df_data["Victim Age"]>100].value_counts()


fig, ax = plt.subplots(figsize=(20,10)) 
ax = sns.countplot(x=df_data["Year"], color="Pink")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Year'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.countplot(x=df_data["Victim Race"], color="Pink")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Victim Race'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
ax = sns.countplot(x=df_data["Victim Ethnicity"],color="pink")
plt.xticks(rotation=30)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Victim Ethnicity'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.countplot(x=df_data["Perpetrator Sex"], color="Pink")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Perpetrator Sex'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(30,10)) 
ax = sns.countplot(x=df_data["Perpetrator Age"], color="Pink")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Perpetrator Age'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.countplot(x=df_data["Perpetrator Race"], color="Pink")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Perpetrator Race'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.countplot(x=df_data["Relationship"], color="Pink")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Relationship'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.countplot(x=df_data["Weapon"], color="Pink")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Weapon'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.countplot(x=df_data["Victim Count"], color="Pink")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Victim Count'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
fig, ax = plt.subplots(figsize=(10,10)) 
ax = sns.countplot(x=df_data["Crime Solved"], color="Pink")
plt.xticks(rotation=75)

# Get current axis on current figure
axi = plt.gca()

# ylim max value to be set
y_max = df_data['Crime Solved'].value_counts().max() 
axi.set_ylim([0, roundup(y_max)])

# Iterate through the list of axes' patches
for p in axi.patches:
    axi.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
#scatterplot
sns.set()
cols = ['Agency Type', 
       'State', 'Year', 'Month', 'Crime Type', 'Crime Solved',
       'Victim Sex', 'Victim Age', 'Victim Race', 'Victim Ethnicity',
       'Perpetrator Sex', 'Perpetrator Age', 'Perpetrator Race',
       'Perpetrator Ethnicity', 'Relationship', 'Weapon', 'Victim Count',
       'Perpetrator Count', 'Record Source']
sns.pairplot(df_data[cols], size = 2.5)
plt.show();