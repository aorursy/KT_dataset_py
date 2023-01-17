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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
td = pd.concat([train_data, test_data], ignore_index=True, sort = False) # Concantinate train and test data
import seaborn as sns # Import Seaborn

import matplotlib.pyplot as plt

td.isnull().sum() # Check the number of missing values in the data set

sns.heatmap(td.isnull(), cbar = False).set_title("Missing values heatmap") # Building a heatmap of missing values

# Setting up visualisations
sns.set_style(style='white') 
sns.set(rc={
    'figure.figsize':(10,5), 
    'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '.9',
    'axes.linewidth': 1,
    'grid.linestyle': u'-'},font_scale=1.5)
custom_colors = ["#3498db", "#95a5a6","#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)
td.nunique() # Study number of unique values
(train_data.Survived.value_counts(normalize=True) * 100).plot.barh().set_title("Training Data - Percentage of people survived and Deceased")
fig_pclass = train_data.Pclass.value_counts().plot.pie().legend(labels=["Class 3","Class 1","Class 2"], loc='lower right', bbox_to_anchor=(3.25, 0.5)).set_title("Training Data - People travelling in different classes")
pclass_1_survivor_distribution = round((train_data[train_data.Pclass == 1].Survived == 1).value_counts()[1]/len(train_data[train_data.Pclass == 1]) * 100, 2) 
pclass_2_survivor_distribution = round((train_data[train_data.Pclass == 2].Survived == 1).value_counts()[1]/len(train_data[train_data.Pclass == 2]) * 100, 2)
pclass_3_survivor_distribution = round((train_data[train_data.Pclass == 3].Survived == 1).value_counts()[1]/len(train_data[train_data.Pclass == 3]) * 100, 2)
pclass_perc_df = pd.DataFrame(
    { "Percentage Survived":{"Class 1": pclass_1_survivor_distribution,"Class 2": pclass_2_survivor_distribution, "Class 3": pclass_3_survivor_distribution},  
     "Percentage Not Survived":{"Class 1": 100-pclass_1_survivor_distribution,"Class 2": 100-pclass_2_survivor_distribution, "Class 3": 100-pclass_3_survivor_distribution}})
pclass_perc_df.plot.bar().set_title("Training Data - Percentage of people survived on the basis of class")
for x in [1,2,3]:    ## for 3 classes
    train_data.Age[train_data.Pclass == x].plot(kind="kde")
plt.title("Age density in classes")
plt.legend(("1st","2nd","3rd"))
for x in ["male","female"]:
    td.Pclass[td.Sex == x].plot(kind="kde")
plt.title("Training Data - Gender density in classes")
plt.legend(("Male","Female"))
pclass_perc_df
