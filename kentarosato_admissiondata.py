# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

plt.style.use("ggplot") 
plt.rcParams['figure.figsize'] = [8,8]  #Want some bigger pictures



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df.head(5)
df.columns
sns.heatmap(df.corr())
sns.stripplot(df["SOP"],df["Chance of Admit "],hue=df['Research'])
sns.stripplot(df["LOR "],df["Chance of Admit "],hue=df['Research'])
sns.stripplot(df["University Rating"],df["Chance of Admit "],hue=df['Research'])
sns.stripplot(df["CGPA"],df["Chance of Admit "],hue=df['Research'])
df['CGPA'].describe()
def CGPA_cate(x):
    if x >= 6 and x < 8:
        return '6-8'
    elif x >= 8 and x < 9:
        return '8-9'
    else:
        return '9-'
df['CGPA_cat'] = df['CGPA'].apply(CGPA_cate)
sns.stripplot(df["CGPA_cat"],df["Chance of Admit "],hue=df['Research'],order=['6-8','8-9','9-'])
