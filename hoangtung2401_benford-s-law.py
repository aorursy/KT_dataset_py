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
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(12,9)})
df = pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")
df.head()
digit_counts = df["views"].dropna().astype(str).str[0].value_counts(normalize=True)
digit_counts
my_variable_name = "Variable Description Here"
benford_df = pd.DataFrame({"Leading Digit": ['1', '2', '3', '4', '5', '6', '7', '8', '9'], "Proportion": [.301, .176, .125, .097, .079, .067, .058, .051, .046], "Distribution": "Benford"})
my_variable_df = pd.DataFrame({"Leading Digit": digit_counts.index, "Proportion": digit_counts.values, "Distribution": my_variable_name})
pd.concat([benford_df, my_variable_df]).pivot("Leading Digit", "Distribution", "Proportion").plot(kind="bar", width=.65)
plt.rcParams["font.weight"] = "bold"
plt.title("Comparing %s to Benford's Law"%my_variable_name, fontsize=24, fontweight="bold")
plt.xlabel("Leading Digit", fontsize=18, fontweight="bold")
plt.ylabel("Proportion", fontsize=18, fontweight="bold")
plt.xticks(fontsize=18, fontweight="bold", rotation=0)
plt.yticks(fontsize=18, fontweight="bold")
plt.legend(fontsize=18)
