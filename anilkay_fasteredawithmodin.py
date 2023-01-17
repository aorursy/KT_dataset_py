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
data=pd.read_csv("/kaggle/input/flaredown-autoimmune-symptom-tracker/fd-export.csv")
data.head()
!pip install modin
import modin.pandas as pd2
data=pd2.read_csv("/kaggle/input/flaredown-autoimmune-symptom-tracker/fd-export.csv")
data.head()
counts=data["trackable_name"].value_counts().sort_values(ascending=False)
counts[0:20]
data["trackable_name"].unique()
set(data["trackable_name"])
counts[20:40]
data[data["trackable_name"]=="Shoulder pain"]["sex"].value_counts()
data[data["trackable_name"]=="Knee pain"]["sex"].value_counts()
data["sex"].value_counts()
data[data["trackable_name"]=="Übelkeit"]
germany=data[data["country"]=="DE"]
set(germany[germany["trackable_type"]=="Condition"]["trackable_name"])
germany["sex"].value_counts()
germany[germany["trackable_type"]=="Condition"]["trackable_name"].value_counts().sort_values(ascending=False)[0:30]
data[data["trackable_name"]=="Knee pain"].groupby(by="trackable_name").median()["age"]
condition_median_age=data[data["trackable_type"]=="Condition"].groupby(by="trackable_name").median()["age"]
condition_median_age[0:30]
condition_median_age.to_csv("conditionmedianage.csv")
data["country"].value_counts().sort_values(ascending=False)[0:10]
gb=data[data["country"]=="GB"]
gb[gb["trackable_type"]=="Condition"]["trackable_name"].value_counts().sort_values(ascending=False)[0:30]
canada=data[data["country"]=="CA"]
canada[canada["trackable_type"]=="Condition"]["trackable_name"].value_counts().sort_values(ascending=False)[0:30]
