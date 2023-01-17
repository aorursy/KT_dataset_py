import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 



parts = pd.read_csv("../input/parts.csv")

part_cat = pd.read_csv("../input/part_categories.csv")
merged_parts = pd.merge(parts,part_cat, left_on= parts["part_cat_id"], right_on=part_cat["id"])
merged_parts.dtypes
sns.countplot(x= merged_parts["name_y"])