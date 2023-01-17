import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset_colors=pd.read_csv("../input/colors.csv")

dataset_colors.isnull().sum()
dataset_colors.head()
dataset_colors.shape
dataset_colors["is_trans"].value_counts()
dataset_parts=pd.read_csv("../input/parts.csv")

dataset_parts.isnull().sum()
dataset_parts.head
dataset_parts.shape
dataset_parts.describe()
dataset_inventory_parts=pd.read_csv("../input/inventory_parts.csv")

dataset_inventory_parts.isnull().sum()
dataset_sets=pd.read_csv("../input/sets.csv")

dataset_sets.isnull().sum()
dataset_themes=pd.read_csv("../input/themes.csv")

dataset_themes.isnull().sum()
dataset_inventories=pd.read_csv("../input/inventories.csv")

dataset_inventories.isnull().sum()
dataset_inventory_sets=pd.read_csv("../input/inventory_sets.csv")

dataset_inventory_sets.isnull().sum()
dataset_part_categories=pd.read_csv("../input/part_categories.csv")

dataset_part_categories.isnull().sum()