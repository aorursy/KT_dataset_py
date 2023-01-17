# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import warnings
warnings.filterwarnings('ignore')
os.listdir('../input/')
sets = pd.read_csv('../input/sets.csv')
inventory_parts = pd.read_csv('../input/inventory_parts.csv')
part_categories = pd.read_csv('../input/part_categories.csv')
parts = pd.read_csv('../input/parts.csv')
themes = pd.read_csv('../input/themes.csv')
colors = pd.read_csv('../input/colors.csv')
inventories = pd.read_csv('../input/inventories.csv')
inventory_sets = pd.read_csv('../input/inventory_sets.csv')
## print('Shape of sets: (%s,%s)' % sets.shape)
print('Shape of inventory_parts: (%s,%s)' % inventory_parts.shape)
print('Shape of part_categories: (%s,%s)' % part_categories.shape)
print('Shape of parts: (%s,%s)' % parts.shape)
print('Shape of themes: (%s,%s)' % themes.shape)
print('Shape of colors: (%s,%s)' % colors.shape)
print('Shape of inventories: (%s,%s)' % inventories.shape)
print('Shape of inventory_sets: (%s,%s)' % inventory_sets.shape)
print('Info of sets:')
sets.info()
sets.describe()
print('Info of inventory_parts:')
inventory_parts.info()
inventory_parts.describe()
print('Info of part_categories:')
part_categories.info()
part_categories.describe()
print('Info of parts:')
parts.info()
parts.describe()
print('Info of themes:')
themes.info()
themes.describe()
print('Info of colors:')
colors.info()
colors.describe()
print('Info of inventories:')
inventories.info()
inventories.describe()
print('Info of inventory_sets:')
inventory_sets.info()
inventory_sets.describe()
