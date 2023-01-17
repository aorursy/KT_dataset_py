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
# Read Data
import numpy as np     # Linear Algebra (calculate the mean and standard deviation)
import pandas as pd    # manipulate data, data processing, load csv file I/O (e.g. pd.read_csv)

# Visualization
# Seaborn
import seaborn as sns
# matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
# Plotly
import plotly
import plotly.express as px
import plotly.graph_objs as go

# style
plt.style.use("fivethirtyeight")
sns.set_style("darkgrid")

# ML model building; Pre Processing & Evaluation
from sklearn.model_selection import train_test_split    # split  data into training and testing sets
from sklearn.ensemble import RandomForestClassifier     # this will make a Random Forest classificaiton
from sklearn.metrics import confusion_matrix            # this creates a confusion matrix
df = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head()
df.shape
df.info()
df.dtypes
df.count()
df.isnull().sum()
# A count plot for single categorical variables
sns.countplot(x='room_type',  data=df)
# A count plot for two categorical variables
sns.countplot(x='room_type', hue="neighbourhood_group", data=df)
def groupPrice(price):
    if price < 100:
        return "Low Cost"
    elif price >=100 and price < 200:
        return "Middle Cost"
    else:
        return "High Cost"
      
price_group = df['price'].apply(groupPrice)
df.insert(10, "price_group", price_group, True)
df.head(5)
sns.catplot(x="room_type", hue="neighbourhood_group", col="price_group", data=df, kind="count", height=5, aspect=1)
plt.show()