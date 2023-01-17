import pandas as pd
import numpy as np
data=pd.read_csv('../input/mushrooms.csv')
data
data.head()
data['class'].nunique()
data.shape        # display the dimensionality
data.count()
data['class'].value_counts()
data['class'].value_counts().plot.bar()
data.describe()
pd.plotting.parallel_coordinates(data, "class")
import seaborn as sns
sns.pairplot(data, hue="class")
pd.plotting.andrews_curves(data, "class")