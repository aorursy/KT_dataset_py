import numpy as np 
import pandas as pd 
import seaborn as sb
from matplotlib import pyplot as plt
%matplotlib inline

sb.set(style="whitegrid")

import warnings
warnings.filterwarnings('ignore')

titanic_train = pd.read_csv("../input/train.csv") #importing file
titanic_test = pd.read_csv("../input/test.csv") #importing file
titanic_train.head()
titanic_test.head()
titanic_train.info()
titanic_train.describe()

%matplotlib inline
import matplotlib.pyplot as plt
titanic_train.hist(bins=50, figsize=(20,15))
plt.show
corr_matrix = titanic_train.corr()
corr_matrix["Survived"].sort_values(ascending=False)
