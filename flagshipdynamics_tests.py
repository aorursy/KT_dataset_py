# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
import pandas as pd
df=pd.read_csv('../input/Concrete_Data_Yeh.csv', sep=',',header=None)
df.values
from IPython.display import display

%matplotlib inline
try:
    data = pd.read_csv("../input/Concrete_Data_Yeh.csv")
    #data.drop(['age'], axis = 1, inplace = True)
    print("Loaded Dataset.")
except:
    print("Dataset could not be loaded. Is the dataset missing?")
    
data = data.sample(n=1000)

# Display a description of the dataset
display(data.describe())
# TODO: Select three indices of your choice you wish to sample from the dataset
# indices = [ 10,20,30]

# Create a DataFrame of the chosen samples
samples = data.sample(n=5)

print("Chosen samples of wholesale customers dataset:")
display(samples)
import seaborn as sns

percentiles_data = 100*data.rank(pct=True)
percentiles_samples = percentiles_data.iloc[indices]
sns.heatmap(percentiles_samples, annot=True)
features = data.drop(['age'],1)
wildcard = data['age']

# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, wildcard, test_size=0.25, random_state=23)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=2, random_state=23)
regressor.fit(X_train, y_train)
results = regressor.predict(X_test)
display(results)
# TODO: Report the score of the prediction using the testing set
from sklearn.metrics import r2_score
score = regressor.score(X_test, y_test)
print("score")
display(score)
# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (18,12), diagonal = 'kde');