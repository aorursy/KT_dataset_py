import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Drop unused columns
train.drop(["ID", "target"], inplace=True, axis=1)
test.drop(["ID"], inplace=True, axis=1)
# Function to plot nan of a dataset
def plot_nan(data):
    # Get NaN and subsample for plotting
    s = data.isnull().sample(20000)
    
    # Count NaN (just for sorting proposes)
    s['num_nan'] = s.sum(axis=1)
    
    # Create a sorted version and drop num_nan
    ss = s.sort_values(['num_nan'])
    ss.drop(['num_nan'], inplace=True, axis=1)
    
    # Plot values (aspect correction: to http://stackoverflow.com/questions/13384653/imshow-extent-and-aspect)
    return plt.matshow(ss, extent=[0,100,0,1], aspect=100)
%matplotlib inline
plt.figure()
plot_nan(train)
plt.title("Train")

plt.figure()
plot_nan(test)
plt.title("Test")

plt.show()


