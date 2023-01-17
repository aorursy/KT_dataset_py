import numpy as np 
import pandas as pd

import seaborn as sns

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
import itertools



dataset = pd.read_csv('../input/land_train.csv') 
# Preprocessing
# Let Separate Features and Target for machine Learning
# Step1 


features = list(dataset.columns[:12])            
target = dataset.columns[12]                      

print('Features:',features)
print('Target:',target)

# store feature matrix in "X"
X = dataset.iloc[:,:12]                          # slicing: all rows and 1 to 12 cols

# store response vector in "Y"
Y = dataset.iloc[:,12]                            # slicing: all rows and 13th col



X.isnull().describe()
dataset.groupby('target').size()
# Correlation 
total_features = X.shape[1]
#sets the number of features considered
size = total_features

#create a dataframe with only 'size' features
data=dataset.iloc[:,:size] 

#get the names of all the columns
cols=data.columns 

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()

# Set the threshold to select only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

print("pair of features in order of decreasing correlation")
#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))



# Scatter plot of only the highly correlated pairs
total_plots = 3

for v,i,j in s_corr_list[:total_plots]:
    sns.pairplot(dataset, hue="target", height=9, x_vars=cols[i],y_vars=cols[j],markers=['P','*','|','v'] )
    plt.show()
dataset.describe()