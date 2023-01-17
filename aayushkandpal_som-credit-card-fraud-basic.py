# Unsupervised learning  Predicted Credit card fraud using SOM

!pip install minisom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing our dataset
dataset=pd.read_csv('../input/credit-card-train-data/Credit_Card_Applications.csv')
dataset
#Checking for missing values
dataset.count()
# From the below result you can see that there are no missing values ,good
# We are using SOM so this is a part of unsupervised learning( We are only using x in the model , in my next book I will 
#  use an Artificial neural network ( Supervised learning and try and show you the comparison)
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# Feature scaling our data using MinMaxSclaer
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1)) # MEntioning 0,1 is not really necessary but it is is a good habit
x=sc.fit_transform(x)
# We will be using a library called Minisom which is a simple and basic SOM, In Upcoming notebooks I will use boltzmann machines and 
# and autoencoders.
# To decide the parameters look at the documentation

from minisom import MiniSom
ms=MiniSom(15,15,learning_rate=0.5,input_len=15,sigma=1.0,random_seed=0)
 # Inititlaising with a random weight
ms.random_weights_init(x)
ms.train_random(data=x,num_iteration=100)



# Visualisng the results
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(ms.distance_map().T)
colorbar()
 
# The plot below basically shows yout the outliers (MEAN INTERNEURON DISTANCES (MID) ) 
# THE VALUES ARE BETWEEN 0,1 BECUASE THE VALUES HAVE BEEN SCALED AS YOU MAY HAVE SEEN BEFORE# Greater the MID meaning they are less likely to be a part of a cluster hence are potential frauds




mappings=ms.win_map(x)
frauds=np.concatenate((mappings[(4,3)],mappings[(5,3)],mappings[(4,4)],mappings[(5,4)],mappings[(13,11)]),axis=0)
frauds=sc.inverse_transform(frauds)
frauds=pd.DataFrame(frauds)
# I have only considered 4 outliers , you can consider more to see what happens


x=pd.DataFrame(x)
s1 = pd.merge(x, frauds, how='inner', on=[0]) # To find out the common users from both the lsit
    
frauds
# Above is the lsit of credit card users that our model has predicted to commit fraud
# It must be noted that as you increase your scope of outliers depending on the banks in this case you
# the number of fraud prediction will change
# Since the algorithm in minisom is based on k_means clustering 
# https://pypi.org/project/MiniSom/  check out the documnetation for more 
# Upvote if you liked the approach

