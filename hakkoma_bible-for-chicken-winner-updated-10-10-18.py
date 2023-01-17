#Import library and data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#check train data set
train.head(10)
#check test data set
test.head(10)
#characteristics of variables
train.info()
#plot walkDistance
plt.figure(figsize=(15,10))
plt.title('Walking Distance Distribution')
sns.distplot(train['walkDistance'], kde = False)
plt.show()
#plot heals
plt.figure(figsize=(15,10))
plt.title('Heals Distribution')
sns.distplot(train['heals'], kde = False)
plt.show()
#plot Damage Dealt
plt.figure(figsize=(15,10))
plt.title('Damage Dealt Distribution')
sns.distplot(train['damageDealt'], kde = False)
plt.show()
#plot DBNOs
plt.figure(figsize=(15,10))
plt.title('Number of Kills distribution')
sns.distplot(train['DBNOs'], kde = False)
plt.show()
