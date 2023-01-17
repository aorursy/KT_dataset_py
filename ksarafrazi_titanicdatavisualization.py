import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#checking the training data
train = pd.read_csv('../input/train.csv')
train.head()
#visualizing the missing data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Exploring the data
#Number of survivors of each sex
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
#Number of survivors of each class
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
#Age distribution
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
#Fair distribution
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
