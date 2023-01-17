# importing data
import pandas as pd

data=pd.read_csv('../input/train.csv')
data.head()
# Lets covert sex to numeric data by 'male'=1 and 'female'=0 
def convert(row):
    if row=='male':
        return 1
    else:
        return 0
    
data['Sex_']=data['Sex'].apply(convert)
data.head()
# feature matrix ( I think these 3 features are very important)
features=data[['Pclass','Sex_','Age']]
features.head()
print(len(features))
# response vector
response=data['Survived']
response.head()
print(len(response))
# lets visualize the data
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize']=(12,10)
fig,ax=plt.subplots(3,3)
for i in range(3):
    for j in range(3):
        if i!=j:
            for t,s,c in zip([1,0],'ox','br'):
                ax[i,j].scatter(features.iloc[:,i][response==t],features.iloc[:,j][response==t],marker=s,c=c,label=t)
                ax[i,j].set_xlabel(features.columns[i])
                ax[i,j].set_ylabel(features.columns[j])
                
        elif i==j:
            features.iloc[:,i][response==1].plot(kind='hist',ax=ax[i,j])
            features.iloc[:,i][~(response==1)].plot(kind='hist',ax=ax[i,j])
            ax[i,j].set_xlabel(features.columns[i])
    
import numpy as np
features['Pclass'].plot.hist(bins=5)