import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style="white", color_codes=True)
irisdf = pd.read_csv('../input/Iris.csv',header=0,sep=',')

irisdf.head()
## Plot 

sns.distplot(irisdf.SepalLengthCm,kde=False,rug=True)
sns.boxplot(data=irisdf.SepalLengthCm)
sns.pairplot(irisdf[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']],hue='Species')
sns.lmplot(x = 'SepalLengthCm',y = 'PetalLengthCm',data =irisdf)
sns.lmplot(x = 'SepalLengthCm',y = 'SepalWidthCm',data =irisdf)