import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy .stats import norm 

df=pd.read_csv('/kaggle/input/data_1.csv')

df
%matplotlib inline

plt.hist(df.Height,bins=20,rwidth=0.8)

plt.xlabel('height(inches)')

plt.ylabel('count')

plt.show()
from scipy.stats import norm

import numpy as np

from matplotlib import pyplot as plt

plt.hist(df.Height,bins=20,rwidth=0.8)

plt.xlabel('height(inches)')

plt.ylabel('count')

rng=np.arange(df.Height.min(),df.Height.max(),0.1)

plt.plot(rng,norm.pdf(rng,df.Height.mean(),df.Height.std()))

plt.show()

#matplotlib. rcParams['figure.figsize']=(10,6)



#i dont know why the bell curve isnt plotting in Kaggle(was plotting in JN),Trouble shoot and let me know
#max height

df.Height.max()

#mean height

df.Height.mean()

#std. deviation of height

df.Height.std()
#so my upper limit will be my mean value plus 3 sigma

upper_limit=df.Height.mean()+3*df.Height.std()

upper_limit
#my lowar limit will be my mean - 3 sigma

lowar_limit=df.Height.mean()-3*df.Height.std()

lowar_limit
#now that my outliers are defined, i want to see what are my outliers

df[(df.Height>upper_limit)|(df.Height<lowar_limit)]
#now we will visualise the good data

new_data=df[(df.Height<upper_limit)& (df.Height>lowar_limit)]

new_data
#shape of our new data

new_data.shape
#shape of our outliers

df.shape[0]-new_data.shape[0]
#now we will calculate the z score of all our datapoints and display in a dataframe

df['zscore']=(df.Height-df.Height.mean())/df.Height.std()

df
#figuring out all the datapoints more than 3

df[df['zscore']>3]
#figuring out all the datapoints less than 3

df[df['zscore']<-3]
#displaying the outliers with respect to the zscores

df[(df.zscore<-3)|(df.zscore>3)]
new_data_1=df[(df.zscore>-3)& (df.zscore<3)]

new_data_1
df

df=df.drop(['zscore'],axis=1)
df
df.describe()
Q1=df.Height.quantile(0.25)

Q3=df.Height.quantile(0.75)

Q1,Q3

#WHICH MEANS THAT Q1 CORRESPONDS TO 25% OF ALL THE HEIGHT DISTRIBUTION IS BELOW 63.50

#Q3 CORRESPONDS TO 75% OF ALL THE HEIGHT DISTRIBUTION IS BELOW 69.174
#NOW WE WILL CALCULATE THE IQR

IQR=Q3-Q1

IQR
#NOW WE WILL DEFINE THE UPPER LIMITS AND LOWAR LIMITS

LOWAR_LIMIT=Q1-1.5*IQR

UPPER_LIMIT=Q3+1.5*IQR

LOWAR_LIMIT,UPPER_LIMIT
#NOW WE SHALL DISPLY THE OUTLIERS HEIGHTS

df[(df.Height<LOWAR_LIMIT)|(df.Height>UPPER_LIMIT)]
#NOW WE WILL DISPLAY THE REMAINING SAMPLES ARE WITHIN THE RANGE

df[(df.Height>LOWAR_LIMIT)&(df.Height<UPPER_LIMIT)]
