# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy .stats import norm 



df = pd.read_csv('../input/preprocess-choc/dfn.csv')

df
%matplotlib inline

plt.hist(df.rating,bins=20,rwidth=0.8)

plt.xlabel('rating')

plt.ylabel('count')

plt.show()
from scipy.stats import norm

import numpy as np

from matplotlib import pyplot as plt

plt.hist(df.rating,bins=20,rwidth=0.8)

plt.xlabel('rating')

plt.ylabel('count')

rng=np.arange(df.rating.min(),df.rating.max(),0.1)

plt.plot(rng,norm.pdf(rng,df.rating.mean(),df.rating.std()))

plt.show()

#matplotlib. rcParams['figure.figsize']=(10,6)



#i dont know why the bell curve isnt plotting in Kaggle(was plotting in JN),Trouble shoot and let me know
#max rating df.rating.max()



#mean rating df.rating.mean()



#std. deviation of rating df.rating.std()

#so my upper limit will be my mean value plus 3 sigma

upper_limit=df.rating.mean()+3*df.rating.std()

upper_limit
#my lowar limit will be my mean - 3 sigma

lowar_limit=df.rating.mean()-3*df.rating.std()

lowar_limit
#now that my outliers are defined, i want to see what are my outliers

df[(df.rating>upper_limit)|(df.rating<lowar_limit)]
#now we will visualise the good data

new_data=df[(df.rating<upper_limit)& (df.rating>lowar_limit)]

new_data
#shape of our new data

new_data.shape
#shape of our outliers

df.shape[0]-new_data.shape[0]
#now we will calculate the z score of all our datapoints and display in a dataframe

df['zscore']=(df.rating-df.rating.mean())/df.rating.std()

df
![z.png]
#figuring out all the datapoints more than 3

df[df['zscore']>3]
#figuring out all the datapoints less than 3

df[df['zscore']<-3]
#displaying the outliers with respect to the zscores

df[(df.zscore<-3)|(df.zscore>3)]
new_data_1=df[(df.zscore>-3)& (df.zscore<3)]

new_data_1
figure=df.boxplot(column="rating", figsize=(20,20))
figure=new_data_1.boxplot(column="rating", figsize=(20,20))
from scipy.stats import norm

import numpy as np

from matplotlib import pyplot as plt

plt.hist(df.rating,bins=20,rwidth=0.8)

plt.xlabel('rating')

plt.ylabel('count')

rng=np.arange(df.rating.min(),df.rating.max(),0.1)

plt.plot(rng,norm.pdf(rng,df.rating.mean(),df.rating.std()))

plt.show()

#matplotlib. rcParams['figure.figsize']=(10,6)



#i dont know why the bell curve isnt plotting in Kaggle(was plotting in JN),Trouble shoot and let me know
from scipy.stats import norm

import numpy as np

from matplotlib import pyplot as plt

plt.hist(new_data_1.rating,bins=20,rwidth=0.8)

plt.xlabel('rating')

plt.ylabel('count')

rng=np.arange(new_data_1.rating.min(),new_data_1.rating.max(),0.1)

plt.plot(rng,norm.pdf(rng,new_data_1.rating.mean(),new_data_1.rating.std()))

plt.show()

#matplotlib. rcParams['figure.figsize']=(10,6)



#i dont know why the bell curve isnt plotting in Kaggle(was plotting in JN),Trouble shoot and let me know
df=df.drop(['zscore'],axis=1)
df.describe()
Q1=df.rating.quantile(0.25)

Q3=df.rating.quantile(0.75)

Q1,Q3

#WHICH MEANS THAT Q1 CORRESPONDS TO 25% OF ALL THE HEIGHT DISTRIBUTION IS BELOW 3.0

#Q3 CORRESPONDS TO 75% OF ALL THE HEIGHT DISTRIBUTION IS BELOW 3.5
#NOW WE WILL CALCULATE THE IQR

IQR=Q3-Q1

IQR
#NOW WE WILL DEFINE THE UPPER LIMITS AND LOWAR LIMITS

LOWAR_LIMIT=Q1-1.5*IQR

UPPER_LIMIT=Q3+1.5*IQR

LOWAR_LIMIT,UPPER_LIMIT
#NOW WE SHALL DISPLY THE OUTLIERS rating

df[(df.rating<LOWAR_LIMIT)|(df.rating>UPPER_LIMIT)]
#NOW WE WILL DISPLAY THE REMAINING SAMPLES ARE WITHIN THE RANGE

Without_outliers_data = df[(df.rating>LOWAR_LIMIT)&(df.rating<UPPER_LIMIT)]
figure=df.boxplot(column="rating", figsize=(20,20))

figure=Without_outliers_data.boxplot(column="rating", figsize=(20,20))
from scipy.stats import norm

import numpy as np

from matplotlib import pyplot as plt

plt.hist(df.rating,bins=20,rwidth=0.8)

plt.xlabel('rating')

plt.ylabel('count')

rng=np.arange(df.rating.min(),df.rating.max(),0.1)

plt.plot(rng,norm.pdf(rng,df.rating.mean(),df.rating.std()))

plt.show()

#matplotlib. rcParams['figure.figsize']=(10,6)



#i dont know why the bell curve isnt plotting in Kaggle(was plotting in JN),Trouble shoot and let me know
from scipy.stats import norm

import numpy as np

from matplotlib import pyplot as plt

plt.hist(Without_outliers_data.rating,bins=20,rwidth=0.8)

plt.xlabel('rating')

plt.ylabel('count')

rng=np.arange(Without_outliers_data.rating.min(),df.rating.max(),0.1)

plt.plot(rng,norm.pdf(rng,Without_outliers_data.rating.mean(),Without_outliers_data.rating.std()))

plt.show()

#matplotlib. rcParams['figure.figsize']=(10,6)



#i dont know why the bell curve isnt plotting in Kaggle(was plotting in JN),Trouble shoot and let me know
df_enc = pd.read_csv('../input/preprocess-choc/10 best RD_Feature')

df_enc
a = df_enc.loc[:,~df_enc.columns.duplicated()]

a


b = a.drop('rating', axis = 1)
X = b.iloc[:,0:11]  

y = a.iloc[:,2]    





from sklearn.model_selection import train_test_split



X_train,y_train, X_test,y_test = train_test_split(X, y, test_size=0.3)
y


from scipy import stats



from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor



import matplotlib.dates as md

from scipy.stats import norm

%matplotlib inline 

import seaborn as sns 

sns.set_style("whitegrid") #possible choices: white, dark, whitegrid, darkgrid, ticks



import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

pd.set_option('float_format', '{:f}'.format)

pd.set_option('max_columns',250)

pd.set_option('max_rows',150)
clf = IsolationForest(max_samples='auto', random_state = 1, contamination= 0.02)

preds = clf.fit_predict(X)

df['isoletionForest_outliers'] = preds

df['isoletionForest_outliers'] = df['isoletionForest_outliers'].astype(str)

df['isoletionForest_scores'] = clf.decision_function(X)

print(df['isoletionForest_outliers'].value_counts())

df[152:156]
!pip install eif
import eif as iso
fig, ax = plt.subplots(figsize=(20, 7))

ax.set_title('Distribution of Extended Isolation Scores', fontsize = 15, loc='center')

sns.distplot(df['isoletionForest_scores'],color='red',label='if',hist_kws = {"alpha": 0.5});




fig, ax = plt.subplots(figsize=(30, 7))

ax.set_title('Extended Outlier Factor Scores Outlier Detection', fontsize = 15, loc='center')



plt.scatter(X.iloc[:, 0], X.iloc[:, 1], color='g', s=3., label='Data points')

radius = (df['isoletionForest_scores'].max() - df['isoletionForest_scores']) / (df['isoletionForest_scores'].max() - df['isoletionForest_scores'].min())

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=2000 * radius, edgecolors='r', facecolors='none', label='Outlier scores')

plt.axis('tight')

legend = plt.legend(loc='upper left')

legend.legendHandles[0]._sizes = [10]

legend.legendHandles[1]._sizes = [20]

plt.show();







clf = LocalOutlierFactor(n_neighbors=11)

y_pred = clf.fit_predict(X)



df['localOutlierFactor_outliers'] = y_pred.astype(str)

print(df['localOutlierFactor_outliers'].value_counts())

df['localOutlierFactor_scores'] = clf.negative_outlier_factor_



fig, ax = plt.subplots(figsize=(20, 7))

ax.set_title('Distribution of Local Outlier Factor Scores', fontsize = 15, loc='center')

sns.distplot(df['localOutlierFactor_scores'],color='red',label='eif',hist_kws = {"alpha": 0.5});





fig, ax = plt.subplots(figsize=(30, 7))

ax.set_title('Local Outlier Factor Scores Outlier Detection', fontsize = 15, loc='center')



plt.scatter(X.iloc[:, 0], X.iloc[:, 1], color='g', s=3., label='Data points')

radius = (df['localOutlierFactor_scores'].max() - df['localOutlierFactor_scores']) / (df['localOutlierFactor_scores'].max() - df['localOutlierFactor_scores'].min())

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=2000 * radius, edgecolors='r', facecolors='none', label='Outlier scores')

plt.axis('tight')

legend = plt.legend(loc='upper left')

legend.legendHandles[0]._sizes = [10]

legend.legendHandles[1]._sizes = [20]

plt.show();


