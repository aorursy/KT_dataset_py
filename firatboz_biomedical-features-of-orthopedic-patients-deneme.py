# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/column_2C_weka.csv')
df.head()
df.rename(columns={'class':'class1'}, inplace=True)
df.shape
# Let's see whether there are any NaN value and length of this data so lets look at info

df.info()
color=[]

from pandas.plotting import scatter_matrix

colors_palette = {'Abnormal': "red", 'Normal': "green"}

for _ in df.class1:

    color.append(colors_palette[_]);

matrix_of_scatterplots = scatter_matrix(df, figsize=(15, 15),

                                        color=color,alpha=0.5,marker = '*',

                                        s=200,edgecolor= "black");
#sns.pairplot(df,hue='class1',kind="reg");

markers_dict=['o','s']

sns.pairplot(df,hue='class1',markers=markers_dict);
#Scatterplot of sacral slope and pelvic incidence with pandas

color_choice=['g' if _ =='Normal' else 'r' for _ in df.class1]

df.plot.scatter(x='sacral_slope',y='pelvic_incidence',figsize=(8, 8),title=' Relation of sacral slope & pelvic incidence',

                marker='s',s=30,color=color_choice,alpha=0.5);

#Scatterplot of sacral slope and pelvic incidence with seaborn

sns.scatterplot(df.pelvic_incidence, df.sacral_slope, data=df,hue='class1');
#Filter class with the feature abnormal

df_abnormal=df[df.class1=='Abnormal']
from sklearn.linear_model import LinearRegression

x=df_abnormal.pelvic_incidence.values.reshape(-1,1)

y=df_abnormal.sacral_slope.values.reshape(-1,1)

lr=LinearRegression()

lr.fit(x,y)

y_head=lr.predict(x)
#Calculate and print r_square value

from sklearn.metrics import r2_score

r_squared=r2_score(y, y_head)

print('r_square value is:',r_squared)
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)[source]

# Return evenly spaced numbers over a specified interval.

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html



prediction_x_vals=np.linspace(x.min(),x.max()).reshape(-1,1)

y_predict=lr.predict(prediction_x_vals)
fig,ax =plt.subplots(figsize=(8,8));

ax.plot(prediction_x_vals,y_predict,color='red',linewidth=3);

ax.scatter(x,y,s=50)

ax.set_xlabel('pelvic incidence');

ax.set_ylabel('sacral_slope');

ax.set_title('regression analysis of pelvic incidence and sacral slope for annormal class');

ax.grid()

ax.set_facecolor('whitesmoke')
