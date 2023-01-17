# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def univar_plot(df,column):

    '''

    Function to make 3 plots about a numeric column, the plots are:

    - Boxplot(data), Histogram(data) , Histogram(log(data))

    '''

    f, ax = plt.subplots(3, figsize=(12,7))



    sns.set(rc={'figure.figsize':(12,8)})

    sns.boxplot(x=df[column], ax = ax[0])

    ax[0].set_title("{} Boxplot".format(column))

    sns.distplot(a=df[column], kde = False, ax = ax[1])

    ax[1].set_title("{} Histogram".format(column))

    sns.distplot(a=np.log1p(df[column]), kde = False, ax = ax[2])

    ax[2].set_title("Log1p transformed {} Histogram".format(column))

    f.tight_layout()

    return None



#univar_plot(df,'Weight')
import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv('/kaggle/input/fish-market/Fish.csv')
df.head()
df['Species'].value_counts()
df.groupby('Species').mean()
sns.boxplot(data = df,x = 'Species',y = 'Height')
#sns.boxplot(data = df,x = 'Species',y = 'Length1')

#sns.boxplot(data = df,x = 'Species',y = 'Length2')

#sns.boxplot(data = df,x = 'Species',y = 'Length3')



df['mean_length'] = (df['Length1'] + df['Length2'] + df['Length3'])/3

sns.boxplot(data = df,x = 'Species',y = 'mean_length')
sns.boxplot(data = df,x = 'Species',y = 'Width')
df.head()
df['density_height'] = df['Weight']/((2)*3.14*df['Height']*(df['Width']/2))

#df['density_length'] = df['Weight']/((2)*3.14*df['mean_length']*df['Width'])
#sns.lmplot(data = df,x = 'Species',y = 'density')

sns.lmplot(data = df ,x = 'Weight', y = 'density_length',hue = 'Species',fit_reg = False)
sns.lmplot(data = df ,x = 'mean_length', y = 'density_length',hue = 'Species',fit_reg = False)
def distplot_hue(df,hue,y_var,kde_dist = True):

    unique_classes = df[hue].unique()

    for c in unique_classes:

        sns.distplot(df[df[hue] == c][y_var], label = c,kde = kde_dist)

    plt.legend()

    plt.grid()

    return None



distplot_hue(df,hue = 'Species',y_var = 'Weight')
def rename_peixes(series):

    new_dict = {}

    for i,fish in enumerate(series.unique()):

        new_dict[fish] = i

    print(new_dict)

    return series.replace(new_dict)

            
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler



X = df.drop(['Species'],axis=1)

y = rename_peixes(df['Species'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ss = StandardScaler()

X_train = ss.fit_transform(X_train)

X_test = ss.transform(X_test)



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
pd.DataFrame(lr.coef_,columns = X.columns,

            index = df['Species'].unique())
print(confusion_matrix(y_pred,y_test))

sns.heatmap(confusion_matrix(y_pred,y_test)/confusion_matrix(y_pred,y_test).sum(axis=0))