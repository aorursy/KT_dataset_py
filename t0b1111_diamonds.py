# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plot Graphs and visualization of data and results

import seaborn as sns # Graphical representation of data and results.

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import precision_recall_fscore_support as error_metric

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,f1_score

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')

# print(df.head())

catagorical_col = df.select_dtypes(exclude=['number']).columns

print(catagorical_col)



X = df.drop(columns=['Unnamed: 0','price'])

print(X.head())



y = df['price']

print(y.head())



print(df.info())

def missing_datas(dataset):

    total = dataset.isnull().sum().sort_values(ascending=False)

    percent = (dataset.isnull().sum()/dataset.isnull().count()*100).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data





missing_data = missing_datas(df)

print(missing_data)
num_col = X.select_dtypes(include=['number'])

cat_col = X.select_dtypes(exclude=['number'])



print(num_col)

print('-'*20)

print(cat_col)
fig, ax = plt.subplots(2,3,sharex='col',figsize=(20, 12))

color = ['blue','orange','violet','pink','yellow']

i = 0





for p in range(2):

    for q in range(3):

        z = np.random.randint(low=0,high=4)

        ax[p,q].scatter(y,num_col[num_col.columns[i]],color = color[z])

        

        

        ax[p,q].set_title('Price Vs. {}'.format(num_col.columns[i]))

        ax[p,q].set_xlabel('Price')

        ax[p,q].set_ylabel(num_col.columns[i])

        

        i+=1

            

        

fig, ax = plt.subplots(2,3,sharex='col',figsize=(20, 12))

i = 0



for p in range(2):

    for q in range(3):

        sns.distplot(num_col[num_col.columns[i]],ax=ax[p,q],bins=40)

        i+=1

fig, ax = plt.subplots(2,3,sharex='col',figsize=(20, 12))

i = 0



for p in range(2):

    for q in range(3):

        

        sns.boxplot(data=num_col[num_col.columns[i]],ax=ax[p,q])

        ax[p,q].set_title('{}'.format(num_col.columns[i]))

        

        i+=1
fig,axs = plt.subplots(1,3,figsize=(20, 12))





for i in range(3):

    sns.boxplot(data=df , x=catagorical_col[i], y='price',ax=axs[i])

    sns.violinplot(data=df, x=catagorical_col[i], y='price',ax=axs[i], palette=["lightblue", "lightpink","lightyellow"])

mask1 = X['depth'] <= 65

mask2 = X['y'] <= 10

mask3 = X['z'] <= 6

mask4 = X['x'] <= 9

mask5 = 3 < X['y']

mask6 = 59 < X['depth'] 

mask7 = 2 < X['z']

mask8 = 3 < X['x'] 

mask9 = X['table'] <= 65

mask10 = 53 < X['depth'] 



X = X[mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7 & mask8 & mask9 & mask10]





y = y.iloc[X.index]
le = LabelEncoder()

for cols in cat_col:

    le.fit(X[cols])

    X[cols] = le.transform(X[cols])

    

    

    

print(X.head())
sns.heatmap(X.corr())

X = X.drop(columns=['x','y','z'])



sns.heatmap(X.corr())
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



def plot_learning_curves(model, X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):

        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])

        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))

        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")

    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

    print(min(np.sqrt(val_errors)))
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

plot_learning_curves(lin_reg, X, y)