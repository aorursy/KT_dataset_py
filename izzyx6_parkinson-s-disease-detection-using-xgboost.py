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
#importing all the necessary libraries to be used in our code

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score,cross_validate

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier
#importing the parkinsons data downloaded from UCI machine learning repository

path = 'parkinsons.data'

df = pd.read_csv("/kaggle/input/parkinsons-data-set/parkinsons.data")

df.head()
df.info() #checking the information of the columns
df.shape #Cecking th shape of of data i.e rows and column
df.describe() #Getting the statistical summary of features in our dataset
sns.countplot(df['status']) #ploting a count of the target variable with seaborn

plt.show()
#Visualization is always the best way to explain figures.

#Using seaborn heatmap for plotting the correlation of the features 

plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True,fmt=".2f",linewidths="1.2")

plt.show()
#Using pair plot to showing the relationship between variables that are highly correlected (+ve and -ve) with the target (status)

plt.figure(figsize = (15,10))

sns.pairplot(df, vars=['MDVP:Fo(Hz)','MDVP:Flo(Hz)','HNR','PPE','spread1','spread2'],hue='status',palette='Dark2')

plt.savefig('Relationship')

plt.show()
#Dividing our dataset into X (features) and y (Target)

X = np.array(df.drop(['name','status'], axis = 1))

y = np.array(df['status'])

print(f'X shape: {X.shape} Y Shape: {y.shape}')
#scaling the features so they are of the same scale. The target doesnt need to be scaled

scaler = MinMaxScaler()

scaled_X = scaler.fit_transform(X)
#writting a function for performing cross validation 

def crossValidate(model):

    #Using StratifiedKFold to ensure that the divided folds are shuffled

    strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

    

    #Getting just specific scores for perfromance evualation.

    scoring = ["accuracy","precision","recall","f1","roc_auc"]

    cv = cross_validate(model, scaled_X, y, cv = strat_k_fold, scoring = scoring)

    

    '''

    for score in cv:

        print(f'{score}: {round(cv[score].mean(),3)}')

    '''

    

    result = [round(cv[score].mean(),3) for score in cv]

    return result

    
model = XGBClassifier()

result = crossValidate(model)#passing the model to the cross validate function
result[2:]
#Giving a plot of the performance metrics used

plt.figure(figsize = (6,2))

model_preformance = pd.Series(data=result[2:], 

        index=['Accuracy','Precision','Recall','F1-Score','AUC (ROC)'])

model_preformance.sort_values().plot.barh()

plt.title('Model Performance')