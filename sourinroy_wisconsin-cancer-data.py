# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# reading the data and specifying features and target 



# column dropped because it had a lot of NaN values

data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv',index_col='id')

data.drop(['Unnamed: 32'],axis=1,inplace = True)



data['diagnosis'].replace({'M' : 1 , 'B' : 0}, inplace=True)

y = data.diagnosis

features = ['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']

X = data[features]
#generate heatmap to identify useless features



plt.figure(figsize=(25,15))

corr_matrix = X.corr()

sns.heatmap(data = corr_matrix, annot = True, fmt = "0.2f")
#splitting data into training and testing data



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=11)
#Using RandomForestClassification



from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=200,max_depth=10,random_state=33)

model.fit(X_train,y_train)



test_preds = model.predict(X_test)

error = mean_absolute_error(test_preds,y_test)

error