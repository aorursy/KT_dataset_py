# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading dataset

#making our ID as an index so it doesn't mess up with our model.

train = pd.read_csv("../input/learn-together/train.csv", index_col='Id')

test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')



#making our traning/testing dataset ready.

X_train = pd.DataFrame(train)

Y_train = train['Cover_Type']

X_train.drop(columns = ['Cover_Type'],inplace=True)

X_test = pd.DataFrame(test)  
test.head(10)
#number of features, samples

train.shape, test.shape
import matplotlib.pyplot as plt

#histogram with no soils

cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',

       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',

       'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']

hist_data = test[cols]

hist_data.hist(figsize=(15,10))

plt.tight_layout()
test['Wilderness_Area2'].value_counts()
train.info()
train['Cover_Type'].unique()
#check for all soil columns are redundant or not

for name in range(1,41):

    cur_soil_name = "Soil_Type"+ str(name)

    soil = pd.DataFrame(train[cur_soil_name])

    print(soil.nunique())      
train["Soil_Type15"].unique()
X_train.drop(columns = ['Soil_Type7','Soil_Type15'], inplace=True)

X_test.drop(columns = ['Soil_Type7','Soil_Type15'], inplace=True)
X_train['Soil_Type12'].value_counts()
X_train.describe()
#try different models evaluate them using cross_val_score

from sklearn.linear_model import LogisticRegression  

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection

import warnings

warnings.filterwarnings(action="ignore")



scoring='accuracy'

models = []

models.append(('LR',  LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART',DecisionTreeClassifier()))

models.append(('SVM', SVC()))

# evaluate each model in turn

results = []

names = []

for name, model in models:

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=10, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
#more advance models

from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier



models = []

models.append(('RF', RandomForestClassifier()))

models.append(('GB', GradientBoostingClassifier()))

models.append(('AB', AdaBoostClassifier()))

models.append(('ET', ExtraTreeClassifier()))

# evaluate each model in turn

results = []

names = []

for name, model in models:

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=10, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state = 42)
from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier(random_state= 42,oob_score=False)



# max_fea = np.arange(1,9)

params =  {"n_estimators": [150,100], 

           "max_depth": [15,None]

          } 

#with max_features, min_samples_leaf it was taking too long and results merely changed,

#so removed them.

#max_depth above 10 and below 20 is great.

#will try with oob_score later.



gscv = GridSearchCV(estimator = rfc, param_grid = params, cv = 10, n_jobs =-1)

gscv.fit(x_train,y_train)



print("GSCV Score: %.2f%% "% gscv.score(x_val,y_val))

gscv.best_params_ 
#final model with GridSearchCV findings 

rf = RandomForestClassifier(n_estimators = 150, random_state= 42, max_depth=None)

cv_results = model_selection.cross_val_score(rf, X_train, Y_train, cv=10, scoring=scoring)

print("Random Forest Result: %.2f%%" % cv_results.mean())
# Calculate MAE

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

rf.fit(x_train, y_train)

y_pred = rf.predict(x_val)

mae_score = mean_absolute_error(y_val, y_pred)

acc_score = accuracy_score(y_val, y_pred)

print("Accuracy Score: %.2f%%" % acc_score)

print("Mean Absolute Error: %.2f%%" % mae_score)
#feature importance

rf.feature_importances_
#submission

rf.fit(x_train, y_train)

pred = rf.predict(X_test)

test_ids = X_test.index

output = pd.DataFrame({'Id': test_ids,

                       'Cover_Type': pred})

output.to_csv('submission.csv', index=False)

output.head(10)