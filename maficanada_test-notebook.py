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
# LOAD IN THE TRAINING DATA.

# THIS IS THE DATA THAT WE WILL BUILD THE RULES TO MAKE THE PREDICTIONS

# LETS LOAD IT IN AND TAKE A LOOK AT THE FIRST 5 ROWS



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
# LOAD IN THE TESTING DATA

# THIS IS THE DATA THAT WE WILL BE MAKING PREDICTIONS ON FOR THE COMPETITION

# LETS LOAD IT IN AND TAKE A LOOK AT THE FIRST 5 ROWS



test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
#TO MAKE OUR FIRST PREDICTION ABOUT THE NUMBER OF PASSENGERS WHO SURVIVE, WE WILL USE THE RANDOM FIRST MODEL. 



from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



rf1_died = np.count_nonzero(predictions == 0)

rf1_survived = np.count_nonzero(predictions == 1)



print(f"Model 1 predicts that {rf1_died} people will die")

print(f"Model 1 predicts that {rf1_survived} people will survive")

print(f"Out of a Total of {rf1_died + rf1_survived} Passengers")
# This is good for a start but the machine learning model is very basic and can be improved. 

# We need to define a metric by which we can assess our improvement and we will use the MAE metric

# We therefore need to calculate this on the TRAINING data that we have loaded in already


