# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
gender_data = pd.read_csv("../input/gender_submission.csv")

gender_data.describe()
data = pd.read_csv("../input/train.csv")

data.describe()
y = data.Survived



features = ['Pclass', 'Age']

X = data[features]





cols_with_missing = [col for col in X.columns 

                                 if X[col].isnull().any()]

reduced_X_train = X.drop(cols_with_missing, axis=1)

#reduced_y_train = y.drop(cols_with_missing, axis=1)





model = RandomForestRegressor(random_state=1)



model.fit(reduced_X_train, y)





test_data = pd.read_csv("../input/test.csv")

test_data.describe()





val_y = gender_data.Survived

val_X = test_data[features]



reduced_X_test  = val_X.drop(cols_with_missing, axis=1)





val_predictions = model.predict(reduced_X_test)



from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(val_predictions, val_y)



print("MAE: " + str(mae))



# Generate Submission File 

PassengerId = test_data['PassengerId']

Submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': val_predictions })

Submission.to_csv("StackingSubmission.csv", index=False)


