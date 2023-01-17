!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@quick-changes
import sys

sys.path.append('/kaggle/working')
import pandas as pd

from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex8 import *

print("Setup complete")
pulsar_data = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

pulsar_data.head()
y = pulsar_data['target_class']

X = pulsar_data.drop('target_class', axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size=.2)
from sklearn.ensemble import RandomForestClassifier



# Define the model. Set random_state to 1

model = ____



# Fit your model

____



step_1.check()
# The lines below will show you a hint or the solution.

#step_1.hint() 

#step_1.solution()
#%%RM_IF(PROD)%%



from sklearn.ensemble import RandomForestClassifier



# Define the model. Set random_state to 1

model = RandomForestClassifier(random_state=1)



# fit your model on the training data

model.fit(train_X, train_y)



step_1.assert_check_passed()
# Get predictions from the trained model using the validation features

pred_y = ____



# Calculate the accuracy of the trained model with the validation targets and predicted targets

accuracy = ____



print("Accuracy: ", accuracy)



step_2.check()
# The lines below will show you a hint or the solution.

#step_2.hint()

#step_2.solution()
#%%RM_IF(PROD)%%

# Get predictions from the trained model using the validation features

pred_y = model.predict(val_X)



# Calculate the accuracy of the trained model with the validation targets and predicted targets

accuracy = metrics.accuracy_score(val_y, pred_y)



print("Accuracy: ", accuracy)

step_2.assert_check_passed()
(val_y==0).mean()
confusion = metrics.confusion_matrix(val_y, pred_y)

print(f"Confusion matrix:\n{confusion}")



# Normalizing by the true label counts to get rates

print(f"\nNormalized confusion matrix:")

for row in confusion:

    print(row / row.sum())
#step_3.solution()
#step_4.solution()