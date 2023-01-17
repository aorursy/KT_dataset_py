from learntools.ml_explainability.ex4 import *

print("Setup Complete")
import pandas as pd

data = pd.read_csv('../input/hospital-readmissions/train.csv')

data.columns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



data = pd.read_csv('../input/hospital-readmissions/train.csv')



y = data.readmitted



base_features = [c for c in data.columns if c != "readmitted"]



X = data[base_features]



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
# Your code here

____
# Run this code cell to receive credit!

q_1.solution()
# Your Code Here

____
# Check your answer (Run this code cell to receive credit!)

q_2.solution()
# Your Code Here

____
# Check your answer (Run this code cell to receive credit!)

q_3.solution()
# Your Code Here

____
# q_4.hint()
# Check your answer (Run this code cell to receive credit!)

q_4.solution()
# Your Code Here

____
# q_5.hint()
# Check your answer (Run this code cell to receive credit!)

q_5.solution()