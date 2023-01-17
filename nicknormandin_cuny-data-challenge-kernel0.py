import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os
os.listdir('../input/')
d = pd.read_csv('../input/inspections_train.csv')
print(d.shape)



print(d.columns)
d.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, log_loss



x_train0, x_test0 = train_test_split(d, test_size=0.25)
percent_passed = x_train0.passed.mean()

print(f"Percent of inspections with 'A' grade: {100*percent_passed:.2f}%")
test_solution0 = np.ones(x_test0.passed.shape)

loss0a = log_loss(x_test0.passed.values, test_solution0)

print(f'log loss: {loss0a:.3f}')
test_solution0 = np.zeros(x_test0.passed.shape)

loss0b = log_loss(x_test0.passed.values, test_solution0)

print(f'log loss: {loss0b:.3f}')
test_solution0 = np.ones(x_test0.passed.shape) * percent_passed

loss0c = log_loss(x_test0.passed.values, test_solution0)

print(f'log loss: {loss0c:.3f}')
def probability_soln(shape, val): return np.ones(shape) * val

prediction_values = np.linspace(0.01, 0.99, 100)

plt.plot(prediction_values,

        [log_loss(x_test0.passed, probability_soln(x_test0.shape[0], _)) for _ in prediction_values])

plt.scatter([percent_passed], [loss0c], color='black')

plt.xlabel('probability'); plt.ylabel('log loss')

plt.title('Log loss at different prediction values')

plt.show()
# load the test data

test_data = pd.read_csv('../input/inspections_test.csv')



# take just the `id` columns (since that's all we need)

submission = test_data[['id']].copy()



# create a `Predicted` column. in this case it's just the `percent_passed` that we calculated earlier

submission['Predicted'] = percent_passed



# IMPORTANT: Kaggle expects you to name the columns `Id` and `Predicted`, so let's make sure here

submission.columns = ['Id', 'Predicted']



# write the submission to a csv file so that we can submit it after running the kernel

submission.to_csv('submission0.csv', index=False)



# let's take a look at our submission to make sure it's what we want

submission.head()