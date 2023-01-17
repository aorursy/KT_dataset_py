import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, log_loss





d = pd.read_csv('../input/inspections_train.csv')

x_train0, x_test0 = train_test_split(d, test_size=0.25)
violations = pd.read_csv('../input/violations.csv')

violations.head()
violation_counts = violations.groupby(['camis', 'inspection_date']).size()

violation_counts = violation_counts.reset_index().set_index(['camis', 'inspection_date'])

violation_counts.columns = ['n_violations']



x_train1 = x_train0.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)

x_test1 = x_test0.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
x_train1.n_violations.hist()

plt.show()
test_solution1 = ((x_test1.n_violations < 3).map(int) * 0.5) + 0.4

loss1 = log_loss(x_test1.passed.values, test_solution1)

print(f'log loss: {loss1:.3f}')
# edit these 3 variables

cut_off = 3

lower_prob = 0.9

upper_prob = 0.4





# don't change anything down here

def decision_rule(val):

    if val < cut_off: return lower_prob

    else: return upper_prob



custom_solution = x_test1.n_violations.map(decision_rule)

custom_loss = log_loss(x_test1.passed.values, custom_solution)

print(f'Custom loss: {custom_loss:.3f}')



loss_delta = loss1 - custom_loss

if loss_delta > 0: print(f'Loss improved {loss_delta*100 / loss1:.2f}% !')

elif loss_delta < 0: print('Loss did not improve')
# load the test data and add the `n_violations` feature

test_data = pd.read_csv('../input/inspections_test.csv')

test_data = test_data.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)







# take just the `id` and `n_violations` columns (since that's all we need)

submission = test_data[['id', 'n_violations']].copy()



# create a `Predicted` column

# for this example, we're using the custom decision rule defined by you above

submission['Predicted'] = submission.n_violations.map(decision_rule)



# drop the n_violations columns

submission = submission.drop('n_violations', axis=1)



# IMPORTANT: Kaggle expects you to name the columns `Id` and `Predicted`, so let's make sure here

submission.columns = ['Id', 'Predicted']



# write the submission to a csv file so that we can submit it after running the kernel

submission.to_csv('submission1.csv', index=False)



# let's take a look at our submission to make sure it's what we want

submission.head()