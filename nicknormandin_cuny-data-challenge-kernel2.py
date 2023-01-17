import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, log_loss





d = pd.read_csv('../input/inspections_train.csv')

x_train0, x_test0 = train_test_split(d, test_size=0.25)
from sklearn.linear_model import LogisticRegression
violations = pd.read_csv('../input/violations.csv')

violation_counts = violations.groupby(['camis', 'inspection_date']).size()

violation_counts = violation_counts.reset_index().set_index(['camis', 'inspection_date'])

violation_counts.columns = ['n_violations']



x_train2 = x_train0.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)

x_test2 = x_test0.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
# create the model named 'classifier'

classifier = LogisticRegression(solver='lbfgs')



# train the model to predict 'passed' based on 'n_violatoins'

classifier.fit(x_train2[['n_violations']], x_train2['passed'])
sample_example = [2]



# predict method just gives a 0 or 1

print(classifier.predict([sample_example])[0])



# predict_proba method gives us the probability between 0 and 1

print(classifier.predict_proba([sample_example])[0][1])
test_solution2 = classifier.predict_proba(x_test2[['n_violations']])

loss2 = log_loss(x_test2.passed.values, test_solution2)

print(f'log loss: {loss2:.3f}')
n_violations = np.linspace(0, 15, 16)

plt.plot(n_violations, [_[1] for _ in classifier.predict_proba(n_violations.reshape(-1, 1))])

plt.xlabel('number of violations'); plt.ylabel('predicted probability of pass')

plt.show()
# load the test data and add the `n_violations` feature

test_data = pd.read_csv('../input/inspections_test.csv')

test_data = test_data.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)



# take just the `id` and `n_violations` columns (since that's all we need)

submission = test_data[['id', 'n_violations']].copy()



# create a `Predicted` column

# for this example, we're using the model we previously trained

submission['Predicted'] = [_[1] for _ in classifier.predict_proba(submission.n_violations.values.reshape(-1, 1))]



# drop the n_violations columns

submission = submission.drop('n_violations', axis=1)



# IMPORTANT: Kaggle expects you to name the columns `Id` and `Predicted`, so let's make sure here

submission.columns = ['Id', 'Predicted']



# write the submission to a csv file so that we can submit it after running the kernel

submission.to_csv('submission2.csv', index=False)



# let's take a look at our submission to make sure it's what we want

submission.head()