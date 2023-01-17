import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
X_test = pd.read_csv('../input/test.csv')

X_test = X_test.drop(["ID"], axis=1)
print ('Number of duplicates among test set: {}, {}'

       .format(X_test.duplicated().sum(), X_test.duplicated(keep=False).sum()))
print ('Percentage of duplicates among test set:')

print ('{:.2%}'.format(X_test.duplicated().sum()/len(X_test)))

print ('{:.2%}'.format(X_test.duplicated(keep=False).sum()/len(X_test)))
valid = [i for i in range (100,136)] + [136, 136, 137, 137, 137]

model_1 = [i for i in range (101,137)] + [136.7,136.7,137.5,137.5,137.5]

model_2 = [i+0.5 for i in range (100,136)] + [138.5, 138.5, 139, 139, 139]
print ('Percentage of duplicates among valid set:')

print ('{:.2%}'.format((3/len(valid))))

print ('{:.2%}'.format((5/len(valid))))
plt.plot(valid, 'b.')

plt.plot(model_1, 'r.')

plt.plot(model_2, 'g.')

plt.ylim(ymin=95, ymax=145)

plt.show()
from sklearn.metrics import r2_score
print ('Calculating r2 score for both models:')

print ('Model_1: {:.2%}'.format(r2_score(valid, model_1)))

print ('Model_2: {:.2%}'.format(r2_score(valid, model_2)))
valid_LB1 = valid[0:8]

model_1_LB1 = model_1[0:8]

model_2_LB1 = model_2[0:8]



valid_PB1 = valid[8:]

model_1_PB1 = model_1[8:]

model_2_PB1 = model_2[8:]
print ('Public LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_LB1, model_1_LB1), r2_score(valid_LB1, model_2_LB1)))

print ('Private LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_PB1, model_1_PB1), r2_score(valid_PB1, model_2_PB1)))
valid_LB2 = valid[-8:]

model_1_LB2 = model_1[-8:]

model_2_LB2 = model_2[-8:]



valid_PB2 = valid[:-8]

model_1_PB2 = model_1[:-8]

model_2_PB2 = model_2[:-8]
print ('Public LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_LB2, model_1_LB2), r2_score(valid_LB2, model_2_LB2)))

print ('Private LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_PB2, model_1_PB2), r2_score(valid_PB2, model_2_PB2)))
valid_LB3 = valid[:5] + valid[-3:]

model_1_LB3 = model_1[:5] + model_1[-3:]

model_2_LB3 = model_2[:5] + model_2[-3:]



valid_PB3 = valid[5:] + valid[:-3]

model_1_PB3 = model_1[5:] + model_1[:-3]

model_2_PB3 = model_2[5:] + model_2[:-3]
print ('Public LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_LB3, model_1_LB3), r2_score(valid_LB3, model_2_LB3)))

print ('Private LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_PB3, model_1_PB3), r2_score(valid_PB3, model_2_PB3)))
valid_LB4 = valid[:5] + valid[-5:-2]

model_1_LB4 = model_1[:5] + model_1[-5:-2]

model_2_LB4 = model_2[:5] + model_2[-5:-2]



valid_PB4 = valid[5:-5] + valid[-2:]

model_1_PB4 = model_1[5:-5] + model_1[-2:]

model_2_PB4 = model_2[5:-5] + model_2[-2:]
print ('Public LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_LB4, model_1_LB4), r2_score(valid_LB4, model_2_LB4)))

print ('Private LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_PB4, model_1_PB4), r2_score(valid_PB4, model_2_PB4)))
valid_LB5 = valid[:5] + valid[-5:-4] + valid[-3:-1]

model_1_LB5 = model_1[:5] + model_1[-5:-4] + model_1[-3:-1]

model_2_LB5 = model_2[:5] + model_2[-5:-4] + model_2[-3:-1]



valid_PB5 = valid[5:-5] + valid[-4:-3] + valid[-1:]

model_1_PB5 = model_1[5:-5] + model_1[-4:-3] + model_1[-1:]

model_2_PB5 = model_2[5:-5] + model_2[-4:-3] + model_2[-1:]
print ('Public LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_LB5, model_1_LB5), r2_score(valid_LB5, model_2_LB5)))

print ('Private LB scores. Model_1: {:.2%}, Model_2: {:.2%}'.format(r2_score(valid_PB5, model_1_PB5), r2_score(valid_PB5, model_2_PB5)))