# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import precision_recall_curve, auc

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('/kaggle/input/titanic/train.csv')

train = train.replace({'male': 1, 'female': 0})

test = train.sample(frac = 0.2)

train = train[~train.index.isin(test.index)]





train = train[['Survived', 'Sex', 'Fare', 'Pclass']]

test_variables = test[['Sex', 'Fare', 'Pclass']]

test_target = test['Survived']
##Both classes have equal examples.



zero_data = train[train['Survived'] == 0][:250]

one_data =  train[train['Survived'] == 1][:250]

non_skewed_data = one_data.append(zero_data)



from sklearn.linear_model import LogisticRegression

non_skewed_model = LogisticRegression(random_state=0, solver='lbfgs',

                         multi_class='multinomial').fit(non_skewed_data[['Sex', 'Fare', 'Pclass']],

                                                        non_skewed_data['Survived'])



predictions = non_skewed_model.predict(test_variables)

predicted_prob = non_skewed_model.predict_proba(test_variables)

prob_series = pd.Series([i[1] for i in predicted_prob])



precision, recall, thresholds = precision_recall_curve(test['Survived'], prob_series) 

   #retrieve probability of being 1(in second column of probs_y)

pr_auc = auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")

plt.plot(thresholds, precision[: -1], "b--", label="Precision")

plt.plot(thresholds, recall[: -1], "r--", label="Recall")

plt.ylabel("Precision, Recall")

plt.xlabel("Threshold")

plt.legend(loc="lower left")

plt.ylim([0,1])

plt.show()
####Using data Skewed towards 0 class.



zero_data = train[train['Survived'] == 0]#[:100]#[:100]

one_data =  train[train['Survived'] == 1][:100]

zero_skewed_data = one_data.append(zero_data)

from sklearn.linear_model import LogisticRegression

zero_skewed_model = LogisticRegression(random_state=0, solver='lbfgs',

                         multi_class='multinomial').fit(zero_skewed_data[['Sex', 'Fare', 'Pclass']],

                                                        zero_skewed_data['Survived'])



predictions = zero_skewed_model.predict(test_variables)

predicted_prob = zero_skewed_model.predict_proba(test_variables)

prob_series = pd.Series([i[1] for i in predicted_prob])



precision, recall, thresholds = precision_recall_curve(test['Survived'], prob_series) 

   #retrieve probability of being 1(in second column of probs_y)

pr_auc = auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")

plt.plot(thresholds, precision[: -1], "b--", label="Precision")

plt.plot(thresholds, recall[: -1], "r--", label="Recall")

plt.ylabel("Precision, Recall")

plt.xlabel("Threshold")

plt.legend(loc="lower left")

plt.ylim([0,1])

plt.show()
####Using data Skewed towards 1 class.



zero_data = train[train['Survived'] == 0][:100]#[:100]

one_data =  train[train['Survived'] == 1]

one_skewed_data = one_data.append(zero_data)

from sklearn.linear_model import LogisticRegression

one_skewed_model = LogisticRegression(random_state=0, solver='lbfgs',

                         multi_class='multinomial').fit(one_skewed_data[['Sex', 'Fare', 'Pclass']],

                                                        one_skewed_data['Survived'])



predictions = one_skewed_model.predict(test_variables)

predicted_prob = one_skewed_model.predict_proba(test_variables)

prob_series = pd.Series([i[1] for i in predicted_prob])



precision, recall, thresholds = precision_recall_curve(test['Survived'], prob_series) 

   #retrieve probability of being 1(in second column of probs_y)

pr_auc = auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")

plt.plot(thresholds, precision[: -1], "b--", label="Precision")

plt.plot(thresholds, recall[: -1], "r--", label="Recall")

plt.ylabel("Precision, Recall")

plt.xlabel("Threshold")

plt.legend(loc="lower left")

plt.ylim([0,1])

plt.show()