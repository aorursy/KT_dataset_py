import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
from sklearn.preprocessing import OneHotEncoder as enc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
#initial read in of comma separated data
df = pd.read_csv('../input/horse.csv')
df.head()
#some null values, but we can use a separate dummy variable for them
df.info()
#split a few initial features of interest into separate dataframes

surgery = pd.DataFrame(df['surgery'])
age = pd.DataFrame(df['age'])
temp_of_extremities = pd.DataFrame(df['temp_of_extremities'])
outcome = pd.DataFrame(df['outcome'])
peripheral_pulse = pd.DataFrame(df['peripheral_pulse'])
mucous_membrane = pd.DataFrame(df['mucous_membrane'])
abdominal_distention = pd.DataFrame(df['abdominal_distention'])

peripheral_pulse.head()
#use get_dummies to create 0/1 binary variable for each category in each feature
surgery = pd.get_dummies(surgery,prefix=['surgery'],drop_first=False)
age = pd.get_dummies(age,prefix=['age'],drop_first=False)
temp_of_extremities = pd.get_dummies(temp_of_extremities,prefix=['temp_of_extremities'],drop_first=False, dummy_na=True)
outcome = pd.get_dummies(outcome,prefix=['outcome'],drop_first=False)
peripheral_pulse = pd.get_dummies(peripheral_pulse,prefix=['peripheral_pulse'],drop_first=False, dummy_na=True)
mucous_membrane = pd.get_dummies(mucous_membrane,prefix=['mucous_membrane'],drop_first=False, dummy_na=True)
abdominal_distention = pd.get_dummies(abdominal_distention,prefix=['abdominal_distention'],drop_first=False, dummy_na=True)

peripheral_pulse.head()
#check our classifier for "died" outcome based on split of data into two sets, 25% of data as our training data

data = pd.concat([surgery, age, temp_of_extremities, peripheral_pulse, mucous_membrane], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    data, outcome['outcome_died'], test_size=0.25, random_state=0);

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y1_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on "died": {:.2f}'.format(logreg.score(X_test, y_test)))
#check our classifier for "lived" outcome based on split of data into two sets, 25% of data as our training data
data = pd.concat([surgery, age, temp_of_extremities, peripheral_pulse, mucous_membrane], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    data, outcome['outcome_lived'], test_size=0.25, random_state=0);

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y2_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on "lived": {:.2f}'.format(logreg.score(X_test, y_test)))
#check our classifier for "euthanized" outcome based on split of data into two sets, 25% of data as our training data
data = pd.concat([surgery, age, temp_of_extremities, peripheral_pulse, mucous_membrane], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    data, outcome['outcome_euthanized'], test_size=0.25, random_state=0);

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on "euthanized": {:.2f}'.format(logreg.score(X_test, y_test)))