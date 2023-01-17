%matplotlib notebook

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, confusion_matrix
dataset = pd.read_csv('../input/201710-CAH_PulseOfTheNation.csv')
dataset.head()
# First, the classes that are already discrete. Let's check the existing values with .unique()

unique_vals = {}
columns = {'gender': 'Gender', 
           'agerng': 'Age Range', 
           'polit': 'Political Affiliation ', 
           'race': 'What is your race?',
          'educ': 'What is your highest level of education?'}

for col, colname in columns.items():
    unique_vals[col] = dataset[colname].unique()
    print("{0}: {1}".format(colname, unique_vals[col]))
# Now let's create a numerical index for each of them
for col, colname in columns.items():
    dataset[col + '_num'] = np.array(list(map(lambda x: list(unique_vals[col]).index(x), dataset[colname].values)))
# For Income Class we need something different...

incomes = np.nan_to_num(dataset['Income'].values) # Fixes NaNs as zeroes

fig = plt.figure()
ax = fig.add_subplot(111)

ax.hist(incomes[np.where(incomes > 0)], bins=10) # Ten bins, plus one for the "zeroes" (NaNs)
# We 'digitize' the class

non_zero = np.where(incomes > 0)
income_class = np.zeros(incomes.shape)

_, bins = np.histogram(incomes[non_zero])
income_class[non_zero] = np.digitize(incomes[non_zero], bins)

dataset['income_num'] = income_class.astype(int)
# Now let's consider the classes we want to use as input

Xcols = [c for c in dataset.columns if '_num' in c]
dataset[Xcols].head()
question = 'Who would you prefer as president of the United States, Darth Vader or Donald Trump?'

print('Given answers: ', dataset[question].unique())

dataset['ans'] = list(map(lambda x: x == 'Donald Trump', dataset[question].values))
train_set, test_set = train_test_split(dataset)
# The model, and a scan on its parameter alpha
model = MultinomialNB()
params = {
    'alpha': np.linspace(1e-10, 2.0, 20)
}

gsearch = GridSearchCV(model, params, scoring='accuracy')
gsearch.fit(train_set[Xcols], train_set['ans'])

print("Best value for alpha: {0}".format(gsearch.best_params_['alpha']))
model = MultinomialNB(**gsearch.best_params_)
cv_scores = cross_val_score(model, X=train_set[Xcols], y=train_set['ans'], cv=5, scoring='accuracy')
print("Average CV score: {0}".format(np.average(cv_scores)))
model.fit(train_set[Xcols], train_set['ans'])
pr, rec, thr = precision_recall_curve(test_set['ans'], model.predict_proba(test_set[Xcols])[:,1])
fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_title("PR curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.axhline(sum(test_set['ans'])*1.0/len(test_set), c=(0,0,0), ls='--', lw=0.5)
ax.plot(rec, pr)
