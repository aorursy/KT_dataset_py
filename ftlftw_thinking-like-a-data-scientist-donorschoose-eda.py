import pandas as pd, numpy as np

import os,sys

%matplotlib inline
from IPython.display import display, HTML
import matplotlib.pyplot as plt, seaborn as sns

datadir = '../input/'
resources = pd.read_csv(datadir + 'resources.csv')
sample_submission = pd.read_csv(datadir + 'sample_submission.csv', index_col='id')
test = pd.read_csv(datadir + 'test.csv', index_col='id')
train = pd.read_csv(datadir + 'train.csv', index_col='id')
print("Sample submission has %d rows. "
      "Validation - this should be the same as length of test set, which has %d." % (
        len(sample_submission), len(test)))
print("Sample submission has same items as test set:" , set(sample_submission.index) == set(test.index))
print("Sample submission has same order as test set:" , all(sample_submission.index == test.index))
print("Sample submission is disjoint from train indices:", len(set(sample_submission.index) & set(train.index)) == 0)
print("Sample submission is subset of resources ids:", len(set(sample_submission.index) - set(resources['id'])) == 0)

display(sample_submission.head())
display(sample_submission.describe())
display(resources.head())
display(resources.tail())
# First, just check for missing values in columns:
for col in resources.columns:
  print("%s has %d missing values." % (col, resources[col].isnull().sum()))
resources['description'].fillna('', inplace=True)
resources['description'].isnull().sum() # Should be 0 now
# Total cost: quantity x price
resources['total_cost'] = resources['quantity']*resources['price'] 

# After grouping by id, we can sum the price to get the total cost of a proposal and count the items in it.
proposals = pd.DataFrame({
  'summed_cost_in_proposal':resources.groupby('id').sum()['total_cost'],
  'num_items_in_proposal':resources.groupby('id').count()['total_cost']
})

# Length of description in characters, treating it as string.
resources['desc_len'] = resources['description'].str.len()
resources['desc_len_words'] = resources['description'].map(lambda x: len( x.split(' ')))

# Stats for all of the numeric columns so far
display(resources.describe())
display(proposals.describe())
from pandas.api.types import is_numeric_dtype
for df in resources, proposals:
  for col in df.columns:
    values = df[col]
    if is_numeric_dtype(values):
      ax = sns.distplot(np.log1p(values), kde=False, axlabel='log_'+col)
      plt.show(ax)
display(resources[ resources['quantity'] > 100])
display(resources[ resources['quantity'] > 500])
for col in ['price','total_cost','desc_len_words', 'desc_len']:
  display( resources.sort_values(by=col).head(10))
  display( resources.sort_values(by=col).tail(10))
cheapest_total_id = proposals['summed_cost_in_proposal'].idxmin()
most_expensive_total_id = proposals['summed_cost_in_proposal'].idxmax()
biggest_proposal_id = proposals['num_items_in_proposal'].idxmax()

display(resources[resources['id']==cheapest_total_id])
display(resources[resources['id']==most_expensive_total_id])
display(resources[resources['id']==biggest_proposal_id])
display(train.head())
display(test.head())
print("%d train examples, %d test examples" % (len(train), len(test)))
print("How many examples in train have essays 3 and 4?")
display(train['project_essay_3'].isnull().value_counts())
display(train['project_essay_4'].isnull().value_counts())
print("How many examples in test have essays 3 and 4?")
display(test['project_essay_3'].isnull().value_counts())
display(test['project_essay_4'].isnull().value_counts())
train['project_is_approved'].value_counts()
categorical_variables = ['teacher_prefix','school_state','project_grade_category', 'project_subject_categories','project_subject_subcategories']
for var in categorical_variables:
  print("Exploring", var)
  print("There are %d categories in this column in training dataset:" % train[var].nunique() )
  print("There are %d categories in this column in test dataset:" % test[var].nunique() )
  
  counts = train[var].fillna("MISSING").value_counts()
  freqs =  counts/counts.sum()
  frac_accepted = train.fillna("MISSING").groupby(var).mean()['project_is_approved']
  result = pd.DataFrame({'count':counts, 'frequency':freqs, 'acceptance_rate':frac_accepted, 'value':counts.index},
                        index=counts.index)
  
  print("Value counts in train:" )
  display(result) # Show the values
  
  # plot the frequency
  result.plot.bar(x='value',y='frequency')
  plt.xlabel(var)
  plt.ylabel('frequency')
  plt.show()
  plt.close()
  
  # plot acceptance rate
  result.plot.bar(x='value',y='acceptance_rate')
  plt.xlabel(var)
  plt.ylabel('acceptance rate')
  plt.show()
  plt.close()
  
  print("Value counts in test:")
  counts = test[var].fillna("MISSING").value_counts()
  freqs =  counts/counts.sum()
  result = pd.DataFrame({'count':counts, 'frequency':freqs,'value':counts.index})
  display(result)
  
  result.plot.bar(x='value',y='frequency')
  plt.xlabel(var)
  plt.ylabel('frequency')
  plt.show()
  plt.close()
    
display(train['teacher_number_of_previously_posted_projects'].describe())
print("Correlation:", train[['teacher_number_of_previously_posted_projects', 'project_is_approved']].corr().iloc[0]['project_is_approved'])
# First, get the folds
from sklearn.model_selection import KFold
num_folds = 5

all_indices = pd.Series(train.index)
kf = KFold(n_splits = num_folds, shuffle=True, random_state = 12345)
fold_indices = [
                [all_indices[x] for x in train_test]
                for train_test in kf.split(all_indices)
                ]
# fold_indices is a list of pairs, train and test, to be used for cross_validation
# next, put making features into a function - that can be called on both train/test, and train/validate data.
def featurize_train_test(train, test, resources):
  """
  This function takes as input a training dataframe, a test dataframe, and the resources dataframe.
  It returns a train dataframe, test dataframe, and targets, ready for input into an ML model.
  
  For the EDA, I'm only going to pick out a few simple features and run logistic regression. 
  So the features I'm using are the number of previously posted projects, the grade, total cost, number of items.
  I'm deliberately using a lot of things from the resources file to ensure that I catch any information leaks early.
  """
  train_labels = train['project_is_approved'].values
  
  # Set this up to work on both train and test data, for validation and real use
  if 'project_is_approved' in test.columns:
    test_labels = test['project_is_approved']
  else:
    test_labels = pd.Series(index=test.index, data=np.nan)
  
  train_features = pd.DataFrame(index=train.index)
  test_features = pd.DataFrame(index=test.index)
  
  train_features['prev_proj'] = train['teacher_number_of_previously_posted_projects']
  test_features['prev_proj'] = test['teacher_number_of_previously_posted_projects']
  
  def grade_to_int(x):
    return {'Grades PreK-2':0, "Grades 3-5": 1, "Grades 6-8": 2, "Grades 9-12":3}[x]
  
  train_features['grade_int'] = train['project_grade_category'].map(grade_to_int)
  test_features['grade_int'] = test['project_grade_category'].map(grade_to_int)
  
  resources['total_cost'] = resources['quantity']*resources['price']
  proposals = pd.DataFrame({
    'summed_cost_in_proposal':resources.groupby('id').sum()['total_cost'],
    'num_items_in_proposal':resources.groupby('id').count()['total_cost']
  })
  
  train_features['total_cost'] = [proposals.loc[i, 'summed_cost_in_proposal'] for i in train_features.index]
  test_features['total_cost'] = [proposals.loc[i, 'summed_cost_in_proposal'] for i in test_features.index]
  train_features['num_items'] = [proposals.loc[i, 'num_items_in_proposal'] for i in train_features.index]
  test_features['num_items'] = [proposals.loc[i, 'num_items_in_proposal'] for i in test_features.index]
  
  return (train_features.values, train_labels, test_features.values, test_labels)
  
# Make a function that makes a basic model
from sklearn.linear_model import LogisticRegression
def train_test_basic_model(train_features, train_labels, test_features):
  """
  This function should train the model on the training data, 
  test it on the test data, and return the predicted probabilities.
  """
  model = LogisticRegression()
  model.fit(train_features, train_labels)
  return model.predict_proba(test_features)[:,1]

from sklearn.metrics import roc_auc_score
def evaluate(true_labels, predicted_probs):
  """
  Possibly the most important function you'll write - evaluation. 
  Takes predicted probabilities and true labels, and gets the AUC.
  See https://www.kaggle.com/c/donorschoose-application-screening#evaluation . 
  
  Here, our evaluation function is trivial and passed on to sklearn.
  """
  return roc_auc_score(true_labels, predicted_probs)

aucs = []
for train_ind, test_ind in fold_indices:
  tr, tr_l, te, te_l = featurize_train_test(train.loc[train_ind], train.loc[test_ind], resources)
  predicted = train_test_basic_model(tr, tr_l, te)
  aucs.append(evaluate(te_l, predicted))

print("The crossvalidated AUC is", np.mean(aucs))
tr, tr_l, te, te_l = featurize_train_test(train, test, resources)
predicted = train_test_basic_model(tr, tr_l, te)

new_submission = sample_submission.copy()
new_submission['project_is_approved'] = predicted

new_submission.to_csv("cv_test_lr_predictions.csv")