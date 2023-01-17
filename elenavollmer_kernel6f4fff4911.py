# Import essential libraries
import os
import pandas as pd
import numpy as np
import sklearn

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load data
train = pd.read_csv("../input/modeltrap/train.csv",low_memory = False) #import train data
test = pd.read_csv("../input/modeltrap/test.csv",low_memory = False) #import test data
# Look at the data
train.head(10)
# Calculate loan defaults
print(f'Percentage of loans in default is {train.default.mean()*100:.2f}%') # use mean() function on default column of train dataset
# ZIP code with highest default rate
train.groupby('ZIP').default.mean().idxmax() # use groupby(), idxmax() functions to select ZIP code with highest default rate
# See which years are available
train.year.unique()
# Calculate average 
print(f"Default rate for year 0 is {train[train.year==0].default.mean()*100 :.2f}%") # use mean() function on default for first year of train dataset
# Correlation between age and income
print(f"Correlation between age and income is {train.age.corr(train.income):.4f}") # use corr() function to find correlation
# Define features to use
predictors = ['ZIP', 'rent', 'education',  'income', 'loan_size', 'payment_timing', 'job_stability','occupation']

# Define X 
X_train = pd.get_dummies(train[predictors]) # encode categorical variables to dummies using get_dummies() function

# Define y
y_train = train.default
# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Define classifier
clf = RandomForestClassifier(n_estimators=100, # number of trees in bag
                            max_depth=4, # max depth of each individual tree
                            random_state=42, # use random state 42
                            oob_score=True, # get classifier to return out-of-bag score
                            n_jobs=-1) # ask engine to use all available cores to minimise runtime

# Train classifier by calling .fit() 
clf.fit(X_train,y_train)
# Import accuracy score metric
from sklearn.metrics import accuracy_score

# Get predictions from the trained model
y_tr_pred = clf.predict(X_train) 

# Get accuracy score
print(f'The in-sample accuracy score is {accuracy_score(y_train,y_tr_pred):.4f}')
# Show the out-of-bag score
print(f"Out-of-bag score for model is {clf.oob_score_:.4f}")
# Define features (X) and target (y) for test set
X_test = pd.get_dummies(test[predictors])
y_test = test.default

# Get predictions from the trained model
y_tst_pred = clf.predict(X_test)

# Get accuracy score
print(f'The Out-of-bag accuracy score is {accuracy_score(y_test,y_tst_pred):.4f}')
# Add default predictions to test dataset
test['predicted_default'] = y_tst_pred

# Find predicted default for non-minority members
print(f"Predicted average default probability for non-minority members is {test[test.minority==0].predicted_default.mean()*100:.4f}%")
# use mean() function on default column of test dataset for non-minority members

# Find predicted default for minority members 
print(f"Predicted average default probability for minority members is {test[test.minority==1].predicted_default.mean()*100:.2f}%") 
# use mean() function on default column of test dataset for minority members
# Positive outcome = getting a loan

# Women vs men
wom_pos_rate = sum((test.sex==1) & (test.predicted_default == False))/sum(test.sex == 1) # positive rate for women 
wom_neg_rate = sum((test.sex==1) & (test.predicted_default == True))/sum(test.sex == 1) # positive rate for women 

men_pos_rate = sum((test.sex==0) & (test.predicted_default == False))/sum(test.sex == 0) # positive rate for men
men_neg_rate = sum((test.sex==0) & (test.predicted_default == True))/sum(test.sex == 0) # positive rate for men

print(f"The share of approved women is {wom_pos_rate*100:.2f}% and the share of approved men is {men_pos_rate*100:.2f}%") # compare positive rates
# Minority vs non-minority
min_pos_rate = sum((test.minority==1) & (test.predicted_default == False))/sum(test.minority == 1) # positive rate for non-minority members
min_neg_rate = sum((test.minority==1) & (test.predicted_default == True))/sum(test.minority == 1) # positive rate for non-minority members

maj_pos_rate = sum((test.minority==0) & (test.predicted_default == False))/sum(test.minority == 0) # positive rate for minority members
maj_neg_rate = sum((test.minority==0) & (test.predicted_default == True))/sum(test.minority == 0) # positive rate for minority members

print(f"The share of approved minority applicants is {min_pos_rate*100:.2f}% and the share of approved non-minority applicants is {maj_pos_rate*100:.2f}%") # compare positive rates
# Positive outcome = no default
# True positive = giving a loan to an applicant that will not default

# Minority vs non-minority
min_true_pos_rate = sum((test.minority==1) & (test.predicted_default==False) & (test.default==False))/sum((test.minority==1) & (test.predicted_default==False)) # true positive rate for minority members

min_false_pos_rate = sum((test.minority==1) & (test.predicted_default==False) & (test.default==True))/sum((test.minority==1) & (test.predicted_default==False)) # false positive rate for minority members

min_false_neg_rate = sum((test.minority==1) & (test.predicted_default==True) & (test.default==False))/sum((test.minority==1) & (test.predicted_default==True)) # false positive rate for minority members

maj_true_pos_rate = sum((test.minority==0) & (test.predicted_default==False) & (test.default==False))/sum((test.minority==0) & (test.predicted_default==False)) # true positive rate for non-minority members

maj_false_pos_rate = sum((test.minority==0) & (test.predicted_default==False) & (test.default==True))/sum((test.minority==0) & (test.predicted_default==False)) # true positive rate for non-minority members

maj_false_neg_rate = sum((test.minority==0) & (test.predicted_default==True) & (test.default==False))/sum((test.minority==0) & (test.predicted_default==True)) # true positive rate for non-minority members

print(f"The share of approved minority applicants that would repay loan is {min_true_pos_rate*100:.2f}%\
 and the share of approved non-minority applicants that would repay loan is {maj_true_pos_rate*100:.2f}%") # compare true positive rates
# Women vs men
wom_true_pos_rate = sum((test.sex==1) & (test.predicted_default == False) & (test.default == False)) / sum((test.sex==1) & (test.predicted_default == False)) # true positive rate for women
wom_false_pos_rate = sum((test.sex==1) & (test.predicted_default == True) & (test.default == False)) / sum((test.sex==1) & (test.predicted_default == False)) # true positive rate for women
wom_false_neg_rate = sum((test.sex==1) & (test.predicted_default == False) & (test.default == True)) / sum((test.sex==1) & (test.predicted_default == True)) # true positive rate for women

men_true_pos_rate = sum((test.sex==0) & (test.predicted_default == False) & (test.default == False))/sum((test.sex==0) & (test.predicted_default == False)) # true positive rate for men
men_false_pos_rate = sum((test.sex==0) & (test.predicted_default == True) & (test.default == False))/sum((test.sex==0) & (test.predicted_default == False)) # true positive rate for men
men_false_neg_rate = sum((test.sex==0) & (test.predicted_default == False) & (test.default == True))/sum((test.sex==0) & (test.predicted_default == True)) # true positive rate for men

print(f"The share of approved women applicants that would repay loan is {wom_true_pos_rate*100:.2f}%\
 and the share of approved men applicants that would repay loan is {men_true_pos_rate*100:.2f}%") # compare true positive rates
# Bar chart - Demographic Parity
import matplotlib.pyplot as plt

x = ['Minority', 'Non-minority']
height = [min_pos_rate, maj_pos_rate]
#x = ['Women', 'Men']
# height = [min_neg_rate, maj_neg_rate]
# height = [wom_pos_rate, men_pos_rate]
# height = [wom_neg_rate, men_neg_rate]

plt.bar(x, height)
plt.title('Demographic parity')
plt.ylabel('%')
plt.show
# Bar chart - Equal opportunity
import matplotlib.pyplot as plt

x = ['Minority', 'Non-minority']
height = [min_true_pos_rate, maj_true_pos_rate]
#x = ['Women', 'Men']
#height = [min_false_pos_rate, maj_false_pos_rate]
#height = [min_false_neg_rate, maj_false_neg_rate]
#height = [wom_true_pos_rate, men_true_pos_rate]
#height = [wom_false_pos_rate, men_false_pos_rate]
#height = [wom_false_neg_rate, men_false_neg_rate]

plt.bar(x, height)
plt.title('Equal opportunity')
plt.ylabel('%')
plt.show
# Correlation plot
import seaborn as sns
sns.set(style="white")
# Compute the correlation matrix
corr = train.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# How to find distribution of group within feature
# To plot, probably easier to use Excel

feature = 'ZIP'
group = 'minority'
for i in train[group].unique():
    for j in train[feature].unique():
        print(i,j)
        print( len(train[(train[group]==i) & (train[feature] == j) ]) )
        print('\n')
