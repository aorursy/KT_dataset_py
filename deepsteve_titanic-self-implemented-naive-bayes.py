import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
# convert the Sex column to zeros and ones
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
train.head()
# Check the correlations between all variables
sn.heatmap(train.corr(), annot=True, center=0, cmap = "RdBu")
plt.show()
# Check for NaN values in the data columns
for column in train.columns:
    print(column, train[column].isna().sum())

# Age and Cabin both have NaNs. Neither have strong correlation so lets leave them out for now. 
# One possible idea for the age is to split the ages into child/adult, since children were given preference
# but then what do we set as the cut-off for a child?
n_obs = len(train.index)
print(n_obs)
survived = train.loc[train['Survived'] == 1]
prob_c1 = len(survived.index)/n_obs # Probability of survival
print("Prob of survival : ", prob_c1)
died = train.loc[train['Survived'] == 0]
prob_c0 = len(died.index)/n_obs
print("Prob of death : ", prob_c0) # Probability of death
female_and_survived = train.loc[(train['Survived'] == 1) & (train['Sex'] == 1)]
prob_female_given_survived = len(female_and_survived.index)/len(survived.index)
print("Conditional probability of being female and surviving : ", prob_female_given_survived)

female_and_died = train.loc[(train['Survived'] == 0) & (train['Sex'] == 1)]
prob_female_given_death = len(female_and_died.index)/len(died.index)
print("Conditional probability of being female and dieing : ", prob_female_given_death)

male_and_survived = train.loc[(train['Survived'] == 1) & (train['Sex'] == 0)]
prob_male_given_survived = len(male_and_survived.index)/len(survived.index)
print("Conditional probability of being male and surviving : ", prob_male_given_survived)

male_and_died = train.loc[(train['Survived'] == 0) & (train['Sex'] == 0)]
prob_male_given_death = len(male_and_died.index)/len(died.index)
print("Conditional probability of being male and dieing : ", prob_male_given_death)
# Check distribution of Fare
plt.hist(train.Fare)
print(np.max(np.log(train.Fare)))
print(np.min(np.log(train.Fare)))
# min is -inf, so we need to fix this. First find the index where this occurs.
print(np.where(np.log(train.Fare) == np.min(np.log(train.Fare))))
# Check the value at these row for the fare
rows = np.where(np.log(train.Fare) == np.min(np.log(train.Fare)))
for row in rows:
    print(train.iloc[row, 9])
# They are all zero, so lets give it a new value. Lets check what class they were in, and take the mean value
# for the class as the fare value
class_ = train.iloc[rows[0], 2].values
print(class_)
for clas, row in zip(class_, rows[0]):
    # get the mean
    Pclass = train.loc[(train['Pclass'] == clas)]
    c_mean = np.mean(Pclass['Fare'])
    # assign the value to the proper row
    train.iloc[row, 9] = c_mean
# Now we check again the distribution
print(np.max(np.log(train.Fare)))
print(np.min(np.log(train.Fare)))
plt.hist(np.log(train.Fare))
from scipy.stats import normaltest
alpha = 0.005
s, p = normaltest(np.log(train.Fare))
if p < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")
# Then a function for the fare variable which we assume follows a gaussian distribution
def gaussian(y, fare):
    temp = train.loc[(train['Survived'] == y)]

    mean = np.mean(np.log(temp.Fare))
    sd = np.std(np.log(temp.Fare))
    return (np.exp((((np.log(fare)-mean)/sd)**2)*-0.5)) / (sd * np.sqrt(2*np.pi))
# ceate function which returns the proper outcome given a sex input
def P_sex_given_outcome(y, sex):
    joint = train.loc[(train['Survived'] == y) & (train['Sex'] == sex)]
    if y == 0:
        return len(joint.index)/len(died.index)
    else:
        return len(joint.index)/len(survived.index)
# First we need P(class)
def P_c(y):
    temp = train.loc[train['Survived'] == y]
    return len(temp.index)/n_obs
def bayes(fare_value, sex, y):
    """Instead of taking the multiplication of likelihoods and probabilities, we take the sum of their logs.

    """
    return np.log(P_sex_given_outcome(y, sex)) + np.log(gaussian(y, fare_value)) + np.log(P_c(y))
def classification(fare_paid, sex, n_classes = 2):
    """Returns the class which has the highest score given.
    
    Parameters
    ----------
    fare_paid : float
        The amount of money paid by the passenger
    sex : int
        The sex of the passenger. 1 = female, 0 = male
    n_classes : int
        Number of classes in the classification.
    
    """
    return np.argmax(np.array([[bayes(fare_paid, sex, i)] for i in range(n_classes)]))
correct = 0
for i, row in train.iterrows():
    fare = round(row['Fare'])
    sex = row['Sex']
    try:
        pred = classification(fare, sex)
    except Exception as e:
        #print(e)
        pass
        
    if row['Survived'] == pred:
        #print('Correctly predicted as %d \n' % pred)
        correct += 1
        #percent_correct = (correct/(i+1))*100
        #print("%.1f percent correctly predicted by the model" % percent_correct)
    else:
        pass
        #percent_correct = (correct/(i+1))*100
        #print('Incorrectly predicted as %d \n' % pred)
        #print("%.1f percent correctly predicted by the model" % percent_correct)
print(correct/len(train.index))
test = pd.read_csv('/kaggle/input/titanic/test.csv')
# convert the Sex column to zeros and ones
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
test.head()
# See where fare is equal to zero again
rows = np.where(test.Fare == 0)
for row in rows:
    print(test.iloc[row, 8])
# Check what class they were in, and take the mean value for that class as the fare value
class_ = test.iloc[rows[0], 1].values
print(class_)
for clas, row in zip(class_, rows[0]):
    # get the mean
    Pclass = test.loc[(test['Pclass'] == clas)]
    c_mean = np.mean(Pclass['Fare'])
    # assign the value to the proper row
    test.iloc[row, 8] = c_mean
# Check for NaN values in the data columns
for column in test.columns:
    print(column, test[column].isna().sum())
# assign the fare as the mean of the class the passenger was in

# find the index of the NaN entry
rows = test['Fare'].index[test['Fare'].apply(np.isnan)]
print(rows)
class_ = test.iloc[rows, 1].values
print(class_)
for clas, row in zip(class_, rows):
    # get the mean
    Pclass = test.loc[(test['Pclass'] == clas)]
    c_mean = np.mean(Pclass['Fare'])
    # assign the value to the proper row
    test.iloc[row, 8] = c_mean
#check tat the value was fixed
test.iloc[rows[0], 8]
# make final predictions
predictions = []
pass_ids = []
for i, row in test.iterrows():
    fare = round(row['Fare'])
    sex = row['Sex']
    predictions.append(classification(fare, sex))
    pass_ids.append(row['PassengerId'])
predictions[0:6]
import csv
rows = zip(*[predictions, pass_ids])
        
with open('/kaggle/working/output.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for row in rows:
        wr.writerow(row)
