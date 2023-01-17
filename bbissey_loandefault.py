#commented code without "bb: " preceding was written by UCFAI datascience club

#commented code with "bb: " was written by brett bissey. I wasn't present last week 

#    so I'm trying to catch up using this walkthrough.





import pandas as pd

import matplotlib.pyplot as plt





train = pd.read_csv("../input/ucfai-dsg-fa19-default/train.csv")

test = pd.read_csv("../input/ucfai-dsg-fa19-default/test.csv")



#train = pd.read_csv("train.csv")

#test = pd.read_csv("test.csv")

ID_test = test['id']
train['GOOD_STANDING'].value_counts()
import numpy as np



# Give me the -length - of the subset of -train- made up of entries with GOOD_STANDING == 0 

# In otherwords, how many bad loans are there?

bad_standing_len = len(train[train["GOOD_STANDING"] == 0])



# Give me the index of the subset of train where good_standing == 1 

# In otherwords, give me the index of all the good loans

good_standing_index = train[train['GOOD_STANDING'] == 1].index



# Randomly choose indices of good loans equal to the number of bad loans

random_index = np.random.choice(good_standing_index, bad_standing_len, replace=False)



# Give me the index of all the bad loans in train

bad_standing_index = train[train['GOOD_STANDING'] == 0].index



# Concatonate the indices of bad loans, and our randomly sampled good loans

under_sample_index = np.concatenate([bad_standing_index, random_index])



# Create a new pandas dataframe made only of these indices 

under_sample = train.loc[under_sample_index]



#bb: We are randomly sampling indices of good loans from our dataset 

#    so that bad loans and good loans have == sample sizes. 



# Make sure it works, and make this undersampled dataframe our train

print(train['GOOD_STANDING'].value_counts())

print(under_sample['GOOD_STANDING'].value_counts())

train = under_sample
print(train.head())

train_len = len(train)

dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

dataset.describe()
import seaborn as sns

%matplotlib inline 

#sns.relplot(x="loan_amnt", y="GOOD_STANDING", data=dataset);



sns.barplot(x='grade',y='GOOD_STANDING',data=dataset)



#train = pd.read_csv("train.csv")

g = sns.FacetGrid(dataset, row='GOOD_STANDING', col='grade')

g.map(sns.distplot, "loan_amnt")

plt.show()
g = sns.FacetGrid(dataset, row='GOOD_STANDING', col='grade')

g.map(sns.distplot, "funded_amnt")

plt.show()
dataset['issue_d'] = pd.to_datetime(dataset['issue_d']) 



#sns.barplot(x='application_type',y='GOOD_STANDING',data=dataset)
dataset.select_dtypes(exclude=['int', 'float']).columns
missingData = dataset.isna()
numMissing = missingData.sum()

numMissing / len(dataset)
dataset.info()


sns.violinplot(x="GOOD_STANDING", y="loan_amnt", data=dataset);
sns.hist(dataset.grade)
sns.distplot(dataset.funded_amnt_inv)