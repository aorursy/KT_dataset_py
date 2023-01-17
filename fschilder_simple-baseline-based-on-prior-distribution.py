# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test_stage_1 = pd.read_csv("../input/test_stage_1.tsv", sep="\t")

test_stage_2 = pd.read_csv("../input/test_stage_2.tsv", sep="\t")
gap_test = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", delimiter='\t')

gap_valid = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv", delimiter='\t')
gap_test[0:5]
def get_prior(df):

    # count how many times neither antecedent is correct for the pronoun

    Neither_count = len(df) - sum(df["A-coref"]  |  df["B-coref"])

    # count the  A coreferences

    A_count = sum(df["A-coref"])

    # count the B coreferences

    B_count = sum(df["B-coref"])

    # total number of samples

    test_total = len(df)

    # compute the prior probabilities of the three classes

    Neither_prior = Neither_count/test_total

    A_prior = A_count/test_total

    B_prior = B_count/test_total

    print("Prior probabilities:")

    print("Neither: "+str(Neither_prior),"A: "+str(A_prior),"B: "+str(B_prior))

    # sanity check whether everything adds up

    assert Neither_count + A_count + B_count == test_total

    return A_prior, B_prior, Neither_prior



A_prior,B_prior,Neither_prior = get_prior(gap_test)



sample_submission = pd.read_csv("../input/sample_submission_stage_1.csv")

def assign_prior(df):

    sub = pd.DataFrame()

    for index, row in df.iterrows():

        sub.loc[index, "ID"] = row["ID"]

        sub.loc[index, "A"] = A_prior

        sub.loc[index, "B"] = B_prior

        sub.loc[index, "NEITHER"] = Neither_prior

    return sub
train = assign_prior(gap_test)

valid = assign_prior(gap_valid)
from sklearn.metrics import log_loss



def get_gold(df):

    gold = []

    for index, row in df.iterrows():

        if (row["A-coref"]):

            gold.append("A") 

        else:

            if (row["B-coref"]):

                gold.append("B") 

            else:

                gold.append("NEITHER")

    return gold
train_gold = get_gold(gap_test)

valid_gold = get_gold(gap_valid)
train_pred = train[["A","B","NEITHER"]]

log_loss(train_gold,train_pred)
valid_pred = valid[["A","B","NEITHER"]]

log_loss(valid_gold,valid_pred)
sub1 = assign_prior(test_stage_1)
sub1[0:4]
sub1.to_csv("submission_1.csv", index=False)
set(gap_test["Pronoun"]).union(set(gap_valid["Pronoun"])).union(set(test_stage_1["Pronoun"]))
female_pronouns = ['she','her','hers']

male_pronouns = ['he','him','his']
female_gap_test = gap_test[gap_test["Pronoun"].str.lower().isin(female_pronouns)]

male_gap_test = gap_test[gap_test["Pronoun"].str.lower().isin(male_pronouns)]

female_gap_valid = gap_valid[gap_valid["Pronoun"].str.lower().isin(female_pronouns)]

male_gap_valid = gap_valid[gap_valid["Pronoun"].str.lower().isin(male_pronouns)]

len(female_gap_test) == len(male_gap_test)
len(female_gap_valid) == len(male_gap_valid)
train_female = assign_prior(female_gap_test)

train_male = assign_prior(male_gap_test)

valid_female = assign_prior(female_gap_valid)

valid_male = assign_prior(male_gap_valid)
train_gold_female = get_gold(female_gap_test)

train_gold_male = get_gold(male_gap_test)
train_pred_female = train_female[["A","B","NEITHER"]]

log_loss(train_gold_female,train_pred_female)
train_pred_male = train_male[["A","B","NEITHER"]]

log_loss(train_gold_male,train_pred_male)
valid_gold_female = get_gold(female_gap_valid)

valid_gold_male = get_gold(male_gap_valid)
valid_pred_female = valid_female[["A","B","NEITHER"]]

log_loss(valid_gold_female,valid_pred_female)
valid_pred_male = valid_male[["A","B","NEITHER"]]

log_loss(valid_gold_male,valid_pred_male)
female_test_stage_2 = test_stage_2[test_stage_2["Pronoun"].str.lower().isin(female_pronouns)]

male_test_stage_2 = test_stage_2[test_stage_2["Pronoun"].str.lower().isin(male_pronouns)]
len(female_test_stage_2)
len(male_test_stage_2)
sub2 = assign_prior(test_stage_2)
sub2.head()
sub2.to_csv("submission.csv", index=False)