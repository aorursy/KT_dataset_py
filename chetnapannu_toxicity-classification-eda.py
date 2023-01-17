import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
print("Number of rows in training data: {!s}".format(train_data.shape[0]))

print("Number of columns in training data: {!s}".format(train_data.shape[1]))

print("'{}'".format("', '".join(train_data.columns.values)))



train_data.head(5)
print("Number of non-Nan values in each column")

train_data.count()
train_data['comment_length'] = train_data['comment_text'].str.len()

ax = train_data.hist(column='comment_length', bins=25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)

ax = ax[0]

for x in ax:

    # Add title

    x.set_title("Comment Length Distribution")



    # Set x-axis label

    x.set_xlabel("Number of characters", labelpad=20, weight='bold', size=12)



    # Set y-axis label

    x.set_ylabel("Comments", labelpad=20, weight='bold', size=12)
cols = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 'jewish', 'latino', 'male', 'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation', 'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 'white', 'funny', 'wow', 'sad', 'likes', 'disagree', 'sexual_explicit', 'identity_annotator_count', 'toxicity_annotator_count', 'comment_length']

target_col = 'target'

pairwise_correlation = []



for col in cols:

    # Compute pairwise correlation of columns, excluding NA/null values.

    corr = train_data[col].corr(train_data[target_col])

    #print("Correlation between {} and {}: {:f}".format(target_col, col, corr))

    pairwise_correlation.append({'Score': corr, 'Feature': col})



correlation_df = pd.DataFrame(pairwise_correlation)

correlation_df.sort_values(by=['Score'], ascending=False)
# Generate Kernel Density Estimate plot using Gaussian kernels.

# In statistics, kernel density estimation (KDE) is a non-parametric way to estimate the probability density function (PDF) of a random variable. This function uses Gaussian kernels and includes automatic bandwidth determination.



train_data['target'].plot.kde()