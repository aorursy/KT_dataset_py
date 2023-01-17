import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv("../input/train.csv")

train_comments = train_df['comment_text']

train_df.head(3)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# let's check out the length, may have some kind of correlation

train_df['length'] = train_df['comment_text'].apply(len)

train_df['length'].plot(bins=100, kind="hist")
colors = ('b','g','r','c','m','y')

f, axes = plt.subplots(1, 6, sharey=True, figsize=(15,5))
f.suptitle('Comment Length vs Toxic')

for i, label in enumerate(labels):
    
    data = train_df.groupby(label).mean()['length']
    
    axes[i].bar(data.index.values, data, width=0.5, color=colors[i])
    
    axes[i].set_xlabel(label)
    
def get_correlation_df_for_feature(name, feature_name):
    
    # remember labels here is defined at the top
    rows = [{label:train_df[feature_name].corr(train_df[label]) for label in labels}]

    return pd.DataFrame(rows, index=[name])


df_length_corr = get_correlation_df_for_feature('Comment Length Correlation', 'length')

df_length_corr
def count_capitals(comment):
    count = 0
    
    for letter in comment:
        if letter.isupper():
            count += 1
    
    return count

train_df['capitals'] = train_df['comment_text'].apply(count_capitals)
train_df['capital_ratio'] = train_df.apply(lambda row: float(row['capitals'])/float(row['length']), axis=1)
train_df.head(3)
for i, label in enumerate(labels):
    
    data = train_df[[label,'capital_ratio']]
    
    data.hist(column='capital_ratio', by=label, bins=10, figsize=(10,4), color=colors[i])
    
    plt.suptitle(label, x=0.5, y=1.05, ha='center', fontsize='xx-large')
capital_ratio_corr_df = get_correlation_df_for_feature('%ge Capitals in Message Correlation', 'capital_ratio')

capital_ratio_corr_df
for i, label in enumerate(labels):
    
    entries_in_cat = train_df[train_df[label] == 1]
    entries_not_in_cat = train_df[train_df[label] == 0]
    
    num_over_50_for_cat = len(entries_in_cat[entries_in_cat['capital_ratio'] > 0.5])
    num_over_50_for_not_cat = len(entries_not_in_cat[entries_not_in_cat['capital_ratio'] > 0.5])
    
    num_in_cat = len(entries_in_cat)
    num_total = len(train_df)
    
    perc_over_50_in_cat = num_over_50_for_cat / float(num_in_cat)
    perc_over_50_not_in_cat = num_over_50_for_not_cat / float(num_total - num_in_cat)
    
    print('% over 50% caps rate for ' + label + ': ' + str(perc_over_50_in_cat))
    print('% over 50% caps rate for not ' + label + ': ' + str(perc_over_50_not_in_cat) + '\n')
