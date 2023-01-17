#Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#import datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#Separate columns into groups

ID_target_comment_text = ['id', 'target', 'comment_text']



main_indicators = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']



identity_columns = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian',

                    'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',

                    'intellectual_or_learning_disability', 'jewish', 'latino', 'male',

                    'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity',

                    'other_religion', 'other_sexual_orientation', 'physical_disability',

                    'psychiatric_or_mental_illness', 'transgender', 'white']



# Only identities with more than 500 examples in the test set (combined public and private)

# will be included in the evaluation calculation. 

main_identities = ['male', 'female', 'homosexual_gay_or_lesbian',

                    'christian', 'jewish', 'muslim', 'white', 'black',

                    'psychiatric_or_mental_illness']



comment_properties = ['created_date', 'publication_id', 'parent_id', 'article_id']



reactions = ['funny', 'wow', 'sad', 'likes', 'disagree', 'rating']



annotators = ['identity_annotator_count', 'toxicity_annotator_count']
train.head(10)
#Find missing values

nan_dict = dict()

for column in train.columns:

    nan_dict[column] = train[column].isna().sum()/len(train[column])



#Find unique values

unique_values_dict = dict()

for column in train.columns:

    unique_values_dict[column] = len(train[column].unique())/train[column].count()



columns = list(unique_values_dict.keys())

y_pos = np.arange(len(columns))

unique_percentage = list(unique_values_dict.values())

nan_percentage = list(nan_dict.values())



#fig, ax = plt.subplots(figsize=(8, 10))



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10), sharey=True)

ax1.barh(y_pos, nan_percentage, color='green', ecolor='black')

ax2.barh(y_pos, unique_percentage, color='green', ecolor='black')



ax1.set_title('NaN percentage in train dataset')

ax2.set_title('Unique values percentage in train dataset')



ax1.set_xlabel('nan_percentage percentage')

ax2.set_xlabel('unique_values percentage')



ax1.set_yticks(y_pos)

ax1.set_yticklabels(columns)

ax1.invert_yaxis()





plt.show()
for column in main_indicators + main_identities:

    print('-'*5, column, '-'*5, '\n')

    comment, target, column_value = train[['comment_text', 'target', column]][train[column] == train[column].max()].iloc[0]

    print('target:', target)

    print(str(column)+':', column_value)

    print(comment, '\n')
test.head()
#BIGGER IDENTITY GROUPS

#This is how I would classify the columns. You might desagree. Feel free to change.

#Physical disability and other disability have a single category, so it doesn't need to be grouped.

religion_columns = ['atheist', 'buddhist', 'christian', 'hindu', 'jewish', 'muslim', 'other_religion']

gender_columns = ['male', 'female']

sexuality_columns = ['heterosexual', 'homosexual_gay_or_lesbian', 'other_gender', 'other_sexual_orientation', 'transgender']

ethinicity_columns = ['black', 'latino', 'white', 'asian', 'other_race_or_ethnicity']

mental_disability_columns = ['intellectual_or_learning_disability', 'psychiatric_or_mental_illness']



train['is_religion_related'] = (train[religion_columns] > 0).sum(axis=1)

train['is_gender_related'] = (train[gender_columns] > 0).sum(axis=1)

train['is_sexuality_related'] = (train[sexuality_columns] > 0).sum(axis=1)

train['is_ethinicity_related'] = (train[ethinicity_columns] > 0).sum(axis=1)

train['is_mental_disability_related'] = (train[mental_disability_columns] > 0).sum(axis=1)



#LIKES RATIO

pd.options.mode.chained_assignment = None  # desible copy warning - default='warn'

train['disagree_to_likes'] = 0

train['funny_to_likes'] = 0

train['wow_to_likes'] = 0

train['sad_to_likes'] = 0

train['all_to_likes'] = 0

train['disagree_to_likes'][train['likes'] > 0] = train['disagree'][train['likes'] > 0] / train['likes'][train['likes'] > 0]

train['funny_to_likes'][train['likes'] > 0] = train['funny'][train['likes'] > 0] / train['likes'][train['likes'] > 0]

train['wow_to_likes'][train['likes'] > 0] = train['wow'][train['likes'] > 0] / train['likes'][train['likes'] > 0]

train['sad_to_likes'][train['likes'] > 0] = train['sad'][train['likes'] > 0] /train['likes'][train['likes'] > 0]

train['all_to_likes'][train['likes'] > 0] = train[['disagree', 'funny', 'wow', 'sad']][train['likes'] > 0].sum(axis = 1) / train['likes'][train['likes'] > 0]



#COMMENTS PROPERTIES

#rating

train['rating'] = train['rating'].apply(lambda x: 1 if x =='approved' else 0)



#has_parent_id

train['has_parent_id'] = train['parent_id'].apply(lambda x: 1 if x > 0 else 0)



#date

train['created_date'] = pd.to_datetime(train['created_date'])

earliest_date = train['created_date'].min()

train['created_date'] = train['created_date'].apply(lambda x: (x - earliest_date).days)



#TOXICITY RELATED TO IDENTITY



#identity_degree

train['identity_degree'] = (train[identity_columns] > 0).sum(axis=1)



#identity_weight

train['identity_weight'] = train[identity_columns].sum(axis=1)



#is_identity_related

train['is_identity_related'] = train['identity_degree'].apply(lambda x: 1 if x>0 else 0)



#is_main_identity_related

train['in_main_identity_related'] = 0

for identity in main_identities:

    train['in_main_identity_related'] += train[identity].apply(lambda x: 1 if x>0 else 0)



train['in_main_identity_related'] = train['in_main_identity_related'].apply(lambda x: 1 if x >0 else 0)



#FILLING NaN

train.fillna(0, inplace = True)
#CORRELATION HEAT MAP

columns = [column for column in train.columns if column not in ['id', 'comment_text', 'created_date']]



colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train[columns].astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=False)
#If you want to see the correlation matrix, uncomment the code below

correlation_matrix = train[columns].corr()

#correlation_matrix
#Plots target correlation

target_correlation = correlation_matrix['target'].copy()

target_correlation = target_correlation.sort_values(ascending = False)



plt.rcdefaults()

fig, ax = plt.subplots(figsize=(4, 10))



# Example data

columns = list(target_correlation.index)

y_pos = np.arange(len(columns))

nan_percentage = list(target_correlation.values)



ax.barh(y_pos, nan_percentage, color='green', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(columns, fontsize = 9)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Correlation')

ax.set_title('Target correlation')

plt.show()
columns_to_make_binary = ['target'] + main_indicators



train_binary = train[columns_to_make_binary].copy()



binary_threshold = 0.5

for column in columns_to_make_binary:

    train_binary[column] = train_binary[column].apply(lambda x: 1 if x >= binary_threshold else 0)



fig, ax = plt.subplots(figsize=(10, 4))

for column in main_indicators:

    ax.bar(column, (train_binary[column] == 1).sum())



ax.set_title('Main indicators occurrencies')

plt.tight_layout()

plt.show()
#FIRST GRAPH

train_binary = train[columns_to_make_binary].copy()



train_binary['target'] = train_binary['target'].apply(lambda x: 1 if x >= 0.5 else 0)



all_positives = (train_binary['target'] == 1).sum()



true_positive_dict = dict()

false_negative_dict = dict()



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))



for column in main_indicators:

    true_positive_dict[column] = []

    false_negative_dict[column] = []



    for binary_threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        train_binary[column] = train[column].apply(lambda x: 1 if x >= binary_threshold else 0)

        all_negatives = (train_binary['target'][train[column] > 0] == 0).sum()

        

        true_positives = (train_binary[column][train_binary['target'] == 1] == 1).sum()

        false_positive = (train_binary[column][(train_binary['target'] == 0) & (train[column] > 0)] == 1).sum()

        

        true_positive_dict[column].append(true_positives/all_positives)

        false_negative_dict[column].append(false_positive/all_negatives)

    

        ax1.annotate(binary_threshold, (false_positive/all_negatives, true_positives/all_positives))

    

    ax1.plot(false_negative_dict[column], true_positive_dict[column], label = column)

    

ax1.set_ylabel('True Positive Rate', fontsize=12)

ax1.set_xlabel('False Positive Rate', fontsize=12)

    



#SECOND GRAPH

train_temp = train[columns_to_make_binary][train['in_main_identity_related'] > 0].copy()



train_binary = train_temp.copy()

train_binary['target'] = train_binary['target'].apply(lambda x: 1 if x >= 0.5 else 0)



all_positives = (train_binary['target'] == 1).sum()



true_positive_dict = dict()

false_negative_dict = dict()



for column in main_indicators:

    true_positive_dict[column] = []

    false_negative_dict[column] = []



    for binary_threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        train_binary[column] = train[column].apply(lambda x: 1 if x >= binary_threshold else 0)

        all_negatives = (train_binary['target'][train_temp[column] > 0] == 0).sum()

        

        true_positives = (train_binary[column][train_binary['target'] == 1] == 1).sum()

        false_positive = (train_binary[column][(train_binary['target'] == 0) & (train_temp[column] > 0)] == 1).sum()



        true_positive_dict[column].append(true_positives/all_positives)

        false_negative_dict[column].append(false_positive/all_negatives)

        ax2.annotate(binary_threshold, (false_positive/all_negatives, true_positives/all_positives))

            

    ax2.plot(false_negative_dict[column], true_positive_dict[column], label = column)





ax1.set_title('Main indicators ROC-AUC all comments')

ax2.set_title('Main Indicators ROC-AUC comments with main identities')

ax2.set_xlabel('False Positive Rate', fontsize=12)

ax2.legend(bbox_to_anchor=(1.10, 1), loc=2, borderaxespad=0.)    

plt.show()
train_binned = train.copy()

train_binned['target_binned'] = pd.cut(train_binned['target'], bins = 10)

train_binned = train_binned.groupby('target_binned').mean().copy()



identity_big_groups = ['is_identity_related', 'is_religion_related', 'is_gender_related', 'is_sexuality_related',

            'is_ethinicity_related', 'is_mental_disability_related', 'identity_degree',

           'identity_weight']



likes_ratio = ['disagree_to_likes', 'funny_to_likes', 'wow_to_likes', 'sad_to_likes', 'all_to_likes', 'rating']

titles = ['Identity Big Groups', 'Main Identities', 'Reactions', 'Likes Ratio', 'Annotators']

for columns, title in zip([identity_big_groups, main_identities, reactions, likes_ratio, annotators], titles):

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()



    ax1.hist(train['target'], color = 'grey', bins = np.array(range(11))/10, label = 'Target Histogram')

    ax1.set_ylabel('Target Frequency', fontsize=12)

    for i in range(len(columns)):

        ax2.plot(train_binned['target'], train_binned[columns[i]], label = columns[i])

    ax1.legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)

    ax2.legend(bbox_to_anchor=(1.15, 0.9), loc=2, borderaxespad=0.)

    ax2.set_ylabel('Feature Value', fontsize=12)

    ax1.set_xlabel('Target', fontsize=12)

    fig.suptitle(title, fontsize=20)

    plt.show()
train_binary = train.copy()



columns_to_make_binary = ['target'] + main_indicators + main_identities



for column in columns_to_make_binary:

    train_binary[column] = train_binary[column].apply(lambda x: 1 if x>= 0.5 else 0)

    

main_col="target"

corr_mats=[]

for other_col in columns_to_make_binary:

    if other_col == 'target':

        continue

    confusion_matrix = pd.crosstab(train_binary[main_col], train_binary[other_col])

    corr_mats.append(confusion_matrix)

out = pd.concat(corr_mats,axis=1,keys=columns_to_make_binary[1:])



#cell highlighting

out = out.style.highlight_min(axis=0)

out.columns.names = [None, 'Classification']

out