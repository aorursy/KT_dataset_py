# Packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
import sklearn.metrics as metrics
import os # to access data files (found in the "../input/" directory)
import re  #regular expressions

# More Packages from https://colab.research.google.com/notebooks/mlcc/sparsity_and_l1_regularization.ipynb?hl=en#scrollTo=pb7rSrLKIjnS
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

# Data Files  NOTE:  My file paths here may look different since I'm using files from multiple data sources 
training_dataset = pd.read_csv('../input/donorschoose-application-screening/train.csv', sep=',')
resources_dataset = pd.read_csv('../input/donorschoose-application-screening/resources.csv', sep=',')
test_dataset = pd.read_csv('../input/donorschoose-application-screening/test.csv', sep=',')
#English Word Frequency
dictionary_dataset = pd.read_csv('../input/english-word-frequency/unigram_freq.csv', sep=',')

#Unix Words
#https://stackoverflow.com/questions/3277503/in-python-how-do-i-read-a-file-line-by-line-into-a-list
with open('../input/unix-words/words/en') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

unix_set = set(content)

#Brown Corpus
USING_BROWN = False  #Though Brown has "1 million words," most of them are NOT unique.  So this wasn't that helpful
if USING_BROWN:
    #https://www.kaggle.com/alvations/testing-1000-files-datasets-from-nltk
    #^Code to get data into pandas dataframe:
    from nltk.corpus import (LazyCorpusLoader, CategorizedTaggedCorpusReader)

    import nltk
    # Removing the original path
    if '/usr/share/nltk_data' in nltk.data.path:
        nltk.data.path.remove('/usr/share/nltk_data')
    nltk.data.path.append('../input/brown-corpus/brown')
    nltk.data.path

    brown = LazyCorpusLoader('brown', CategorizedTaggedCorpusReader, r'c[a-z]\d\d',
                             cat_file='cats.txt', tagset='brown', encoding="ascii",
                            nltk_data_subdir='brown-corpus/brown')

    list_of_lists_of_tuples = brown.tagged_sents()

    list_sents = []
    for list_of_tuples in list_of_lists_of_tuples:
        sentence = " ".join([tupl[0] for tupl in list_of_tuples])
        list_sents.append(sentence)

    list_sents

    pd_brown = pd.DataFrame({'sentence': list_sents})
    
    #some text formatting:
    pd_brown['sentence'] = pd_brown['sentence'].replace('(\-)|(/)|(\.\.\.\.)|(\.\.\.)', ' ', regex=True)
    pd_brown['sentence'] = pd_brown['sentence'].replace('["#$%&\'()*+,.:!;<>?@\\^_`{|}~]', '', regex=True)
    pd_brown['sentence'] = pd_brown['sentence'].replace('(   )|(  )', ' ', regex=True)
    pd_brown['sentence'] = pd_brown['sentence'].str.lower()
    pd_brown[0:10]
resources_dataset[0:9]
resources_dataset[37602:37612]
resources_dataset.isnull().any()
resources_dataset.loc[resources_dataset["description"].isnull()]
resources_dataset = resources_dataset.fillna({'description' : 'no_detail'})
resources_dataset.isnull().any()
# Make a cost column (quantity * price)
resources_dataset['cost'] = resources_dataset['quantity'] * resources_dataset['price']
resources_dataset[0:9]
#create a total_cost column

grouped_ids = resources_dataset.groupby(['id'], as_index=False)
resources_condensed = grouped_ids.agg({'cost' : 'sum'}).rename(columns={'cost' : 'total_cost'})
#resources_condensed.loc[resources_condensed['id'] == 'p069063']
resources_condensed[69060:69065]
#Join together all of the columns
group_the_ids = resources_dataset.groupby(['id'], as_index=False)
all_condensed = grouped_ids.agg({'description' : lambda x: ' '.join(x), #there's no simple single keyword option like 'sum'
                                 'quantity' : 'sum',
                                 'cost' : 'sum'}).rename(columns={'description' : 'full_description',
                                                                  'quantity' : 'total_quantity',
                                                                  'cost' : 'total_cost'})
all_condensed[69060:69065]
#Show that p069063's full_description actually contains the joined text:
entry = all_condensed.loc[all_condensed['id'] == 'p069063'].reset_index()
entry.loc[0, 'full_description']
combined_training_dataset = pd.merge(training_dataset, all_condensed, on='id')
combined_training_dataset[0:5]
combined_test_dataset = pd.merge(test_dataset, all_condensed, on='id')
combined_test_dataset[0:5]
combined_training_dataset['teacher_prefix'] = combined_training_dataset['teacher_prefix'].fillna('none')
combined_test_dataset['teacher_prefix'] = combined_test_dataset['teacher_prefix'].fillna('none')

combined_training_dataset = combined_training_dataset.fillna('')
combined_test_dataset = combined_test_dataset.fillna('')
combined_training_dataset.loc[combined_training_dataset['project_essay_4'] != '', 'project_essay_1'] = combined_training_dataset['project_essay_1'] + ' ' + combined_training_dataset['project_essay_2']
combined_training_dataset.loc[combined_training_dataset['project_essay_4'] != '', 'project_essay_2'] = combined_training_dataset['project_essay_3'] + ' ' + combined_training_dataset['project_essay_4']

#and do the same for the test data:
combined_test_dataset.loc[combined_test_dataset['project_essay_4'] != '', 'project_essay_1'] = combined_test_dataset['project_essay_1'] + ' ' + combined_test_dataset['project_essay_2']
combined_test_dataset.loc[combined_test_dataset['project_essay_4'] != '', 'project_essay_2'] = combined_test_dataset['project_essay_3'] + ' ' + combined_test_dataset['project_essay_4']

combined_training_dataset[16:20]
combined_training_dataset.loc[18, 'project_essay_2']
USING_DICTIONARY_CODE = True
if USING_DICTIONARY_CODE == False:
    #OLD Text preprocessing
    the_essays = [
        'project_essay_1',
        'project_essay_2',
        'full_description',  #not an essay... but has the same problems
        'project_resource_summary'  # probably has the same issues
    ]

    def tidy_essays(dataset):
        for col_name in the_essays:
            #strip out the irritating new line stuff:
            dataset[col_name] = dataset[col_name].replace(r'(\\r)|(\\n)', ' ', regex=True)
            #get everything down to just one space (hopefully):
            dataset[col_name] = dataset[col_name].replace(r'  ', ' ', regex=True)
            dataset[col_name] = dataset[col_name].replace(r'  ', ' ', regex=True)
            dataset[col_name] = dataset[col_name].replace(r'  ', ' ', regex=True)

    #do both training and test datasets:
    tidy_essays(combined_training_dataset)
    tidy_essays(combined_test_dataset)
    
    #OLD finish cleaning up all the text
    text_columns = [
        'project_title',
        'project_essay_1',
        'project_essay_2',
        'project_resource_summary',
        'project_subject_categories',
        'project_subject_subcategories',
        'full_description'
    ]

    punc_pattern = '["#$%&\'()*+,.:;<=>?@\\\\^_`{|}~]'

    def text_edits(dataset):
        for col_name in text_columns:
            #lowercase all
            dataset[col_name] = dataset[col_name].str.lower()
            #treat !'s as separate words just in case the model picks up on something:
            dataset[col_name] = dataset[col_name].replace(r'!', ' !', regex=True)
            #don't remove hyphens and smash words together; keep the words separate
            #same (hopefully) for /
            dataset[col_name] = dataset[col_name].replace(r'(\-)|(/)', ' ', regex=True)  #ADDED so hopefully works
            #replace the rest of the punctuation:
            dataset[col_name] = dataset[col_name].replace(punc_pattern, '', regex=True)

    #do both training and test datasets:
    text_edits(combined_training_dataset)
    text_edits(combined_test_dataset)

    combined_training_dataset[16:22]   
#USING_DICTIONARY_CODE located in cell above
if USING_DICTIONARY_CODE:
    text_columns = [
        'project_title',
        'project_essay_1',
        'project_essay_2',
        'project_resource_summary',
        'project_subject_categories',
        'project_subject_subcategories',
        'full_description'
    ]

    def tidy_text(dataset):
        for col_name in text_columns:
            #new idea:  handle weird utf8 symbol characters like smiley faces
            #  replace with ?, then strip out that question mark and replace it with a space:
            #first, remove actual ?'s and replace with no space '' (this is because some URLs contain ?'s):
            dataset[col_name] = dataset[col_name].replace('\?', '', regex=True)
            dataset[col_name] = dataset[col_name].str.encode('ascii', 'replace')  # replaces with a ? mark
            dataset[col_name] = dataset[col_name].str.decode('utf8') #turn back to normal string
                #new ?'s will be removed in a later regex
            #new idea:  handle acronyms since they're not going to show up in dictionaries:
                #is the actual acronym important?  I'm assuming not.  But perhaps you should work on a copy of the data
            dataset[col_name] = dataset[col_name].replace('([A-Z][.])+', '', regex=True)
            #0th - lowercase everything:
            dataset[col_name] = dataset[col_name].str.lower()
            #1st - strip out the irritating new lines that literally became \\r\\n in the text:
            #replace with a space so that words aren't unfairly smashed together and count as an error
            dataset[col_name] = dataset[col_name].replace('(\\\\r)|(\\\\n)', ' ', regex=True)
            #2nd - remove all the website links
            # .com AND .org AND .gov AND .edu AND .net
            dataset[col_name] = dataset[col_name].replace('[^ ]*\.com[^ ]*', '', regex=True)
            dataset[col_name] = dataset[col_name].replace('[^ ]*\.org[^ ]*', '', regex=True)
            dataset[col_name] = dataset[col_name].replace('[^ ]*\.gov[^ ]*', '', regex=True)
            dataset[col_name] = dataset[col_name].replace('[^ ]*\.edu[^ ]*', '', regex=True)
            dataset[col_name] = dataset[col_name].replace('[^ ]*\.net[^ ]*', '', regex=True)
            #2ndb - also remove tweet hashtags:
            dataset[col_name] = dataset[col_name].replace('#[a-z][^ ]*', '', regex=True)
            #3rd - remove other \\ affecting things like " marks
            dataset[col_name] = dataset[col_name].replace('\\\\', '', regex=True)
            #4th - don't remove hyphens/slashes and smash words together; instead, keep the words separate
            #same with ellipsis
            dataset[col_name] = dataset[col_name].replace('(\-)|(/)|(\.\.\.\.)|(\.\.\.)', ' ', regex=True)
            #5th - some people didn't space correctly.  Replace those specific instances involving punctuation marks
            #with a = sign to better handle that particular typo later on when looking for misspelled words
            #(first, remove any equal signs already in the text just in case)
            dataset[col_name] = dataset[col_name].replace('=', '', regex=True)
            dataset[col_name] = dataset[col_name].replace('(?<=\w)[,.;!"](?=\w)', '=', regex=True)
            #6th - treat !'s as separate words in case the model picks up on something related to too many exclamatory remarks:
            dataset[col_name] = dataset[col_name].replace('!', ' ! ', regex=True)
            #7th remove the remaining punctuation:
            dataset[col_name] = dataset[col_name].replace('["#$%&\'()*+,.:;[\]<>?@\\^_`{|}~]', '', regex=True)
            #8th - make everything have a separation of just one space:
            dataset[col_name] = dataset[col_name].replace(' *   *', ' ', regex=True)
            #and no space at the end:
            dataset[col_name] = dataset[col_name].str.strip()

        return

    tidy_text(combined_training_dataset)
    tidy_text(combined_test_dataset)
combined_training_dataset['project_essay_2'].iloc[3]
if USING_DICTIONARY_CODE:
    COLUMNS_TO_CHECK = [
        'project_title',
        'project_essay_1',
        'project_essay_2',
        'project_resource_summary'
    ]

    def total_typing_errors(df):
        new_col_names = []
        for col in COLUMNS_TO_CHECK:
            new_col_name = col + '_typing_errors'
            new_col_name = new_col_name[8:]  #drop 'project_' from the name to make it shorter
            df[new_col_name] = df[col].str.count('=')
            new_col_names.append(new_col_name)

        df['total_typing_errors'] = df[new_col_names].sum(axis=1)

        return

    total_typing_errors(combined_training_dataset)
    total_typing_errors(combined_test_dataset)
    
    #remove those = signs now:
    for col_name in text_columns:
        combined_training_dataset[col_name] = combined_training_dataset[col_name].replace('=', ' ', regex=True)
        combined_test_dataset[col_name] = combined_test_dataset[col_name].replace('=', ' ', regex=True)
combined_training_dataset[3:4]
combined_training_dataset['project_essay_2'].iloc[3]
if USING_DICTIONARY_CODE:
    join_cols = [
        'project_title',
        'project_essay_1',
        'project_essay_2',
        'project_resource_summary',
        'full_description'
    ]
    #first, join the rows of the test dataset with the training dataset to get a full look at all the words in the data
    all_dataset = pd.concat([combined_training_dataset[join_cols], combined_test_dataset[join_cols]])
    
    def get_unique_word_sets(df):
        new_set = set()
        df.str.split().apply(new_set.update)

        return new_set

    title_set = get_unique_word_sets(all_dataset['project_title'])  
    essay_1_set = get_unique_word_sets(all_dataset['project_essay_1'])  
    essay_2_set = get_unique_word_sets(all_dataset['project_essay_2']) 
    summary_set = get_unique_word_sets(all_dataset['project_resource_summary'])
    description_set = get_unique_word_sets(all_dataset['full_description'])
    print(len(title_set))
    print(len(essay_1_set))
    print(len(essay_2_set))
    print(len(summary_set))
    print(len(description_set))
if USING_DICTIONARY_CODE:
    #Contains 333,332 words
    #And '!' as acceptable word
    dictionary_set = set(dictionary_dataset['word'])
    dictionary_set.add('!')
    print(len(dictionary_set))
if USING_DICTIONARY_CODE:
    title_set = title_set - dictionary_set - unix_set  
    essay_1_set = essay_1_set - dictionary_set - unix_set
    essay_2_set = essay_2_set - dictionary_set - unix_set
    summary_set = summary_set - dictionary_set - unix_set
    description_set = description_set - dictionary_set - unix_set
    print(len(title_set))
    print(len(essay_1_set))
    print(len(essay_2_set))
    print(len(summary_set))
    print(len(description_set))
if USING_DICTIONARY_CODE:
    title_set = title_set - description_set
    essay_1_set = essay_1_set - description_set
    essay_2_set = essay_2_set - description_set
    summary_set = summary_set - description_set
    print(len(title_set))
    print(len(essay_1_set))
    print(len(essay_2_set))
    print(len(summary_set))
if USING_DICTIONARY_CODE:
    def make_set_alpha(word_set):
        words_to_drop = set()
        for word in word_set:
            if word.isalpha() == False:
                words_to_drop.add(word)

        return word_set - words_to_drop

    title_set = make_set_alpha(title_set)
    essay_1_set = make_set_alpha(essay_1_set)
    essay_2_set = make_set_alpha(essay_2_set)
    summary_set = make_set_alpha(summary_set)
    print(len(title_set))
    print(len(essay_1_set))
    print(len(essay_2_set))
    print(len(summary_set))
if USING_DICTIONARY_CODE:
    #for each column is a matching set of 'misspelled' words:
    COLS_AND_SETS = {
        'project_title' : title_set,
        'project_essay_1' : essay_1_set,
        'project_essay_2' : essay_2_set,
        'project_resource_summary' : summary_set
    }
    
    def count_spelling_errors(string, word_set):
        list_words = string.split()
        error_count = 0
        for word in list_words:
            if word in word_set:
                error_count += 1

        return error_count

    def total_spelling_errors(df):
        new_col_names = []
        for col, word_set in COLS_AND_SETS.items():
            new_col_name = col + '_spelling_errors'
            new_col_name = new_col_name[8:]  #drop 'project_' from the name to make it shorter
            df[new_col_name] = df[col].apply(count_spelling_errors, args=(word_set,))
            new_col_names.append(new_col_name)

        df['total_spelling_errors'] = df[new_col_names].sum(axis=1)

        return
        
    total_spelling_errors(combined_training_dataset)
    total_spelling_errors(combined_test_dataset)
if USING_DICTIONARY_CODE:
    combined_training_dataset.loc[combined_training_dataset['total_spelling_errors'] > 12]
combined_training_dataset['project_essay_2'].loc[110575]
combined_training_dataset['project_is_approved'].loc[110575]
combined_training_dataset['project_essay_2'].loc[172786]
#https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-from-pandas-datetime-column-python
#https://stackoverflow.com/questions/17950374/converting-a-column-within-pandas-dataframe-from-int-to-string
#First, get the project_submitted_datetime column into a format that Pandas can work with
combined_training_dataset['project_submitted_datetime'] = pd.to_datetime(combined_training_dataset['project_submitted_datetime'])
combined_training_dataset['year_week'] = combined_training_dataset['project_submitted_datetime'].map(lambda x: 100*(x.year - 2000) + x.week).apply(str)

combined_test_dataset['project_submitted_datetime'] = pd.to_datetime(combined_test_dataset['project_submitted_datetime'])
combined_test_dataset['year_week'] = combined_test_dataset['project_submitted_datetime'].map(lambda x: 100*(x.year - 2000) + x.week).apply(str)
combined_training_dataset[0:5]
import matplotlib.pyplot as plt
#Prefixes and Approved vs Unapproved Application
grouped_prefixes = combined_training_dataset.groupby(["teacher_prefix", "project_is_approved"])
grouped_prefixes = grouped_prefixes.agg({'teacher_prefix' : 'count'}).rename(columns={'teacher_prefix' : 'count'})
grouped_prefixes
arr_counts = grouped_prefixes["count"].tolist()
#split arr_counts into two separate lists:
y_no = []
y_yes = []
for i in range(0, len(arr_counts) - 1):
    if i % 2 == 0:
        y_no.append(arr_counts[i])
    else:
        y_yes.append(arr_counts[i])
y_yes.append(arr_counts[i+1])
y_no.append(0)
x_labels = ['Dr', 'Mr.', 'Mrs', 'Ms', 'Teacher', 'None']

#Make a multiple bar graph
#via https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
N = len(x_labels)
ind = np.arange(N)  # the x locations for the groups
width = 0.4      # the width of the bars
fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(ind, y_yes, width, color='#7CFC00')
rects2 = ax.bar(ind+width, y_no, width, color='#DC143C')

ax.set_ylabel('Submissions Count')
ax.set_xticks(ind+width)
ax.set_xticklabels(x_labels)
ax.legend( (rects1[0], rects2[0]), ('Approved', 'NOT Approved') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.title('Submissions Approved or NOT Approved Based on Name Prefix')
plt.show()
#Chance to get approved by prefix
print('Chance to get approved by prefix')
arr_total_applications = []
for i in range(0, len(y_yes)):
    arr_total_applications.append(y_yes[i] + y_no[i])
    print(x_labels[i] + ': ' + "%f" % (y_yes[i] / arr_total_applications[i]))
bins = [0, 150, 200, 250, 300, 350, 400, 500, 600, 750, 1000, 1000000]
combined_training_dataset['binned_costs'] = pd.cut(combined_training_dataset['total_cost'], bins)

grouped_costs = combined_training_dataset.groupby(["binned_costs", "project_is_approved"])
grouped_costs = grouped_costs.agg({'binned_costs' : 'count'}).rename(columns={'binned_costs' : 'count'})

arr_percent_yes = []
for i in range(0, (len(bins) - 1) * 2):
    if i % 2 == 0:
        k = i // 2
        label = 'no'
        no_count = grouped_costs["count"].values[i]
        print("%d+ - %s: %d" % (bins[k], label, no_count))
    else:
        label = 'yes'
        yes_count = grouped_costs["count"].values[i]
        percent_yes = yes_count / (no_count + yes_count)
        arr_percent_yes.append(percent_yes)
        print("%d+ - %s: %d, percent: %f" % (bins[k], label, yes_count, percent_yes))
arr_x_labels = []
for i in range(0, len(bins)-2):
    arr_x_labels.append("%d+" % (bins[i]))
arr_x_labels.append("_1000+") #add the _ so that the plot doesn't alphabetize the numbers
 
plt.scatter(arr_x_labels, arr_percent_yes)
plt.xlabel("Cost of Request")
plt.ylabel("Yes Acceptance Rate")
plt.show()
#function code via https://colab.research.google.com/notebooks/mlcc/sparsity_and_l1_regularization.ipynb?hl=en#scrollTo=bLzK72jkNJPf
def quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]

quantile_bins = quantile_based_buckets(combined_training_dataset["total_cost"], 12)
#^But why does the DataFrame later show NaN for the lowest range (from 0 to 133)?
#...Looks like Pandas requires you to specify the lower/upper limits if you pass in a list to pd.cut?
quantile_bins.insert(0, 0)
quantile_bins.append(1000000)

combined_training_dataset['quantile_costs'] = pd.cut(combined_training_dataset['total_cost'], quantile_bins)

grouped_costs = combined_training_dataset.groupby(["quantile_costs", "project_is_approved"])
grouped_costs = grouped_costs.agg({'quantile_costs' : 'count'}).rename(columns={'quantile_costs' : 'qcount'})

arr_percent_yes = []
for i in range(0, (len(quantile_bins) - 1) * 2):
    if i % 2 == 0:
        k = i // 2
        label = 'no'
        no_count = grouped_costs["qcount"].values[i]
        print("%d+ - %s: %d" % (quantile_bins[k], label, no_count))
    else:
        label = 'yes'
        yes_count = grouped_costs["qcount"].values[i]
        percent_yes = yes_count / (no_count + yes_count)
        arr_percent_yes.append(percent_yes)
        print("%d+ - %s: %d, percent: %f" % (quantile_bins[k], label, yes_count, percent_yes))


arr_x_labels = []
for i in range(0, len(quantile_bins)-2):
    arr_x_labels.append("%d+" % (quantile_bins[i]))
arr_x_labels.append("_1000+") #add the _ so that the plot doesn't alphabetize the numbers
 
plt.scatter(arr_x_labels, arr_percent_yes)
plt.xlabel("Cost of Request")
plt.ylabel("Yes Acceptance Rate")
plt.show()
grouped_previous_submissions = combined_training_dataset.groupby(["teacher_number_of_previously_posted_projects", "project_is_approved"])
grouped_previous_submissions = grouped_previous_submissions.agg({'teacher_number_of_previously_posted_projects' : 'count'}).rename(columns={'teacher_number_of_previously_posted_projects' : 'count'})
grouped_previous_submissions[0:202]
arr_percents_yes = []
for i in range(0,202):
    if i % 2 == 0:
        k = i / 2
        label = 'no'
        no_count = grouped_previous_submissions["count"].values[i]
        print("%d - %s: %d" % (k, label, no_count))
    else:
        label = 'yes'
        yes_count = grouped_previous_submissions["count"].values[i]
        percent_yes = yes_count / (no_count + yes_count)
        arr_percents_yes.append(percent_yes)
        print("%d - %s: %d, percent: %f" % (k, label, yes_count, percent_yes))
    
x_num_previous = np.arange(0, 101)
#y is arr_percents_yes

plt.plot(x_num_previous, arr_percents_yes)

plt.xlabel("Number of Previous Submissions")
plt.ylabel("Chance of Accepted Submission")
plt.title("Chance of Accepted Submission Based on Previous Number of Submissions")
plt.show()
grouped_states = combined_training_dataset.groupby(["school_state", "project_is_approved"])
#a way to eliminate the multi-index?:
#https://stackoverflow.com/questions/39778686/pandas-reset-index-after-groupby-value-counts
grouped_states = grouped_states.size().rename('count').reset_index()

arr_percents = []
for i in range(0,102):
    if i % 2 == 0:
        no_count = grouped_states["count"].values[i]
    else:
        yes_count = grouped_states["count"].values[i]
        total_count = no_count + yes_count
        percent_no = no_count / total_count
        percent_yes = yes_count / total_count
        arr_percents.append(percent_no)
        arr_percents.append(percent_yes)
        
grouped_states['chances'] = pd.Series(arr_percents, index=grouped_states.index)
grouped_states

#Find the lowest approval rates:
grouped_states.loc[(grouped_states['project_is_approved'] == 1) & (grouped_states['chances'] < .83)]
#And the highest ones:
grouped_states.loc[(grouped_states['project_is_approved'] == 1) & (grouped_states['chances'] > .868)]
#Make a stacked bar chart of Top 5 vs Bottom 5 acceptance rates because...  I just wanna see what it looks like...

idxes = ['1 DE/DC', '2 WY/TX', '3 OH/NM', '4 CT/FL', '5 WA/MT']
lowest = [.812639, .815670, .822052, .824500, .828125]
highest = [.891341, .875706, .871467, .871294, .868050]

#apparently the 2nd bar just paints over the first, so the 2nd must be smaller or it gets hidden
plt.bar(idxes, highest, label="DE, WY, OH, CT, WA", color='#87CEFA')
plt.bar(idxes, lowest, label="DC, TX, NM, FL, MT", color='#B22222')

plt.plot()

#make the scale more useful
#https://stackoverflow.com/questions/3777861/setting-y-axis-limit-in-matplotlib
axes = plt.gca()
axes.set_ylim([.75, .92])

plt.title('Submission Acceptance Rates for Top 5 States vs Bottom 5 States')
plt.legend()
plt.xlabel('State')
plt.ylabel("Chance of Accepted Submission")
plt.show()
grouped_grades = combined_training_dataset.groupby(["project_grade_category", "project_is_approved"])
#a way to eliminate the multi-index?:
#https://stackoverflow.com/questions/39778686/pandas-reset-index-after-groupby-value-counts
grouped_grades = grouped_grades.size().rename('count').reset_index()
grouped_grades
arr_no = []
arr_yes = []
arr_labels = []
arr_all = []
for i in range(0, len(grouped_grades["project_grade_category"].values)):
    if i % 2 == 0:
        arr_labels.append("%s - No" % (grouped_grades["project_grade_category"].values[i]))
        arr_no.append(grouped_grades["count"][i])
    else:
        arr_labels.append("%s - Yes" % (grouped_grades["project_grade_category"].values[i]))
        arr_yes.append(grouped_grades["count"][i])
    arr_all.append(grouped_grades["count"][i])

colors = ['#DC143C', '#7CFC00']
plt.pie(arr_all, labels=arr_labels, colors=colors,
        startangle=50,
        explode = (.3, .3, .3, .3, .3, .3, .3, .3),
        autopct = '%1.2f%%',
        shadow=True)

plt.axis('equal')
plt.title('Pie Chart Example')
plt.show()

#combined_training_dataset['project_submitted_datetime'] = pd.to_datetime(combined_training_dataset['project_submitted_datetime'])
  #^done in an earlier step when preprocessing the data
#try grouping by month:
#https://stackoverflow.com/questions/44908383/how-can-i-group-by-month-from-a-date-field-using-python-pandas

grouped_months = combined_training_dataset.groupby([pd.Grouper(key='project_submitted_datetime', freq='1M'), "project_is_approved"])
grouped_months = grouped_months.size().rename('count').reset_index()

arr_percents_month = []
for i in range(0,26):
    if i % 2 == 0:
        no_count = grouped_months["count"].values[i]
    else:
        yes_count = grouped_months["count"].values[i]
        total_count = no_count + yes_count
        percent_no = no_count / total_count
        percent_yes = yes_count / total_count
        arr_percents_month.append(percent_no)
        arr_percents_month.append(percent_yes)
        
grouped_months['chances'] = pd.Series(arr_percents_month, index=grouped_months.index)
grouped_months
x_num_previous_m = np.arange(0, 13)

arr_percents_m_yes = []
for i in range(0,26):
    if not (i % 2 == 0):
        arr_percents_m_yes.append(arr_percents_month[i])


plt.plot(x_num_previous_m, arr_percents_m_yes)

plt.xlabel("Month")
plt.ylabel("Chance of Accepted Submission")
plt.title("Chance of Accepted Submission Based on Month")
plt.show()
#only difference here is freq='1W' instead of '1M'
grouped_weeks = combined_training_dataset.groupby([pd.Grouper(key='project_submitted_datetime', freq='1W'), "project_is_approved"])
grouped_weeks = grouped_weeks.size().rename('count').reset_index()

arr_percents_w = []
for i in range(0,106):
    if i % 2 == 0:
        no_count = grouped_weeks["count"].values[i]
    else:
        yes_count = grouped_weeks["count"].values[i]
        total_count = no_count + yes_count
        percent_no = no_count / total_count
        percent_yes = yes_count / total_count
        arr_percents_w.append(percent_no)
        arr_percents_w.append(percent_yes)
        
grouped_weeks['chances'] = pd.Series(arr_percents_w, index=grouped_weeks.index)
grouped_weeks
x_num_previous_w = np.arange(0, 53)

arr_percents_yes_w = []
for i in range(0,106):
    if not (i % 2 == 0):
        arr_percents_yes_w.append(arr_percents_w[i])


plt.plot(x_num_previous_w, arr_percents_yes_w)

plt.xlabel("Week")
plt.ylabel("Chance of Accepted Submission")
plt.title("Chance of Accepted Submission Based on Week")
plt.show()
#going to also try by day of the week:
#https://stackoverflow.com/questions/13740672/in-pandas-how-can-i-groupby-weekday-for-a-datetime-column
combined_training_dataset['weekday'] = combined_training_dataset['project_submitted_datetime'].dt.weekday

grouped_weekday = combined_training_dataset.groupby(["weekday", "project_is_approved"])
#grouped_dates = grouped_dates.agg({'project_submitted_datetime' : 'count'})
grouped_weekday = grouped_weekday.size().rename('count').reset_index()

arr_percents_weekday = []
for i in range(0,14):
    if i % 2 == 0:
        no_count = grouped_weekday["count"].values[i]
    else:
        yes_count = grouped_weekday["count"].values[i]
        total_count = no_count + yes_count
        percent_no = no_count / total_count
        percent_yes = yes_count / total_count
        arr_percents_weekday.append(percent_no)
        arr_percents_weekday.append(percent_yes)
        
grouped_weekday['chances'] = pd.Series(arr_percents_weekday, index=grouped_weekday.index)
grouped_weekday
if USING_DICTIONARY_CODE:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    #convert data back to int:
    pd_plot = combined_training_dataset[['total_spelling_errors', 'total_typing_errors', 'project_is_approved']].copy()
    pd_plot['total_spelling_errors'] = pd.to_numeric(pd_plot['total_spelling_errors'])
    pd_plot['total_typing_errors'] = pd.to_numeric(pd_plot['total_typing_errors'])
    
    fig, axes = plt.subplots()
    # plot violin. 'project_is_approved' is according to x axis, 
    # 'comma_count' is y axis, data is training_dataset. ax - is axes instance
    sns.violinplot('project_is_approved','total_spelling_errors', data=pd_plot, ax = axes)
    axes.set_title('Spelling Errors vs Approval')

    axes.yaxis.grid(True)
    axes.set_xlabel('project_is_approved')
    axes.set_ylabel('total-spelling_errors')

    plt.show()
if USING_DICTIONARY_CODE:
    fig, axes = plt.subplots()
    # plot violin. 'project_is_approved' is according to x axis, 
    # 'comma_count' is y axis, data is training_dataset. ax - is axes instance
    sns.violinplot('project_is_approved','total_typing_errors', data=pd_plot, ax = axes)
    axes.set_title('Typing Errors vs Approval')

    axes.yaxis.grid(True)
    axes.set_xlabel('project_is_approved')
    axes.set_ylabel('total-typing_errors')

    plt.show()
# Data sets are:
# combined_training_dataset
# combined_test_dataset

combined_training_dataset = combined_training_dataset.reindex(
    np.random.permutation(combined_training_dataset.index))
USING_OLD_MODELS = False

#combined_training_dataset.shape  -->  182,080 examples
#split 80% training, 20% validation --> 145,664 training, 36,416 validation
HEAD = 145664
TAIL = 36416
TARGET_STR = "project_is_approved"

training_examples = combined_training_dataset.head(HEAD)
validation_examples = combined_training_dataset.tail(TAIL)

if USING_OLD_MODELS:  
    #new models = don't separate examples from targets here
    #old models require targets to be separate:
    training_targets = combined_training_dataset[[TARGET_STR]].head(HEAD)
    validation_targets = combined_training_dataset[[TARGET_STR]].tail(TAIL)
    
# GLOBAL CONSTANTS
#USING_OLD_MODELS = False  #(located in cell above)

# ***  DESIRED_FEATURES  ***
#To make it easier to swap features in and out (and even create and toss in new ones), many functions throughout
#the rest of this Notebook only require you to update this DESIRED_FEATURES list.
# DESIRED_FEATURES is a list whose elements are a tuple of the form:
#       (feature_name, feature_type, num_bucket), so (str, str, int)
# feature_type - types of features include:
WORDS_STRING = 'words_string' #for strings that should be looked at as individual words
WHOLE_STRING = 'whole_string' # for strings that should NOT be looked at as individual words
BUCKET_INT = 'bucket_int' # ints that will be bucketized
BUCKET_FLOAT = 'bucket_float' # floats that will be bucketized
NUM_INT = 'num_int' #treat the int as a normal, individual number
NUM_FLOAT = 'num_float' # treat the float as a normal, individual number

# num_bucket is meant only for the bucket_int and bucket_float features; how many buckets to split the data into
# alternatively, you can set num_bucket equal to an array of your choosing for the buckets, like:  [1.0, 2.0, 5.0]
ERROR_BUCKETS = [0, 1, 2, 4, 7, 10, 20, 50]

if USING_OLD_MODELS:
    DESIRED_FEATURES = [
            #'id', not used, not handled
            #'teacher_id', not used, not handled
    ('teacher_prefix', WHOLE_STRING, None),
    ('school_state', WHOLE_STRING, None),
            #'project_submitted_datetime', not used
    #('project_grade_category', WHOLE_STRING, None),
            #('project_subject_categories', WORDS_STRING, None), not handled in old model
            #('project_subject_subcategories', WORDS_STRING), not handled in old model
            #('project_title', WORDS_STRING), not handled in old model
            #('project_essay_1', WORDS_STRING), not handled in old model
            #('project_essay_2', WORDS_STRING, None), not handled in old model
            #'project_essay_3', not used, not handled
            #'project_essay_4', not used, not handled
            #('project_resource_summary', WORDS_STRING), not handled in old model
    ('teacher_number_of_previously_posted_projects', BUCKET_INT, 6),
            #('full_description', WORDS_STRING), not handled in old model
    #('total_quantity', BUCKET_INT, 10),
    #('total_cost', BUCKET_FLOAT, 12),
    #('year_week', WHOLE_STRING, None),  #effectively takes the place of project_submitted_datetime
    ('project_is_approved', NUM_INT, None) #target
]
else:
    DESIRED_FEATURES = [
            #'id', not used, not handled
            #'teacher_id', not used, not handled
    #('teacher_prefix', WHOLE_STRING, None),
    #('school_state', WHOLE_STRING, None),
            #'project_submitted_datetime', not used, not handled
    #('project_grade_category', WHOLE_STRING, None),
    #('project_subject_categories', WORDS_STRING, None), #because some belong to multiple categories
    #('project_subject_subcategories', WORDS_STRING, None),
    #('project_title', WORDS_STRING, None),
    ('project_essay_1', WORDS_STRING, None),
    ('project_essay_2', WORDS_STRING, None),
            #'project_essay_3', not used, not handled
            #'project_essay_4', not used, not handled
    #('project_resource_summary', WORDS_STRING, None),
    ('teacher_number_of_previously_posted_projects', BUCKET_INT, 6),
    ('full_description', WORDS_STRING, None),
    #('total_quantity', BUCKET_INT, 10),
    ('total_cost', BUCKET_FLOAT, 12),
    #('year_week', WHOLE_STRING, None),  #effectively takes the place of project_submitted_datetime
#POSSIBLE NEW FEATURES:  (ints are now WHOLE_STRING since treated like vocabulary list)
    #('comma_count',  WHOLE_STRING, 6),
    #('hyphen_count', WHOLE_STRING, 2),
    #('sentence_count', WHOLE_STRING, 6),
    #('avg_sentence_length', BUCKET_FLOAT, 5),
    #('plural_pronouns', WHOLE_STRING, 5),
    #('wrong_article', WHOLE_STRING, 1),  #the error ones = don't have enough to make buckets
    #('cap_errors', WHOLE_STRING, 1),
    #('singular_pronouns', WHOLE_STRING, 5),
    #('repeat_words', WHOLE_STRING, 1),
    #('avg_word_size', BUCKET_FLOAT, 6),
    #('num_charged_words', NUM_INT, 1),
#NEW ERROR-RELATED FEATURES:
    #('title_typing_errors', BUCKET_INT, ERROR_BUCKETS),
    ('title_spelling_errors', BUCKET_INT, ERROR_BUCKETS),
    #('essay_1_typing_errors', BUCKET_INT, ERROR_BUCKETS),
    ('essay_1_spelling_errors', BUCKET_INT, ERROR_BUCKETS),
    #('essay_2_typing_errors', BUCKET_INT, ERROR_BUCKETS),
    ('essay_2_spelling_errors', BUCKET_INT, ERROR_BUCKETS),
    #('resource_summary_typing_errors', BUCKET_INT, ERROR_BUCKETS),
    #('resource_summary_spelling_errors', BUCKET_INT, ERROR_BUCKETS),
    ('total_typing_errors', BUCKET_INT, ERROR_BUCKETS),
    ('total_spelling_errors', BUCKET_INT, ERROR_BUCKETS),
    ('project_is_approved', NUM_INT, None) #target
]


#some functions (like the ones saving to a TFRecord) just want the string column names, so grab those:
DESIRED_COLUMNS = []
for str_name, str_type, dontcare in DESIRED_FEATURES:
    DESIRED_COLUMNS.append(str_name)

STR_TARGET = 'project_is_approved' #target, which needs to be singled out and kept out of some functions

# construct_feature_columns (shown later) is designed to work with either the Linear Classifer
# or the DNN Classifier depending on the following constant
# That is, if training the DNN Classifier, turn categorical_column_with_vocabulary_list into embeddings:
IS_DNN_CLASSIFIER = False
def get_quantile_based_buckets(feature_values, num_buckets):
    """
    Args:
        feature_values:  Pandas DataFrame (one column)
        num_buckets:  how many buckets
        
    Returns:
        An array of the quantiles
    """
    quantiles = feature_values.quantile(
        [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]
for str_name, str_type, num_bucket in DESIRED_FEATURES:
    print(str_name)
    if (str_type == BUCKET_INT) or (str_type == BUCKET_FLOAT):
        if (type(num_bucket) == list) == False:
            print(get_quantile_based_buckets(training_examples[str_name], num_bucket))
        else:
            print(num_bucket)
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: True or False. Whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
        Tuple of (features, labels) for next data batch
    """
    
    # Grab only the features specified in DESIRED_FEATURES:
    selected_features_data = features[DESIRED_COLUMNS]
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(selected_features_data).items()}                                            

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
def vocab_lists_for_each_feature():
    vocab_for_feature = {}
    for str_name, str_type, dont_care in DESIRED_FEATURES:
        #features to treat as full strings:
        if (str_type == WHOLE_STRING):
            vocab_for_feature[str_name] = training_examples[str_name].unique().tolist()
        #string features that should be split into individual words
        elif (str_type == WORDS_STRING):
            new_set = set()
            training_examples[str_name].str.split().apply(new_set.update)
            vocab_for_feature[str_name] = new_set
        #everything else is a number that doesn't need a vocabulary list
                  
    return vocab_for_feature

vocab_lists_for_features = vocab_lists_for_each_feature()
def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.

    Returns:
        A set of feature columns
    """
    
    arr_num_columns = []
    arr_vocabulary_columns = []
    #for use in determining embedding dimensions for each embedding_column:
    arr_dimensions = []
    
    for str_name, str_type, num_bucket in DESIRED_FEATURES:
        # don't create a feature column for the target:
        if (str_name == STR_TARGET):
            continue
        # create normal numeric_columns for normal numbers
        elif (str_type == NUM_INT) or (str_type == NUM_FLOAT):
            arr_num_columns.append(tf.feature_column.numeric_column(str_name))
        # create bucketized columns for bucket numbers:
        elif (str_type == BUCKET_INT) or (str_type == BUCKET_FLOAT):
            #print(str_name)  for debugging irritating training error when it doesn't like the values comprising the buckets
            if (type(num_bucket) == list) == False:  #then get the buckets
                bucket_column = tf.feature_column.bucketized_column(
                    tf.feature_column.numeric_column(str_name),
                    boundaries=get_quantile_based_buckets(training_examples[str_name], num_bucket))
            else:
                bucket_column = tf.feature_column.bucketized_column(
                    tf.feature_column.numeric_column(str_name), boundaries=num_bucket)
            arr_num_columns.append(bucket_column)
        # create categorical vocab columns for strings:
        elif (str_type == WHOLE_STRING) or (str_type == WORDS_STRING):
            arr_vocabulary_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(key=str_name, vocabulary_list=vocab_lists_for_features[str_name]))
            #for DNN Classifier, also calculate the number of dimensions for embedding:
            if IS_DNN_CLASSIFIER:
                fourth_root = int(math.sqrt(math.sqrt(len(vocab_lists_for_features[str_name]))))
                if (fourth_root == 0):
                    fourth_root += 1
                arr_dimensions.append(fourth_root)

    #turn vocab_columns into embedding_columns
    #arr_vocab and arr_dimensions will match by index
    end = len(arr_dimensions)
    arr_embedding_columns = []
    if IS_DNN_CLASSIFIER:
        for i in range(0, end):
            arr_embedding_columns.append(tf.feature_column.embedding_column(arr_vocabulary_columns[i], dimension=arr_dimensions[i]))
        arr_vocabulary_columns = arr_embedding_columns
        
    feature_columns = set(arr_num_columns + arr_vocabulary_columns)

    return feature_columns
def train_linear_classifier_model(
    learning_rate,
    regularization_strength,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    periods=1):
    """Trains a linear classifier model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
        learning_rate: A `float`, the learning rate.
        regularization_strength: A `float` that indicates the strength of the L1
            regularization. A value of `0.0` means no regularization.
        steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
        feature_columns: A `set` specifying the input feature columns to use.
        training_examples: A `DataFrame` containing one or more columns to use as input features for training.
        training_targets: A `DataFrame` containing exactly one column from to use as target for training.
        validation_examples: A `DataFrame` containing one or more columns to use as input features for validation.
        validation_targets: A `DataFrame` containing exactly one column to use as target for validation.
        periods: A integer, the number of times to train the model.  #Programming Exercises had periods = 7

    Returns:
    A `LinearClassifier` object trained on the training data.
    """

    #steps_per_period = steps / periods  SKIPPING PERIODS
    
    # Create a linear classifier object.
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )
    
    # Create input function for training (dependent on batch_size passed in)
    training_input_fn = lambda: my_input_fn(training_examples, 
                      training_targets[TARGET_STR], 
                      batch_size=batch_size)
    
    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on validation data):")
    #training_log_losses = []  SKIPPING PERIODS
    #validation_log_losses = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps
        )
        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        # Compute training and validation loss.
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  Training loss for period %02d: %0.2f" % (period, training_log_loss))
        print("  Validation loss for period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.  SKIPPING PERIODS
        #training_log_losses.append(training_log_loss)
        #validation_log_losses.append(validation_log_loss)
        
    print("Model training finished.")

    # Periods slow down Kaggle, so only do one; makes this graph pointless
    # Output a graph of loss metrics over periods.
    #plt.ylabel("LogLoss")
    #plt.xlabel("Periods")
    #plt.title("LogLoss vs. Periods")
    #plt.tight_layout()
    #plt.plot(training_log_losses, label="training")
    #plt.plot(validation_log_losses, label="validation")
    #plt.legend()

    return linear_classifier
# Create input functions.
predict_training_input_fn = lambda: my_input_fn(training_examples, 
                              training_targets[TARGET_STR], 
                              num_epochs=1, 
                              shuffle=False)
predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                validation_targets[TARGET_STR], 
                                num_epochs=1, 
                                shuffle=False)
if USING_OLD_MODELS & (not IS_DNN_CLASSIFIER):
    linear_classifier = train_linear_classifier_model(
        learning_rate=.1,
        regularization_strength=0.1,
        steps=1000,  
        batch_size=140,
        periods=1,
        feature_columns=construct_feature_columns(),
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets) 

if USING_OLD_MODELS & (not IS_DNN_CLASSIFIER):
    #training_metrics = linear_classifier.evaluate(input_fn=predict_training_input_fn)
    validation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

    #print("AUC on the training set: %0.2f" % training_metrics['auc'])
    #print("Accuracy on the training set: %0.2f" % training_metrics['accuracy'])

    print("AUC on the validation set: %0.2f" % validation_metrics['auc'])
    print("Accuracy on the validation set: %0.2f" % validation_metrics['accuracy'])
if USING_OLD_MODELS & (not IS_DNN_CLASSIFIER):
    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    # Get just the probabilities for the positive class.
    validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
        validation_targets, validation_probabilities)
    plt.plot(false_positive_rate, true_positive_rate, label="our model")
    plt.plot([0, 1], [0, 1], label="random classifier")
    _ = plt.legend(loc=2)
#functions to write correct values to TFRecord
#depending on the column value, TFRecord either needs to store a bytes_list, int64_list, or float_list value
def make_bytes_list_append(col_index, value, example):
    example.features.feature[DESIRED_COLUMNS[col_index]].bytes_list.value.append(value.encode())

#for the data with actual sentences, split them into individual words
def make_bytes_list_extend(col_index, value, example):
    arr_strings = value.split(' ')
    arr_bstrings = list(map(lambda x: x.encode(), arr_strings))
    example.features.feature[DESIRED_COLUMNS[col_index]].bytes_list.value.extend(arr_bstrings)
    
def make_int64_list(col_index, value, example):
    example.features.feature[DESIRED_COLUMNS[col_index]].int64_list.value.append(value)

def make_float_list(col_index, value, example):
    example.features.feature[DESIRED_COLUMNS[col_index]].float_list.value.append(value)


#function to create an list of the right function to call depending on the feature
def match_feature_with_tfrecord_function():
    arr_functions = []
    for str_name, str_type, dontcare in DESIRED_FEATURES:
        #features consisting of a full string = bytes, append
        if (str_type == WHOLE_STRING):
            arr_functions.append(make_bytes_list_append)
        #features consisting of text that should be broken down into words = bytes, extend
        elif (str_type == WORDS_STRING):
            arr_functions.append(make_bytes_list_extend)
        #features consisting of non-integer numbers = float
        elif (str_type == NUM_FLOAT) or (str_type == BUCKET_FLOAT):
            arr_functions.append(make_float_list)
        #features consisting of integers = int64
        elif (str_type == NUM_INT) or (str_type == BUCKET_INT):
            arr_functions.append(make_int64_list)
        
    return arr_functions

#loop through all the rows in a dataset, convert each individual cell to the right data type that TFRecord requires,
#take those tf.examples and save them into a TFRecord file
#NOTE:  REQUIRES arr_tfrecord_funcs = match_feature_with_tfrecord_function() to be called first
def save_rows_as_TFRecord(array_of_rows, tfrecord_file_name):
    with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
        last_column = len(DESIRED_COLUMNS)
        for row in array_of_rows:
            example = tf.train.Example()
            for col_index in range(0, last_column):
                arr_tfrecord_funcs[col_index](col_index, row[col_index], example)  #for each feature, call the corresponding function

            writer.write(example.SerializeToString())
    return

#just a test function to make sure all the above works
def TEST_save_rows_as_TFRecord(array_of_rows, tfrecord_file_name):
    last_column = len(DESIRED_COLUMNS)
    for row in array_of_rows:
        example = tf.train.Example()
        for col_index in range(0, last_column):
            arr_tfrecord_funcs[col_index](col_index, row[col_index], example)  #for each feature, call the corresponding function
        print(example)
#testing new functions:
arr_tfrecord_funcs = match_feature_with_tfrecord_function()
testdata = training_examples[DESIRED_COLUMNS][0:2]

TEST_save_rows_as_TFRecord(testdata.values, 'test')
combined_test_dataset['project_is_approved'] = 0
combined_test_dataset[0:5]
#Files to save:
TRAINING_TFRECORD = "training.tfrecords" #for training_examples
VALIDATION_TFRECORD = "validation.tfrecords" #for validation_examples
TEST_TFRECORD = "test.tfrecords" #for combined_test_dataset

arr_tfrecord_funcs = match_feature_with_tfrecord_function() #grab the conversion functions to call for each cell
#make the TFRecord for training data:
save_rows_as_TFRecord(training_examples[DESIRED_COLUMNS].values, TRAINING_TFRECORD)
#make the TFRecord for validation data:
save_rows_as_TFRecord(validation_examples[DESIRED_COLUMNS].values, VALIDATION_TFRECORD)
#make the TFRecord for test data:
save_rows_as_TFRecord(combined_test_dataset[DESIRED_COLUMNS].values, TEST_TFRECORD)
def parse_function(record):
    """Extracts features and labels from a TFRecord file.

    Args:
        record: file name of the TFRecord file    
    Returns:
        A `tuple` `(features, labels)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features_to_parse = {}
    for str_name, str_type, dontcare in DESIRED_FEATURES:
        #features consisting of a full string that wasn't split apart
        if (str_type == WHOLE_STRING):
            features_to_parse[str_name] = tf.FixedLenFeature(shape=[1], dtype=tf.string)
        #features consisting of a string split apart into pieces of varying lengths.  Thus, need the VarLenFeature:
        elif (str_type == WORDS_STRING):
            features_to_parse[str_name] = tf.VarLenFeature(dtype=tf.string)
        #features consisting of non-integer numbers = float
        elif (str_type == NUM_FLOAT) or (str_type == BUCKET_FLOAT):
            features_to_parse[str_name] = tf.FixedLenFeature(shape=[1], dtype=tf.float32)
        #features consisting of integers = int64
        elif (str_type == NUM_INT) or (str_type == BUCKET_INT):
            features_to_parse[str_name] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)

    parsed_features = tf.parse_single_example(record, features_to_parse)
    
    final_features = {}
    for str_name, str_type, dontcare in DESIRED_FEATURES:
        #don't include target in features; handle later as label
        if (str_name == STR_TARGET):
            continue
        ##DO NOT FORGET .values for the VarLenFeature strings (doing so creates a cryptic error that was hard to figure out)
        elif (str_type == WORDS_STRING):
            final_features[str_name] = parsed_features[str_name].values
        #everything else as normal
        else:
            final_features[str_name] = parsed_features[str_name]
    
    #grab the targets:
    labels = parsed_features[STR_TARGET]
    
    return final_features, labels
ds = tf.data.TFRecordDataset(TRAINING_TFRECORD)
ds = ds.map(parse_function)

n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()
sess.run(n)
# Create an input_fn that parses the tf.Examples from the given files,
# and split them into features and targets.
def tfrecord_input_fn(input_filenames, batch_size=100, num_epochs=None, shuffle=True):
    # Create a dataset and map features and labels.
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary.
    ds = ds.padded_batch(batch_size, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
def train_linear_classifier_text(learning_rate=.05, steps=2000, batch_size=100):
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    feature_columns = construct_feature_columns()

    classifier = tf.estimator.LinearClassifier(
      feature_columns=feature_columns,
      optimizer=my_optimizer,
    )

    classifier.train(
      input_fn=lambda: tfrecord_input_fn(TRAINING_TFRECORD, batch_size=batch_size),
      steps=steps)

    predict_training_input_fn = lambda: tfrecord_input_fn(TRAINING_TFRECORD,
                                  batch_size=batch_size,
                                  num_epochs=1, 
                                  shuffle=False)
    predict_validation_input_fn = lambda: tfrecord_input_fn(VALIDATION_TFRECORD, 
                                    batch_size=batch_size,
                                    num_epochs=1, 
                                    shuffle=False)
    
    return classifier, [predict_training_input_fn, predict_validation_input_fn]
                    #return these last two so you can call them later without editing hyperparameters
if (USING_OLD_MODELS == False) & (IS_DNN_CLASSIFIER == False):
    classifier, arr_funcs = train_linear_classifier_text(learning_rate=.01,
                                                        steps=1500,
                                                        batch_size=100)
if (USING_OLD_MODELS == False) & (IS_DNN_CLASSIFIER == False):
    #evaluation_metrics = classifier.evaluate(input_fn=arr_funcs[0])

    #print("Training set metrics:")
    #for m in evaluation_metrics:
    #    print(m, evaluation_metrics[m])
    #print("---")

    evaluation_metrics = classifier.evaluate(input_fn=arr_funcs[1])

    print("Test set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")
def train_dnn_classifier_text(learning_rate=.05, steps=2000, batch_size=100, hidden_units=[20,20], dropout=.1):
    #my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    feature_columns = construct_feature_columns()

##WHAT'S CHANGED:##
    classifier = tf.estimator.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=hidden_units,  #how many hidden units?:  https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
      optimizer=my_optimizer,
      dropout=dropout
    )

    classifier.train(
      input_fn=lambda: tfrecord_input_fn(TRAINING_TFRECORD, batch_size=batch_size),
      steps=steps)

    predict_training_input_fn = lambda: tfrecord_input_fn(TRAINING_TFRECORD,
                                  batch_size=batch_size,
                                  num_epochs=1, 
                                  shuffle=False)
    predict_validation_input_fn = lambda: tfrecord_input_fn(VALIDATION_TFRECORD, 
                                    batch_size=batch_size,
                                    num_epochs=1, 
                                    shuffle=False)
    
    return classifier, [predict_training_input_fn, predict_validation_input_fn]
                    #return these last two so you can call them later without editing hyperparameters
if (USING_OLD_MODELS == False) & (IS_DNN_CLASSIFIER):
    dnn_classifier, dnn_arr_funcs = train_dnn_classifier_text(learning_rate=.005,
                                                        steps=1500,
                                                        batch_size=100,
                                                        hidden_units=[6],
                                                        dropout=.1)
if (USING_OLD_MODELS == False) & (IS_DNN_CLASSIFIER):
    #evaluation_metrics = dnn_classifier.evaluate(input_fn=dnn_arr_funcs[0])

    #print("Training set metrics:")
    #for m in evaluation_metrics:
    #    print(m, evaluation_metrics[m])
    #print("---")

    evaluation_metrics = dnn_classifier.evaluate(input_fn=dnn_arr_funcs[1])

    print("Test set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")
#https://www.kaggle.com/skleinfeld/getting-started-with-the-donorschoose-data-set
predict_test_input_fn = lambda: tfrecord_input_fn(TEST_TFRECORD, 
                                    num_epochs=1, 
                                    shuffle=False)

if (USING_OLD_MODELS == False):
    if (IS_DNN_CLASSIFIER == False):
        predictions_generator = classifier.predict(input_fn=predict_test_input_fn)
    else:
        predictions_generator = dnn_classifier.predict(input_fn=predict_test_input_fn)
        
    predictions_list = list(predictions_generator)
    probabilities = [p["probabilities"][1] for p in predictions_list]
    print('Now have the probabilities.')
submission_values = pd.DataFrame({'id': combined_test_dataset['id'], 'project_is_approved': probabilities})
submission_values[0:5]
submission_values.to_csv('linear-spelling.csv', index=False)