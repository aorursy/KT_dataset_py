import os

import pandas as pd

import numpy as np

from functools import reduce

import matplotlib.pyplot as plt

import time

from IPython.display import display

pd.options.display.max_colwidth = 120

%matplotlib inline
input_root_dir = '../input/cord-19-eda-parse-json-and-generate-clean-csv'

input_files_list = [f for f in os.listdir(input_root_dir) if os.path.splitext(f)[1] == '.csv']

df_list = []

for input_file in input_files_list:

    df_next = pd.read_csv(os.path.join(input_root_dir, input_file))

    df_next['source_dataset'] = input_file

    df_list.append(df_next)

df_all_papers = pd.concat(df_list).reset_index(drop=True)
df_all_papers.head(3)
df_all_papers.describe()
print(f"Number of missing texts: {df_all_papers['text'].isnull().sum()}")
print(f"Number of missing titles: {df_all_papers['title'].isnull().sum()}")
print(f"Number of missing authors: {df_all_papers['authors'].isnull().sum()}")
source_datasets_of_duplicates = list(df_all_papers.loc[df_all_papers['text'].duplicated(keep=False), 'source_dataset'].values)

duplicated_text_bool_idx = df_all_papers['text'].duplicated()

df_all_papers = df_all_papers[np.logical_not(duplicated_text_bool_idx)]

print(f'{sum(duplicated_text_bool_idx)} papers with duplicated text were removed.')
removed_duplicates_total_count = 10
missing_abstract_bool_idx = df_all_papers['abstract'].isnull()

print(f"Number of missing abstracts: {missing_abstract_bool_idx.sum()}")
indices = df_all_papers.loc[~df_all_papers['abstract'].isnull() & df_all_papers['abstract'].duplicated(keep=False),

                  'abstract'].sort_values().index



def get_word_count(text):

    # when missing

    if isinstance(text, float):

        return 0

    return len(text.replace('\n', ' ').split())

    

def check_duplicates_by_sorted_indices(indices, duplication_field, max_number_of_groups=50, max_group_size=20, return_shown_indices=False):

    cols = ['title', 'authors', 'abstract', 'text']

    if not duplication_field in cols:

        cols += [duplication_field]

    duplicates_to_check = df_all_papers.loc[indices, cols]    

    duplication_field_i = duplicates_to_check.columns.to_list().index(duplication_field)    

    row_i = 0

    duplication_count = 0

    group_i = 0

    if return_shown_indices:

        indices_shown = []

    while row_i + duplication_count + 1 < len(duplicates_to_check) and group_i < max_number_of_groups:

        while (row_i + duplication_count + 1 < len(duplicates_to_check) and 

               duplicates_to_check.iloc[row_i, duplication_field_i] == duplicates_to_check.iloc[row_i + duplication_count + 1, duplication_field_i]):

            duplication_count += 1 

        group_i += 1

        if duplication_count + 1 > max_group_size:

            continue

        

        print(f'Group {group_i + 1} of potential duplicates')

        dups_group = duplicates_to_check.iloc[row_i: row_i + duplication_count + 1]

        if return_shown_indices:

            indices_shown.extend(dups_group.index)

        display(dups_group)

        text_lens = dups_group['text'].map(get_word_count).values

        if reduce(lambda x, y: x == y, text_lens):

            print('Text have equal length.')

        else:

            print(f"Text lengths are {', '.join(map(str, text_lens))}.")

        print('='*90)

        row_i += duplication_count + 1

        duplication_count = 0

    if return_shown_indices:

        return indices_shown

                  

check_duplicates_by_sorted_indices(indices, 'abstract')
missing_affiliations_count = df_all_papers['affiliations'].isnull().sum()

print(f"Number of missing affiliations: {missing_affiliations_count}")

non_missing_affiliations = df_all_papers.loc[np.logical_not(df_all_papers['affiliations'].isnull()), 'affiliations']

duplicated_affiliations = non_missing_affiliations[non_missing_affiliations.duplicated()].unique()

print(f"Number of duplicated affiliations: {len(duplicated_affiliations)}")
duplicated_affiliations_bool_idx = df_all_papers['affiliations'].isin(duplicated_affiliations)

title_affilation_series = (df_all_papers['title'].map(lambda x: [x]) +

                           df_all_papers['affiliations'].map(lambda x: [x]))

removal_candidates_title_affiliation = title_affilation_series[duplicated_affiliations_bool_idx & 

                                                               title_affilation_series.map(tuple).duplicated(keep=False)]
print(f'Number of duplicates: {len(removal_candidates_title_affiliation)}.')
indices = (removal_candidates_title_affiliation

           .map(lambda x: ', '.join(filter(lambda i: not isinstance(i, float), x)))

           .sort_values().index)



check_duplicates_by_sorted_indices(indices, 'affiliations')
merged_to_parent_papers_total_count = 0

source_datasets_of_merged_supps = []

source_datasets_of_parents_for_merged_supps = []



def drop_strategy_developed_for_affiliations(indices):

    global merged_to_parent_papers_total_count

    global removed_duplicates_total_count

    global source_datasets_of_duplicates

    global source_datasets_of_merged_supps

    global source_datasets_of_parents_for_merged_supps

    

    merged_to_parent_papers_total_count_before = merged_to_parent_papers_total_count

    indices_to_drop = []

    duplicates_to_check = df_all_papers.loc[indices, ['title', 'abstract', 'text']]

    for row_i in range(0, len(duplicates_to_check), 2):

        dups_pair = duplicates_to_check.iloc[row_i: row_i + 2]

        text_lens = dups_pair['text'].map(get_word_count).values

        missing_abstracts = dups_pair['abstract'].map(lambda x: isinstance(x, float)).values

        if text_lens[0] == text_lens[1]:

            if missing_abstracts[0]:

                indices_to_drop.append(duplicates_to_check.index[row_i])

            else:

                indices_to_drop.append(duplicates_to_check.index[row_i + 1]) # including the case when both abstracts are present

            source_datasets_of_duplicates.extend(df_all_papers.loc[dups_pair.index, 'source_dataset'].values)

        else:

            missing_titles = dups_pair['title'].map(lambda x: isinstance(x, float)).values

            shorter_paper_i = np.argmin(text_lens)

            if not missing_titles[0] and not missing_titles[1]:

                shorter_paper_index = duplicates_to_check.index[row_i + shorter_paper_i]

                indices_to_drop.append(shorter_paper_index)

                merged_to_parent_papers_total_count += 1

                source_datasets_of_merged_supps.append(df_all_papers.loc[shorter_paper_index, 'source_dataset'])

                if not missing_abstracts[(shorter_paper_i+1) % 2] and missing_abstracts[shorter_paper_i]:

                    longer_paper_index = duplicates_to_check.index[row_i + (shorter_paper_i+1) % 2]

                    df_all_papers.loc[longer_paper_index, 'text'] += ' ' + df_all_papers.loc[shorter_paper_index, 'text']

                    source_datasets_of_parents_for_merged_supps.append(df_all_papers.loc[longer_paper_index, 'source_dataset'])

    df_all_papers.drop(indices_to_drop, inplace=True)

    number_of_merged_papers = merged_to_parent_papers_total_count - merged_to_parent_papers_total_count_before

    removed_duplicates_total_count += len(indices_to_drop) - number_of_merged_papers

    print(f'{len(indices_to_drop) - number_of_merged_papers} items were removed.')

    print(f'{number_of_merged_papers} items were merged to parents.')



drop_strategy_developed_for_affiliations(indices)
non_missing_authors = df_all_papers.loc[np.logical_not(df_all_papers['authors'].isnull()), 'authors']

duplicated_authors = non_missing_authors[non_missing_authors.duplicated()].unique()

duplicated_authors_bool_idx = df_all_papers['authors'].isin(duplicated_authors)

title_authors_series = (df_all_papers['title'].map(lambda x: [x]) +

                        df_all_papers['authors'].map(lambda x: [x]))

removal_candidates_title_author = title_authors_series[duplicated_authors_bool_idx & 

                                                       title_authors_series.map(tuple).duplicated(keep=False)]

print(f'Number of duplicates: {len(removal_candidates_title_author)}.')
indices = (removal_candidates_title_author

           .map(lambda x: ', '.join(filter(lambda i: not isinstance(i, float), x)))

           .sort_values().index)



check_duplicates_by_sorted_indices(indices, 'authors')
drop_strategy_developed_for_affiliations(indices)
indices = df_all_papers.loc[~df_all_papers['title'].isnull() & df_all_papers['title'].duplicated(keep=False),

                            'title'].sort_values().index

print(f'Number of papers with non-unique title: {len(indices)}.')              
check_duplicates_by_sorted_indices(indices, 'title', max_number_of_groups=20)
indices = df_all_papers.loc[~df_all_papers['title'].isnull() & 

                            df_all_papers['title'].duplicated(keep=False) & 

                            (df_all_papers['title'].map(get_word_count) > 11),

                            'title'].sort_values().index

indices = check_duplicates_by_sorted_indices(indices, 'title', max_group_size=2, return_shown_indices=True)
drop_strategy_developed_for_affiliations(indices)
fig, ax = plt.subplots(figsize=(7, 7))



# credits: https://stackoverflow.com/questions/6170246/how-do-i-use-matplotlib-autopct

def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.0f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct



processed_duplicates_counts = [removed_duplicates_total_count, merged_to_parent_papers_total_count]

ax.pie(processed_duplicates_counts, labels=['Duplicates removed', 'Merged to parent as supplementary material'], 

       autopct=make_autopct(processed_duplicates_counts), shadow=True, startangle=90)

ax.axis('equal')

_ = ax.set_title('How the discovered duplicates were processed', fontsize=20)
duplicate_source_dataset_counts = pd.Series(source_datasets_of_duplicates).value_counts()

duplicate_source_dataset_counts.plot(kind='barh',

                                     title='Counts of removed duplicates per dataset', 

                                     color=['dodgerblue'])
merged_source_dataset_counts = pd.Series(source_datasets_of_merged_supps).value_counts()

merged_source_dataset_counts.plot(kind='barh',

                                  title='Counts of the merged shorter papers per dataset', 

                                  color=['dodgerblue'])
parent_for_merged_source_dataset_counts = pd.Series(source_datasets_of_parents_for_merged_supps).value_counts()

parent_for_merged_source_dataset_counts.plot(kind='barh',

                                             title='Counts of the merged longer papers per dataset', 

                                             color=['dodgerblue'])
output_file_name = f"master_dataset_cleaned_{time.strftime('%Y%m%d_%H%M')}.csv"

df_all_papers.to_csv(output_file_name, index=None)