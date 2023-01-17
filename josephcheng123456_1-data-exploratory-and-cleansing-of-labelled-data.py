# Import and read Metadata
import pandas as pd
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from os import listdir
from tqdm import tqdm
import numpy as np

# Key Data Paths
data_path = '../input/CORD-19-research-challenge'
# data_path = './551982_1230614_bundle_archive/'
priority_question_path = data_path + '/Kaggle/target_tables/2_relevant_factors/'
# Loading the meta data and relative questions path

meta_data=pd.read_csv(data_path + '/metadata.csv')
print("Column names: {}".format(meta_data.columns))
print("number of rows: ", len(meta_data))
meta_data.head(5)
plt.figure(figsize=(20,10))
na_analysis = meta_data.isna().sum()
na_analysis.sort_values().plot(kind='bar', stacked=True, x = 'columns', y = 'count')
question_1 = 'Effectiveness of a multifactorial strategy to prevent secondary transmission.csv'
priority_question_1 =pd.read_csv(priority_question_path + question_1,index_col=0)
print("Column names: {}".format(priority_question_1.columns))
print("number of rows: ", len(priority_question_1))
priority_question_1.head(10)
priority_question_1.shape
# Testing match between metadata and question 1
match_list = []
for index, row in priority_question_1.iterrows():
    url_row = meta_data.loc[meta_data['url'] == row['Study Link']]['url'].any()
    title_row = meta_data.loc[meta_data['title'] == row['Study']]['title'].any()
    if url_row != False or title_row != False:
        match_list.append(True)
    else:
        match_list.append(False)
print("percentage of matched list is", sum(match_list), "/", len(match_list))
# Testing the duplication relationship between priority question 1 and literatures
duplicateRowsDF = priority_question_1[priority_question_1.duplicated(['Study'])]
print("Duplicate Rows based on a single column are:", duplicateRowsDF, sep='\n')
# Listing all the priority questions 
priority_question_list = [f for f in listdir(priority_question_path) if isfile(join(priority_question_path, f))]
for question in priority_question_list:
    match_list = []
    priority_question_df = pd.read_csv(priority_question_path + question,index_col=0)
    for index, row in priority_question_df.iterrows():
        url_row = meta_data.loc[meta_data['url'] == row['Study Link']]['url'].any()
        title_row = meta_data.loc[meta_data['title'] == row['Study']]['title'].any()
        if url_row != False or title_row != False:
            match_list.append(True)
        else:
            match_list.append(False)
    print(question)
    print("percentage of matched list is", sum(match_list), "/", len(match_list), " = ", sum(match_list)/len(match_list))
# Checking the quality of the csv file for each questions
for question in priority_question_list:
    print("file name is: ",question)
    priority_question_df = pd.read_csv(priority_question_path + question,index_col=0)
    priority_question_df.info()
# Obtaining all files names from all priority_question list into one dataframe
# Metadata -> We want pdf_json_files, pmc_json_files

pdf_json_files_list = [];
pmc_json_files_list = [];
research_topic_list = [];
topic_id_list = [];
topic_id = 1
combined_ques_literature = [];
for question in tqdm(priority_question_list):
    print("file name is: ",question)
    priority_question_df = pd.read_csv(priority_question_path + question,index_col=0)
    combined_ques_literature.append(priority_question_df)
    for index, row in priority_question_df.iterrows():
        # Match using URL or title
        url_match_row = meta_data.loc[meta_data['url'] == row['Study Link']]
        title_match_row = meta_data.loc[meta_data['title'] == row['Study']]
        if not url_match_row.empty or not title_match_row.empty:
            if not url_match_row.empty:
                pdf_json_files_list.append(url_match_row['pdf_json_files'].values[0])
                pmc_json_files_list.append(url_match_row['pmc_json_files'].values[0])
            else:
                pdf_json_files_list.append(title_match_row['pdf_json_files'].values[0])
                pmc_json_files_list.append(title_match_row['pmc_json_files'].values[0])
        else:
            pdf_json_files_list.append(" ")
            pmc_json_files_list.append(" ")
        topic_id_list.append(topic_id)
        research_topic_list.append(question.split('.')[0])
    topic_id = topic_id + 1

# combining all datarfame into one big dataframe
combined_ques_literature = pd.concat(combined_ques_literature);
# Checking if previous result provides the correct dimension for the output
print(len(topic_id_list), len(pmc_json_files_list), len(pdf_json_files_list), len(research_topic_list))
print("before adding columns: ", combined_ques_literature.shape)
print(combined_ques_literature.columns)
combined_ques_literature.insert(1, "topic_id", topic_id_list, True)
combined_ques_literature.insert(2, "research_topic", research_topic_list, True)
combined_ques_literature.insert(3, "pdf_json_files", pdf_json_files_list, True)
combined_ques_literature.insert(4, "pmc_json_files", pmc_json_files_list, True)

print("after adding columns: ", combined_ques_literature.shape)
combined_ques_literature[['Influential','Infuential','Influential (Y/N)']].info()
combined_ques_literature[['Factors', 'Factors Described']].info()
combined_ques_literature[['Date', 'Date Published']].info()
combined_ques_literature['Influential'] = combined_ques_literature['Influential'].fillna(combined_ques_literature['Infuential'])
combined_ques_literature['Influential'] = combined_ques_literature['Influential'].fillna(combined_ques_literature['Influential (Y/N)'])            
combined_ques_literature['Factors'] = combined_ques_literature['Factors'].fillna(combined_ques_literature['Factors Described'])
combined_ques_literature['Date'] = combined_ques_literature['Date'].fillna(combined_ques_literature['Date Published'])
combined_ques_literature = combined_ques_literature.drop(['Infuential', 'Influential (Y/N)', 'Factors Described', 'Date Published'], axis=1)
print(combined_ques_literature.info())
# Drop all the na and spaces on the pdf_json_file to make sure all labelled data is linked to a literature stored in database

scoped_categorised_literature = combined_ques_literature.dropna(subset=['pdf_json_files'])
scoped_categorised_literature = scoped_categorised_literature[~scoped_categorised_literature['pdf_json_files'].str.isspace()] 
scoped_categorised_literature.info()
scoped_categorised_literature.to_pickle("./1_scoped_cat_lit.pkl")
