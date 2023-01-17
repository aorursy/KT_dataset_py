# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
# Any results you write to the current directory are saved as output.
from datetime import datetime
from IPython.core.display import display, HTML
import math
#import json
bln_ready_to_commit = True
bln_create_estimate_files = True
bln_upload_input_estimates = True
bln_recode_variables = True
pd.set_option("display.max_rows", 101)
pd.set_option("display.max_columns", 25)

df_time_check = pd.DataFrame(columns=['Stage','Start','End', 'Seconds', 'Minutes'])
int_time_check = 0
dat_start = datetime.now()
dat_program_start = dat_start

if not bln_ready_to_commit:
    int_read_csv_rows = 100000
else:
    int_read_csv_rows= None
    
# generate crosstabs  {0 = nothing; 1 = screen}
int_important_crosstab = 1
int_past_crosstab = 0
int_current_crosstab = 1

print('input:\n', os.listdir("../input"))
print('\nga-customer-revenue-prediction:\n', os.listdir("../input/ga-customer-revenue-prediction"))
print('\ndd8-files:\n', os.listdir("../input/dd8-files"))
import csv
import bq_helper
from bq_helper import BigQueryHelper
#bq_train = BigQueryHelper("kaggle-public-datasets", "ga_train_set")
#bq_test = BigQueryHelper("kaggle-public-datasets", "ga_test_set")
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
#google_analytics = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
#                                   dataset_name="data:google_analytics_sample")
#bq_assistant.head("ga_sessions_20160801", num_rows=3)

bln_run_stage1 = False
bln_run_stage2 = False
bln_run_stage3 = False
bln_run_stage4 = False
bln_run_stage5 = False
bln_run_stage6 = False
bln_run_stage7 = False
bln_run_stage8 = False
bln_run_stage9 = False
bln_run_stage10 = False
bln_run_stage11 = False
    

flt_query_limit = 0.1
int_sample_records = 15

csv_query_info = open('dd8_query_info.csv', 'w')
query_writer = csv.writer(csv_query_info)
query_writer.writerow( ['query', 'size1_gb', 'size2_gb'] )

def select_from_dataset_table(str_dataset, str_table):
    return str_dataset + "." + str_table  

def create_simple_query1(str_dataset, str_columns, str_table, str_where):
    str_select_from_dataset_table = select_from_dataset_table(str_dataset, str_table)
    query = "SELECT " + str_columns + \
        """ \nFROM `""" + str_select_from_dataset_table + """` """ + \
        "\nWHERE " + str_where + " "
    return query

def create_simple_query2(str_dataset, str_select_columns, str_table, str_group_by_columns):
    str_select_from_dataset_table = select_from_dataset_table(str_dataset, str_table)
    query = "SELECT " + str_select_columns + \
            "\nFROM `" + str_select_from_dataset_table + "` " + \
            "\nGROUP BY " + str_group_by_columns
    return query

def create_simple_query3(int_start_date, int_end_date, str_train_or_test, str_select_columns, str_where):
    str_dataset = get_train_or_test_dataset(str_train_or_test)

    str_return_query = "WITH SUB1 AS ( \n"
    csv_file_read = open('../input/dd8-files/dd8_table_list_' + str_train_or_test + '.csv', 'r')
    csv_reader = csv.reader(csv_file_read, delimiter=',')

    int_row = 0
    for row in csv_reader:
        if int_row > 0:
            int_id = row[0]
            str_table = row[1]
            int_table_date = int(str_table[-8:])
            if int_table_date >= int_start_date and int_table_date <= int_end_date:
                str_select_from_table = str_table
                query_temp = create_simple_query1(str_dataset, str_select_columns, str_select_from_table, str_where)
                if int_table_date < int_end_date:
                    query_temp = query_temp + "UNION ALL \n"
                str_return_query = str_return_query + query_temp
            
        int_row += 1
    str_return_query = str_return_query + ") \n"
    return str_return_query

def create_simple_query4(str_select_columns, str_table, str_group_by_columns):
    query = "SELECT " + str_select_columns + \
            "\nFROM `" + str_table + "` " + \
            "\nGROUP BY " + str_group_by_columns
    return query

def get_flt_query_size_mb(flt_size):
    flt_return = flt_size * 1000
    return flt_return

def get_str_query_size_mb(flt_size):
    str_return = str( get_flt_query_size_mb(flt_size) )
    return str_return

def get_query_size(flt_size):
    # assuming there is a minimum - can't check at time of setting this up (22aug2018)
    if flt_size < 0.01:
        return 0.01
    else:
        return flt_size

def get_train_or_test_dataset(str_train_or_test):
    if str_train_or_test == 'train':
        str_dataset = "kaggle-public-datasets.ga_train_set"
    elif str_train_or_test == 'test':
        str_dataset = "kaggle-public-datasets.ga_test_set"
    else:
        str_dataset = "not_defined"
    return str_dataset

def get_train_or_test_bigqueryhelper(str_train_or_test):
    if str_train_or_test == 'train':
        bigqueryhelper = BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_train_set")
    elif str_train_or_test == 'test':
        bigqueryhelper = BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_test_set")
    else:
        bigqueryhelper = None
    return bigqueryhelper

def run_queries_2(bln_create_df, str_train_or_test, int_start_date, int_end_date, str_select_columns, str_group_by_columns):
    str_dataset = get_train_or_test_dataset(str_train_or_test)
    bigqueryhelper = get_train_or_test_bigqueryhelper(str_train_or_test)
    csv_file_read = open('../input/dd8-files/dd8_table_list_' + str_train_or_test + '.csv', 'r')

    csv_reader = csv.reader(csv_file_read, delimiter=',')

    flt_est_query_size_total = 0.0
    int_query_count = 0
    print('created dataframes:', bln_create_df)
    print('sample queries run are shown below:\n')
    int_row = 0
    for row in csv_reader:
        if int_row > 0:
            int_id = row[0]
            str_table = row[1]
            int_table_date = int(str_table[-8:])
            if int_table_date >= int_start_date and int_table_date <= int_end_date:
                str_select_from_table = str_table
                if str_group_by_columns != "":
                    query = create_simple_query2(str_dataset, str_select_columns, str_select_from_table, str_group_by_columns)
                else:
                    query = create_simple_query1(str_dataset, str_select_columns, str_select_from_table)
                flt_est1_query_size = bigqueryhelper.estimate_query_size(query)
                flt_est2_query_size = get_query_size(flt_est1_query_size) 
                flt_est_query_size_total += flt_est2_query_size
                int_query_count += 1
                if int_table_date == int_start_date or int_table_date == int_end_date or (int_query_count % 20 == 0):
                    print(query)
                    print('size1mb:', get_str_query_size_mb(flt_est1_query_size), ' size2mb: ', get_str_query_size_mb(flt_est2_query_size) ) 
                    print()
            
                if bln_create_df:
                    df_temp = bigqueryhelper.query_to_pandas_safe(query, max_gb_scanned=flt_query_limit)
                    if int_query_count == 1:
                        df_query = df_temp
                    else:
                        df_query = pd.concat([df_query, df_temp])
            
                query_writer.writerow( [query, flt_est1_query_size, flt_est2_query_size] )
        int_row += 1
    csv_file_read.close()
    print('the number of lines read in input table list file: ', int_row)
    print('the number of queries run: ', int_query_count)
    print('total estimated size of queries run (mb):', get_str_query_size_mb(flt_est_query_size_total) )
    if bln_create_df:
        return df_query

def get_sample_train_data(str_columns, str_date):
    # darryldias 19sep2018
    query = "SELECT fullVisitorId, " + str_columns + " \
        FROM `kaggle-public-datasets.ga_train_set.ga_sessions_" + str_date + "`" 
    ds_current = BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_train_set")
    flt_est1_query_size = ds_current.estimate_query_size(query)
    flt_est2_query_size = get_query_size(flt_est1_query_size)
    print('size1', get_str_query_size_mb(flt_est1_query_size))
    print('size2', get_str_query_size_mb(flt_est2_query_size))

    df_return = ds_current.query_to_pandas_safe(query, max_gb_scanned=flt_query_limit)
    return df_return
    
    
def add_csv_data(df_input, str_csv_file, bln_show_message):
    if bln_show_message:
        print ('processing ' + str_csv_file)
    df_temp = pd.read_csv(str_csv_file)
    df_input = pd.concat([df_input, df_temp])
    return df_input

def get_csv_row_count(str_file):
    csv_file_read = open(str_file, 'r')
    csv_reader = csv.reader(csv_file_read, delimiter=',')
    int_row = 0
    for row in csv_reader:
        int_row += 1
    csv_file_read.close()
    return int_row    

def dict_get_count(dict_input, str_key):
    int_return = 0
    for key, value in dict_input.items():
        if key == str_key:
            int_return = value
            break
    return int_return

def create_csv_from_dict(dict_input, str_filename, list_input_headings):
    csvfile_w1 = open(str_filename, 'w')
    writer1 = csv.writer(csvfile_w1)
    writer1.writerow( list_input_headings )
    for key, value in dict_input.items():
        writer1.writerow( [key, value] )
    csvfile_w1.close()
    print('\nCreated file ', str_filename)

#bln_testing = False
#int_start_date = 20160801
#if bln_testing:
#    int_end_date = 20160802
#else:
#    int_end_date = 20170731
def get_translations_analysis_description(df_input, str_language, str_group, int_code):
    # created by darryldias 25may2018
    df_temp = df_input[(df_input['language']==str_language) & (df_input['group']==str_group) & (df_input['code']==int_code)] \
                    ['description']
    return df_temp.iloc[0]

#translations_analysis = pd.read_csv('../input/ulabox-translations-analysis/translations_analysis.csv')
strg_count_column = 'count'   #get_translations_analysis_description(translations_analysis, str_language, 'special', 2)

def start_time_check():
    # created by darryldias 21may2018 - updated 8june2018
    global dat_start 
    dat_start = datetime.now()
    
def end_time_check(dat_start, str_stage):
    # created by darryldias 21may2018 - updated 8june2018
    global int_time_check
    global df_time_check
    int_time_check += 1
    dat_end = datetime.now()
    diff_seconds = (dat_end-dat_start).total_seconds()
    diff_minutes = diff_seconds / 60.0
    df_time_check.loc[int_time_check] = [str_stage, dat_start, dat_end, diff_seconds, diff_minutes]

def create_topline(df_input, str_item_column, str_count_column):
    # created by darryldias 21may2018; updated by darryldias 29may2018
    str_percent_column = 'percent'   #get_translations_analysis_description(translations_analysis, str_language, 'special', 3)
    df_temp = df_input.groupby(str_item_column).size().reset_index(name=str_count_column)
    df_output = pd.DataFrame(columns=[str_item_column, str_count_column, str_percent_column])
    int_rows = df_temp.shape[0]
    int_columns = df_temp.shape[1]
    int_total = df_temp[str_count_column].sum()
    flt_total = float(int_total)
    for i in range(int_rows):
        str_item = df_temp.iloc[i][0]
        int_count = df_temp.iloc[i][1]
        flt_percent = round(int_count / flt_total * 100, 1)
        df_output.loc[i] = [str_item, int_count, flt_percent]
    
    df_output.loc[int_rows] = ['total', int_total, 100.0]
    return df_output        

def get_dataframe_info(df_input, bln_output_csv, str_filename):
    # created by darryldias 24may2018 - updated 7june2018
    int_rows = df_input.shape[0]
    int_cols = df_input.shape[1]
    flt_rows = float(int_rows)
    
    df_output = pd.DataFrame(columns=["Column", "Type", "Not Null", 'Null', '% Not Null', '% Null'])
    df_output.loc[0] = ['Table Row Count', '', int_rows, '', '', '']
    df_output.loc[1] = ['Table Column Count', '', int_cols, '', '', '']
    int_table_row = 1
    for i in range(int_cols):
        str_column_name = df_input.columns.values[i]
        str_column_type = df_input.dtypes.values[i]
        int_not_null = df_input[str_column_name].count()
        int_null = sum( pd.isnull(df_input[str_column_name]) )
        flt_percent_not_null = round(int_not_null / flt_rows * 100, 1)
        flt_percent_null = round(100 - flt_percent_not_null, 1)
        int_table_row += 1
        df_output.loc[int_table_row] = [str_column_name, str_column_type, int_not_null, int_null, flt_percent_not_null, flt_percent_null]

    if bln_output_csv:
        df_output.to_csv(str_filename)
        print ('Dataframe information output created in file: ' + str_filename)
        return None
    return df_output

def check_numeric_var(str_question, int_groups):
    # created by darryldias 3jul2018  
    #print(df_output.iloc[3][2])
    flt_min = application_all[str_question].min()
    flt_max = application_all[str_question].max()
    flt_range = flt_max - flt_min 
    flt_interval = flt_range / int_groups 
    df_output = pd.DataFrame(columns=['interval', 'value', 'count', 'percent', 'code1', 'code2'])

    int_total = application_all[ (application_all[str_question] <= flt_max) ][str_question].count()
    for i in range(0, int_groups + 1):
        flt_curr_interval = i * flt_interval
        flt_value = flt_min + flt_curr_interval
        int_count = application_all[ (application_all[str_question] <= flt_value) ][str_question].count()
        flt_percent = int_count /  int_total * 100.0
        str_code_value = "{0:.6f}".format(flt_value)
        str_code1 = "if row['" + str_question + "'] <= " + str_code_value + ":"
        str_code2 = "return '(x to " + str_code_value + "]'"
        df_output.loc[i] = [flt_curr_interval, flt_value, int_count, flt_percent, str_code1, str_code2]

    return df_output

def get_column_analysis(int_analysis, int_code):
    # created by darryldias 24jul2018 
    if int_code == 1:
        return ['overall', 'test', 'train', 'no rev', 'rev', '(000-025]', '(025-050]', '(050-100]', '(100+']
    elif int_code == 2:
        return ['overall', 'train or test', 'train or test', 'rev_sum_div_s1d', 'rev_sum_div_s1d', \
                'rev_sum_div_s2d', 'rev_sum_div_s2d', 'rev_sum_div_s2d', 'rev_sum_div_s2d']
    elif int_code == 3:
        return ['yes', 'test', 'train', 'no rev', 'rev', '(000 - 025]', '(025 - 050]', '(050 - 100]', '(100 +']
    else:
        return None

def create_crosstab_type1(df_input, str_row_question, int_output_destination):
    # created by darryldias 10jun2018 - updated 27sep2018 
    # got some useful code from:
    # https://chrisalbon.com/python/data_wrangling/pandas_missing_data/
    # https://www.tutorialspoint.com/python/python_lists.htm
    # https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points

    if int_output_destination == 0:
        return None
    
    str_count_desc = 'count'  #get_translations_analysis_description(translations_analysis, str_language, 'special', 3)
    str_colpercent_desc = 'col percent'
    
    list_str_column_desc = get_column_analysis(1, 1)
    list_str_column_question = get_column_analysis(1, 2)
    list_str_column_category = get_column_analysis(1, 3)
    int_columns = len(list_str_column_desc)
    list_int_column_base = []
    list_flt_column_base_percent = []
    
    df_group = df_input.groupby(str_row_question).size().reset_index(name='count')
    int_rows = df_group.shape[0]

    for j in range(int_columns):
        int_count = df_input[ df_input[str_row_question].notnull() & (df_input[list_str_column_question[j]]==list_str_column_category[j]) ] \
                                [list_str_column_question[j]].count()
        list_int_column_base.append(int_count)
        if int_count == 0:
            list_flt_column_base_percent.append('')
        else:
            list_flt_column_base_percent.append('100.0')
        
    list_output = []
    list_output.append('row_question')
    list_output.append('row_category')
    list_output.append('statistic')
    for k in range(1, int_columns+1):
        str_temp = 'c' + str(k)
        list_output.append(str_temp)
    df_output = pd.DataFrame(columns=list_output)

    int_row = 1
    list_output = []
    list_output.append(str_row_question)
    list_output.append('')
    list_output.append('')
    for k in range(int_columns):
        list_output.append(list_str_column_desc[k])
    df_output.loc[int_row] = list_output
    
    int_row = 2
    list_output = []
    list_output.append(str_row_question)
    list_output.append('total')
    list_output.append(str_count_desc)
    for k in range(int_columns):
        list_output.append(list_int_column_base[k])
    df_output.loc[int_row] = list_output
    
    int_row = 3
    list_output = []
    list_output.append(str_row_question)
    list_output.append('total')
    list_output.append(str_colpercent_desc)
    for k in range(int_columns):
        list_output.append(list_flt_column_base_percent[k])
    df_output.loc[int_row] = list_output

    for i in range(int_rows):
        int_row += 1
        int_count_row = int_row
        int_row += 1
        int_colpercent_row = int_row

        str_row_category = df_group.iloc[i][0]

        list_int_column_count = []
        list_flt_column_percent = []
        for j in range(int_columns):
            int_count = df_input[ (df_input[str_row_question]==str_row_category) & \
                                  (df_input[list_str_column_question[j]]==list_str_column_category[j]) ] \
                                [list_str_column_question[j]].count()
            list_int_column_count.append(int_count)
            flt_base = float(list_int_column_base[j])
            if flt_base > 0:
                flt_percent = round(100 * int_count / flt_base,1)
                str_percent = "{0:.1f}".format(flt_percent)
            else:
                str_percent = ''
            list_flt_column_percent.append(str_percent)
        
        list_output = []
        list_output.append(str_row_question)
        list_output.append(str_row_category)
        list_output.append(str_count_desc)
        for k in range(int_columns):
            list_output.append(list_int_column_count[k])
        df_output.loc[int_count_row] = list_output
        
        list_output = []
        list_output.append(str_row_question)
        list_output.append(str_row_category)
        list_output.append(str_colpercent_desc)
        for k in range(int_columns):
            list_output.append(list_flt_column_percent[k])
        df_output.loc[int_colpercent_row] = list_output
        
    return df_output        

def get_ct_statistic2(df_input, str_row_question, str_col_question, str_col_category, str_statistic):
    # created by darryldias 17jul2018
    if str_statistic == 'total':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].isnull().count() 
    elif str_statistic == 'notnull':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].count() 
    elif str_statistic == 'null':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].isnull().sum() 
    elif str_statistic == 'mean':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].mean() 
    elif str_statistic == 'median':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].median() 
    elif str_statistic == 'minimum':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].min() 
    elif str_statistic == 'maximum':
        int_temp = df_input[ (df_input[str_col_question] == str_col_category) ][str_row_question].max() 
    else:
        int_temp = None
    return int_temp
 
def create_crosstab_type2(df_input, str_row_question, int_output_destination):
    # created by darryldias 24jul2018
    if int_output_destination == 0:
        return None

    list_str_column_desc = get_column_analysis(1, 1)
    list_str_column_question = get_column_analysis(1, 2)
    list_str_column_category = get_column_analysis(1, 3)
    int_analysis_columns = len(list_str_column_question)

    list_str_statistics = ['total', 'notnull', 'null', 'mean', 'median', 'minimum', 'maximum']
    list_str_counts = ['total', 'notnull', 'null']
    int_statistics = len(list_str_statistics)

    df_output = pd.DataFrame(columns=['row_question', 'row_category', 'statistic', 'c1', 'c2', 'c3', 'c4', 'c5'])
    int_row = 1

    list_values = []
    list_values.append(str_row_question)
    list_values.append('')
    list_values.append('')
    for j in range(int_analysis_columns):
        list_values.append(list_str_column_desc[j])
    df_output.loc[int_row] = list_values

    for i in range(int_statistics):
        str_statistic = list_str_statistics[i] 
        list_values = []
        list_values.append(str_row_question)
        if str_statistic in list_str_counts:
            list_values.append(str_statistic)
            list_values.append('count')
        else:
            list_values.append('numeric')
            list_values.append(str_statistic)
    
        for j in range(int_analysis_columns):
            str_col_question = list_str_column_question[j]
            str_col_category = list_str_column_category[j]
            num_statistic = get_ct_statistic2(df_input, str_row_question, str_col_question, str_col_category, str_statistic)
            list_values.append(num_statistic)
        int_row += 1
        df_output.loc[int_row] = list_values
    return df_output


def percent_summary_1 (row, str_input_column):
    # created by darryldias 27may2018   
    if row[str_input_column] == 0 :   
        return 'no'
    if row[str_input_column] > 0 :
        return 'yes'
    return 'Unknown'

def month_description (row, str_input_column):
    # created by darryldias 1june2018   
    if row[str_input_column] == 1 :   
        return 'Jan'
    if row[str_input_column] == 2 :   
        return 'Feb'
    if row[str_input_column] == 3 :   
        return 'Mar'
    if row[str_input_column] == 4 :   
        return 'Apr'
    if row[str_input_column] == 5 :   
        return 'May'
    if row[str_input_column] == 6 :   
        return 'Jun'
    if row[str_input_column] == 7 :   
        return 'Jul'
    if row[str_input_column] == 8 :   
        return 'Aug'
    if row[str_input_column] == 9 :   
        return 'Sep'
    if row[str_input_column] == 10 :   
        return 'Oct'
    if row[str_input_column] == 11 :   
        return 'Nov'
    if row[str_input_column] == 12 :   
        return 'Dec'
    return 'Unknown'

def year_month_code1 (row, str_input_column_year, str_input_column_month):
    # created by darryldias 1june2018   
    if row[str_input_column_month] <= 9 :   
        return int(str(row[str_input_column_year]) + '0' + str(row[str_input_column_month]))
    if row[str_input_column_month] <= 12 :   
        return int(str(row[str_input_column_year]) + str(row[str_input_column_month]))
    return 0

def year_month_code2 (row, str_input_column):
    str_date = str(row[str_input_column])
    return int(str_date[:6])

def year_month_code3 (row, str_input_column):
    int_date = row[str_input_column]
    if int_date > 0:
        str_date = str(int_date)
        return int(str_date[:6])
    else:
        return None
    
def n_0_1_summary (row, str_input_column):
    # created by darryldias 11jun2018   
    if row[str_input_column] == 0 :   
        return '0'
    if row[str_input_column] == 1 :
        return '1'
    return 'Unknown'

def n_0_1_summary2 (row, str_input_column):
    # created by darryldias 28jun2018   
    if row[str_input_column] <= 0.1 :   
        return '(0 to 0.1]'
    if row[str_input_column] <= 0.2 :   
        return '(0.1 to 0.2]'
    if row[str_input_column] <= 0.3 :   
        return '(0.2 to 0.3]'
    if row[str_input_column] <= 0.4 :   
        return '(0.3 to 0.4]'
    if row[str_input_column] <= 0.5 :   
        return '(0.4 to 0.5]'
    if row[str_input_column] <= 0.6 :   
        return '(0.5 to 0.6]'
    if row[str_input_column] <= 0.7 :   
        return '(0.6 to 0.7]'
    if row[str_input_column] <= 0.8 :   
        return '(0.7 to 0.8]'
    if row[str_input_column] <= 0.9 :   
        return '(0.8 to 0.9]'
    if row[str_input_column] <= 1.0 :   
        return '(0.9 to 1.0]'
    return 'UNKNOWN'

def n_0_10_summary (row, str_input_column):
    # created by darryldias 29jun2018   
    for i in range(11):
        if row[str_input_column] == i :   
            return str(i)
    return 'UNKNOWN'

def expm1_s1d (row):  
    return math.expm1( row['abc'] )

def log1p_s1d (row):  
    flt_revenue = row['transactionRevenue']
    if np.isnan(flt_revenue):
        flt_revenue = 0.0
    return math.log1p( flt_revenue )

def rev_sum_div_s1d (row):  
    str_train_or_test = row['train or test']
    flt_rev = row['totals_transactionRevenue_sum_div']
    if str_train_or_test == 'train':
        if flt_rev > 0:
            return 'rev'
        else:
            return 'no rev'
    else:
        return 'na test'

def rev_sum_div_s2d (row):  
    str_train_or_test = row['train or test']
    flt_rev = row['totals_transactionRevenue_sum_div']
    if str_train_or_test == 'train':
        if flt_rev > 0:
            if flt_rev <= 25:
                return '(000 - 025]'
            elif flt_rev <= 50:
                return '(025 - 050]'
            elif flt_rev <= 100:
                return '(050 - 100]'
            else:
                return '(100 +'
        else:
            return 'no rev'
    else:
        return 'na test'
    
def sessions_s1d (row):
    int_sessions = row['fullVisitorId_count'] 
    if int_sessions == 1 :   
        return '1'
    elif int_sessions == 2 :   
        return '2'
    elif int_sessions == 3 :   
        return '3'
    elif int_sessions == 4 :   
        return '4'
    else:
        return '5 or more'

def date_diff_days_s1d (row):
    int_days = row['date_diff_days'] 
    if int_days == 0 :   
        return '00'
    elif int_days >= 1 and int_days <= 10:   
        return '01 - 10'
    else:
        return '11 or more'

def totals_hits_avg_s1d (row):
    int_hits = row['totals_hits_avg'] 
    if int_hits <= 1 : # min is actually 1   
        return '(00 - 01]'
    elif int_hits <= 3:   
        return '(01 - 03]'
    elif int_hits <= 10:   
        return '(03 - 10]'
    else:
        return '(10 +'

def totals_pageviews_avg_s1d (row):
    int_pvs = row['totals_pageviews_avg'] 
    if int_pvs <= 1 : # min is actually 1   
        return '(00 - 01]'
    elif int_pvs <= 3:   
        return '(01 - 03]'
    elif int_pvs <= 10:   
        return '(03 - 10]'
    elif int_pvs > 10:   
        return '(10 +'
    else:
        return 'unknown'

def rev_count_s1d (row):  
    str_train_or_test = row['train or test']
    flt_rev = row['revenue_sum_div']
    flt_count = row['revenue_count']
    if str_train_or_test == 'train':
        if flt_rev > 0:
            if flt_count == 1:
                return '1'
            else:
                return '2+'
        else:
            return '0'
    else:
        return 'na test'

def date_min_s2d (row):
    int_yyyymm = row['date_min_s1d'] 
    if int_yyyymm >= 201608 and int_yyyymm <= 201610:   
        return '201608 - 201610'
    elif int_yyyymm >= 201611 and int_yyyymm <= 201701:   
        return '201611 - 201701'
    elif int_yyyymm >= 201702 and int_yyyymm <= 201804:   
        return '201702 - 201804'
    else:
        return 'other'

def sp1_s1d (row):
    str_date_min_s2d = row['date_min_s2d'] 
    str_rev_sum_div_s1d = row['rev_sum_div_s1d'] 
    if str_date_min_s2d == '201608 - 201610': 
        if str_rev_sum_div_s1d == 'rev':
            return 'sp1 rev'
        else:
            return 'sp1 no rev'
    else:
        return 'other/na'

def sp1_s2d (row):
    str_sp1_s1d = row['sp1_s1d'] 
    int_yyyymm_rev_min = row['revenue_date_min_s1d']
    if str_sp1_s1d == 'sp1 rev': 
        if int_yyyymm_rev_min >= 201702:
            return 'rev min 201702 or later'
        elif int_yyyymm_rev_min == 201701:
            return 'rev min 201701'
        elif int_yyyymm_rev_min == 201612:
            return 'rev min 201612'
        elif int_yyyymm_rev_min == 201611:
            return 'rev min 201611'
        elif int_yyyymm_rev_min == 201610:
            return 'rev min 201610'
        elif int_yyyymm_rev_min == 201609:
            return 'rev min 201609'
        elif int_yyyymm_rev_min == 201608:
            return 'rev min 201608'
        else:
            return 'rev unknown'
    elif str_sp1_s1d == 'sp1 no rev':
        return 'sp1 no rev'
    else:
        return 'other/na'
   
df_train_id = pd.read_csv('../input/dd8-files/dd8_visitorid_train.csv', nrows=int_read_csv_rows, dtype={'fullVisitorId': 'str'})
df_test_id = pd.read_csv('../input/dd8-files/dd8_visitorid_test.csv', nrows=int_read_csv_rows, dtype={'fullVisitorId': 'str'})
df_train_p01 = pd.read_csv('../input/dd8-files/dd8_visitor_train_p01_s08.csv', nrows=int_read_csv_rows, dtype={'fullVisitorId': 'str'})
df_test_p01 = pd.read_csv('../input/dd8-files/dd8_visitor_test_p01_s09.csv', nrows=int_read_csv_rows, dtype={'fullVisitorId': 'str'})
df_train_p02 = pd.read_csv('../input/dd8-files/dd8_visitor_train_p02_s10.csv', nrows=int_read_csv_rows, dtype={'fullVisitorId': 'str'})
df_train_p01['train or test'] = 'train'
df_test_p01['train or test'] = 'test'
df_train_p01 = pd.merge(df_train_p01, df_train_id, how='left', on=['fullVisitorId'])
df_test_p01 = pd.merge(df_test_p01, df_test_id, how='left', on=['fullVisitorId'])
df_train_p01 = pd.merge(df_train_p01, df_train_p02, how='left', on=['fullVisitorId'])

df_all_p01 = pd.concat([df_train_p01, df_test_p01], sort=False)
df_all_p01['overall'] = 'yes'
df_all_p01['rev_sum_div_s1d'] = df_all_p01.apply(rev_sum_div_s1d, axis=1)
df_all_p01['rev_sum_div_s2d'] = df_all_p01.apply(rev_sum_div_s2d, axis=1)
df_all_p01['rev_count_s1d'] = df_all_p01.apply(rev_count_s1d, axis=1)

df_all_p01['sessions_s1d'] = df_all_p01.apply(sessions_s1d, axis=1)
df_all_p01['date_min_s1d'] = df_all_p01.apply(year_month_code2, axis=1, str_input_column='date_min')
df_all_p01['date_max_s1d'] = df_all_p01.apply(year_month_code2, axis=1, str_input_column='date_max')
df_all_p01['date_min'] = pd.to_datetime(df_all_p01['date_min'].astype(str), format='%Y%m%d')
df_all_p01['date_max'] = pd.to_datetime(df_all_p01['date_max'].astype(str), format='%Y%m%d')
df_all_p01['date_diff_days'] = (df_all_p01['date_max'] - df_all_p01['date_min']).dt.days
df_all_p01['date_diff_days_s1d'] = df_all_p01.apply(date_diff_days_s1d, axis=1)
df_all_p01['totals_hits_avg_s1d'] = df_all_p01.apply(totals_hits_avg_s1d, axis=1)
df_all_p01['totals_pageviews_avg_s1d'] = df_all_p01.apply(totals_pageviews_avg_s1d, axis=1)
df_all_p01['date_min_s2d'] = df_all_p01.apply(date_min_s2d, axis=1)
df_all_p01['revenue_date_min_s1d'] = df_all_p01.apply(year_month_code3, axis=1, str_input_column='revenue_date_min')
df_all_p01['revenue_date_max_s1d'] = df_all_p01.apply(year_month_code3, axis=1, str_input_column='revenue_date_max')
df_all_p01['sp1_s1d'] = df_all_p01.apply(sp1_s1d, axis=1)
df_all_p01['sp1_s2d'] = df_all_p01.apply(sp1_s2d, axis=1)

create_crosstab_type1(df_all_p01, 'overall', int_important_crosstab)
create_crosstab_type1(df_all_p01, 'train or test', int_important_crosstab)
create_crosstab_type1(df_all_p01, 'rev_sum_div_s1d', int_important_crosstab)
create_crosstab_type1(df_all_p01, 'rev_sum_div_s2d', int_important_crosstab)
create_crosstab_type1(df_all_p01, 'rev_count_s1d', int_current_crosstab)
create_crosstab_type1(df_all_p01, 'sessions_s1d', int_current_crosstab)
create_crosstab_type1(df_all_p01, 'date_min_s1d', int_current_crosstab)
create_crosstab_type1(df_all_p01, 'date_max_s1d', int_current_crosstab)
create_crosstab_type1(df_all_p01, 'date_diff_days_s1d', int_current_crosstab)
create_crosstab_type1(df_all_p01, 'totals_hits_avg_s1d', int_current_crosstab)
create_crosstab_type1(df_all_p01, 'totals_pageviews_avg_s1d', int_current_crosstab)
create_crosstab_type1(df_all_p01, 'date_min_s2d', int_current_crosstab)
create_crosstab_type1(df_all_p01, 'sp1_s1d', int_current_crosstab)
create_crosstab_type1(df_all_p01, 'sp1_s2d', int_current_crosstab)
#df_all_p01.sample(10)
#df_all_p01.info()
if bln_run_stage1:
    str_filename = 'dd8_table_list_all.csv'
    csvfile1 = open(str_filename, 'w')
    writer1 = csv.writer(csvfile1)
    writer1.writerow( ['version', 'id', 'train_or_test', 'name', 'records', 'est_qry_size'] )
    
    str_select_columns = "COUNT(*)"
    int_counter = 0

    for str_train_or_test in ['train', 'test']:
        bigqueryhelper = get_train_or_test_bigqueryhelper(str_train_or_test)
        str_dataset = get_train_or_test_dataset(str_train_or_test)
        list_tables = bigqueryhelper.list_tables()
    
        for table in list_tables:
            int_counter += 1
            print(int_counter, table, str_train_or_test)
            str_select_from_table = table
            query = create_simple_query1(str_dataset, str_select_columns, str_select_from_table)
            flt_est_query_size = bigqueryhelper.estimate_query_size(query)
            df_query = bigqueryhelper.query_to_pandas_safe(query, max_gb_scanned=flt_query_limit)
            int_table_count = df_query.iloc[0][0]
            writer1.writerow( [1, int_counter, str_train_or_test, table, int_table_count, flt_est_query_size] )

    csvfile1.close()

if bln_run_stage2:
    ds_current = BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_train_set")
    df_table_schema = ds_current.table_schema('ga_sessions_20170801')
    df_table_schema.to_csv('dd8_table_schema_train.csv', index=False)
    ds_current = BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_test_set")
    df_table_schema = ds_current.table_schema('ga_sessions_20180430')
    df_table_schema.to_csv('dd8_table_schema_test.csv', index=False)

if bln_run_stage3:
    str_train_or_test = 'test'
    if str_train_or_test == 'train':
        ds_current = BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_train_set")
        str_dataset = "kaggle-public-datasets.ga_train_set"
        int_start_date = 20160801
        int_end_date = 20170801
        str_output_csv = 'dd8_train_s3.csv'
        str_select_columns = "sessionId, fullVisitorId, date, totals.transactionRevenue, visitNumber"
        csv_file_read = open('../input/dd8-files/dd8_table_list_train.csv', 'r')
    else:
        ds_current = BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_test_set")
        str_dataset = "kaggle-public-datasets.ga_test_set"
        int_start_date = 20170802
        int_end_date = 20180430
        str_output_csv = 'dd8_test_s3.csv'
        str_select_columns = "sessionId, fullVisitorId, date, visitNumber"
        csv_file_read = open('../input/dd8-files/dd8_table_list_test.csv', 'r')
    
    
    csv_reader = csv.reader(csv_file_read, delimiter=',')

    print('sample queries run are shown below:')
    int_row = 0
    for row in csv_reader:
        if int_row > 0:
            int_id = row[0]
            str_table = row[1]
            int_table_date = int(str_table[-8:])
            if int_table_date >= int_start_date and int_table_date <= int_end_date:
                str_select_from_table = str_table
                query = create_simple_query1(str_select_columns, str_select_from_table)
                flt_est1_query_size = ds_current.estimate_query_size(query)
                flt_est2_query_size = get_query_size(flt_est1_query_size) 
                flt_est_query_size_total += flt_est2_query_size
                int_query_count += 1
                if int_table_date == int_start_date or int_table_date == int_end_date or (int_query_count % 20 == 0):
                    print(query)
                    print('size1mb:', get_str_query_size_mb(flt_est1_query_size), ' size2mb: ', get_str_query_size_mb(flt_est2_query_size) ) 
                    print()
            
                df_temp = ds_current.query_to_pandas_safe(query, max_gb_scanned=flt_query_limit)
                if int_query_count == 1:
                    df_query = df_temp
                else:
                    df_query = pd.concat([df_query, df_temp])
            
                query_writer.writerow( [query, flt_est1_query_size, flt_est2_query_size] )
        int_row += 1
    df_query.to_csv(str_output_csv, index=False)

    csv_file_read.close()
    print('the number of lines read in input table list file: ', int_row)
    print('the number of queries run: ', int_query_count)
    print('total estimated size of queries run (mb):', get_str_query_size_mb(flt_est_query_size_total) )

if bln_run_stage4:
    df_train_temp = pd.read_csv('../input/dd8-files/dd8_train_s3.csv', nrows=int_read_csv_rows, dtype={'fullVisitorId': 'str'})
    df_test_temp = pd.read_csv('../input/dd8-files/dd8_test_s3.csv', nrows=int_read_csv_rows, dtype={'fullVisitorId': 'str'})
    df_train_temp['train or test'] = 'train'
    df_test_temp['train or test'] = 'test'
    df_all = pd.concat([df_train_temp, df_test_temp], sort=False)
    df_all['overall'] = 'yes'
    print(df_all.groupby(['train or test'])['sessionId'].count())
    df_all.to_csv('dd8_all_s4.csv', index=False) # fixed name after running this stage
   
if bln_run_stage5:
    df_all = pd.read_csv('../input/dd8-files/dd8_all_s4.csv', nrows=int_read_csv_rows, dtype={'fullVisitorId': 'str'})
    df_all['year_month_code'] = df_all.apply(year_month_code2, axis=1)
    print(df_all.groupby(['train or test'])['sessionId'].count(),'\n')
    print(df_all.groupby(['train or test', 'year_month_code'])['sessionId'].count())
if bln_run_stage6:
    bln_create_dataframe = True
    str_select_columns = "fullVisitorId, count(*) as count"
    str_group_by_columns = "fullVisitorId"

    str_train_or_test = 'train'
    int_start_date = 20160801
    int_end_date = 20170801
    df_train = run_queries_2(bln_create_dataframe, str_train_or_test, int_start_date, int_end_date, str_select_columns, str_group_by_columns)

    df_train = df_train.groupby(['fullVisitorId']).count().reset_index()
    df_train['k_VisitorId']= df_train.index + 1000001
    df_train = df_train[['fullVisitorId', 'k_VisitorId']]   
    df_train.to_csv('dd8_visitorid_train.csv', index=False)
    
if bln_run_stage6:
    print(df_train.info())
    print(df_train.sample(10))
if bln_run_stage7:
    bln_create_dataframe = True
    str_select_columns = "fullVisitorId, count(*) as count"
    str_group_by_columns = "fullVisitorId"

    str_train_or_test = 'test'
    int_start_date = 20170802
    int_end_date = 20180430
    df_test = run_queries_2(bln_create_dataframe, str_train_or_test, int_start_date, int_end_date, str_select_columns, str_group_by_columns)
    
    df_test = df_test.groupby(['fullVisitorId']).count().reset_index()
    df_test['k_VisitorId'] = df_test.index + 2000001
    df_test = df_test[['fullVisitorId', 'k_VisitorId']]   
    df_test.to_csv('dd8_visitorid_test.csv', index=False)  

if bln_run_stage7:
    print(df_test.info())
    print(df_test.sample(10))
    df_sample_submission = pd.read_csv('../input/google-analytics-customer-revenue-prediction/sample_submission.csv', \
                      nrows=int_read_csv_rows, dtype={'fullVisitorId': 'str'})
    print(df_sample_submission.info())    
if bln_run_stage8:
    #bln_create_dataframe = True
    #df_train = run_queries_2(bln_create_dataframe, str_train_or_test, int_start_date, int_end_date, str_select_columns, str_group_by_columns)
    str_train_or_test = 'train'
    int_start_date = 20160801
    int_end_date = 20170801
    str_select_columns_sub1 = "fullVisitorId, visitNumber, date, totals.transactionRevenue as totals_transactionRevenue, " + \
        "totals.visits as totals_visits, totals.hits as totals_hits, totals.timeOnSite as totals_timeOnSite, " + \
        "totals.pageviews as totals_pageviews"
    str_select_columns = "fullVisitorId, " + \
                         "COUNT(fullVisitorId) AS fullVisitorId_count, " + \
                         "SUM(totals_transactionRevenue) AS totals_transactionRevenue_sum, " + \
                         "COUNT(totals_transactionRevenue) AS totals_transactionRevenue_count, " + \
                         "AVG(totals_transactionRevenue) AS totals_transactionRevenue_avg, " + \
                         "SUM(totals_transactionRevenue)/1000000 AS totals_transactionRevenue_sum_div, " + \
                         "AVG(totals_transactionRevenue)/1000000 AS totals_transactionRevenue_avg_div, " + \
                         "MAX(visitNumber) AS visitNumber_max, " + \
                         "MIN(date) AS date_min, " + \
                         "MAX(date) AS date_max, " + \
                         "SUM(totals_visits) AS totals_visits_sum, " + \
                         "AVG(totals_hits) AS totals_hits_avg, " + \
                         "AVG(totals_timeOnSite) AS totals_timeOnSite_avg, " + \
                         "AVG(totals_pageviews) AS totals_pageviews_avg"
    str_select_from_table = "SUB1"
    str_group_by_columns = "fullVisitorId"

    query_sub1 = create_simple_query3(int_start_date, int_end_date, str_train_or_test, str_select_columns_sub1)
    query = query_sub1 + create_simple_query4(str_select_columns, str_select_from_table, str_group_by_columns)
    print('start and end of query:')
    print(query[:1011], '\n...\n\n', query[-1013:])

    ds_current = BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_train_set")
    flt_est1_query_size = ds_current.estimate_query_size(query)
    flt_est2_query_size = get_query_size(flt_est1_query_size)
    print('\nsize1mb', get_str_query_size_mb(flt_est1_query_size))
    print('size2mb', get_str_query_size_mb(flt_est2_query_size))    
    
if bln_run_stage8:
    df_train = ds_current.query_to_pandas_safe(query, max_gb_scanned=flt_query_limit)
    print(df_train.info())
    print('\nTotal sessions (check):', df_train['fullVisitorId_count'].sum(), '\n')
    print(df_train.sample(10))
    df_temp = df_train[ df_train['totals_transactionRevenue_sum'] > 0  ]
    #df_temp2 = df_temp[ df_temp['totals_transactionRevenue_count'] > 1  ]
    print(df_temp.sample(10))
    df_train.to_csv('dd8_visitor_train_p01_s08.csv', index=False)   
if bln_run_stage9:
    str_train_or_test = 'test'
    int_start_date = 20170802
    int_end_date = 20180430
    str_select_columns_sub1 = "fullVisitorId, visitNumber, date, NULL as totals_transactionRevenue, " + \
        "totals.visits as totals_visits, totals.hits as totals_hits, totals.timeOnSite as totals_timeOnSite, " + \
        "totals.pageviews as totals_pageviews"
    str_select_columns = "fullVisitorId, " + \
                         "COUNT(fullVisitorId) AS fullVisitorId_count, " + \
                         "NULL AS totals_transactionRevenue_sum, " + \
                         "NULL AS totals_transactionRevenue_count, " + \
                         "NULL AS totals_transactionRevenue_avg, " + \
                         "NULL AS totals_transactionRevenue_sum_div, " + \
                         "NULL AS totals_transactionRevenue_avg_div, " + \
                         "MAX(visitNumber) AS visitNumber_max, " + \
                         "MIN(date) AS date_min, " + \
                         "MAX(date) AS date_max, " + \
                         "SUM(totals_visits) AS totals_visits_sum, " + \
                         "AVG(totals_hits) AS totals_hits_avg, " + \
                         "AVG(totals_timeOnSite) AS totals_timeOnSite_avg, " + \
                         "AVG(totals_pageviews) AS totals_pageviews_avg"
    str_select_from_table = "SUB1"
    str_group_by_columns = "fullVisitorId"

    query_sub1 = create_simple_query3(int_start_date, int_end_date, str_train_or_test, str_select_columns_sub1)
    query = query_sub1 + create_simple_query4(str_select_columns, str_select_from_table, str_group_by_columns)
    print('start and end of query:')
    print(query[:1011], '\n...\n\n', query[-1013:])

    bigqueryhelper = get_train_or_test_bigqueryhelper(str_train_or_test)
    flt_est1_query_size = bigqueryhelper.estimate_query_size(query)
    flt_est2_query_size = get_query_size(flt_est1_query_size)
    print('\nsize1mb', get_str_query_size_mb(flt_est1_query_size))
    print('size2mb', get_str_query_size_mb(flt_est2_query_size))    

if bln_run_stage9:
    df_test = bigqueryhelper.query_to_pandas_safe(query, max_gb_scanned=flt_query_limit)
    print(df_test.info())
    print('\nTotal sessions (check):', df_test['fullVisitorId_count'].sum(), '\n')
    print(df_test.sample(10))
    df_test.to_csv('dd8_visitor_test_p01_s09.csv', index=False)   
if bln_run_stage10:
    str_train_or_test = 'train'
    int_start_date = 20160801
    int_end_date = 20170801
    str_select_columns_sub1 = "fullVisitorId, date, totals.transactionRevenue as revenue" 
    str_where_sub1 = "totals.transactionRevenue > 0"
    str_select_columns = "fullVisitorId, " + \
                         "SUM(revenue) AS revenue_sum, " + \
                         "COUNT(revenue) AS revenue_count, " + \
                         "AVG(revenue) AS revenue_avg, " + \
                         "SUM(revenue)/1000000 AS revenue_sum_div, " + \
                         "AVG(revenue)/1000000 AS revenue_avg_div, " + \
                         "MIN(date) AS revenue_date_min, " + \
                         "MAX(date) AS revenue_date_max" 
    str_select_from_table = "SUB1"
    str_group_by_columns = "fullVisitorId"

    query_sub1 = create_simple_query3(int_start_date, int_end_date, str_train_or_test, str_select_columns_sub1, str_where_sub1)
    query = query_sub1 + create_simple_query4(str_select_columns, str_select_from_table, str_group_by_columns)
    print('start and end of query:')
    print(query[:511], '\n...\n\n', query[len(query)-500:])

    bigqueryhelper = BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_train_set")
    flt_est1_query_size = bigqueryhelper.estimate_query_size(query)
    flt_est2_query_size = get_query_size(flt_est1_query_size)
    print('\nsize1mb', get_str_query_size_mb(flt_est1_query_size))
    print('size2mb', get_str_query_size_mb(flt_est2_query_size))    

if bln_run_stage10:
    df_train = bigqueryhelper.query_to_pandas_safe(query, max_gb_scanned=flt_query_limit)
    print(df_train.info())
    df_train.to_csv('dd8_visitor_train_p02_s10.csv', index=False)   
if bln_run_stage11:
    # totals.bounces   totals.newVisits   trafficSource.referralPath   trafficSource.campaign
    # trafficSource.source   trafficSource.adwordsClickInfo.page  trafficSource.isTrueDirect
    # device.browser   device.operatingSystem   device.isMobile   geoNetwork.continent
    
    str_train_or_test = 'train'
    int_start_date = 20160801
    int_end_date = 20160803 #20170801
    str_select_columns_sub1 = "fullVisitorId, 'yes' as overall, 'train' as train_or_test, " + \
                              "'device_browser' as variable, device.browser as category"
    str_select_columns = "variable, category, " + \
                         "COUNTIF(overall='yes') AS overall_count, " + \
                         "COUNTIF(train_or_test='train') AS train_count, " + \
                         "COUNTIF(train_or_test='test') AS test_count"
    str_select_from_table = "SUB1"
    str_group_by_columns = "variable, category"

    query_sub1 = create_simple_query3(int_start_date, int_end_date, str_train_or_test, str_select_columns_sub1)
    query = query_sub1 + create_simple_query4(str_select_columns, str_select_from_table, str_group_by_columns)
    print('start and end of query:')
    print(query[:1011], '\n...\n\n', query[-1013:])

    bigqueryhelper = get_train_or_test_bigqueryhelper(str_train_or_test)
    flt_est1_query_size = bigqueryhelper.estimate_query_size(query)
    flt_est2_query_size = get_query_size(flt_est1_query_size)
    print('\nsize1mb', get_str_query_size_mb(flt_est1_query_size))
    print('size2mb', get_str_query_size_mb(flt_est2_query_size))    
    
if bln_run_stage11:
    df_train = bigqueryhelper.query_to_pandas_safe(query, max_gb_scanned=flt_query_limit)
    print(df_train.info())
    #print('\nTotal sessions (check):', df_train['fullVisitorId_count'].sum(), '\n')
    print(df_train.sample(24))
    #df_temp = df_train[ df_train['totals_transactionRevenue_sum'] > 0  ]
    ##df_temp2 = df_temp[ df_temp['totals_transactionRevenue_count'] > 1  ]
    #print(df_temp.sample(10))
    #df_train.to_csv('dd8_visitor_train_p01_s08.csv', index=False)   
    
if bln_run_stage11:
    query = """WITH SUB1 AS ( 
SELECT fullVisitorId, 'yes' as overall, 'train' as train_or_test, 'device_browser' as variable, device.browser as category 
FROM `kaggle-public-datasets.ga_train_set.ga_sessions_20160801` UNION ALL SELECT fullVisitorId, 'yes' as overall, 'train' as train_or_test, 'device_browser' as variable, device.browser as category 
FROM `kaggle-public-datasets.ga_train_set.ga_sessions_20160802` UNION ALL SELECT fullVisitorId, 'yes' as overall, 'test' as train_or_test, 'device_browser' as variable, device.browser as category 
FROM `kaggle-public-datasets.ga_test_set.ga_sessions_20180430` ) 
SELECT variable, category, COUNTIF(overall='yes') AS overall_count, COUNTIF(train_or_test='train') AS train_count, COUNTIF(train_or_test='test') AS test_count
FROM `SUB1` 
GROUP BY variable, category"""

    flt_est1_query_size = bigqueryhelper.estimate_query_size(query)
    print(get_str_query_size_mb(flt_est1_query_size))
# totals.timeOnSite
#df_all.to_csv('all.csv', index=False)
#for i in range(850000,850999):
#    x = json.loads(df_train.iloc[i]['totals'])
#    print(x['newVisits'])
#df_query.sample(10)  
#df_test_temp.info()
#df_test_temp['sessionId'].nunique()
#df_all.info()
#df_temp = df_all[ df_all['transactionRevenue']>0 ]
#df_temp.sample(10)
#create_crosstab_type1(df_all, 'overall', int_current_crosstab)
#df_temp = get_sample_train_data("device.browser", "20170720")    
#df_temp.sample(80)
#df_all_p01['totals_transactionRevenue_sum_div'].describe()
#df_temp = df_all_p01[ df_all_p01['totals_pageviews_avg_s1d'] == 'unknown' ]
#df_temp[['totals_pageviews_avg', 'totals_pageviews_avg_s1d']].sample(20)
end_time_check(dat_program_start, 'overall')
df_time_check