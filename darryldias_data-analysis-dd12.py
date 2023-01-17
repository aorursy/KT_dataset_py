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
import csv
bln_create_df_all_csv_file = True
bln_create_words_csv_file = False
int_df_all_version = 6
bln_ready_to_commit = True
bln_create_estimate_files = False
bln_upload_input_estimates = False
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
        return ['overall', 'test', 'train', 'negative', 'zero', 'positive', '2A', '2B', '2C', '2D', '2E', '2F', '2G', '3A', '3B']
    elif int_code == 2:
        return ['overall', 'train_or_test', 'train_or_test', 'target_s1d', 'target_s1d', 'target_s1d', \
                'target_s2d', 'target_s2d', 'target_s2d', 'target_s2d', 'target_s2d', 'target_s2d', 'target_s2d', 'target_s3d', 'target_s3d']
    elif int_code == 3:
        return ['yes', 'test', 'train', 'negative', 'zero', 'positive', '2A', '2B', '2C', '2D', '2E', '2F', '2G', '3A', '3B']
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

def year_month_code (row, str_input_column):
    value = (row[str_input_column].year) * 100 + row[str_input_column].month
    return value

def category_s1d (row, str_input_column, str_input_category):
    str_value = row[str_input_column] 
    if str_value == str_input_category :   
        return 1
    elif pd.isnull(str_value):
        return None
    else:
        return 0

def category_s2d (row, str_input_column):
    flt_value = row[str_input_column] 
    if flt_value <= 0.2 :   
        return '{1} [0.0-0.2]'
    elif flt_value <= 0.4 :   
        return '{2} (0.2-0.4]'
    elif flt_value <= 0.6 :   
        return '{3} (0.4-0.6]'
    elif flt_value <= 0.8 :   
        return '{4} (0.6-0.8]'
    elif flt_value <= 1.0 :   
        return '{5} (0.8-1.0]'
    else:
        return '{6} none'

def target_s1d (row):
    flt_target = row['target'] 
    if flt_target < 0 :   
        return 'negative'
    elif flt_target == 0 :   
        return 'zero'
    elif flt_target > 0 :   
        return 'positive'
    else:
        return 'test'

def target_s2d (row):
    flt_target = row['target'] 
    if flt_target < -1.35 :   
        return '2A'
    elif flt_target < -0.5 :   
        return '2B'
    elif flt_target < 0.0 :   
        return '2C'
    elif flt_target == 0.0 :   
        return '2D'
    elif flt_target <= 0.5 :   
        return '2E'
    elif flt_target <= 1.35 :   
        return '2F'
    elif flt_target > 1.35 :   
        return '2G'
    else:
        return 'test'

def target_s3d (row):
    flt_target = row['target'] 
    if flt_target < -30 :   
        return '3A'
    elif flt_target >= -30 :   
        return '3B'
    else:
        return 'test'

def ht_count_s1d (row):
    int_count = row['ht_count'] 
    if int_count <= 25 :   
        return '001-025'
    elif int_count <= 50 :   
        return '026-050'
    elif int_count <= 100 :   
        return '051-100'
    elif int_count >= 101 :   
        return '101+'
    else:
        return 'unknown'

def nt_count_s1d (row):
    int_count = row['nt_count'] 
    if int_count <= 2 :   
        return '1-2'
    elif int_count <= 4 :   
        return '3-4'
    elif int_count <= 8 :   
        return '5-8'
    elif int_count >= 9 :   
        return '9+'
    else:
        return '0'

def ht_pur_amt_adj_sum_s1d (row):
    flt_value = row['ht_pur_amt_adj_sum'] 
    if flt_value <= 3.0 :   
        return '[00.0-03.0]'
    elif flt_value <= 7.0 :   
        return '(03.0-07.0]'
    elif flt_value <= 15.0 :   
        return '(07.0-15.0]'
    elif flt_value > 15.0 :   
        return '(15.0+'
    else:
        return 'unknown'

def nt_pur_amt_adj_sum_s1d (row):
    flt_value = row['nt_pur_amt_adj_sum'] 
    if flt_value <= 0.2 :   
        return '[0.0-0.2]'
    elif flt_value <= 0.5 :   
        return '(0.2-0.5]'
    elif flt_value <= 1.0 :   
        return '(0.5-1.0]'
    elif flt_value <= 2.0 :   
        return '(1.0-2.0]'
    elif flt_value <= 4.0 :   
        return '(2.0-4.0]'
    elif flt_value > 4.0 :   
        return '(4.0+'
    else:
        return 'none'

def first_active_month_s2d (row):
    flt_value = row['first_active_month_s1d'] 
    if flt_value >= 201710 :   
        return '201710+'
    elif flt_value >= 201707 :   
        return '201707-201709'
    elif flt_value >= 201702 :   
        return '201702-201706'
    elif flt_value >= 201608 :   
        return '201608-201701'
    elif flt_value >= 201111 :   
        return '201111-201607'
    else:
        return 'unknown'

def nt_pur_date_max_s2d (row):
    flt_value = row['nt_pur_date_max_s1d'] 
    if flt_value >= 201804 :   
        return '201804'
    elif flt_value >= 201803 :   
        return '201803'
    elif flt_value >= 201802 :   
        return '201802'
    elif flt_value >= 201801 :   
        return '201801'
    elif flt_value >= 201711 :   
        return '201711-201712'
    elif flt_value >= 201708 :   
        return '201708-201710'
    elif flt_value >= 201703 :   
        return '201703-201707'
    else:
        return 'none'

def ht_pur_date_max_s2d (row):
    flt_value = row['ht_pur_date_max_s1d'] 
    if flt_value >= 201802 :   
        return '201802'
    elif flt_value >= 201801 :   
        return '201801'
    elif flt_value >= 201711 :   
        return '201711-201712'
    elif flt_value >= 201709 :   
        return '201709-201710'
    elif flt_value >= 201706 :   
        return '201706-201708'
    elif flt_value >= 201702 :   
        return '201702-201705'
    else:
        return 'none'

def has_new_tran (row):
    int_count = row['nt_count'] 
    if int_count >= 1 :   
        return 'yes'
    else:
        return 'no'

def nt_category_2_mean_s1d (row):
    flt_value = row['nt_category_2_mean'] 
    str_value = row['has_new_tran'] 
    if flt_value == 1.0 :   
        return '{1} 1.0'
    elif flt_value <= 2.0 :   
        return '{2} (1.0-2.0]'
    elif flt_value <= 3.0 :   
        return '{3} (2.0-3.0]'
    elif flt_value <= 4.0 :   
        return '{4} (3.0-4.0]'
    elif flt_value <= 5.0 :   
        return '{5} (4.0-5.0]'
    else:
        if str_value == 'no':
            return '{7} none'
        else:
            return '{6} unknown'

df_train = pd.read_csv('../input/train.csv', nrows=int_read_csv_rows, parse_dates=["first_active_month"])
df_test = pd.read_csv('../input/test.csv', nrows=int_read_csv_rows, parse_dates=["first_active_month"])
df_train['train_or_test'] = 'train'
df_test['train_or_test'] = 'test'
df_all = pd.concat([df_train, df_test], sort=False)
df_all['overall'] = 'yes'
df_all['target_s1d'] = df_all.apply(target_s1d, axis=1)
df_all['target_s2d'] = df_all.apply(target_s2d, axis=1)
df_all['target_s3d'] = df_all.apply(target_s3d, axis=1)
df_all['first_active_month_s1d'] = df_all.apply(year_month_code, axis=1, str_input_column='first_active_month')
df_all['first_active_month_s2d'] = df_all.apply(first_active_month_s2d, axis=1)

df_hist_trans = pd.read_csv('../input/historical_transactions.csv', nrows=int_read_csv_rows, parse_dates=["purchase_date"])
flt_min = df_hist_trans['purchase_amount'].min()
df_hist_trans['purchase_amount_adj'] = df_hist_trans['purchase_amount'] - flt_min
grouped = df_hist_trans.groupby('card_id')
df_grouped = grouped['card_id'].count().reset_index(name='ht_count')  
df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
df_grouped = grouped['purchase_amount_adj'].sum().reset_index(name='ht_pur_amt_adj_sum')  
df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
df_grouped = grouped['purchase_date'].max().reset_index(name='ht_pur_date_max')  
df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])


df_new_trans = pd.read_csv('../input/new_merchant_transactions.csv', nrows=int_read_csv_rows, parse_dates=["purchase_date"])
flt_min = df_new_trans['purchase_amount'].min()
df_new_trans['purchase_amount_adj'] = df_new_trans['purchase_amount'] - flt_min
df_new_trans['nt_category_1_N'] = df_new_trans.apply(category_s1d, axis=1, str_input_column='category_1', str_input_category='N')
#df_new_trans['nt_category_3_A'] = df_new_trans.apply(category_s1d, axis=1, str_input_column='category_3', str_input_category='A')
#df_new_trans['nt_category_3_B'] = df_new_trans.apply(category_s1d, axis=1, str_input_column='category_3', str_input_category='B')
#df_new_trans['nt_category_3_C'] = df_new_trans.apply(category_s1d, axis=1, str_input_column='category_3', str_input_category='C')
grouped = df_new_trans.groupby('card_id')
df_grouped = grouped['card_id'].count().reset_index(name='nt_count')  
df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
df_grouped = grouped['purchase_amount_adj'].sum().reset_index(name='nt_pur_amt_adj_sum')  
df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
df_grouped = grouped['purchase_date'].max().reset_index(name='nt_pur_date_max')  
df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
df_grouped = grouped['nt_category_1_N'].mean().reset_index(name='nt_category_1_N_mean')  
df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
df_grouped = grouped['category_2'].mean().reset_index(name='nt_category_2_mean')  
df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
#df_grouped = grouped['nt_category_3_A'].mean().reset_index(name='nt_category_3_A_mean')  
#df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
#df_grouped = grouped['nt_category_3_B'].mean().reset_index(name='nt_category_3_B_mean')  
#df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
#df_grouped = grouped['nt_category_3_C'].mean().reset_index(name='nt_category_3_C_mean')  
#df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])


df_all['has_new_tran'] = df_all.apply(has_new_tran, axis=1)
df_all['ht_count_s1d'] = df_all.apply(ht_count_s1d, axis=1)
df_all['nt_count_s1d'] = df_all.apply(nt_count_s1d, axis=1)
df_all['ht_pur_amt_adj_sum_s1d'] = df_all.apply(ht_pur_amt_adj_sum_s1d, axis=1)
df_all['nt_pur_amt_adj_sum_s1d'] = df_all.apply(nt_pur_amt_adj_sum_s1d, axis=1)
df_all['ht_pur_date_max_s1d'] = df_all.apply(year_month_code, axis=1, str_input_column='ht_pur_date_max')
df_all['ht_pur_date_max_s2d'] = df_all.apply(ht_pur_date_max_s2d, axis=1)
df_all['nt_pur_date_max_s1d'] = df_all.apply(year_month_code, axis=1, str_input_column='nt_pur_date_max')
df_all['nt_pur_date_max_s2d'] = df_all.apply(nt_pur_date_max_s2d, axis=1)
df_all['nt_category_1_N_mean_s1d'] = df_all.apply(category_s2d, axis=1, str_input_column='nt_category_1_N_mean')
df_all['nt_category_2_mean_s1d'] = df_all.apply(nt_category_2_mean_s1d, axis=1)


df_all.info()
#df_train['k_qid'] = df_train.index + 10000001
#df_test['k_qid'] = df_test.index + 20000001
df_all.sample(10)
create_crosstab_type1(df_all, 'overall', int_important_crosstab)
create_crosstab_type1(df_all, 'train_or_test', int_important_crosstab)
create_crosstab_type1(df_all, 'target_s1d', int_important_crosstab)
create_crosstab_type1(df_all, 'target_s2d', int_important_crosstab)
create_crosstab_type1(df_all, 'target_s3d', int_important_crosstab)
create_crosstab_type1(df_all, 'feature_1', int_current_crosstab)
create_crosstab_type1(df_all, 'feature_2', int_current_crosstab)
create_crosstab_type1(df_all, 'feature_3', int_current_crosstab)
create_crosstab_type1(df_all, 'first_active_month_s2d', int_current_crosstab)
create_crosstab_type1(df_all, 'has_new_tran', int_current_crosstab)
create_crosstab_type1(df_all, 'ht_count_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'ht_pur_amt_adj_sum_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'ht_pur_date_max_s2d', int_current_crosstab)
create_crosstab_type1(df_all, 'nt_count_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'nt_pur_amt_adj_sum_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'nt_pur_date_max_s2d', int_current_crosstab)
create_crosstab_type1(df_all, 'nt_category_1_N_mean_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'nt_category_2_mean_s1d', int_current_crosstab)
#df_new_trans.info()
#df_temp = df_new_trans[ df_new_trans['card_id'] == 'C_ID_0c167d84b2' ]
#df_temp[['card_id', 'merchant_id']].sample(5)
#df_new_trans[['category_1', 'nt_category_1_N']].sample(50)
#df_all['nt_category_2_mean'].describe()

#df_new_trans['nt_category_3_C'].describe()
#grouped = df_new_trans.groupby('card_id')
#df_grouped = grouped['nt_category_1_N'].mean().reset_index(name='nt_category_1_N_mean')  
#df_all = pd.merge(df_all, df_grouped, how='left', on=['card_id'])
#df_all['nt_category_3_A_mean_s1d'] = df_all.apply(category_s2d, axis=1, str_input_column='nt_category_3_A_mean')
#df_all['nt_category_3_B_mean_s1d'] = df_all.apply(category_s2d, axis=1, str_input_column='nt_category_3_B_mean')
#create_crosstab_type1(df_all, 'nt_category_3_A_mean_s1d', int_current_crosstab)
#create_crosstab_type1(df_all, 'nt_category_3_B_mean_s1d', int_current_crosstab)
#df_temp = df_all[ df_all['nt_category_2_mean_s1d'] == '{7} none' ]
#df_temp[['nt_category_2_mean', 'nt_category_2_mean_s1d', 'has_new_tran']].sample(20)
#df_temp.info()
#df_temp.max()
#df_all['target'].describe()
#create_topline(df_new_trans, 'category_2', 'count')

end_time_check(dat_program_start, 'overall')
df_time_check