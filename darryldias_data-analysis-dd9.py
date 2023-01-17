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
print('\nembeddings:\n', os.listdir("../input/embeddings"))
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
        return ['overall', 'test', 'train', 'sincere', 'insincere']
    elif int_code == 2:
        return ['overall', 'train_or_test', 'train_or_test', 'target_s1d', 'target_s1d']
    elif int_code == 3:
        return ['yes', 'test', 'train', 'sincere', 'insincere']
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

# new
def target_s1d (row):
    int_target = row['target'] 
    if int_target == 0 :   
        return 'sincere'
    elif int_target == 1 :   
        return 'insincere'
    else:
        return 'na test'

def q_length_s1d (row):
    int = row['q_length'] 
    if int <= 50 : 
        return '001 to 050'
    elif int <= 100:   
        return '051 to 100'
    elif int >= 101:   
        return '101 or more'
    else:
        return 'unknown'

def word_count (row):
    int = len( row['question_text'].split() )
    return int

def word_count_s1d (row):
    int = row['word_count']
    if int <= 10:
        return '01 to 10'
    elif int <= 20:
        return '11 to 20'
    elif int >= 21:
        return '21 or more'
    else:
        return 'unknown'

def qmark_count (row):
    int = row['question_text'].count('?') 
    return int

def qmark_count_s1d (row):
    int = row['qmark_count']
    if int == 0:
        return '0'
    elif int == 1:
        return '1'
    elif int >= 2:
        return '2+'
    else:
        return 'unknown'

def period_count (row):
    int = row['question_text'].count('.') 
    return int

def period_count_s1d (row):
    int = row['period_count']
    if int == 0:
        return '0'
    elif int >= 1:
        return '1+'
    else:
        return 'unknown'

def first_word (row):
    words = row['question_text'].split()
    str = words[0].lower()
    return str

def first_word_s1d (row):
    str_word = row['first_word']
    if str_word in ['what', 'where', 'why', 'how', 'when', 'is', 'will', 'should', 'are', 'can', 'do', 'did', 'who', 'if', \
                    'would', 'which', 'does', 'has', 'was', 'could', 'have']:
        return str_word
    else:
        return 'other'

def cap_word_count (row):
    words = row['question_text'].split()
    int_count = 0
    for word in words:
        char = word[0] 
        if char == char.upper() and char.isalpha():
            int_count += 1
    return int_count

def cap_word_count_s1d (row):
    int = row['cap_word_count']
    if int <= 3:
        return str(int)
    elif int >= 4:
        return '4+'
    else:
        return 'unknown'

def contains_word (row, word):
    # to do setup clean word function 
    words = row['question_text'].lower().split()
    if word in words:
        return 'yes'
    else:
        return 'no'

def create_contains_word_var(str_word):
    temp_var = 'contains_word_' + str_word
    df_all[temp_var] = df_all.apply(contains_word, axis=1, word=str_word)

def equals_count (row):
    int = row['question_text'].count('=') 
    return int

def equals_count_s1d (row):
    int = row['equals_count']
    if int == 0:
        return '0'
    elif int >= 1:
        return '1+'
    else:
        return 'unknown'

def dollar_count (row):
    int = row['question_text'].count('$') 
    return int

def dollar_count_s1d (row):
    int = row['dollar_count']
    if int == 0:
        return '0'
    elif int >= 1:
        return '1+'
    else:
        return 'unknown'

df_train = pd.read_csv('../input/train.csv', nrows=int_read_csv_rows)
df_train['k_qid'] = df_train.index + 10000001
df_test = pd.read_csv('../input/test.csv', nrows=int_read_csv_rows)
df_test['k_qid'] = df_test.index + 20000001
df_train['train_or_test'] = 'train'
df_test['train_or_test'] = 'test'
df_all = pd.concat([df_train, df_test], sort=False)
df_all['overall'] = 'yes'

df_all['target_s1d'] = df_all.apply(target_s1d, axis=1)
df_all['q_length']  = df_all['question_text'].str.len()
df_all['q_length_s1d'] = df_all.apply(q_length_s1d, axis=1)
df_all['word_count'] = df_all.apply(word_count, axis=1)
df_all['word_count_s1d'] = df_all.apply(word_count_s1d, axis=1)
df_all['qmark_count'] = df_all.apply(qmark_count, axis=1)
df_all['qmark_count_s1d'] = df_all.apply(qmark_count_s1d, axis=1)
df_all['period_count'] = df_all.apply(period_count, axis=1)
df_all['period_count_s1d'] = df_all.apply(period_count_s1d, axis=1)
df_all['first_word'] = df_all.apply(first_word, axis=1)
df_all['first_word_s1d'] = df_all.apply(first_word_s1d, axis=1)
df_all['cap_word_count'] = df_all.apply(cap_word_count, axis=1)
df_all['cap_word_count_s1d'] = df_all.apply(cap_word_count_s1d, axis=1)
df_all['equals_count'] = df_all.apply(equals_count, axis=1)
df_all['equals_count_s1d'] = df_all.apply(equals_count_s1d, axis=1)
df_all['dollar_count'] = df_all.apply(dollar_count, axis=1)
df_all['dollar_count_s1d'] = df_all.apply(dollar_count_s1d, axis=1)

df_all['version'] = int_df_all_version

create_crosstab_type1(df_all, 'overall', int_important_crosstab)
create_crosstab_type1(df_all, 'train_or_test', int_important_crosstab)
create_crosstab_type1(df_all, 'target_s1d', int_important_crosstab)
create_crosstab_type1(df_all, 'q_length_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'word_count_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'qmark_count_s1d', int_current_crosstab) 
create_crosstab_type1(df_all, 'period_count_s1d', int_current_crosstab) 
create_crosstab_type1(df_all, 'first_word_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'cap_word_count_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'equals_count_s1d', int_current_crosstab)
create_crosstab_type1(df_all, 'dollar_count_s1d', int_current_crosstab)
df_all.sample(5)
if bln_create_df_all_csv_file:
    print('questions_s1.csv file created (version ' + str(int_df_all_version) + ')')
    df_all.to_csv('questions_s1.csv', index=False)

create_contains_word_var('i')
create_crosstab_type1(df_all, 'contains_word_i', int_current_crosstab)
create_contains_word_var('best')
create_crosstab_type1(df_all, 'contains_word_best', int_current_crosstab)
create_contains_word_var('men')
create_crosstab_type1(df_all, 'contains_word_men', int_current_crosstab)
create_contains_word_var('women')
create_crosstab_type1(df_all, 'contains_word_women', int_current_crosstab)
create_contains_word_var('money')
create_crosstab_type1(df_all, 'contains_word_money', int_current_crosstab)
create_contains_word_var('you')
create_crosstab_type1(df_all, 'contains_word_you', int_current_crosstab)
create_contains_word_var('love')
create_crosstab_type1(df_all, 'contains_word_love', int_current_crosstab)
df_temp = df_all[ df_all['equals_count'] >= 1 ]
df_temp[['question_text', 'equals_count', 'equals_count_s1d']].sample(20)
if bln_create_words_csv_file:
    print('creating words file...')
    str_filename = 'words.csv'
    csvfile1 = open(str_filename, 'w')
    writer1 = csv.writer(csvfile1)
    writer1.writerow( ['train_or_test', 'k_qid', 'target', 'position', 'word_orig', 'word_alt'] )

    #df_temp = df_all[ df_all['k_qid'] > 10700000 ]
    df_temp = df_all[ (df_all['train_or_test'] == 'train') & (df_all['k_qid'] > 10700000)]
    df_temp.reset_index(drop=True)

    int_rows = df_temp.shape[0]
    for i in range(int_rows):
        str_train_or_test = df_temp.iloc[i]['train_or_test']
        int_k_qid = df_temp.iloc[i]['k_qid']
        if str_train_or_test == 'train':
            int_target = int(df_temp.iloc[i]['target'])
        else:
            int_target = None
        str_question_text = df_temp.iloc[i]['question_text']
        words = str_question_text.split()
        int_position = 0
        for word in words:
            int_position += 1
            word_alt = word
            word_alt = word_alt.lower() 
            word_alt = word_alt.replace('?','')
            word_alt = word_alt.replace(',','')
            word_alt = word_alt.replace('.','')
            word_alt = word_alt.replace('/','')
            word_alt = word_alt.replace('(','')
            word_alt = word_alt.replace(')','')
            word_alt = word_alt.replace('\'','')
            word_alt = word_alt.replace('"','')
            word_alt = word_alt.replace('-','')
            word_alt = word_alt.replace('\\','')
            writer1.writerow( [str_train_or_test, int_k_qid, int_target, int_position, word, word_alt] )
            if int_k_qid == 10000001 or int_k_qid == 20000001:
                print(str_train_or_test, int_k_qid, int_target, int_position, word, word_alt)

    csvfile1.close()

#def temp_count (row):
#    int = row['question_text'].count('\\') 
#    return int
#df_all['temp_count'] = df_all.apply(temp_count, axis=1)
#df_train.info()
#df_train.sample(12)
#df_test.info()
#df_test.sample(12)
#df_all.info()
#df_all.sample(12)

#df_all_p01['rev_count_s1d'] = df_all_p01.apply(rev_count_s1d, axis=1)
#df_all_p01['sessions_s1d'] = df_all_p01.apply(sessions_s1d, axis=1)
#df_all_p01['date_min_s1d'] = df_all_p01.apply(year_month_code2, axis=1, str_input_column='date_min')
#df_all_p01['date_min'] = pd.to_datetime(df_all_p01['date_min'].astype(str), format='%Y%m%d')
#df_all_p01['date_diff_days'] = (df_all_p01['date_max'] - df_all_p01['date_min']).dt.days
#df_all_p01['date_diff_days_s1d'] = df_all_p01.apply(date_diff_days_s1d, axis=1)
#df_all_p01['totals_hits_avg_s1d'] = df_all_p01.apply(totals_hits_avg_s1d, axis=1)
#df_all_p01['totals_pageviews_avg_s1d'] = df_all_p01.apply(totals_pageviews_avg_s1d, axis=1)
#df_all_p01['date_min_s2d'] = df_all_p01.apply(date_min_s2d, axis=1)
#df_all_p01['revenue_date_min_s1d'] = df_all_p01.apply(year_month_code3, axis=1, str_input_column='revenue_date_min')
#df_all_p01['sp1_s1d'] = df_all_p01.apply(sp1_s1d, axis=1)

#df_train_p01 = pd.merge(df_train_p01, df_train_id, how='left', on=['fullVisitorId'])
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
#df_temp = df_all[ df_all['dollar_count'] >= 1 ]
#df_temp[['question_text', 'dollar_count', 'dollar_count_s1d']].sample(20)

end_time_check(dat_program_start, 'overall')
df_time_check