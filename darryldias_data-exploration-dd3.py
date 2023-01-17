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
from datetime import datetime
from IPython.core.display import display, HTML
bln_ready_to_commit = True
bln_create_estimate_files = True
bln_upload_input_estimates = True
bln_recode_variables = True

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

# used some code from the following sites:
# https://pythonprogramming.net/pandas-column-operations-calculations/
# https://stackoverflow.com/questions/29077188/absolute-value-for-column-in-python
# https://stackoverflow.com/questions/19758364/rename-a-single-column-header-in-a-pandas-dataframe
# https://stackoverflow.com/questions/20107570/removing-index-column-in-pandas
def get_column_analysis(int_analysis, int_code):
    # created by darryldias 24jul2018 
    if int_code == 1:
        return ['overall', 'test', 'train', 'no pay diff', 'pay diff']
    elif int_code == 2:
        return ['OVERALL', 'TRAIN OR TEST', 'TRAIN OR TEST', 'TARGET DESCRIPTION', 'TARGET DESCRIPTION']
    elif int_code == 3:
        return ['yes', 'test', 'train', 'no payment difficulties', 'payment difficulties']
    else:
        return None

def create_crosstab_type1(df_input, str_row_question, int_output_destination):
    # created by darryldias 10jun2018 - updated 24jul2018 
    # got some useful code from:
    # https://chrisalbon.com/python/data_wrangling/pandas_missing_data/
    # https://www.tutorialspoint.com/python/python_lists.htm
    # https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points

    if int_output_destination == 0:
        return None
    
    str_count_desc = 'count'  #get_translations_analysis_description(translations_analysis, str_language, 'special', 3)
    str_colpercent_desc = 'col percent'
    
    int_columns = 5
    list_str_column_desc = get_column_analysis(1, 1)
    list_str_column_question = get_column_analysis(1, 2)
    list_str_column_category = get_column_analysis(1, 3)
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
        
    
    df_output = pd.DataFrame(columns=['row_question', 'row_category', 'statistic', 'c1', 'c2', 'c3', 'c4', 'c5'])
    int_row = 1
    df_output.loc[int_row] = [str_row_question, '', '', list_str_column_desc[0], list_str_column_desc[1], list_str_column_desc[2], \
                                 list_str_column_desc[3], list_str_column_desc[4]]
    int_row = 2
    df_output.loc[int_row] = [str_row_question, 'total', str_count_desc, list_int_column_base[0], list_int_column_base[1], list_int_column_base[2], \
                                 list_int_column_base[3], list_int_column_base[4]] 
    int_row = 3
    df_output.loc[int_row] = [str_row_question, 'total', str_colpercent_desc, list_flt_column_base_percent[0], list_flt_column_base_percent[1], \
                               list_flt_column_base_percent[2], list_flt_column_base_percent[3], list_flt_column_base_percent[4]] 

    for i in range(int_rows):
        int_row += 1
        int_count_row = int_row
        int_row += 1
        int_colpercent_row = int_row

        str_row_category = df_group.iloc[i][0]

        list_int_column_count = []
        list_flt_column_percent = []
        for j in range(int_columns):
            int_count = df_input[ (df_input[str_row_question]==str_row_category) & (df_input[list_str_column_question[j]]==list_str_column_category[j]) ] \
                                [list_str_column_question[j]].count()
            list_int_column_count.append(int_count)
            flt_base = float(list_int_column_base[j])
            if flt_base > 0:
                flt_percent = round(100 * int_count / flt_base,1)
                str_percent = "{0:.1f}".format(flt_percent)
            else:
                str_percent = ''
            list_flt_column_percent.append(str_percent)
        
        df_output.loc[int_count_row] = [str_row_question, str_row_category, str_count_desc, list_int_column_count[0], list_int_column_count[1], \
                                        list_int_column_count[2], list_int_column_count[3], list_int_column_count[4]]
        df_output.loc[int_colpercent_row] = [str_row_question, str_row_category, str_colpercent_desc, list_flt_column_percent[0], \
                                             list_flt_column_percent[1],list_flt_column_percent[2], list_flt_column_percent[3], list_flt_column_percent[4]]
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

def year_month_code (row, str_input_column_year, str_input_column_month):
    # created by darryldias 1june2018   
    if row[str_input_column_month] <= 9 :   
        return int(str(row[str_input_column_year]) + '0' + str(row[str_input_column_month]))
    if row[str_input_column_month] <= 12 :   
        return int(str(row[str_input_column_year]) + str(row[str_input_column_month]))
    return 0

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

start_time_check()
application_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv', nrows=int_read_csv_rows)
application_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv', nrows=int_read_csv_rows)
#column_descriptions = pd.read_csv('../input/HomeCredit_columns_description.csv', nrows=int_read_csv_rows)
application_train.head(10)
get_dataframe_info(application_train, True, 'info_application_train.csv')
create_topline(application_train, 'TARGET', strg_count_column)   
def target_desc (row):  
    if row['TARGET'] == 1 :
        return 'payment difficulties'
    if row['TARGET'] == 0 :
        return 'no payment difficulties'
    return 'Unknown'

application_train['TARGET DESCRIPTION'] = application_train.apply(target_desc, axis=1)
create_topline(application_train, 'TARGET DESCRIPTION', strg_count_column)   
application_train['TRAIN OR TEST'] = 'train'
application_test.head(10)
get_dataframe_info(application_test, True, 'info_application_test.csv')
application_test['TRAIN OR TEST'] = 'test'
end_time_check(dat_start, 'application train and test')
start_time_check()
application_all = pd.concat([application_train, application_test], sort=False)
application_all['OVERALL'] = 'yes'
application_all.sample(10)
get_dataframe_info(application_all, True, 'info_application_all.csv')
#print('Past crosstabs can be found in the data source section of this kernel.')
display(HTML('<h3>Crosstabs which have been generated previously can be found in the data source section of this kernel: <a href="https://www.kaggle.com/darryldias/data-exploration-dd3/data">data tab</a></h3>'))
# recode variable section
if bln_recode_variables:
    def cnt_children_s1d (row):  
        if row['CNT_CHILDREN'] == 0 :
            return '0'
        if row['CNT_CHILDREN'] == 1 :
            return '1'
        if row['CNT_CHILDREN'] == 2 :
            return '2'
        if row['CNT_CHILDREN'] >= 3 :
            return '3 or more'
        return 'Unknown'
    application_all['CNT_CHILDREN_S1D'] = application_all.apply(cnt_children_s1d, axis=1)
    application_all['FLAG_MOBIL_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_MOBIL')
    application_all['FLAG_EMP_PHONE_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_EMP_PHONE')
    application_all['FLAG_WORK_PHONE_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_WORK_PHONE')
    application_all['FLAG_CONT_MOBILE_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_CONT_MOBILE')
    application_all['FLAG_PHONE_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_PHONE')
    application_all['FLAG_EMAIL_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_EMAIL')
    application_all['FLAG_DOCUMENT_2_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_2')
    application_all['FLAG_DOCUMENT_3_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_3')
    application_all['FLAG_DOCUMENT_4_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_4')
    application_all['FLAG_DOCUMENT_5_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_5')
    application_all['FLAG_DOCUMENT_6_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_6')
    application_all['FLAG_DOCUMENT_7_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_7')
    application_all['FLAG_DOCUMENT_8_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_8')
    application_all['FLAG_DOCUMENT_9_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_9')
    application_all['FLAG_DOCUMENT_10_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_10')
    application_all['FLAG_DOCUMENT_11_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_11')
    application_all['FLAG_DOCUMENT_12_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_12')
    application_all['FLAG_DOCUMENT_13_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_13')
    application_all['FLAG_DOCUMENT_14_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_14')
    application_all['FLAG_DOCUMENT_15_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_15')
    application_all['FLAG_DOCUMENT_16_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_16')
    application_all['FLAG_DOCUMENT_17_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_17')
    application_all['FLAG_DOCUMENT_18_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_18')
    application_all['FLAG_DOCUMENT_19_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_19')
    application_all['FLAG_DOCUMENT_20_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_20')
    application_all['FLAG_DOCUMENT_21_S1D'] = application_all.apply(n_0_1_summary, axis=1, str_input_column='FLAG_DOCUMENT_21')
    application_all['DAYS_BIRTH_POSITIVE'] = 0 - application_all['DAYS_BIRTH']
    application_all['YEARS_BIRTH'] = application_all['DAYS_BIRTH_POSITIVE'] / 365.0
    def years_birth_s1d (row):  
        if row['YEARS_BIRTH'] <= 25 :
            return '(20 to 25]'
        if row['YEARS_BIRTH'] <= 30 :
            return '(25 to 30]'
        if row['YEARS_BIRTH'] <= 35 :
            return '(30 to 35]'
        if row['YEARS_BIRTH'] <= 40 :
            return '(35 to 40]'
        if row['YEARS_BIRTH'] <= 45 :
            return '(40 to 45]'
        if row['YEARS_BIRTH'] <= 50 :
            return '(45 to 50]'
        if row['YEARS_BIRTH'] <= 55 :
            return '(50 to 55]'
        if row['YEARS_BIRTH'] <= 60 :
            return '(55 to 60]'
        if row['YEARS_BIRTH'] <= 65 :
            return '(60 to 65]'
        if row['YEARS_BIRTH'] <= 70 :
            return '(65 to 70]'
        return 'Unknown'
    application_all['YEARS_BIRTH_S1D'] = application_all.apply(years_birth_s1d, axis=1)
    application_all['EXT_SOURCE_1_S1D'] = application_all.apply(n_0_1_summary2, axis=1, str_input_column='EXT_SOURCE_1')
    application_all['EXT_SOURCE_2_S1D'] = application_all.apply(n_0_1_summary2, axis=1, str_input_column='EXT_SOURCE_2')
    application_all['EXT_SOURCE_3_S1D'] = application_all.apply(n_0_1_summary2, axis=1, str_input_column='EXT_SOURCE_3')
    application_all['REGION_RATING_CLIENT_S1D'] = application_all.apply(n_0_10_summary, axis=1, str_input_column='REGION_RATING_CLIENT')
    application_all['REGION_RATING_CLIENT_W_CITY_S1D'] = application_all.apply(n_0_10_summary, axis=1, str_input_column='REGION_RATING_CLIENT_W_CITY')
    def amt_income_total_s1d (row):  
        if row['AMT_INCOME_TOTAL'] <= 75000 :
            return '(0 to 75,000]'
        if row['AMT_INCOME_TOTAL'] <= 100000 :
            return '(075,000 to 100,000]'
        if row['AMT_INCOME_TOTAL'] <= 125000 :
            return '(100,000 to 125,000]'
        if row['AMT_INCOME_TOTAL'] <= 150000 :
            return '(125,000 to 150,000]'
        if row['AMT_INCOME_TOTAL'] <= 175000 :
            return '(150,000 to 175,000]'
        if row['AMT_INCOME_TOTAL'] <= 200000 :
            return '(175,000 to 200,000]'
        if row['AMT_INCOME_TOTAL'] <= 250000 :
            return '(200,000 to 250,000]'
        if row['AMT_INCOME_TOTAL'] > 250000 :
            return '(250,000+'
        return 'Unknown'
    application_all['AMT_INCOME_TOTAL_S1D'] = application_all.apply(amt_income_total_s1d, axis=1)
    def amt_credit_s1d (row):  
        if row['AMT_CREDIT'] <= 200000 :
            return '(0 to 200,000]'
        if row['AMT_CREDIT'] <= 300000 :
            return '(0,200,000 to 300,000]'
        if row['AMT_CREDIT'] <= 450000 :
            return '(0,300,000 to 450,000]'
        if row['AMT_CREDIT'] <= 600000 :
            return '(0,450,000 to 600,000]'
        if row['AMT_CREDIT'] <= 800000 :
            return '(0,600,000 to 800,000]'
        if row['AMT_CREDIT'] <= 1000000 :
            return '(0,800,000 to 1,000,000]'
        if row['AMT_CREDIT'] > 1000000 :
            return '(1,000,000+'
        return 'Unknown'
    application_all['AMT_CREDIT_S1D'] = application_all.apply(amt_credit_s1d, axis=1)
    def amt_annuity_s1d (row):  
        if row['AMT_ANNUITY'] <= 10000 :
            return '(0 to 10,000]'
        if row['AMT_ANNUITY'] <= 15000 :
            return '(10,000 to 15,000]'
        if row['AMT_ANNUITY'] <= 20000 :
            return '(15,000 to 20,000]'
        if row['AMT_ANNUITY'] <= 25000 :
            return '(20,000 to 25,000]'
        if row['AMT_ANNUITY'] <= 30000 :
            return '(25,000 to 30,000]'
        if row['AMT_ANNUITY'] <= 40000 :
            return '(30,000 to 40,000]'
        if row['AMT_ANNUITY'] <= 50000 :
            return '(40,000 to 50,000]'
        if row['AMT_ANNUITY'] > 50000 :
            return '(50,000+]'
        return 'Unknown'
    application_all['AMT_ANNUITY_S1D'] = application_all.apply(amt_annuity_s1d, axis=1)
    def amt_goods_price_s1d (row):  
        if row['AMT_GOODS_PRICE'] <= 150000 :
            return '(0 to 150,000]'
        if row['AMT_GOODS_PRICE'] <= 250000 :
            return '(150,000 to 250,000]'
        if row['AMT_GOODS_PRICE'] <= 350000 :
            return '(250,000 to 350,000]'
        if row['AMT_GOODS_PRICE'] <= 450000 :
            return '(350,000 to 450,000]'
        if row['AMT_GOODS_PRICE'] <= 600000 :
            return '(450,000 to 600,000]'
        if row['AMT_GOODS_PRICE'] <= 800000 :
            return '(600,000 to 800,000]'
        if row['AMT_GOODS_PRICE'] > 800000 :
            return '(800,000+]'
        return 'Unknown'
    application_all['AMT_GOODS_PRICE_S1D'] = application_all.apply(amt_goods_price_s1d, axis=1)
    def own_car_age_s1d (row): 
        if row['OWN_CAR_AGE'] <= 2.000000:
            return '(0 to 2.0]'
        if row['OWN_CAR_AGE'] <= 4.000000:
            return '(02.0 to 4.0]'
        if row['OWN_CAR_AGE'] <= 7.000000:
            return '(04.0 to 7.0]'
        if row['OWN_CAR_AGE'] <= 10.000000:
            return '(07.0 to 10.0]'
        if row['OWN_CAR_AGE'] <= 14.000000:
            return '(10.0 to 14.0]'
        if row['OWN_CAR_AGE'] <= 20.000000:
            return '(14.0 to 20.0]'
        if row['OWN_CAR_AGE'] > 20.000000:
            return '(20.0+]'
        return 'Unknown'
    application_all['OWN_CAR_AGE_S1D'] = application_all.apply(own_car_age_s1d, axis=1)
    def region_population_relative_s1d (row): 
        if row['REGION_POPULATION_RELATIVE'] <= 0.007478:
            return '(0 to 0.007478]'
        if row['REGION_POPULATION_RELATIVE'] <= 0.011091:
            return '(0.007478 to 0.011091]'
        if row['REGION_POPULATION_RELATIVE'] <= 0.018317:
            return '(0.011091 to 0.018317]'
        if row['REGION_POPULATION_RELATIVE'] <= 0.021930:
            return '(0.018317 to 0.021930]'
        if row['REGION_POPULATION_RELATIVE'] <= 0.029155:
            return '(0.021930 to 0.029155]'
        if row['REGION_POPULATION_RELATIVE'] <= 0.036381:
            return '(0.029155 to 0.036381]'
        if row['REGION_POPULATION_RELATIVE'] <= 0.072508:
            return '(0.036381 to 0.072508]'
        return 'Unknown'
    application_all['REGION_POPULATION_RELATIVE_S1D'] = application_all.apply(region_population_relative_s1d, axis=1)
    application_all['YEARS_EMPLOYED'] = application_all['DAYS_EMPLOYED'] / -365.0 
    def years_employed_s1d (row): 
        if row['YEARS_EMPLOYED'] < 0:
            return '< 0'
        if row['YEARS_EMPLOYED'] <= 1.5:
            return '(0 to 1.5]'
        if row['YEARS_EMPLOYED'] <= 3.0:
            return '(01.5 to 3.0]'
        if row['YEARS_EMPLOYED'] <= 5.0:
            return '(03.0 to 5.0]'
        if row['YEARS_EMPLOYED'] <= 7.5:
            return '(05.0 to 7.5]'
        if row['YEARS_EMPLOYED'] <= 10.0:
            return '(07.5 to 10.0]'
        if row['YEARS_EMPLOYED'] <= 15.0:
            return '(10.0 to 15.0]'
        if row['YEARS_EMPLOYED'] <= 50.0:
            return '(15.0 to 50.0]'
        return 'Unknown'
    application_all['YEARS_EMPLOYED_S1D'] = application_all.apply(years_employed_s1d, axis=1)
    application_all['YEARS_REGISTRATION'] = application_all['DAYS_REGISTRATION'] / -365.0 
    def years_registration_s1d (row): 
        if row['YEARS_REGISTRATION'] <= 2.253151:
            return '(0 to 2.253151]'
        if row['YEARS_REGISTRATION'] <= 4.506301:
            return '(02.253151 to 4.506301]'
        if row['YEARS_REGISTRATION'] <= 6.759452:
            return '(04.506301 to 6.759452]'
        if row['YEARS_REGISTRATION'] <= 11.265753:
            return '(06.759452 to 11.265753]'
        if row['YEARS_REGISTRATION'] <= 15.772055:
            return '(11.265753 to 15.772055]'
        if row['YEARS_REGISTRATION'] <= 22.531507:
            return '(15.772055 to 22.531507]'
        if row['YEARS_REGISTRATION'] <= 67.594521:
            return '(22.531507 to 67.594521]'
        return 'Unknown'
    application_all['YEARS_REGISTRATION_S1D'] = application_all.apply(years_registration_s1d, axis=1)
    application_all['YEARS_ID_PUBLISH'] = application_all['DAYS_ID_PUBLISH'] / -365.0 
    def years_id_publish_s1d (row): 
        if row['YEARS_ID_PUBLISH'] <= 2.957671:
            return '(0 to 2.957671]'
        if row['YEARS_ID_PUBLISH'] <= 5.915342:
            return '(02.957671 to 5.915342]'
        if row['YEARS_ID_PUBLISH'] <= 7.887123:
            return '(05.915342 to 7.887123]'
        if row['YEARS_ID_PUBLISH'] <= 10.844795:
            return '(07.887123 to 10.844795]'
        if row['YEARS_ID_PUBLISH'] <= 11.830685:
            return '(10.844795 to 11.830685]'
        if row['YEARS_ID_PUBLISH'] <= 12.816575:
            return '(11.830685 to 12.816575]'
        if row['YEARS_ID_PUBLISH'] <= 19.717809:
            return '(12.816575 to 19.717809]'
        return 'Unknown'
    application_all['YEARS_ID_PUBLISH_S1D'] = application_all.apply(years_id_publish_s1d, axis=1)
    def cnt_fam_members_s1d (row): 
        if row['CNT_FAM_MEMBERS'] == 1:
            return '1'
        if row['CNT_FAM_MEMBERS'] == 2:
            return '2'
        if row['CNT_FAM_MEMBERS'] == 3:
            return '3'
        if row['CNT_FAM_MEMBERS'] == 4:
            return '4'
        if row['CNT_FAM_MEMBERS'] >= 5:
            return '5+'
        return 'Unknown'
    application_all['CNT_FAM_MEMBERS_S1D'] = application_all.apply(cnt_fam_members_s1d, axis=1)

    df_output = application_all[['SK_ID_CURR', 'CNT_CHILDREN_S1D', 'FLAG_MOBIL_S1D', 'FLAG_EMP_PHONE_S1D', 'FLAG_WORK_PHONE_S1D', 'FLAG_CONT_MOBILE_S1D', \
                                 'FLAG_PHONE_S1D', 'FLAG_EMAIL_S1D', 'FLAG_DOCUMENT_2_S1D', 'FLAG_DOCUMENT_3_S1D', 'FLAG_DOCUMENT_4_S1D', \
                                 'FLAG_DOCUMENT_5_S1D', 'FLAG_DOCUMENT_6_S1D', 'FLAG_DOCUMENT_7_S1D', 'FLAG_DOCUMENT_8_S1D', 'FLAG_DOCUMENT_9_S1D', \
                                 'FLAG_DOCUMENT_10_S1D', 'FLAG_DOCUMENT_11_S1D', 'FLAG_DOCUMENT_12_S1D', 'FLAG_DOCUMENT_13_S1D', 'FLAG_DOCUMENT_14_S1D', \
                                 'FLAG_DOCUMENT_15_S1D', 'FLAG_DOCUMENT_16_S1D', 'FLAG_DOCUMENT_17_S1D', 'FLAG_DOCUMENT_18_S1D', 'FLAG_DOCUMENT_19_S1D', \
                                 'FLAG_DOCUMENT_20_S1D', 'FLAG_DOCUMENT_21_S1D', 'DAYS_BIRTH_POSITIVE', 'YEARS_BIRTH', 'YEARS_BIRTH_S1D', \
                                 'EXT_SOURCE_1_S1D', 'EXT_SOURCE_2_S1D', 'EXT_SOURCE_3_S1D', 'REGION_RATING_CLIENT_S1D', 'REGION_RATING_CLIENT_W_CITY_S1D',\
                                 'AMT_INCOME_TOTAL_S1D', 'AMT_CREDIT_S1D', 'AMT_ANNUITY_S1D', 'AMT_GOODS_PRICE_S1D', 'OWN_CAR_AGE_S1D', \
                                 'REGION_POPULATION_RELATIVE_S1D', 'YEARS_EMPLOYED', 'YEARS_EMPLOYED_S1D', 'YEARS_REGISTRATION', 'YEARS_REGISTRATION_S1D',\
                                 'YEARS_ID_PUBLISH', 'YEARS_ID_PUBLISH_S1D', 'CNT_FAM_MEMBERS_S1D' ]]
    str_filename = 'dd3_input_recodes.csv'
    df_output.to_csv(str_filename, index = False)
else:
    input_recodes = pd.read_csv('../input/dd3-input-recodes/dd3_input_recodes.csv', nrows=int_read_csv_rows, \
                                dtype={'REGION_RATING_CLIENT_W_CITY_S1D': object} )
    application_all = pd.merge(application_all, input_recodes, how='left', on=['SK_ID_CURR'])


create_crosstab_type1(application_all, 'OVERALL', int_important_crosstab)
create_crosstab_type1(application_all, 'TRAIN OR TEST', int_important_crosstab)
create_crosstab_type1(application_all, 'TARGET DESCRIPTION', int_important_crosstab)
create_crosstab_type1(application_all, 'NAME_CONTRACT_TYPE', int_past_crosstab)
create_crosstab_type1(application_all, 'CODE_GENDER', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_OWN_CAR', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_OWN_REALTY', int_past_crosstab)
create_crosstab_type1(application_all, 'CNT_CHILDREN_S1D', int_past_crosstab) 
create_crosstab_type1(application_all, 'NAME_TYPE_SUITE', int_past_crosstab)
create_crosstab_type1(application_all, 'NAME_INCOME_TYPE', int_past_crosstab)
create_crosstab_type1(application_all, 'NAME_EDUCATION_TYPE', int_past_crosstab)
create_crosstab_type1(application_all, 'NAME_FAMILY_STATUS', int_past_crosstab)
create_crosstab_type1(application_all, 'NAME_HOUSING_TYPE', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_MOBIL_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_EMP_PHONE_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_WORK_PHONE_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_CONT_MOBILE_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_PHONE_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_EMAIL_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'OCCUPATION_TYPE', int_past_crosstab)
create_crosstab_type1(application_all, 'WEEKDAY_APPR_PROCESS_START', int_past_crosstab)
create_crosstab_type1(application_all, 'ORGANIZATION_TYPE', int_past_crosstab)  # ***** 
create_crosstab_type1(application_all, 'FONDKAPREMONT_MODE', int_past_crosstab)
create_crosstab_type1(application_all, 'HOUSETYPE_MODE', int_past_crosstab)
create_crosstab_type1(application_all, 'WALLSMATERIAL_MODE', int_past_crosstab)
create_crosstab_type1(application_all, 'EMERGENCYSTATE_MODE', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_2_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_3_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_4_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_5_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_6_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_7_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_8_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_9_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_10_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_11_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_12_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_13_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_14_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_15_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_16_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_17_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_18_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_19_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_20_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'FLAG_DOCUMENT_21_S1D', int_past_crosstab)
create_crosstab_type1(application_all, 'YEARS_BIRTH_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'YEARS_BIRTH', int_past_crosstab)
create_crosstab_type1(application_all, 'EXT_SOURCE_1_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'EXT_SOURCE_1', int_past_crosstab)
create_crosstab_type1(application_all, 'EXT_SOURCE_2_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'EXT_SOURCE_2', int_past_crosstab)
create_crosstab_type1(application_all, 'EXT_SOURCE_3_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'EXT_SOURCE_3', int_past_crosstab)
create_crosstab_type1(application_all, 'OWN_CAR_AGE_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'OWN_CAR_AGE', int_past_crosstab)
create_crosstab_type1(application_all, 'REGION_RATING_CLIENT_S1D', int_past_crosstab) 
create_crosstab_type2(application_all, 'REGION_RATING_CLIENT', int_past_crosstab) 
create_crosstab_type1(application_all, 'REGION_RATING_CLIENT_W_CITY_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'REGION_RATING_CLIENT_W_CITY', int_past_crosstab)
create_crosstab_type1(application_all, 'AMT_INCOME_TOTAL_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'AMT_INCOME_TOTAL', int_past_crosstab)
create_crosstab_type1(application_all, 'AMT_CREDIT_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'AMT_CREDIT', int_past_crosstab)
create_crosstab_type1(application_all, 'AMT_ANNUITY_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'AMT_ANNUITY', int_past_crosstab)
create_crosstab_type1(application_all, 'AMT_GOODS_PRICE_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'AMT_GOODS_PRICE', int_past_crosstab)
create_crosstab_type1(application_all, 'REGION_POPULATION_RELATIVE_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'REGION_POPULATION_RELATIVE', int_past_crosstab)
create_crosstab_type1(application_all, 'YEARS_EMPLOYED_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'YEARS_EMPLOYED', int_past_crosstab)
create_crosstab_type1(application_all, 'YEARS_REGISTRATION_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'YEARS_REGISTRATION', int_past_crosstab)
create_crosstab_type1(application_all, 'YEARS_ID_PUBLISH_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'YEARS_ID_PUBLISH', int_past_crosstab)
create_crosstab_type1(application_all, 'CNT_FAM_MEMBERS_S1D', int_past_crosstab)
create_crosstab_type2(application_all, 'CNT_FAM_MEMBERS', int_past_crosstab)
create_crosstab_type2(application_all, 'HOUR_APPR_PROCESS_START', int_current_crosstab)
create_crosstab_type2(application_all, 'REG_REGION_NOT_LIVE_REGION', int_current_crosstab)
create_crosstab_type2(application_all, 'REG_REGION_NOT_WORK_REGION', int_current_crosstab)
create_crosstab_type2(application_all, 'LIVE_REGION_NOT_WORK_REGION', int_current_crosstab)
create_crosstab_type2(application_all, 'REG_CITY_NOT_LIVE_CITY', int_current_crosstab)
create_crosstab_type2(application_all, 'REG_CITY_NOT_WORK_CITY', int_current_crosstab)
create_crosstab_type2(application_all, 'LIVE_CITY_NOT_WORK_CITY', int_current_crosstab)
create_crosstab_type2(application_all, 'OBS_30_CNT_SOCIAL_CIRCLE', int_current_crosstab)
create_crosstab_type2(application_all, 'DEF_30_CNT_SOCIAL_CIRCLE', int_current_crosstab)
create_crosstab_type2(application_all, 'OBS_60_CNT_SOCIAL_CIRCLE', int_current_crosstab)
create_crosstab_type2(application_all, 'DEF_60_CNT_SOCIAL_CIRCLE', int_current_crosstab)
create_crosstab_type2(application_all, 'DAYS_LAST_PHONE_CHANGE', int_current_crosstab)
create_crosstab_type2(application_all, 'AMT_REQ_CREDIT_BUREAU_HOUR', int_current_crosstab)
create_crosstab_type2(application_all, 'AMT_REQ_CREDIT_BUREAU_DAY', int_current_crosstab)
create_crosstab_type2(application_all, 'AMT_REQ_CREDIT_BUREAU_WEEK', int_current_crosstab)
create_crosstab_type2(application_all, 'AMT_REQ_CREDIT_BUREAU_MON', int_current_crosstab)
create_crosstab_type2(application_all, 'AMT_REQ_CREDIT_BUREAU_QRT', int_current_crosstab)
create_crosstab_type2(application_all, 'AMT_REQ_CREDIT_BUREAU_YEAR', int_current_crosstab)
create_crosstab_type2(application_all, 'APARTMENTS_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'BASEMENTAREA_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'YEARS_BEGINEXPLUATATION_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'YEARS_BUILD_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'COMMONAREA_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'ELEVATORS_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'ENTRANCES_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'FLOORSMAX_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'FLOORSMIN_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'LANDAREA_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'LIVINGAPARTMENTS_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'LIVINGAREA_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'NONLIVINGAPARTMENTS_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'NONLIVINGAREA_AVG', int_current_crosstab)
create_crosstab_type2(application_all, 'APARTMENTS_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'BASEMENTAREA_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'YEARS_BEGINEXPLUATATION_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'YEARS_BUILD_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'COMMONAREA_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'ELEVATORS_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'ENTRANCES_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'FLOORSMAX_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'FLOORSMIN_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'LANDAREA_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'LIVINGAPARTMENTS_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'LIVINGAREA_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'NONLIVINGAPARTMENTS_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'NONLIVINGAREA_MODE', int_current_crosstab)
create_crosstab_type2(application_all, 'APARTMENTS_MEDI', int_current_crosstab)
create_crosstab_type2(application_all, 'BASEMENTAREA_MEDI', int_current_crosstab)
create_crosstab_type2(application_all, 'YEARS_BEGINEXPLUATATION_MEDI', int_current_crosstab)
end_time_check(dat_start, 'application all')
start_time_check()
if bln_upload_input_estimates:
    input_estimates = pd.read_csv('../input/dd3-input-estimates/dd3_input_estimates.csv', nrows=int_read_csv_rows)
    application_all = pd.merge(application_all, input_estimates, how='left', on=['SK_ID_CURR'])
else:
    application_all['TARGET_GROUP'] = 0
    application_all['TARGET_ESTIMATE'] = 0.5

application_all['TARGET_DIFF_ACTUAL'] = application_all['TARGET_ESTIMATE'] - application_all['TARGET'] 
application_all['TARGET_DIFF_ABS'] = application_all['TARGET_DIFF_ACTUAL'].abs() 
application_all[['SK_ID_CURR', 'TRAIN OR TEST','TARGET_GROUP', 'TARGET', 'TARGET_ESTIMATE', 'TARGET_DIFF_ACTUAL', 'TARGET_DIFF_ABS']].sample(20)
grouped = application_all.groupby(['TRAIN OR TEST','TARGET_GROUP'])   
dfGrouped = grouped['TARGET_DIFF_ABS'].mean().reset_index(name='TARGET_DIFF_ABS_MEAN')  
dfGrouped
if bln_create_estimate_files:
    df_output = application_all[['SK_ID_CURR', 'TRAIN OR TEST','TARGET_GROUP', 'TARGET', 'TARGET_ESTIMATE', 'TARGET_DIFF_ACTUAL', 'TARGET_DIFF_ABS']]
    str_filename = 'all_estimates.csv'
    df_output.to_csv(str_filename, index = False)
    print (str_filename + ' file created ')
    df_output = application_all[( application_all['TRAIN OR TEST']=='test' )]
    df_output = df_output[['SK_ID_CURR', 'TARGET_ESTIMATE']]
    df_output.rename(columns={'TARGET_ESTIMATE':'TARGET'}, inplace=True)
    str_filename = 'sample_submission.csv'
    df_output.to_csv(str_filename, index = False)
    print (str_filename + ' file created ')
df_temp = application_all[['SK_ID_CURR', 'OVERALL', 'TRAIN OR TEST', 'TARGET DESCRIPTION', 'EXT_SOURCE_1_S1D', 'OWN_CAR_AGE', 'YEARS_BIRTH_S1D']]
df_temp.to_csv('temp.csv', index = False)
end_time_check(dat_start, 'testing and learning')
end_time_check(dat_program_start, 'overall')
df_time_check
#testing
#application_test['SK_ID_CURR'].describe()
#int_count = df_input[str_row_question].count()
#int_count = df_input[(df_input['CODE_GENDER']=='M')]['OVERALL'].count()
#create_crosstab_type1(application_all, '')
#input_estimates.sample(10)
#application_all[['DAYS_BIRTH', 'DAYS_EMPLOYED']].describe()
#check_numeric_var('YEARS_REGISTRATION', 20)
#application_all[(application_all['REGION_RATING_CLIENT_W_CITY']>=1)]['REGION_RATING_CLIENT_W_CITY'].count()
#application_all[['AMT_INCOME_TOTAL', 'AMT_INCOME_TOTAL_S1D']].sample(30)
#create_topline(application_all, '', strg_count_column) 
#def x_x_s1d (row): 
#    if row['x_x'] <= x:
#        return '(x to x]'
#    return 'Unknown'
#application_all['x_x_S1D'] = application_all.apply(x_x_s1d, axis=1)