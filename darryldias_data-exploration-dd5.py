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
import bq_helper
ds_current = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "nhtsa_traffic_fatalities")

from datetime import datetime
from IPython.core.display import display, HTML
bln_ready_to_commit = True
bln_create_estimate_files = True
bln_upload_input_estimates = True
bln_recode_variables = False
pd.options.display.max_rows = 110

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
str_language = 'en'
#print(os.listdir("../input/dd5-translations-analysis"))
#print(os.listdir("../input/nhtsa-traffic-fatalities"))
def get_translations_analysis_description(df_input, str_language, str_group, int_code):
    # created by darryldias 25may2018
    df_temp = df_input[(df_input['language']==str_language) & (df_input['group']==str_group) & (df_input['code']==int_code)] \
                    ['description']
    return df_temp.iloc[0]

translations_analysis = pd.read_csv('../input/nhtsa-traffic-fatalities/dd5_translations_analysis.csv') # bug with kernel
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

# used some code from the following sites:
# https://pythonprogramming.net/pandas-column-operations-calculations/
# https://stackoverflow.com/questions/29077188/absolute-value-for-column-in-python
# https://stackoverflow.com/questions/19758364/rename-a-single-column-header-in-a-pandas-dataframe
# https://stackoverflow.com/questions/20107570/removing-index-column-in-pandas

def get_column_analysis(int_analysis, int_code):
    # created by darryldias 24jul2018 
    if int_code == 1:
        return ['overall', '2016', '2015', 'weekday', 'weekend', '00 to 05', '06 to 11', '12 to 17', '18 to 23', '0 dd', '1+ dd']
    elif int_code == 2:
        return ['overall', 'year_of_crash_s1d', 'year_of_crash_s1d', 'day_of_week_s2d', 'day_of_week_s2d', \
                           'hour_of_crash_s1d', 'hour_of_crash_s1d', 'hour_of_crash_s1d', 'hour_of_crash_s1d', \
                           'number_of_drunk_drivers_s1d', 'number_of_drunk_drivers_s1d']
    elif int_code == 3:
        return ['yes', '2016', '2015', 'weekday', 'weekend', \
                       '00:00 to 05:59', '06:00 to 11:59', '12:00 to 17:59', '18:00 to 23:59', 'None', '1 or more']
    else:
        return None


def create_crosstab_type1(df_input, str_row_question, int_output_destination):
    # created by darryldias 10jun2018 - updated 24jun2018 
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
        
    
    df_output = pd.DataFrame(columns=['row_question', 'row_category', 'statistic', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11'])
    int_row = 1
    df_output.loc[int_row] = [str_row_question, '', '', list_str_column_desc[0], \
                                                        list_str_column_desc[1], \
                                                        list_str_column_desc[2], \
                                                        list_str_column_desc[3], \
                                                        list_str_column_desc[4], \
                                                        list_str_column_desc[5], \
                                                        list_str_column_desc[6], \
                                                        list_str_column_desc[7], \
                                                        list_str_column_desc[8], \
                                                        list_str_column_desc[9], \
                                                        list_str_column_desc[10] ]
    int_row = 2
    df_output.loc[int_row] = [str_row_question, 'total', str_count_desc, list_int_column_base[0], \
                                                                         list_int_column_base[1], \
                                                                         list_int_column_base[2], \
                                                                         list_int_column_base[3], \
                                                                         list_int_column_base[4], \
                                                                         list_int_column_base[5], \
                                                                         list_int_column_base[6], \
                                                                         list_int_column_base[7], \
                                                                         list_int_column_base[8], \
                                                                         list_int_column_base[9], \
                                                                         list_int_column_base[10] ] 
    int_row = 3
    df_output.loc[int_row] = [str_row_question, 'total', str_colpercent_desc, list_flt_column_base_percent[0], \
                                                                              list_flt_column_base_percent[1], \
                                                                              list_flt_column_base_percent[2], \
                                                                              list_flt_column_base_percent[3], \
                                                                              list_flt_column_base_percent[4], \
                                                                              list_flt_column_base_percent[5], \
                                                                              list_flt_column_base_percent[6], \
                                                                              list_flt_column_base_percent[7], \
                                                                              list_flt_column_base_percent[8], \
                                                                              list_flt_column_base_percent[9], \
                                                                              list_flt_column_base_percent[10] ] 

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
        
        df_output.loc[int_count_row] = [str_row_question, str_row_category, str_count_desc, list_int_column_count[0], \
                                                                                            list_int_column_count[1], \
                                                                                            list_int_column_count[2], \
                                                                                            list_int_column_count[3], \
                                                                                            list_int_column_count[4], \
                                                                                            list_int_column_count[5], \
                                                                                            list_int_column_count[6], \
                                                                                            list_int_column_count[7], \
                                                                                            list_int_column_count[8], \
                                                                                            list_int_column_count[9], \
                                                                                            list_int_column_count[10] ]
        df_output.loc[int_colpercent_row] = [str_row_question, str_row_category, str_colpercent_desc, list_flt_column_percent[0], \
                                                                                                      list_flt_column_percent[1], \
                                                                                                      list_flt_column_percent[2], \
                                                                                                      list_flt_column_percent[3], \
                                                                                                      list_flt_column_percent[4], \
                                                                                                      list_flt_column_percent[5], \
                                                                                                      list_flt_column_percent[6], \
                                                                                                      list_flt_column_percent[7], \
                                                                                                      list_flt_column_percent[8], \
                                                                                                      list_flt_column_percent[9], \
                                                                                                      list_flt_column_percent[10] ]
    return df_output        

str_dataset = "bigquery-public-data.nhtsa_traffic_fatalities"

def select_from_dataset_table(str_dataset, str_table):
    return str_dataset + "." + str_table  

def create_simple_query1(str_columns, str_table):
    str_select_from_dataset_table = select_from_dataset_table(str_dataset, str_table)
    query = "SELECT " + str_columns + \
        """ \nFROM `""" + str_select_from_dataset_table + """` """
    return query
   
#def create_simple_query_group1(str_column, str_table):
#    str_select_from_dataset_table = select_from_dataset_table(str_dataset, str_table)
#    query = "SELECT " + str_columns + \
#        """ \nFROM `""" + str_select_from_dataset_table + """` """
#    return query

start_time_check()
str_select_columns = "consecutive_number, state_name, day_of_crash, month_of_crash, year_of_crash, day_of_week, hour_of_crash, number_of_fatalities, \
                      land_use_name, light_condition_name, number_of_drunk_drivers"
str_select_from_table = "accident_2015"
query = create_simple_query1(str_select_columns, str_select_from_table)
ds_current.estimate_query_size(query)

df_temp1 = ds_current.query_to_pandas_safe(query, max_gb_scanned=0.1)
df_temp1['kaggleid'] = 15000000 + df_temp1['consecutive_number']
df_temp1.sample(10)
str_select_from_table = "accident_2016"
query = create_simple_query1(str_select_columns, str_select_from_table)
ds_current.estimate_query_size(query)

df_temp2 = ds_current.query_to_pandas_safe(query, max_gb_scanned=0.1)
df_temp2['kaggleid'] = 16000000 + df_temp2['consecutive_number']
df_temp2.sample(10)
df_current = pd.concat([df_temp1, df_temp2], sort=False)
if bln_recode_variables:
    df_current['overall'] = 'yes'

    def day_of_week_s1d (row): 
        for i in range(1,8):
            if row['day_of_week'] == i:
                return get_translations_analysis_description(translations_analysis, str_language, 'day of week', i)
        return 'Unknown'
    df_current['day_of_week_s1d'] = df_current.apply(day_of_week_s1d, axis=1)

    def day_of_week_s2d (row): 
        if row['day_of_week'] >= 2 and row['day_of_week'] <= 6:
                return get_translations_analysis_description(translations_analysis, str_language, 'weekday summary', 1)
        if row['day_of_week'] == 1 or row['day_of_week'] == 7:
                return get_translations_analysis_description(translations_analysis, str_language, 'weekday summary', 2)
        return 'Unknown'
    df_current['day_of_week_s2d'] = df_current.apply(day_of_week_s2d, axis=1)

    def year_of_crash_s1d (row): 
        if row['year_of_crash'] == 2015:
                return '2015'
        if row['year_of_crash'] == 2016:
                return '2016'
        return 'Unknown'
    df_current['year_of_crash_s1d'] = df_current.apply(year_of_crash_s1d, axis=1)
    
    def month_of_crash_s1d (row): 
        for i in range(1, 13):
            if row['month_of_crash'] == i:
                return get_translations_analysis_description(translations_analysis, str_language, 'month', i)
        return 'Unknown'
    df_current['month_of_crash_s1d'] = df_current.apply(month_of_crash_s1d, axis=1)
    
    def hour_of_crash_s1d (row): 
        if row['hour_of_crash'] <= 5 :
                return '00:00 to 05:59'
        if row['hour_of_crash'] <= 11 :
                return '06:00 to 11:59'
        if row['hour_of_crash'] <= 17 :
                return '12:00 to 17:59'
        if row['hour_of_crash'] <= 23 :
                return '18:00 to 23:59'
        return 'Unknown'
    df_current['hour_of_crash_s1d'] = df_current.apply(hour_of_crash_s1d, axis=1)
 
    def number_of_fatalities_s1d (row):
        if row['number_of_fatalities'] == 1:
                return '1'
        if row['number_of_fatalities'] >= 2:
                return '2 or more'
        return 'Unknown'
    df_current['number_of_fatalities_s1d'] = df_current.apply(number_of_fatalities_s1d, axis=1)

    def number_of_drunk_drivers_s1d (row): 
        if row['number_of_drunk_drivers'] == 0:
                return 'None'
        if row['number_of_drunk_drivers'] >= 1:
                return '1 or more'
        return 'Unknown'
    df_current['number_of_drunk_drivers_s1d'] = df_current.apply(number_of_drunk_drivers_s1d, axis=1)

    def us_division_s1d (row):
        str_state_name = row['state_name']
        if str_state_name == 'Connecticut' or str_state_name == 'Maine' or str_state_name == 'Massachusetts' or \
               str_state_name == 'New Hampshire' or str_state_name == 'Rhode Island' or str_state_name == 'Vermont' :
            return 'New England'
        if str_state_name == 'New Jersey' or str_state_name == 'New York' or str_state_name == 'Pennsylvania' :
            return 'Mid Atlantic'
        if str_state_name == 'Illinois' or str_state_name == 'Indiana' or str_state_name == 'Michigan' or \
               str_state_name == 'Ohio' or str_state_name == 'Wisconsin' :
            return 'East North Central'
        if str_state_name == 'Iowa' or str_state_name == 'Kansas' or str_state_name == 'Minnesota' or \
               str_state_name == 'Missouri' or str_state_name == 'Nebraska' or str_state_name == 'North Dakota' or \
               str_state_name == 'South Dakota' :
            return 'West North Central'
        if str_state_name == 'Delaware' or str_state_name == 'Florida' or str_state_name == 'Georgia' or \
               str_state_name == 'Maryland' or str_state_name == 'North Carolina' or str_state_name == 'South Carolina' or \
               str_state_name == 'Virginia' or str_state_name == 'District of Columbia' or str_state_name == 'West Virginia' :
            return 'South Atlantic'
        if str_state_name == 'Alabama' or str_state_name == 'Kentucky' or str_state_name == 'Mississippi' or \
               str_state_name == 'Tennessee' :
            return 'East South Central'
        if str_state_name == 'Arkansas' or str_state_name == 'Louisiana' or str_state_name == 'Oklahoma' or \
               str_state_name == 'Texas' :
            return 'West South Central'
        if str_state_name == 'Arizona' or str_state_name == 'Colorado' or str_state_name == 'Idaho' or \
               str_state_name == 'Montana' or str_state_name == 'Nevada' or str_state_name == 'New Mexico' or \
               str_state_name == 'Utah' or str_state_name == 'Wyoming' :
            return 'Mountain'
        if str_state_name == 'Alaska' or str_state_name == 'California' or str_state_name == 'Hawaii' or \
               str_state_name == 'Oregon' or str_state_name == 'Washington' :
            return 'Pacific'
        return 'Other'
    df_current['us_division_s1d'] = df_current.apply(us_division_s1d, axis=1)
    
    def us_region_s1d (row):
        if row['us_division_s1d'] == 'New England' or row['us_division_s1d'] == 'Mid Atlantic' :
            return 'Northeast'
        if row['us_division_s1d'] == 'East North Central' or row['us_division_s1d'] == 'West North Central' :
            return 'Midwest'
        if row['us_division_s1d'] == 'South Atlantic' or row['us_division_s1d'] == 'East South Central' or \
            row['us_division_s1d'] == 'West South Central' :
            return 'South'
        if row['us_division_s1d'] == 'Mountain' or row['us_division_s1d'] == 'Pacific' :
            return 'West'
        return 'Other'
    df_current['us_region_s1d'] = df_current.apply(us_region_s1d, axis=1)
    
    df_temp1 = df_current[['kaggleid', 'consecutive_number', 'overall', 'day_of_week_s1d', 'day_of_week_s2d', 'year_of_crash_s1d', \
                           'month_of_crash_s1d', 'hour_of_crash_s1d', 'number_of_fatalities_s1d', 'number_of_drunk_drivers_s1d', \
                           'us_division_s1d', 'us_region_s1d' ]]
    str_filename = 'dd5_input_recodes.csv'
    df_temp1.to_csv(str_filename, index = False)
else:
    df_temp2 = pd.read_csv('../input/dd5-translations-analysis/dd5_input_recodes.csv', nrows=int_read_csv_rows, \
                            dtype={'year_of_crash_s1d': object})    # bug in kernel
    df_current = pd.merge(df_current, df_temp2, how='left', on=['kaggleid'])
#    df_current.drop('year_of_crash_s1d', axis=1, inplace=True)
#    df_current['year_of_crash_s1d'] = df_current['year_of_crash'].astype(str)
# new recodes
create_crosstab_type1(df_current, 'overall', int_important_crosstab)
create_crosstab_type1(df_current, 'year_of_crash_s1d', int_important_crosstab)
create_crosstab_type1(df_current, 'day_of_week_s2d', int_important_crosstab)
create_crosstab_type1(df_current, 'hour_of_crash_s1d', int_important_crosstab)
create_crosstab_type1(df_current, 'number_of_drunk_drivers_s1d', int_important_crosstab)
create_crosstab_type1(df_current, 'day_of_week_s1d', int_past_crosstab)
create_crosstab_type1(df_current, 'hour_of_crash', int_past_crosstab)
create_crosstab_type1(df_current, 'month_of_crash_s1d', int_past_crosstab)
create_crosstab_type1(df_current, 'number_of_fatalities_s1d', int_past_crosstab)
create_crosstab_type1(df_current, 'number_of_fatalities', int_past_crosstab)
create_crosstab_type1(df_current, 'land_use_name', int_past_crosstab)
create_crosstab_type1(df_current, 'light_condition_name', int_past_crosstab)
create_crosstab_type1(df_current, 'number_of_drunk_drivers', int_past_crosstab)
create_crosstab_type1(df_current, 'us_region_s1d', int_current_crosstab)
create_crosstab_type1(df_current, 'us_division_s1d', int_current_crosstab)
create_crosstab_type1(df_current, 'state_name', int_current_crosstab)
create_crosstab_type1(df_current, 'day_of_crash', int_current_crosstab)
#create_crosstab_type1(df_current, '', int_current_crosstab)
end_time_check(dat_start, 'accidents')
end_time_check(dat_program_start, 'overall')
df_time_check
#df_current.info()
#df_current.head(20)
#create_crosstab_type1(df_current, 'state_name', int_current_crosstab)
#create_topline(df_current, '', strg_count_column)
