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
str_language = "es" # en = English; es = Spanish

def get_translations_analysis_description(df_input, str_language, str_group, int_code):
    # created by darryldias 25may2018
    df_temp = df_input[(df_input['language']==str_language) & (df_input['group']==str_group) & (df_input['code']==int_code)] \
                    ['description']
    return df_temp.iloc[0]

translations_analysis = pd.read_csv('../input/ulabox-translations-analysis/translations_analysis.csv')
strg_count_column = get_translations_analysis_description(translations_analysis, str_language, 'special', 2)

def start_time_check(str_stage_i):
    # created by darryldias 21may2018
    global intTimeCheck
    global strStage 
    global datStart 
    intTimeCheck += 1
    strStage = str_stage_i
    datStart = datetime.datetime.now()
    
def end_time_check():
    # created by darryldias 21may2018
    global intTimeCheck
    global strStage
    global datStart
    global dfTimeCheck
    datEnd = datetime.datetime.now()
    diffSeconds = (datEnd-datStart).total_seconds()
    diffMinutes = diffSeconds / 60.0
    dfTimeCheck.loc[intTimeCheck] = [strStage, datStart, datEnd, diffSeconds, diffMinutes]
def create_topline(df_input, str_item_column, str_count_column):
    # created by darryldias 21may2018; updated by darryldias 29may2018
    str_percent_column = get_translations_analysis_description(translations_analysis, str_language, 'special', 3)
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
    
    df_output.loc[int_rows] = ['Total', int_total, 100.0]
    return df_output        

def get_dataframe_info(df_input):
    # created by darryldias 24may2018
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
        flt_percent_null = 100 - flt_percent_not_null
        int_table_row += 1
        df_output.loc[int_table_row] = [str_column_name, str_column_type, int_not_null, int_null, flt_percent_not_null, flt_percent_null]
    
    return df_output

def create_crosstab_type1(df_input, str_row_question):
    # created by darryldias 10jun2018 - updated 21jun2018 
    # got some useful code from:
    # https://chrisalbon.com/python/data_wrangling/pandas_missing_data/
    # https://www.tutorialspoint.com/python/python_lists.htm
    # https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points

    str_total_desc = get_translations_analysis_description(translations_analysis, str_language, 'special', 1)
    str_count_desc = get_translations_analysis_description(translations_analysis, str_language, 'special', 2)
    str_colpercent_desc = get_translations_analysis_description(translations_analysis, str_language, 'special', 4)
    str_rowquestion_desc = get_translations_analysis_description(translations_analysis, str_language, 'special', 5)
    str_rowcategory_desc = get_translations_analysis_description(translations_analysis, str_language, 'special', 6)
    str_statistic_desc = get_translations_analysis_description(translations_analysis, str_language, 'special', 7)
    
    int_columns = 10
    #list_str_column_desc = ['overall', '01-20 items', '21-30 items', '31-40 items', '41+ items', 'weekday', 'weekend', 'morning', 'afternoon', 'evening']
    list_str_column_desc = []
    for j in range(int_columns):
        list_str_column_desc.append( get_translations_analysis_description(translations_analysis, str_language, 'analysis 1', j+1 ) )

    list_str_column_question = ['overall', 'total items summary 2', 'total items summary 2', 'total items summary 2', 'total items summary 2', \
                                'weekday summary', 'weekday summary', 'hour summary 1', 'hour summary 1', 'hour summary 1']

    #list_str_column_category = ['yes', '01-20', '21-30', '31-40', '41+', 'weekday', 'weekend', 'morning', 'afternoon', 'evening']
    list_str_column_category = [] # update for next analysis
    list_str_column_category.append( get_translations_analysis_description(translations_analysis, str_language, 'yes no', 1) )
    list_str_column_category.append( '01-20' )
    list_str_column_category.append( '21-30' )
    list_str_column_category.append( '31-40' )
    list_str_column_category.append( '41+' )
    list_str_column_category.append( get_translations_analysis_description(translations_analysis, str_language, 'weekday summary', 1) )
    list_str_column_category.append( get_translations_analysis_description(translations_analysis, str_language, 'weekday summary', 2) )
    list_str_column_category.append( get_translations_analysis_description(translations_analysis, str_language, 'hour summary 1', 1) )
    list_str_column_category.append( get_translations_analysis_description(translations_analysis, str_language, 'hour summary 1', 2) )
    list_str_column_category.append( get_translations_analysis_description(translations_analysis, str_language, 'hour summary 1', 3) )
    
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
        
    
    df_output = pd.DataFrame(columns=[str_rowquestion_desc, str_rowcategory_desc, str_statistic_desc, 'c1', 'c2', 'c3', 'c4', 'c5', \
                                      'c6', 'c7', 'c8', 'c9', 'c10'])
    int_row = 1
    df_output.loc[int_row] = [str_row_question, '', '', list_str_column_desc[0], list_str_column_desc[1], list_str_column_desc[2], \
                                 list_str_column_desc[3], list_str_column_desc[4], list_str_column_desc[5], list_str_column_desc[6], \
                                 list_str_column_desc[7], list_str_column_desc[8], list_str_column_desc[9] ]
    int_row = 2
    df_output.loc[int_row] = [str_row_question, str_total_desc, str_count_desc, list_int_column_base[0], list_int_column_base[1], list_int_column_base[2], \
                                 list_int_column_base[3], list_int_column_base[4], list_int_column_base[5], list_int_column_base[6], \
                                 list_int_column_base[7], list_int_column_base[8], list_int_column_base[9] ] 
    int_row = 3
    df_output.loc[int_row] = [str_row_question, str_total_desc, str_colpercent_desc, list_flt_column_base_percent[0], list_flt_column_base_percent[1], \
                                list_flt_column_base_percent[2], list_flt_column_base_percent[3], list_flt_column_base_percent[4], \
                                list_flt_column_base_percent[5], list_flt_column_base_percent[6], list_flt_column_base_percent[7], \
                                list_flt_column_base_percent[8], list_flt_column_base_percent[9] ] 

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
                                        list_int_column_count[2], list_int_column_count[3], list_int_column_count[4], list_int_column_count[5], \
                                        list_int_column_count[6], list_int_column_count[7], list_int_column_count[8], list_int_column_count[9] ]
        df_output.loc[int_colpercent_row] = [str_row_question, str_row_category, str_colpercent_desc, list_flt_column_percent[0], \
                                             list_flt_column_percent[1], list_flt_column_percent[2], list_flt_column_percent[3], \
                                             list_flt_column_percent[4], list_flt_column_percent[5], list_flt_column_percent[6], \
                                             list_flt_column_percent[7], list_flt_column_percent[8], list_flt_column_percent[9] ]
    return df_output 
    
def percent_summary_1 (row, str_input_column):
    # created by darryldias 27may2018   
    if row[str_input_column] == 0 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'yes no', 2)
    if row[str_input_column] > 0 :
        return get_translations_analysis_description(translations_analysis, str_language, 'yes no', 1)
    return 'Unknown'

def month_description (row, str_input_column):
    # created by darryldias 1june2018   
    if row[str_input_column] == 1 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 1)
    if row[str_input_column] == 2 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 2)
    if row[str_input_column] == 3 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 3)
    if row[str_input_column] == 4 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 4)
    if row[str_input_column] == 5 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 5)
    if row[str_input_column] == 6 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 6)
    if row[str_input_column] == 7 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 7)
    if row[str_input_column] == 8 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 8)
    if row[str_input_column] == 9 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 9)
    if row[str_input_column] == 10 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 10)
    if row[str_input_column] == 11 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 11)
    if row[str_input_column] == 12 :   
        return get_translations_analysis_description(translations_analysis, str_language, 'month', 12)
    return 'Unknown'

def year_month_code (row, str_input_column_year, str_input_column_month):
    # created by darryldias 1june2018   
    if row[str_input_column_month] <= 9 :   
        return int(str(row[str_input_column_year]) + '0' + str(row[str_input_column_month]))
    if row[str_input_column_month] <= 12 :   
        return int(str(row[str_input_column_year]) + str(row[str_input_column_month]))
    return 0

ulabox_orders = pd.read_csv('../input/ulabox-orders-with-categories-partials-2017/ulabox_orders_with_categories_partials_2017.csv')
ulabox_orders.head(10)
get_dataframe_info(ulabox_orders)
ulabox_orders.describe()
ulabox_orders['overall'] = get_translations_analysis_description(translations_analysis, str_language, 'yes no', 1)

def total_items_summary_1 (row):
    if row['total_items'] <= 10 :
        return '01-10'
    if row['total_items'] <= 20 :
        return '11-20'
    if row['total_items'] <= 30 :
        return '21-30'
    if row['total_items'] <= 40 :
        return '31-40'
    if row['total_items'] <= 50 :
        return '41-50'
    if row['total_items'] > 50 :
        return '51+'
    return 'Unknown'

def total_items_summary_2 (row):
    if row['total_items'] <= 20 :
        return '01-20'
    if row['total_items'] <= 30 :
        return '21-30'
    if row['total_items'] <= 40 :
        return '31-40'
    if row['total_items'] > 40 :
        return '41+'
    return 'Unknown'

ulabox_orders['total items summary 1'] = ulabox_orders.apply(total_items_summary_1, axis=1)
ulabox_orders['total items summary 2'] = ulabox_orders.apply(total_items_summary_2, axis=1)

def weekday_desc (row):
    for i in range(1,8):
        if row['weekday'] == i :
            return get_translations_analysis_description(translations_analysis, str_language, 'day of week', i)
    return 'Unknown'
ulabox_orders['weekday desc'] = ulabox_orders.apply(weekday_desc, axis=1)

def weekday_summary (row):   
    if row['weekday'] <= 5 :
        return get_translations_analysis_description(translations_analysis, str_language, 'weekday summary', 1)
    if row['weekday'] <= 7 :
        return get_translations_analysis_description(translations_analysis, str_language, 'weekday summary', 2)
    return 'Unknown'
ulabox_orders['weekday summary'] = ulabox_orders.apply(weekday_summary, axis=1)

def hour_summary_1 (row):
    if row['hour'] <= 11 :
        return get_translations_analysis_description(translations_analysis, str_language, 'hour summary 1', 1)
    if row['hour'] <= 17 :
        return get_translations_analysis_description(translations_analysis, str_language, 'hour summary 1', 2)
    if row['hour'] <= 23 :
        return get_translations_analysis_description(translations_analysis, str_language, 'hour summary 1', 3)
    return 'Unknown'

def hour_summary_2 (row):
    if row['hour'] <= 8 :
        return get_translations_analysis_description(translations_analysis, str_language, 'hour summary 2', 1)
    if row['hour'] <= 11 :
        return get_translations_analysis_description(translations_analysis, str_language, 'hour summary 2', 2)
    if row['hour'] <= 14 :
        return get_translations_analysis_description(translations_analysis, str_language, 'hour summary 2', 3)
    if row['hour'] <= 17 :
        return get_translations_analysis_description(translations_analysis, str_language, 'hour summary 2', 4)
    if row['hour'] <= 20 :
        return get_translations_analysis_description(translations_analysis, str_language, 'hour summary 2', 5)
    if row['hour'] <= 23 :
        return get_translations_analysis_description(translations_analysis, str_language, 'hour summary 2', 6)
    return 'Unknown'

ulabox_orders['hour summary 1'] = ulabox_orders.apply(hour_summary_1, axis=1)
ulabox_orders['hour summary 2'] = ulabox_orders.apply(hour_summary_2, axis=1)

create_crosstab_type1(ulabox_orders, 'overall')
create_crosstab_type1(ulabox_orders, 'weekday desc')
create_crosstab_type1(ulabox_orders, 'weekday summary')
create_crosstab_type1(ulabox_orders, 'hour summary 1')
create_crosstab_type1(ulabox_orders, 'hour summary 2')
create_crosstab_type1(ulabox_orders, 'total items summary 1')
create_crosstab_type1(ulabox_orders, 'total items summary 2')
ulabox_orders['ordered food'] = ulabox_orders.apply(percent_summary_1, axis=1, str_input_column='Food%')
create_crosstab_type1(ulabox_orders, 'ordered food')
ulabox_orders['ordered fresh'] = ulabox_orders.apply(percent_summary_1, axis=1, str_input_column='Fresh%')
create_crosstab_type1(ulabox_orders, 'ordered fresh')
ulabox_orders['ordered drinks'] = ulabox_orders.apply(percent_summary_1, axis=1, str_input_column='Drinks%')
create_crosstab_type1(ulabox_orders, 'ordered drinks')
ulabox_orders['ordered home'] = ulabox_orders.apply(percent_summary_1, axis=1, str_input_column='Home%')
create_crosstab_type1(ulabox_orders, 'ordered home')
ulabox_orders['ordered beauty'] = ulabox_orders.apply(percent_summary_1, axis=1, str_input_column='Beauty%')
create_crosstab_type1(ulabox_orders, 'ordered beauty')
ulabox_orders['ordered health'] = ulabox_orders.apply(percent_summary_1, axis=1, str_input_column='Health%')
create_crosstab_type1(ulabox_orders, 'ordered health')
ulabox_orders['ordered baby'] = ulabox_orders.apply(percent_summary_1, axis=1, str_input_column='Baby%')
create_crosstab_type1(ulabox_orders, 'ordered baby')
ulabox_orders['ordered pets'] = ulabox_orders.apply(percent_summary_1, axis=1, str_input_column='Pets%')
create_crosstab_type1(ulabox_orders, 'ordered pets')
def discount_summary (row):      
    if row['discount%'] < 0 :
        return get_translations_analysis_description(translations_analysis, str_language, 'discount summary', 1)
    if row['discount%'] == 0 :
        return get_translations_analysis_description(translations_analysis, str_language, 'discount summary', 2)
    if row['discount%'] <= 2.5 :
        return get_translations_analysis_description(translations_analysis, str_language, 'discount summary', 3)
    if row['discount%'] <= 5.0 :
        return get_translations_analysis_description(translations_analysis, str_language, 'discount summary', 4)
    if row['discount%'] <= 10.0 :
        return get_translations_analysis_description(translations_analysis, str_language, 'discount summary', 5)
    if row['discount%'] <= 20.0 :
        return get_translations_analysis_description(translations_analysis, str_language, 'discount summary', 6)
    if row['discount%'] <= 50.0 :
        return get_translations_analysis_description(translations_analysis, str_language, 'discount summary', 7)
    if row['discount%'] <= 99.99 :
        return get_translations_analysis_description(translations_analysis, str_language, 'discount summary', 8)
    if row['discount%'] == 100.00 :
        return get_translations_analysis_description(translations_analysis, str_language, 'discount summary', 9)
    return 'Unknown'

ulabox_orders['discount summary'] = ulabox_orders.apply(discount_summary, axis=1)
create_crosstab_type1(ulabox_orders, 'discount summary')
ulabox_orders.sample(10)
grouped = ulabox_orders.groupby('customer')
ulabox_customers = grouped['order'].count().reset_index(name='order count')  
df_grouped = grouped['total_items'].sum().reset_index(name='total_items sum')   
ulabox_customers = pd.merge(ulabox_customers, df_grouped, how='left', on=['customer'])
df_grouped = grouped['total_items'].mean().reset_index(name='total_items mean')   
ulabox_customers = pd.merge(ulabox_customers, df_grouped, how='left', on=['customer'])
df_grouped = grouped['weekday'].min().reset_index(name='weekday minimum')   
ulabox_customers = pd.merge(ulabox_customers, df_grouped, how='left', on=['customer'])
df_grouped = grouped['weekday'].max().reset_index(name='weekday maximum')   
ulabox_customers = pd.merge(ulabox_customers, df_grouped, how='left', on=['customer'])

get_dataframe_info(ulabox_customers)
ulabox_customers.describe()
def order_count_summary_1 (row):   # ***** FIX *****
    if row['order count'] == 1 :
        return 'ordenó una vez'   # ordered once
    if row['order count'] > 1 :
        return 'ordenado varias veces'   # ordered several times
    return 'Unknown'

ulabox_customers['order count summary 1'] = ulabox_customers.apply(order_count_summary_1, axis=1)
create_topline(ulabox_customers, 'order count summary 1', strg_count_column) 
def order_count_summary_2 (row):   # ***** FIX *****
    if row['order count'] == 1 :
        return '1 orden'   # order/s
    if row['order count'] == 2 :
        return '2 pedidos'
    if row['order count'] == 3 :
        return '3 pedidos'
    if row['order count'] == 4 :
        return '4 pedidos'
    if row['order count'] == 5 :
        return '5 pedidos'
    if row['order count'] > 5 :
        return '6+ pedidos'
    return 'Unknown'

ulabox_customers['order count summary 2'] = ulabox_customers.apply(order_count_summary_2, axis=1)
create_topline(ulabox_customers, 'order count summary 2', strg_count_column) 
def total_items_mean_summary_1 (row):   # ***** FIX *****
    if row['total_items mean'] <= 10 :
        return '>0 y <=10'   # and
    if row['total_items mean'] <= 20 :
        return '>10 y <=20'   
    if row['total_items mean'] <= 30 :
        return '>20 y <=30'   
    if row['total_items mean'] <= 40 :
        return '>30 y <=40'   
    if row['total_items mean'] <= 50 :
        return '>40 y <=50'   
    if row['total_items mean'] > 50 :
        return '>50'   
    return 'Unknown'

ulabox_customers['total items mean summary 1'] = ulabox_customers.apply(total_items_mean_summary_1, axis=1)
create_topline(ulabox_customers, 'total items mean summary 1', strg_count_column) 
def order_weekday_summary_1 (row):   
    if row['weekday minimum'] <= 5 :
        return get_translations_analysis_description(translations_analysis, str_language, 'yes no', 1)   
    if row['weekday minimum'] >= 6 :
        return get_translations_analysis_description(translations_analysis, str_language, 'yes no', 2)      
    return 'Unknown'

ulabox_customers['order weekday summary 1'] = ulabox_customers.apply(order_weekday_summary_1, axis=1)
create_topline(ulabox_customers, 'order weekday summary 1', strg_count_column) 
def order_weekend_summary_1 (row):   
    if row['weekday maximum'] >= 6 :
        return get_translations_analysis_description(translations_analysis, str_language, 'yes no', 1)   
    if row['weekday maximum'] <= 5 :
        return get_translations_analysis_description(translations_analysis, str_language, 'yes no', 2)   
    return 'Unknown'

ulabox_customers['order weekend summary 1'] = ulabox_customers.apply(order_weekend_summary_1, axis=1)
create_topline(ulabox_customers, 'order weekend summary 1', strg_count_column) 
ulabox_customers.sample(15)
# set variable below equal to real order amount column
ulabox_orders['order amount euro'] = (ulabox_orders['total_items']*3.0) * ( 1.0 - (ulabox_orders['discount%']/100.0)  )

def order_amount_euro_summary_1 (row):   # ***** FIX *****
    if row['order amount euro'] <= 25.0 :
        return '000.00 a 025.00'   
    if row['order amount euro'] <= 50.0 :
        return '025.01 a 050.00'   
    if row['order amount euro'] <= 75.0 :
        return '050.01 a 075.00'   
    if row['order amount euro'] <= 100.0 :
        return '075.01 a 100.00'   
    if row['order amount euro'] <= 125.0 :
        return '100.01 a 125.00'   
    if row['order amount euro'] <= 150.0 :
        return '125.01 a 150.00'   
    if row['order amount euro'] <= 200.0 :
        return '150.01 a 200.00'   
    if row['order amount euro'] > 200.0 :
        return '200.01+'   
    return 'Unknown'

ulabox_orders['order amount euro summary 1'] = ulabox_orders.apply(order_amount_euro_summary_1, axis=1)
create_topline(ulabox_orders, 'order amount euro summary 1', strg_count_column) 
ulabox_virtual_data_orders_extra = pd.read_csv('../input/ulabox-virtual-data-orders-extra/virtual_data_orders_extra.csv')
ulabox_orders = pd.merge(ulabox_orders, ulabox_virtual_data_orders_extra, how='left', on=['order'])
def order_region_description (row):   
    if row['order region code'] == 1 :
        return get_translations_analysis_description(translations_analysis, str_language, 'region', 1)   
    if row['order region code'] == 2 :
        return get_translations_analysis_description(translations_analysis, str_language, 'region', 2)   
    if row['order region code'] == 3 :
        return get_translations_analysis_description(translations_analysis, str_language, 'region', 3)   
    return 'Unknown'

ulabox_orders['order region description'] = ulabox_orders.apply(order_region_description, axis=1)
create_topline(ulabox_orders, 'order region description', strg_count_column) 
def order_delivered_late (row):   
    if row['order delivered late'] == 1 :
        return get_translations_analysis_description(translations_analysis, str_language, 'yes no', 1)   
    if row['order delivered late'] == 2 :
        return get_translations_analysis_description(translations_analysis, str_language, 'yes no', 2)      
    return 'Unknown'

ulabox_orders['order delivered late description'] = ulabox_orders.apply(order_delivered_late, axis=1)
create_topline(ulabox_orders, 'order delivered late description', strg_count_column) 
def order_date (row):  
    if row['order'] <= 1654 :
        return datetime(2017, 10, 20)   
    if row['order'] <= 3527 :
        return datetime(2017, 11, 20)   
    if row['order'] <= 5671 :
        return datetime(2017, 12, 20)   
    if row['order'] <= 8003 :
        return datetime(2018, 1, 20)   
    if row['order'] <= 10257 :
        return datetime(2018, 2, 20)   
    if row['order'] <= 12564 :
        return datetime(2018, 3, 20)   
    if row['order'] <= 15042 :
        return datetime(2018, 4, 20)   
    if row['order'] <= 17771 :
        return datetime(2018, 5, 20)   
    if row['order'] <= 20597 :
        return datetime(2018, 6, 20)   
    if row['order'] <= 23618 :
        return datetime(2018, 7, 20)   
    if row['order'] <= 26759 :
        return datetime(2018, 8, 20)   
    if row['order'] <= 30000 :
        return datetime(2018, 9, 20)   
    return 'Unknown'

ulabox_orders['order date'] = ulabox_orders.apply(order_date, axis=1)
ulabox_orders['order year'] = ulabox_orders['order date'].dt.year 
ulabox_orders['order month code'] = ulabox_orders['order date'].dt.month 
ulabox_orders['order month description'] = ulabox_orders.apply(month_description, axis=1, str_input_column='order month code')
ulabox_orders['order year month code'] = ulabox_orders.apply(year_month_code, axis=1, str_input_column_year='order year', \
                                                             str_input_column_month='order month code')
ulabox_orders['order year month description'] = ulabox_orders['order year'].apply(str) + ' ' + ulabox_orders['order month description']
create_topline(ulabox_orders, 'order year', strg_count_column) 
create_topline(ulabox_orders, 'order year month description', strg_count_column) 
ulabox_orders.sample(15)
#create_topline(ulabox_orders, 'hour', 'Count')
#get_dataframe_info(translations_analysis)
#translations_analysis.head(15)
#ulabox_orders[ ulabox_orders['discount%']>0 ].sample(30)
#ulabox_orders[ ulabox_orders['customer']==4549 ].head(100)
#ulabox_orders['total_items'].sum()
