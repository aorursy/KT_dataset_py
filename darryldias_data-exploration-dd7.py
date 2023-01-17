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
df_time_check = pd.DataFrame(columns=['Stage','Start','End', 'Seconds', 'Minutes'])
int_time_check = 0
dat_start = datetime.now()
dat_program_start = dat_start

import csv
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
#google_analytics = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
#                                   dataset_name="data:google_analytics_sample")
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
#bq_assistant.head("ga_sessions_20160801", num_rows=3)
#bq_assistant.table_schema("ga_sessions_20160801")
#response1 = google_analytics.query_to_pandas_safe(query1)
#response1.head(10)
ds_current = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "google_analytics_sample")
bln_run_stage1 = False
bln_run_stage2a = False
bln_run_stage2b = False
bln_run_stage3a = False
bln_run_stage3b = False
bln_testing = False

int_start_date = 20160801
if bln_testing:
    int_end_date = 20160802
else:
    int_end_date = 20170731
    

str_dataset = "bigquery-public-data.google_analytics_sample"
flt_query_limit = 0.1
flt_est_query_size_total = 0.0
int_query_count = 0

int_sample_records = 15

csv_query_info = open('dd7_query_info.csv', 'w')
query_writer = csv.writer(csv_query_info)
query_writer.writerow( ['query', 'size1_gb', 'size2_gb'] )

def select_from_dataset_table(str_dataset, str_table):
    return str_dataset + "." + str_table  

def create_simple_query1(str_columns, str_table):
    str_select_from_dataset_table = select_from_dataset_table(str_dataset, str_table)
    query = "SELECT " + str_columns + \
        """ \nFROM `""" + str_select_from_dataset_table + """` """
    return query

def create_simple_query2(str_select_columns, str_table, str_group_by_columns):
    str_select_from_dataset_table = select_from_dataset_table(str_dataset, str_table)
    query = "SELECT " + str_select_columns + \
            "\nFROM `" + str_select_from_dataset_table + "` " + \
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

if bln_run_stage1:
    str_filename = 'dd7_table_list.csv'
    start_time_check()
    list_tables = bq_assistant.list_tables()

    csvfile1 = open(str_filename, 'w')
    writer1 = csv.writer(csvfile1)
    writer1.writerow( ['id', 'name', 'records', 'est_qry_size'] )

    str_select_columns = "COUNT(*)"

    int_counter = 0
    for table in list_tables:
        int_counter += 1

        str_select_from_table = table
        query = create_simple_query1(str_select_columns, str_select_from_table)
        flt_est_query_size = ds_current.estimate_query_size(query)
        df_query = ds_current.query_to_pandas_safe(query, max_gb_scanned=flt_query_limit)
        int_table_count = df_query.iloc[0][0]

        writer1.writerow( [int_counter, table, int_table_count, flt_est_query_size] )

    csvfile1.close()
    print('Number of tables in dataset: ' + str(len(list_tables)) )
    end_time_check(dat_start, 'list tables')

if bln_run_stage2a:
    csv_file_read = open('../input/dd7_table_list.csv', 'r')
    csv_reader = csv.reader(csv_file_read, delimiter=',')

    print('sample queries run are shown below:')
    int_row = 0
    str_select_columns = "date, count(*) as count"
    str_group_by_columns = "date"
    for row in csv_reader:
        if int_row > 0:
            int_id = row[0]
            str_table = row[1]
            int_table_date = int(str_table[-8:])
            if int_table_date >= 20160801 and int_table_date <= 20170731:
                str_select_from_table = str_table
                query = create_simple_query2(str_select_columns, str_select_from_table, str_group_by_columns)
                flt_est1_query_size = ds_current.estimate_query_size(query)
                flt_est2_query_size = get_query_size(flt_est1_query_size) 
                flt_est_query_size_total += flt_est2_query_size
                int_query_count += 1
                if int_query_count % 20 == 0:
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
    df_query.to_csv('dd7_stage2a.csv', index=False)

    csv_file_read.close()
    print('The number of lines read in file: ', int_row)
    print('The number of queries run: ', int_query_count)
    print('Total estimated size of queries (mb):', get_str_query_size_mb(flt_est_query_size_total) )

if bln_run_stage2b:
    str_input_file = '../input/dd7_stage2a.csv'
    int_rows = get_csv_row_count(str_input_file)
    print('The number of lines read in input csv file: ', int_rows)    
    print('\nSample records read including one or more constructed variables are shown below')
    
    int_record_group = int( int_rows / int_sample_records)
    
    csv_file_read = open(str_input_file, 'r')
    csv_reader = csv.reader(csv_file_read, delimiter=',')
    int_row = 0
    dict_output = {}
    for row in csv_reader:
        int_row += 1
        if int_row > 1:
            str_date = row[0]
            int_count = int(row[1])
            str_year_month = str_date[:6]
            dict_output[str_year_month] = dict_get_count(dict_output, str_year_month) + int_count
            if int_row == 2 or int_row == int_rows or (int_row % int_record_group == 0):
                print (str_date, int_count, str_year_month)

    csv_file_read.close()
        
    str_output_file = 'dd7_stage2b.csv'
    list_headings = ['yearmonth', 'count']
    create_csv_from_dict(dict_output, str_output_file, list_headings)
if bln_run_stage3a:
    str_output_csv = 'dd7_stage3a.csv'
    str_select_columns = "geoNetwork.country as country, count(*) as count"
    str_group_by_columns = "country"
    
    csv_file_read = open('../input/dd7_table_list.csv', 'r')
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
                query = create_simple_query2(str_select_columns, str_select_from_table, str_group_by_columns)
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

if bln_run_stage3b:
    str_input_file = '../input/dd7_stage3a.csv'
    str_output_file = 'dd7_stage3b.csv'

    int_rows = get_csv_row_count(str_input_file)
    print('The number of lines read in input csv file:', int_rows)    
    print('\nSample records read including one or more constructed variables if applicable are shown below')
    
    int_record_group = int( int_rows / int_sample_records)
    
    csv_file_read = open(str_input_file, 'r')
    csv_reader = csv.reader(csv_file_read, delimiter=',')
    int_row = 0
    dict_output = {}
    for row in csv_reader:
        int_row += 1
        if int_row > 1:
            str_country = row[0]
            int_count = int(row[1])
            dict_output[str_country] = dict_get_count(dict_output, str_country) + int_count
            if int_row == 2 or int_row == int_rows or (int_row % int_record_group == 0):
                print (str_country, int_count)

    csv_file_read.close()
        
    list_headings = ['country', 'count']
    create_csv_from_dict(dict_output, str_output_file, list_headings)
    #print(dict_output)
#start_time_check()

#str_select_columns = "visitorId, visitNumber, visitId, visitStartTime, date, totals, trafficSource, device, geoNetwork, hits, " + \
#                     "channelGrouping, socialEngagementType"
#str_select_columns = "visitId, date"
#str_select_from_table = "ga_sessions_20170701"
#query = create_simple_query1(str_select_columns, str_select_from_table)
#flt_est_query_size = ds_current.estimate_query_size(query)
#print('Estimated query size (mb): ' + get_str_query_size_mb(flt_est_query_size) ) 
#print(query)
csv_query_info.close()
end_time_check(dat_program_start, 'overall')
df_time_check