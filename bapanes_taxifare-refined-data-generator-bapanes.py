import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

print(tf.__version__)
header_names_train = ['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
header_names_test = ['key','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
from math import cos, asin, sqrt
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a)) #2*R*asin...
def data_refine_train(input_file, output_file, ini_row, tot_row):
    
    df = pd.read_csv(input_file, sep=',', header = None, names = header_names_train, skiprows = ini_row, nrows = tot_row)
    
    df = df[['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']]
    
    df['distance'] = pd.concat([pd.DataFrame([distance(df['pickup_latitude'][i],df['pickup_longitude'][i],
                                                         df['dropoff_latitude'][i],df['dropoff_longitude'][i])], 
                                               columns=['distance']) for i in range(len(df))], ignore_index=True)
    
    df = df[((df.pickup_longitude >= -75.0) & (df.pickup_longitude <= -72)) 
         & ((df.pickup_latitude >= 38) & (df.pickup_latitude <= 42)) 
         & ((df.dropoff_longitude >= -75.0) & (df.dropoff_longitude <= -72)) 
         & ((df.dropoff_latitude >= 38) & (df.dropoff_latitude <= 42)) 
         & (df.fare_amount > 2.5) & (df.passenger_count > 0)]
    
    #df = df[['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance','key']] 
    
    df.to_csv(output_file, index=False, sep=',')

def data_refine_test(input_file, output_file, ini_row, number_of_rows):
    
    df = pd.read_csv(input_file, sep=',', header = None, names = header_names_test, skiprows = ini_row, nrows = number_of_rows)
   
    df = df[['key','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']]
    
    df['distance'] = pd.concat([pd.DataFrame([distance(df['pickup_latitude'][i],df['pickup_longitude'][i],
                                                         df['dropoff_latitude'][i],df['dropoff_longitude'][i])], 
                                               columns=['distance']) for i in range(len(df))], ignore_index=True)
    
    #df = df[['key','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance']]
        
    df.to_csv(output_file, index=False, sep=',')
input_file_train = '../input/train.csv'
input_file_test = '../input/test.csv'
import os

numOfLines_train = int(os.popen('wc -l < ../input/train.csv').read()[:-1])
numOfLines_test = int(os.popen('wc -l < ../input/test.csv').read()[:-1])
print(numOfLines_train, numOfLines_test)
data_refine_test(input_file_test, 'test_r0.csv', 1, numOfLines_test)
data_refine_train(input_file_train, 'train_r0.csv', 1, 100000)
data_refine_train(input_file_train, 'train_r1.csv', 100001, 100000)
#numOfLines_output_r0 = int(os.popen('wc -l < ../Python_Lab/output/train_r0.csv').read()[:-1])
#numOfLines_output_r0
#numOfLines_output_r1 = int(os.popen('wc -l < ../Python_Lab/output/train_r1.csv').read()[:-1])
#numOfLines_output_r1
