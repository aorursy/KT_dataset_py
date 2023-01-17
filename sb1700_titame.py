import csv as csv
import numpy as np

csv_file_object = csv.reader(open('../csv/train.csv', 'rU')) 
header = next(csv_file_object)
