import numpy as np 
import pandas as pd
import time
import logging 
log_format='%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO)
        

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

## a peek on the train data 
train = pd.read_csv("../input/train.csv", nrows=10000)
#print(train.head()) 
#print(train.shape) 
train.describe().transpose()
train.head(10)
train["hotel_cluster"].value_counts()
