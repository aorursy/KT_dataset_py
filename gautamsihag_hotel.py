import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib
train = pd.read_csv('../input/train.csv',dtype={'is_booking':bool, 'date_time':np.str_ ,'hotel_cluster':np.int32},
                    usecols=['is_booking','hotel_cluster', 'date_time'],
                    chunksize = 1000000)
aggs=[]
for chunk in train:
    agg = chunk.groupby(['hotel_cluster','date_time'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)
aggs = pd.concat(aggs, axis = 0)
