# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#data is in gzip file format

#then a txt file as csv format

#lets get it into pandas!



#start with the smaller traffic station data

#read in traffic station data

traffic_station_df = pd.read_csv('../input/dot_traffic_stations_2015.txt.gz', compression='gzip', 

                                 header=0, sep=',', quotechar='"')
#peek inside

traffic_station_df.head(10)


#read in traffic data--takes a bit, be patient...

traffic_df = pd.read_csv('../input/dot_traffic_2015.txt.gz', compression='gzip', 

                         header=0, sep=',', quotechar='"')





#take a peek

traffic_df.head(10)



#Ok, loaded up--Happy Highway Hacking!