import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 

import csv

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



rows = [];



frame_header = ['user', 'type', 's', 'sBase', 'hotkey0', 'hotkey1','hotkey2','hotkey3','hotkey4','hotkey5','hotkey6','hotkey7','hotkey8','hotkey9']

with open('../input/TRAIN.CSV.gz', 'rt') as f_input:

    csvReader = csv.reader(f_input, delimiter=',',skipinitialspace=True)

    for row in csv.reader(f_input, delimiter=',',skipinitialspace=True):

        count_s = 0 

        count_sBase = 0 

        count_0 = 0 

        count_1 = 0 

        count_2 = 0 

        count_3 = 0 

        count_4 = 0 

        count_5 = 0 

        count_6 = 0 

        count_7 = 0

        count_8 = 0

        count_9 = 0

        for cell in row:

                if cell == 's':

                    count_s= count_s +1

                if cell == 'Base':

                    count_sBase= count_sBase +1

                if cell.startswith( 'hotkey0' ):

                    count_0= count_0 +1

                if cell.startswith( 'hotkey1' ):

                    count_1= count_1 +1

                if cell.startswith( 'hotkey2' ):

                    count_2= count_2 +1

                if cell.startswith( 'hotkey3' ):

                    count_3= count_3 +1

                if cell.startswith( 'hotkey4' ):

                    count_4= count_4 +1

                if cell.startswith( 'hotkey5' ):

                    count_5= count_5 +1

                if cell.startswith( 'hotkey6' ):

                    count_6= count_6 +1

                if cell.startswith( 'hotkey7' ):

                    count_7= count_7 +1

                if cell.startswith( 'hotkey8' ):

                    count_8= count_8 +1

                if cell.startswith( 'hotkey9' ):

                    count_9= count_9 +1

                    

                if cell == 't10':

                    

                      rows.append([row[0], row[1], count_s , count_sBase ,count_0 ,count_1 ,count_2 ,count_3 ,count_4,count_5,count_6, count_7 ,count_8,count_9])

                      break

                    

    frame = pd.DataFrame(rows, columns=frame_header)

    print (frame)