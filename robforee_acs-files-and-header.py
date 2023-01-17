import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls" , "../input"]).decode("utf8"))



import csv    

print('the a file')            

with open('../input/ss14pusa.csv', 'r') as infile:

    reader = csv.DictReader(infile)

    fieldnames = reader.fieldnames

    #print( fieldnames ) 

    i = 0

    for row in reader:        

        i += 1

        for col in fieldnames:

            print( col, ' = ' ,row[col] )

        if i > 0: break

print('the b file')            

with open('../input/ss14pusb.csv', 'r') as infile:

    reader = csv.DictReader(infile)

    fieldnames = reader.fieldnames

    #print( fieldnames ) 

    i = 0

    for row in reader:        

        i += 1

        for col in fieldnames:

            print( col, ' = ' ,row[col] )

        if i > 0: break        
