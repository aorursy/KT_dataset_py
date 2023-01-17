import json

from pprint import pprint

import pandas as pd





with open('../input/test.json') as data_file:    

    data = json.load(data_file)



df_parent = pd.DataFrame()

i = 0

arr = []



for row in data:

    arr.append(row)

    i+=1

    if i%100 == 0:

        temp = pd.DataFrame(arr)

        df_parent = pd.concat([df_parent, temp], ignore_index=True)

        print('Done for:', i)

        arr = []



temp = pd.DataFrame(arr)

df_parent = pd.concat([df_parent, temp], ignore_index=True)

print(df_parent.shape)

df_parent
len(arr)
ls