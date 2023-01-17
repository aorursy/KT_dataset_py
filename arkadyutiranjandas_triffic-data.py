import csv 

import pandas as pd 

import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



count=0

with open("../input/mock-traffic-data/MockTrafficDataForMCNFP.csv") as f:

    reader = csv.reader(f)



    

    def Remove(duplicate): 

        final_list = [] 

        for num in duplicate: 

            if num not in final_list: 

                final_list.append(num) 

        return final_list 



    lst = []

    lstt = [] 

    for x in range(3, 9):

        for row in reader:

            if(row[x]!=""):

                datas = row[1]

                #print(row[1])

                #print(row[3])

                #count = count+1

                lst.append(datas)

        #print(len(lst))

        #duplicate = row[1]

        #print(duplicate)

        #print(len(Remove(lst)))

        

        value = len(Remove(lst))

        #print(value)

        lstt.append(value)

    print(lstt)



    

# with open(r"cardataset.csv") as csv_file:

#     csv_reader = csv.reader(csv_file, delimiter=',') 

#     df = pd.DataFrame([csv_reader], index=None)

# for val in list(df[1]): 

#     print(val) 



objects = ('node1', 'node2', 'node3', 'node4', 'node5', 'node6')

y_pos = np.arange(len(objects))





plt.bar(y_pos, lstt, align='center', alpha=0.6)

plt.xticks(y_pos, objects)

plt.ylabel('Number of cars')

plt.title('Visualization of num of cars that passed a node')



plt.show()