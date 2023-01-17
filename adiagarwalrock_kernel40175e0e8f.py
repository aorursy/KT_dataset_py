import numpy as np

# import matplotlib.pyplot as plt

# import os

import csv



y = []



with open("../input/terrorism-data-1970to2017/terrorismData.csv", encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)



    for row in file_data:

        if 'United States' in row['Country']:    

            if ' ' in row['Killed']:

                print(yes)

            else:

                y.append(row['Killed'])

y = np.array(y)

y[np.where(y=='')] = '0'



y = y.astype(np.float)

y = y.astype(np.int)



for i in y:

    print(i)
import numpy as np

# import matplotlib.pyplot as plt

# import os

import csv



y = []



with open("../input/terrorism-data-1970to2017/terrorismData.csv", encoding='utf8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)



    for row in file_data:

        if 'United States' in row['Country']:    

            if ' ' in row['Killed']:

                print(yes)

            else:

                y.append(row['Killed'])

            

y = np.array(y)

y[np.where(y=='')] = '0'



y = y.astype(np.float)

y = y.astype(np.int)



s=0

for i in y:

    s = s+i

print(s)
import csv



a = []



with open("../input/terrorism-data-1970to2017/terrorismData.csv", encoding='utf-8') as file_obj:

    file_data=csv.reader(file_obj, skipinitialspace=True)



    fList = list(file_data)

    

for row in fList[1:] :

    day = int(float(row[2]))

    

    if day>=10 and day<=20:

        a.append(1)



print(sum(a))
import csv



c = 0



with open("../input/terrorism-data-1970to2017/terrorismData.csv", encoding='utf-8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)



    for row in file_data :

        

        if row['Year'] == '2010' and row['Month'] == '1' and row['Day']!='0' :

            c=c+1

                

print(c)
import csv



casualty = []

city = []

tGroup = []



with open("../input/terrorism-data-1970to2017/terrorismData.csv", encoding='utf-8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)



    for row in file_data :

        

        if row['Year'] == '1999' and row['State'] == 'Jammu and Kashmir':

            if row['Month'] == '5' or row['Month'] == '6' or row['Month'] == '7':

                

                if row['Killed'] == '' :

                    row['Killed'] = '0'   

                    

                if row['Wounded'] == '' :

                    row['Wounded'] = '0'

                

                if row['City'] !='Unknown' and row['Group'] != 'Unknown' :

                    

#                         row['Killed'] = int(float(row['Killed']))

#                         row['Wounded'] = int(float(row['Wounded']))



#                         c = int(float(row['Killed'])) + int(float(row['Wounded']))



                        casualty.append(int(float(row['Killed'])) + int(float(row['Wounded'])))

                        city.append(row['City'])

                        tGroup.append(row['Group'])

                        

i = casualty.index(max(casualty))

print(casualty[i], city[i], tGroup[i])
import csv



casualty = []



with open("../input/terrorism-data-1970to2017/terrorismData.csv", encoding='utf-8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)



    for row in file_data :

        if row['State'] == 'Jharkhand' or row['State'] == 'Odisha' or row['State'] == 'Andhra Pradesh' or row['State'] == 'Chhattisgarh' :

                        

            if row['Killed'] == '' :

                row['Killed'] = '0'   



            if row['Wounded'] == '' :

                row['Wounded'] = '0'



#             row['Killed'] = int(float(row['Killed']))

#             row['Wounded'] = int(float(row['Wounded']))



#             c = row['Killed'] + row['Wounded']



#             casualty.append(c)

            

            casualty.append(int(float(row['Killed'])) + int(float(row['Wounded'])))

            

print(sum(casualty))
import csv



casualty = []

city = []



with open("../input/terrorism-data-1970to2017/terrorismData.csv", encoding='utf-8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)



    for row in file_data :

        if row['Country'] == 'India' and row['City'] != 'Unknown':          

            if row['Killed'] == '' :

                row['Killed'] = '0'   

            if row['Wounded'] == '' :

                row['Wounded'] = '0'



#             row['Killed'] = int(float(row['Killed']))

#             row['Wounded'] = int(float(row['Wounded']))



#             c = row['Killed'] + row['Wounded']



#             casualty.append(c)

            

            casualty.append(int(float(row['Killed'])) + int(float(row['Wounded'])))

            city.append(row['City'])



freq = {}



for i in range(len(city)):

    if city[i] in freq:

        freq[city[i]] = freq[city[i]] + casualty[i]

    else:

        freq[city[i]] = casualty[i]

        

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)



for i in range(5):

    print(*freq[i])
import csv



day = []



with open("../input/terrorism-data-1970to2017/terrorismData.csv", encoding='utf-8') as file_obj:

    file_data=csv.DictReader(file_obj, skipinitialspace=True)



    for row in file_data :

            day.append(row['Day'])



d = {}

for i in range(len(day)):

    if day[i] in d:

        d[day[i]] = d[day[i]] + 1

    else:

        d[day[i]] = 1



d = sorted(d.items(), key=lambda x: x[1], reverse=True)



print(*d[0])