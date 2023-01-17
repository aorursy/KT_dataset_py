import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from datetime import datetime



import os



dataFiles = list()



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        printName = os.path.join(dirname, filename)

        print(printName)

        dataFiles.append(printName)



print(len(dataFiles))
dataDF = list()



for idx in range(len(dataFiles)):

    try:



        dataDict = dict()

        filename = dataFiles[idx]

        

        if 'AllDates' in filename:

            dataDict['Date'] = 'AllDates'

        else:

            dataDict['Date'] = filename.split('_')[-1].split('.')[0]

        

        if 'xlsx' in filename:

            dataDict['Data'] = pd.read_excel(filename)

        else:

            dataDict['Data'] = pd.read_csv(filename)



        dataDF.append(dataDict)

        

    except:

        print(f"Error in {idx}# ", filename.split('/')[-1])
april15Idx = None

mar30Idx = None



for idx in range(len(dataDF)):

    print(f"{idx}# ", dataDF[idx]['Date'])

    if '15Apr' in dataDF[idx]['Date']:

        april15Idx = idx

        

    print(dataDF[idx]['Data'].columns)

    print("#####################################")

    
#Adjusting for 15th April



if april15Idx:

    dataDF[april15Idx]['Data']['Tests'] = round(dataDF[april15Idx]['Data']['Tests /millionpeople']*(dataDF[april15Idx]['Data']['Positive']/dataDF[april15Idx]['Data']['Positive /millionpeople']), 0)

dataDF[april15Idx]['Data'].head()


for idx in range(len(dataDF)):

    if dataDF[idx]['Date'] == 'AllDates':

        continue

        

    dataDF[idx]['Data'].rename(columns={'Country or region': 'Country', 'Country or territory': 'Country',

                         'As of': 'Date', 

                         'Total tests': 'Tested', 'Tests': 'Tested',

                        'Positive/ thousands': 'Positive /millionpeople', 'Positive /millionpeople': 'Positive /millionpeople',

                        'Tests /millionpeople': 'Tested /millionpeople', 'Tests/ million': 'Tested /millionpeople',

                                       'Source': 'Source_1'}, inplace=True)

    try:

        dataDF[idx]['Data']['Positive/Tested %'] = round(100*(dataDF[idx]['Data']['Positive'] / dataDF[idx]['Data']['Tested']),2)

    except:

        print(f"ERROR: In calculating %age {idx}# ", dataDF[idx]['Date'])

    

    if 'Units' not in list(dataDF[idx]['Data'].columns):

        dataDF[idx]['Data']['Units'] = 'NA'

    

    if dataDF[idx]['Date'] == 'Conducted':

        # For the 30March File

        dataDF[idx]['Data'] = dataDF[idx]['Data'][['Country', 'Date', 'Tested','Source_1']]

    else:



        try:

            dataDF[idx]['Data'] = dataDF[idx]['Data'][['Country', 'Date', 'Tested', 'Units','Positive', 'Positive/Tested %','Source_1', 'Source_2']]

        except:

            print(f"ERROR: In subsetting columns {idx}# ", dataDF[idx]['Date'])



    # Normalize the dates to the same format %d %b %Y ie 26 Apr 2020 (final expected)

    

    splitDate = dataDF[idx]['Data']['Date'].str.split().loc[0]

    if len(splitDate) == 2:

        dataDF[idx]['Data']['Date'] = dataDF[idx]['Data']['Date'].str.split().str.join(' ') + ' 2020' 

    else:

        dataDF[idx]['Data']['Date'] = dataDF[idx]['Data']['Date'].str.split().str.join(' ')

    

    dataDF[idx]['Data']['FileDate'] = dataDF[idx]['Date']

    

    dataDF[idx]['Data'].reset_index(inplace=True, drop=True)

    

#     if 'Positive /millionpeople' in dataDF[idx]['Data'].columns.to_list():

#         dataDF[idx]['Data']['Tests_calculated'] = (dataDF[idx]['Data']['Positive']/ dataDF[idx]['Data']['Positive /millionpeople'])
dataDF[1]['Data'].columns
completeDF = pd.DataFrame()



for idx in range(len(dataDF)):

    if dataDF[idx]['Date'] == 'AllDates':

        continue

    completeDF = pd.concat([completeDF, dataDF[idx]['Data']])



completeDF.dropna(subset=['Date'], inplace=True)



completeDF.reset_index(inplace=True, drop=True)

print(completeDF.shape)

completeDF.head()
# Uncomment if you are not comfortable with using the 15th April 2020 data

# completeDF = completeDF.drop(completeDF[completeDF['FileDate'] == '15April2020'].index)

# completeDF['FileDate'].value_counts()
completeDF.to_csv('TestsConducted_AllDates_11May2020.csv', index=False)