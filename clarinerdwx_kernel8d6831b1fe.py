import requests

import pandas as pd

import numpy as np

import re
def GFSMOS(station):

    #Open the data file

    response = requests.get('https://www.nws.noaa.gov/mdl/forecast/text/avnmav.txt')

    

    #Put data into list

    data = response.text

    data = data.split("/n")

    

    #Remove all white spaces and replace with commas then split by comma

    x = re.sub("\s+", ",", data[0]).split(',')

    

    #Get the index of the place that we want

    string = station

    ind = x.index(string)

    

    #Make new list starting from our list

    newstring = x[ind:]

    

    #Get index of 'OBV' string and add 21 to get to last value

    ind2 = newstring.index('OBV')

    ind2 = ind2 + 22

    

    #Put data into new list

    datalist = newstring[0:ind2]

    

    #Get index of each of the main values to make dictionary

    indhr = datalist.index('HR')

    try:

        indxn = datalist.index('X/N')

    except:

        indxn = datalist.index('N/X')

    indtmp = datalist.index('TMP')

    inddpt = datalist.index('DPT')

    indcld = datalist.index('CLD')

    indwdr = datalist.index('WDR')

    indwsp = datalist.index('WSP')

    indp06 = datalist.index('P06')

    indp12 = datalist.index('P12')

    indq06 = datalist.index('Q06')

    indq12 = datalist.index('Q12')

    indt06 = datalist.index('T06')

    indt12 = datalist.index('T12')

    indcig = datalist.index('CIG')

    indvis = datalist.index('VIS')

    indobv = datalist.index('OBV')

    inddt = datalist.index('DT')

    

    #Get the first hour and then put the dates for each hour in. We will be putting any dates beyond day 3 under the header for day 3

    hour = int(datalist[indhr+1])

    

    #Now check to make sure the last date and month are correct for longer times

    if(hour == 18):

        month = datalist[inddt+4]

        month = month[1:]

        lastday = datalist[inddt+5]

        lastday = int(lastday)

    else:

        month = datalist[inddt+3]

        month = month[1:]

        lastday = datalist[inddt+6]

        lastday = int(lastday)

    if(month == 'JAN' and (lastday + 1) == 32):

        month = 'FEB'

        lastday = '01'

    elif(month == 'FEB' and (lastday + 1) == 29):

        month = 'MAR'

        lastday = '01'

    elif(month == 'MAR' and (lastday + 1) == 32):

        month = 'APR'

        lastday = '01'

    elif(month == 'APR' and (lastday + 1) == 31):

        month = 'MAY'

        lastday = '01'

    elif(month == 'MAY' and (lastday + 1) == 32):

        month = 'JUN'

        lastday = '01'

    elif(month == 'JUN' and (lastday + 1) == 31):

        month = 'JUL'

        lastday = '01'

    elif(month == 'JUL' and (lastday + 1) == 32):

        month = 'AUG'

        lastday = '01'

    elif(month == 'AUG' and (lastday + 1) == 32):

        month = 'SEP'

        lastday = '01'

    elif(month == 'SEP' and (lastday + 1) == 31):

        month = 'OCT'

        lastday = '01'

    elif(month == 'OCT' and (lastday + 1) == 32):

        month = 'NOV'

        lastday = '01'

    elif(month == 'NOV' and (lastday + 1) == 31):

        month = 'DEC'

        lastday = '01'

    elif(month == 'DEC' and (lastday + 1) == 32):

        month = 'JAN'

        lastday = '01'

    else:

        month = month

        lastday = str(lastday+1)

    

    

    if(hour == 18):

        days = [datalist[inddt+2],datalist[inddt+2], datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4], datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],lastday +'/'+ month,lastday +'/'+ month,lastday +'/'+ month]   

    elif(hour == 12):

        days = [datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5],datalist[inddt+4]+datalist[inddt+5],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],lastday +'/'+ month,lastday +'/'+ month,lastday +'/'+ month]

    elif(hour == 6):

        days = [datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3], datalist[inddt+4]+datalist[inddt+5],datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],lastday +'/'+ month,lastday +'/'+ month]

    else:

        days = [datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5],datalist[inddt+4]+datalist[inddt+5],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],lastday +'/'+ month]

     

    #Put everything into a     return dfdictionary

    dictionary = {

            'HR': datalist[indhr+1:indxn],

            'X/N': datalist[indxn+1:indtmp],

            'TMP': datalist[indtmp+1:inddpt],

            'DPT': datalist[inddpt+1:indcld],

            'CLD': datalist[indcld+1:indwdr],

            'WDR': datalist[indwdr+1:indwsp],

            'WSP': datalist[indwsp+1:indp06],

            'P06': datalist[indp06+1:indp12],

            'P12': datalist[indp12+1:indq06],

            'Q06': datalist[indq06+1:indq12],

            'Q12': datalist[indq12+1:indt06],

            'T06': datalist[indt06+1:indt12],

            'T12': datalist[indt12+1:indcig],

            'CIG': datalist[indcig+1:indvis],

            'VIS': datalist[indvis+1:indobv],

            'OBV': datalist[indobv+1:],

            }

    

    #Make a pandas dataframe from the dictionary and append the 

    df = pd.DataFrame.from_dict(dictionary, orient='index')

    temp = df[0]['X/N']

    temp2 = df[1]['X/N']

    temp3 = df[2]['X/N']

    temp4 = df[3]['X/N']

    temp5 = df[4]['X/N']

    df[6]['X/N'] = temp

    df[10]['X/N'] = temp2

    df[14]['X/N'] = temp3

    df[18]['X/N'] = temp4

    df[20]['X/N'] = temp5

    df[0]['X/N'] = None

    df[1]['X/N'] = None

    df[2]['X/N'] = None

    df[3]['X/N'] = None

    df[4]['X/N'] = None



    temp = df[0]['P06']

    temp2 = df[1]['P06']

    temp3 = df[2]['P06']

    temp4 = df[3]['P06']

    temp5 = df[4]['P06']

    temp6 = df[5]['P06']

    temp7 = df[6]['P06']

    temp8 = df[7]['P06']

    temp9 = df[8]['P06']

    temp10 = df[9]['P06']

    temp11 = df[10]['P06']    

    df[2]['P06'] = temp

    df[4]['P06'] = temp2 

    df[6]['P06'] = temp3

    df[8]['P06'] = temp4

    df[10]['P06'] = temp5

    df[12]['P06'] = temp6

    df[14]['P06'] = temp7

    df[16]['P06'] = temp8

    df[18]['P06'] = temp9

    df[19]['P06'] = temp10

    df[20]['P06'] = temp11

    df[0]['P06'] = None

    df[1]['P06'] = None

    df[3]['P06'] = None

    df[5]['P06'] = None

    df[7]['P06'] = None

    df[9]['P06'] = None

    df[11]['P06'] = None

    

    temp = df[0]['Q06']

    temp2 = df[1]['Q06']

    temp3 = df[2]['Q06']

    temp4 = df[3]['Q06']

    temp5 = df[4]['Q06']

    temp6 = df[5]['Q06']

    temp7 = df[6]['Q06']

    temp8 = df[7]['Q06']

    temp9 = df[8]['Q06']

    temp10 = df[9]['Q06']

    temp11 = df[10]['Q06']    

    df[2]['Q06'] = temp

    df[4]['Q06'] = temp2 

    df[6]['Q06'] = temp3

    df[8]['Q06'] = temp4

    df[10]['Q06'] = temp5

    df[12]['Q06'] = temp6

    df[14]['Q06'] = temp7

    df[16]['Q06'] = temp8

    df[18]['Q06'] = temp9

    df[19]['Q06'] = temp10

    df[20]['Q06'] = temp11

    df[0]['Q06'] = None

    df[1]['Q06'] = None

    df[3]['Q06'] = None

    df[5]['Q06'] = None

    df[7]['Q06'] = None

    df[9]['Q06'] = None

    df[11]['Q06'] = None

    

    temp = df[0]['P12'] 

    temp2 = df[1]['P12']

    temp3 = df[2]['P12']

    temp4 = df[3]['P12']

    temp5 = df[4]['P12']

    df[6]['P12'] = temp

    df[10]['P12'] = temp2

    df[14]['P12']  = temp3

    df[18]['P12'] = temp4

    df[20]['P12'] = temp5

    df[0]['P12'] = None

    df[1]['P12'] = None

    df[2]['P12'] = None

    df[3]['P12'] = None

    df[4]['P12'] = None

    

    temp = df[0]['Q12'] 

    temp2 = df[1]['Q12']

    temp3 = df[2]['Q12']

    temp4 = df[3]['Q12']

    temp5 = df[4]['Q12']

    df[6]['Q12'] = temp

    df[10]['Q12'] = temp2

    df[14]['Q12']  = temp3

    df[18]['Q12'] = temp4

    df[20]['Q12'] = temp5

    df[0]['Q12'] = None

    df[1]['Q12'] = None

    df[2]['Q12'] = None

    df[3]['Q12'] = None

    df[4]['Q12'] = None



    df.columns = days

    

    #Swap the order of the rows to better match MOS output

    df = df.reindex(['HR','X/N','TMP','DPT','CLD','WDR','WSP','P06','P12','Q06','Q12','CIG','VIS','OBV'])

    return df
GFSMOS("KASH")
def NAMMOS(station):

    #Open the data file

    response = requests.get('https://www.nws.noaa.gov/mdl/forecast/text/nammet.txt')

    #Put data into list

    data = response.text

    data = data.split("/n")

    

    #Remove all white spaces and replace with commas then split by comma

    x = re.sub("\s+", ",", data[0]).split(',')

    

    #Get the index of the place that we want

    string = station

    ind = x.index(string)

    

    #Make new list starting from our list

    newstring = x[ind:]

    

    #Get index of 'OBV' string and add 21 to get to last value

    ind2 = newstring.index('OBV')

    ind2 = ind2 + 22

    

    #Put data into new list

    datalist = newstring[0:ind2]

    

    #Get index of each of the main values to make dictionary

    indhr = datalist.index('HR')

    try:

        indxn = datalist.index('X/N')

    except:

        indxn = datalist.index('N/X')

    indtmp = datalist.index('TMP')

    inddpt = datalist.index('DPT')

    indcld = datalist.index('CLD')

    indwdr = datalist.index('WDR')

    indwsp = datalist.index('WSP')

    indp06 = datalist.index('P06')

    indp12 = datalist.index('P12')

    indq06 = datalist.index('Q06')

    indq12 = datalist.index('Q12')

    indt06 = datalist.index('T06')

    indt12 = datalist.index('T12')

    indcig = datalist.index('CIG')

    indvis = datalist.index('VIS')

    indobv = datalist.index('OBV')

    inddt = datalist.index('DT')

    

    #Get the first hour and then put the dates for each hour in.

    hour = int(datalist[indhr+1])

    

    #Now check to make sure the last date and month are correct for longer times

    if(hour == 18):

        month = datalist[inddt+4]

        month = month[1:]

        lastday = datalist[inddt+5]

        lastday = int(lastday)

    else:

        month = datalist[inddt+3]

        month = month[1:]

        lastday = datalist[inddt+6]

        lastday = int(lastday)

    if(month == 'JAN' and (lastday + 1) == 32):

        month = 'FEB'

        lastday = '01'

    elif(month == 'FEB' and (lastday + 1) == 29):

        month = 'MAR'

        lastday = '01'

    elif(month == 'MAR' and (lastday + 1) == 32):

        month = 'APR'

        lastday = '01'

    elif(month == 'APR' and (lastday + 1) == 31):

        month = 'MAY'

        lastday = '01'

    elif(month == 'MAY' and (lastday + 1) == 32):

        month = 'JUN'

        lastday = '01'

    elif(month == 'JUN' and (lastday + 1) == 31):

        month = 'JUL'

        lastday = '01'

    elif(month == 'JUL' and (lastday + 1) == 32):

        month = 'AUG'

        lastday = '01'

    elif(month == 'AUG' and (lastday + 1) == 32):

        month = 'SEP'

        lastday = '01'

    elif(month == 'SEP' and (lastday + 1) == 31):

        month = 'OCT'

        lastday = '01'

    elif(month == 'OCT' and (lastday + 1) == 32):

        month = 'NOV'

        lastday = '01'

    elif(month == 'NOV' and (lastday + 1) == 31):

        month = 'DEC'

        lastday = '01'

    elif(month == 'DEC' and (lastday + 1) == 32):

        month = 'JAN'

        lastday = '01'

    else:

        month = month

        lastday = str(lastday+1)

    

    

    if(hour == 18):

        days = [datalist[inddt+2],datalist[inddt+2], datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4],datalist[inddt+3]+datalist[inddt+4], datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],datalist[inddt+5]+datalist[inddt+6],lastday +'/'+ month,lastday +'/'+ month,lastday +'/'+ month]   

    elif(hour == 12):

        days = [datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5],datalist[inddt+4]+datalist[inddt+5],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],lastday +'/'+ month,lastday +'/'+ month,lastday +'/'+ month]

    elif(hour == 6):

        days = [datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3], datalist[inddt+4]+datalist[inddt+5],datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],lastday +'/'+ month,lastday +'/'+ month]

    else:

        days = [datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+2] + datalist[inddt+3],datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5], datalist[inddt+4]+datalist[inddt+5],datalist[inddt+4]+datalist[inddt+5],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],datalist[inddt+6]+datalist[inddt+1],lastday +'/'+ month]

     

    #Put everything into a dictionary

    dictionary = {

            'HR': datalist[indhr+1:indxn],

            'X/N': datalist[indxn+1:indtmp],

            'TMP': datalist[indtmp+1:inddpt],

            'DPT': datalist[inddpt+1:indcld],

            'CLD': datalist[indcld+1:indwdr],

            'WDR': datalist[indwdr+1:indwsp],

            'WSP': datalist[indwsp+1:indp06],

            'P06': datalist[indp06+1:indp12],

            'P12': datalist[indp12+1:indq06],

            'Q06': datalist[indq06+1:indq12],

            'Q12': datalist[indq12+1:indt06],

            'T06': datalist[indt06+1:indt12],

            'T12': datalist[indt12+1:indcig],

            'CIG': datalist[indcig+1:indvis],

            'VIS': datalist[indvis+1:indobv],

            'OBV': datalist[indobv+1:],

            }

    

    #Make a pandas dataframe from the dictionary and append the 

    df = pd.DataFrame.from_dict(dictionary, orient='index')

    temp = df[0]['X/N']

    temp2 = df[1]['X/N']

    temp3 = df[2]['X/N']

    temp4 = df[3]['X/N']

    temp5 = df[4]['X/N']

    df[6]['X/N'] = temp

    df[10]['X/N'] = temp2

    df[14]['X/N'] = temp3

    df[18]['X/N'] = temp4

    df[20]['X/N'] = temp5

    df[0]['X/N'] = None

    df[1]['X/N'] = None

    df[2]['X/N'] = None

    df[3]['X/N'] = None

    df[4]['X/N'] = None



    temp = df[0]['P06']

    temp2 = df[1]['P06']

    temp3 = df[2]['P06']

    temp4 = df[3]['P06']

    temp5 = df[4]['P06']

    temp6 = df[5]['P06']

    temp7 = df[6]['P06']

    temp8 = df[7]['P06']

    temp9 = df[8]['P06']

    temp10 = df[9]['P06']

    temp11 = df[10]['P06']    

    df[2]['P06'] = temp

    df[4]['P06'] = temp2 

    df[6]['P06'] = temp3

    df[8]['P06'] = temp4

    df[10]['P06'] = temp5

    df[12]['P06'] = temp6

    df[14]['P06'] = temp7

    df[16]['P06'] = temp8

    df[18]['P06'] = temp9

    df[19]['P06'] = temp10

    df[20]['P06'] = temp11

    df[0]['P06'] = None

    df[1]['P06'] = None

    df[3]['P06'] = None

    df[5]['P06'] = None

    df[7]['P06'] = None

    df[9]['P06'] = None

    df[11]['P06'] = None

    

    temp = df[0]['Q06']

    temp2 = df[1]['Q06']

    temp3 = df[2]['Q06']

    temp4 = df[3]['Q06']

    temp5 = df[4]['Q06']

    temp6 = df[5]['Q06']

    temp7 = df[6]['Q06']

    temp8 = df[7]['Q06']

    temp9 = df[8]['Q06']

    temp10 = df[9]['Q06']

    temp11 = df[10]['Q06']    

    df[2]['Q06'] = temp

    df[4]['Q06'] = temp2 

    df[6]['Q06'] = temp3

    df[8]['Q06'] = temp4

    df[10]['Q06'] = temp5

    df[12]['Q06'] = temp6

    df[14]['Q06'] = temp7

    df[16]['Q06'] = temp8

    df[18]['Q06'] = temp9

    df[19]['Q06'] = temp10

    df[20]['Q06'] = temp11

    df[0]['Q06'] = None

    df[1]['Q06'] = None

    df[3]['Q06'] = None

    df[5]['Q06'] = None

    df[7]['Q06'] = None

    df[9]['Q06'] = None

    df[11]['Q06'] = None

    

    temp = df[0]['P12'] 

    temp2 = df[1]['P12']

    temp3 = df[2]['P12']

    temp4 = df[3]['P12']

    temp5 = df[4]['P12']

    df[6]['P12'] = temp

    df[10]['P12'] = temp2

    df[14]['P12']  = temp3

    df[18]['P12'] = temp4

    df[20]['P12'] = temp5

    df[0]['P12'] = None

    df[1]['P12'] = None

    df[2]['P12'] = None

    df[3]['P12'] = None

    df[4]['P12'] = None

    

    temp = df[0]['Q12'] 

    temp2 = df[1]['Q12']

    temp3 = df[2]['Q12']

    temp4 = df[3]['Q12']

    temp5 = df[4]['Q12']

    df[6]['Q12'] = temp

    df[10]['Q12'] = temp2

    df[14]['Q12']  = temp3

    df[18]['Q12'] = temp4

    df[20]['Q12'] = temp5

    df[0]['Q12'] = None

    df[1]['Q12'] = None

    df[2]['Q12'] = None

    df[3]['Q12'] = None

    df[4]['Q12'] = None



    df.columns = days

    

    #Swap the order of the rows to better match MOS output

    df = df.reindex(['HR','X/N','TMP','DPT','CLD','WDR','WSP','P06','P12','Q06','Q12','CIG','VIS','OBV'])

    return df
NAMMOS("KASH")