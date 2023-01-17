import numpy as np 

import pandas as pd 

import inflect

from num2words import num2words 

import re

import inflect

p = inflect.engine()



train = pd.read_csv("../input/en_train.csv")
def is_num(key):

    if is_float(key) or re.match(r'^-?[0-9]\d*?$', key.replace(',','')): 

        return True

    else: 

        return False



def is_float(string):

    try:

        return float(string.replace(',','')) and "." in string 

    except ValueError:  

        return False



# CARDINAL

def CARDINAL(x):

    try:

        x = str(x)

        text = p.number_to_words(x,decimal='point',andword='', zero='o')

        if re.match(r'^0\.',x): 

            text = 'zero '+text[2:]

        if re.match(r'.*\.0$',x): text = text[:-2]+' zero'

        text = text.replace('-',' ').replace(',','')

        return text.lower()

    except: 

        return x



# DECIMAL

def DECIMAL(x):

    try:

        x = str(x)

        numsplit = x.split(" ")

        if len(numsplit) == 1:    

            numsOnly = x.split(".")

            beforeDecimal = ""

            if len(numsOnly) > 1:

                if numsOnly[0] != "":

                    beforeDecimal = CARDINAL(numsOnly[0]) + ' '

                afterDecimal = []

                for digit in numsOnly[1]:

                    afterDecimal.append(CARDINAL(digit).replace("zero", "o"))

                return beforeDecimal + 'point ' + " ".join(afterDecimal)

            else:

                beforeDecimal = CARDINAL(numsOnly[0])

                return beforeDecimal

        else:

            return DECIMAL(numsplit[0]) + ' ' + numsplit[1]

    except:

        return x



# DIGIT

def DIGIT(x):

    try:    

        x = str(x)

        numsOnly = re.sub('[^0-9]', '', x)

        digits = []

        for num in numsOnly:

            digits.append(CARDINAL(num).replace("zero", "o"))

        return ' '.join(digits)

    except:

        return x



# ORDINAL

def ORDINAL(x):

    try:

        numsOnly = int(re.sub('[^0-9]', '', x))

        ordinalNum = num2words(numsOnly, ordinal = True)

        return ordinalNum.replace(' and','').replace('-',' ').replace(',','')

    except:

        return x



#MEASURE

dict_m = {'"': 'inches', "'": 'feet', 'km/s': 'kilometers per second', 'AU': 'units', 'BAR': 'bars', 'CM': 'centimeters', 'mm': 'millimeters', 'FT': 'feet', 'G': 'grams', 

     'GAL': 'gallons', 'GB': 'gigabytes', 'GHZ': 'gigahertz', 'HA': 'hectares', 'HP': 'horsepower', 'HZ': 'hertz', 'KM':'kilometers', 'km3': 'cubic kilometers',

     'KA':'kilo amperes', 'KB': 'kilobytes', 'KG': 'kilograms', 'KHZ': 'kilohertz', 'KM²': 'square kilometers', 'KT': 'knots', 'KV': 'kilo volts', 'M': 'meters',

      'KM2': 'square kilometers','Kw':'kilowatts', 'KWH': 'kilo watt hours', 'LB': 'pounds', 'LBS': 'pounds', 'MA': 'mega amperes', 'MB': 'megabytes',

     'KW': 'kilowatts', 'MPH': 'miles per hour', 'MS': 'milliseconds', 'MV': 'milli volts', 'kJ':'kilojoules', 'km/h': 'kilometers per hour',  'V': 'volts', 

     'M2': 'square meters', 'M3': 'cubic meters', 'MW': 'megawatts', 'M²': 'square meters', 'M³': 'cubic meters', 'OZ': 'ounces',  'MHZ': 'megahertz', 'MI': 'miles',

     'MB/S': 'megabytes per second', 'MG': 'milligrams', 'ML': 'milliliters', 'YD': 'yards', 'au': 'units', 'bar': 'bars', 'cm': 'centimeters', 'ft': 'feet', 'g': 'grams', 

     'gal': 'gallons', 'gb': 'gigabytes', 'ghz': 'gigahertz', 'ha': 'hectares', 'hp': 'horsepower', 'hz': 'hertz', 'kWh': 'kilo watt hours', 'ka': 'kilo amperes', 'kb': 'kilobytes', 

     'kg': 'kilograms', 'khz': 'kilohertz', 'km': 'kilometers', 'km2': 'square kilometers', 'km²': 'square kilometers', 'kt': 'knots','kv': 'kilo volts', 'kw': 'kilowatts', 

     'lb': 'pounds', 'lbs': 'pounds', 'm': 'meters', 'm2': 'square meters','m3': 'cubic meters', 'ma': 'mega amperes', 'mb': 'megabytes', 'mb/s': 'megabytes per second', 

     'mg': 'milligrams', 'mhz': 'megahertz', 'mi': 'miles', 'ml': 'milliliters', 'mph': 'miles per hour','ms': 'milliseconds', 'mv': 'milli volts', 'mw': 'megawatts', 'm²': 'square meters',

     'm³': 'cubic meters', 'oz': 'ounces', 'v': 'volts', 'yd': 'yards', 'µg': 'micrograms', 'ΜG': 'micrograms', 'kg/m3': 'kilograms per meter cube'}



def MEASURE(key):

    try:

        if key.endswith('%'):

            percentKey = key.split('%')

            return DECIMAL(percentKey[0]).strip() + ' percent'

        else:

            unittest = key.split()

            if unittest in dict_m.keys():

                unit = dict_m[unittest()[-1]]

                val = unittest()[0]

            else:

                unit = unittest[-1]

                val = unittest[0]

            if is_num(val):

                val = DECIMAL(val)

                text = val + ' ' + unit

            else: text = key

            return text

    except:

        return(key)



# ELECTRONIC

def ELECTRONIC(key):

    key = key.replace('.',' dot ').replace('/',' slash ').replace('-',' dash ').replace(':',' colon ').replace('_',' underscore ')

    key = key.split()

    lis2 = ['dot','slash','dash','colon']

    for i in range(len(key)):

        if key[i] not in lis2:

            key[i]=" ".join(key[i])

    text = " ".join(key)

    return text.lower()



# MONEY

def MONEY(key):

    try:    

        v = key.replace('$','').replace('US$','').split()

        if len(v) == 2: 

            if is_num(v[0]):

                text = DECIMAL(v[0]) + ' '+ v[1] + ' '+ 'dollars'

        elif is_num(v[0]):

            text = DECIMAL(v[0]) + ' '+ 'dollars'

        else:

            if 'm' in key or 'M' in key or 'million':

                text = p.number_to_words(key).replace(',','').replace('-',' ').replace(' and','') + ' million dollars'

            elif 'bn' in key:

                text = p.number_to_words(key).replace(',','').replace('-',' ').replace(' and','') + ' billion dollars'

            else: text = key

        return text.lower()

    except:

        return(key)



# TELEPHONE

def TELEPHONE(x):

    try:

        telNum = []

        for i in range(0,len(x)):

            if re.match('[0-9]+', x[i]):

                telNum.append(CARDINAL(x[i]))

            elif telNum[-1] != 'sil':

                telNum.append('sil')

        return ' '.join(telNum)  

    except:

        return x



# DATE

dict_mon = {'jan': "January", "feb": "February", "mar ": "march", "apr": "april", "may": "may ","jun": "june", "jul": "july", "aug": "august","sep": "september",

            "oct": "october","nov": "november","dec": "december", "january":"January", "february":"February", "march":"march","april":"april", "may": "may", 

            "june":"june","july":"july", "august":"august", "september":"september", "october":"october", "november":"november", "december":"december"}

def DATE(key):

    try:

        v =  key.split('-')

        if len(v)==3:

            if v[1].isdigit():

                try:

                    date = datetime.strptime(key , '%Y-%m-%d')

                    text = 'the '+ p.ordinal(p.number_to_words(int(v[2]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')

                    if int(v[0])>=2000 and int(v[0]) < 2010:

                        text = text  + ' '+CARDINAL(v[0])

                    else: 

                        text = text + ' ' + CARDINAL(v[0][0:2]) + ' ' + CARDINAL(v[0][2:])

                except:

                    text = key

                return text.lower()    

        else:   

            v = re.sub(r'[^\w]', ' ', key).split()

            if v[0].isalpha():

                try:

                    if len(v)==3:

                        text = dict_mon[v[0].lower()] + ' '+ p.ordinal(p.number_to_words(int(v[1]))).replace('-',' ')

                        if int(v[2])>=2000 and int(v[2]) < 2010:

                            text = text  + ' '+CARDINAL(v[2])

                        else: 

                            text = text + ' ' + CARDINAL(v[2][0:2]) + ' ' + CARDINAL(v[2][2:])   

                    elif len(v)==2:



                        if int(v[1])>=2000 and int(v[1]) < 2010:

                            text = dict_mon[v[0].lower()]  + ' '+ CARDINAL(v[1])

                        else: 

                            if len(v[1]) <=2:

                                text = dict_mon[v[0].lower()] + ' ' + CARDINAL(v[1])

                            else:

                                text = dict_mon[v[0].lower()] + ' ' + CARDINAL(v[1][0:2]) + ' ' + CARDINAL(v[1][2:])

                    else: text = key

                except: text = key

                return text.lower()

            else: 

                key = re.sub(r'[^\w]', ' ', key)

                v = key.split()

                try:

                    date = datetime.strptime(key , '%d %b %Y')

                    text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]

                    if int(v[2])>=2000 and int(v[2]) < 2010:

                        text = text  + ' '+CARDINAL(v[2])

                    else: 

                        text = text + ' ' + CARDINAL(v[2][0:2]) + ' ' + CARDINAL(v[2][2:])

                except:

                    try:

                        date = datetime.strptime(key , '%d %B %Y')

                        text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+ dict_mon[v[1].lower()]

                        if int(v[2])>=2000 and int(v[2]) < 2010:

                            text = text  + ' '+CARDINAL(v[2])

                        else: 

                            text = text + ' ' + CARDINAL(v[2][0:2]) + ' ' + CARDINAL(v[2][2:])

                    except:

                        try:

                            date = datetime.strptime(key , '%d %m %Y')

                            text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')

                            if int(v[2])>=2000 and int(v[2]) < 2010:

                                text = text  + ' '+CARDINAL(v[2])

                            else: 

                                text = text + ' ' + CARDINAL(v[2][0:2]) + ' ' + CARDINAL(v[2][2:])

                        except:

                            try:

                                date = datetime.strptime(key , '%d %m %y')

                                text = 'the '+ p.ordinal(p.number_to_words(int(v[0]))).replace('-',' ')+' of '+datetime.date(date).strftime('%B')

                                v[2] = datetime.date(date).strftime('%Y')

                                if int(v[2])>=2000 and int(v[2]) < 2010:

                                    text = text  + ' '+CARDINAL(v[2])

                                else: 

                                    text = text + ' ' + CARDINAL(v[2][0:2]) + ' ' + CARDINAL(v[2][2:])

                            except:text = key

                return text.lower() 

    except:

        return(key)



# LETTERS 

def LETTERS(x):

    try:

        lettersOnly = re.sub('[^A-Za-z]', '', x)

        if (lettersOnly[-1] == 's') & (lettersOnly[-2] != lettersOnly[-2].lower):

            letterList = []

            for letter in lettersOnly[0:len(lettersOnly)-2]:

                letterList.append(letter.lower())

            lastTwo = lettersOnly[-2].lower() + "'" + lettersOnly[-1]

            letterList.append(lastTwo)

            return ' '.join(letterList)

        else:

            letterList = []

            for letter in lettersOnly:

                letterList.append(letter.lower())

            return ' '.join(letterList)

    except:

        return x



# ADDRESS

def ADDRESS(x):

    try:

        noPunct = re.sub('[^0-9a-zA-z]', '', x)

        charList = []

        for char in noPunct[0:len(noPunct)-2]:

            if char.isalpha():

                charList.append(char.lower())

            else:

                charList.append(DIGIT(char))

        if noPunct[-2].isalpha() | noPunct[-1].isalpha():

            if noPunct[-2].isalpha():

                charList.append(noPunct[-2].lower())

            else:

                charList.append(DIGIT(noPunct[-2]))

            if noPunct[-1].isalpha():

                charList.append(noPunct[-1].lower())

            else:

                charList.append(DIGIT(noPunct[-1]))

        else: 

            charList.append(CARDINAL(noPunct[-2:]))

        return ' '.join(charList)

    except:

        return x



#FRACTION

def FRACTION(x):

    try:

        x = str(x)

        y = x.split('/')

        result_string = ''

        y[0] = CARDINAL(y[0])

        y[1] = ORDINAL(y[1])

        if y[1] == 4:

            result_string = y[0] + ' quarters'

        else:    

            result_string = y[0] + ' ' + y[1] + 's'

        return(result_string)

    except:    

        return(x)



#PLAIN

def PLAIN(x):

    try:

        return(x)

    except:

        return(x)

    

#PUNCT

def PUNCT(x):

    try:

        return(x)

    except:

        return(x)



#VERBATIM

def VERBATIM(x):

    try:

        return(x)

    except:

        return(x)

    

#TIME

def TIME(x):

    try:

        return(x)

    except:

        return(x)
classes = list(set(train['class']))



for tclass in classes:    

    df_num = train[train['class'] == tclass]

    ct = 0

    pred_list = []

    for i in range(df_num.shape[0]):

        tfunction = tclass + "(df_num.iloc[i,3])"

        pred_list.append((eval(tfunction), df_num.iloc[i,4] ))

    pred_list = list(set(pred_list))

    

    for i in range(len(pred_list)):

        if pred_list[i][0] !=  pred_list[i][1]: ct += 1

        

    print (str(tclass) + " - Rows: " + str(len(train[train['class'] == tclass])) + " Percentage Wrong: " + str(round(100*ct*1.0/len(pred_list), 2)))