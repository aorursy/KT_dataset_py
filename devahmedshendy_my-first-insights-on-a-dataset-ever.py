import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from math import isnan

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading datasets

salaries = pd.read_csv('../input/Salaries.csv')
salaries.info()
# Collection of methods used by this Notebook

def isBernardFatooh(name):

    return True if name.title() == 'A Bernard Fatooh' else False



def toCamelCase(string):

    return string.title()



def isNaN(arg):

    try:

        number = float(arg)

        return isnan(number)

    except ValueError:

        return False



def isNotNaN(arg):

    return True if not isNaN(arg) else False



def isFloatNumber(arg):

    try:

        number = float(arg)

        return not isnan(number)

    except ValueError:

        return False



def toFloat64(arg):

    return float(arg) if isFloatNumber(arg) else 0.0
## STEP

# All values in Notes column is NaN, no need for it



salaries.drop('Notes', axis=1, inplace=True)
## STEP

# I noticed some strings are UPPERCASE and LOWERCASE for the same string in EmployeeName column



salaries[salaries['EmployeeName'].apply(isBernardFatooh)]
# So, let'e clean the EmployeeName column, and JobTitle column as you notice it too



salaries['EmployeeName'] = salaries['EmployeeName'].apply(toCamelCase)

salaries['JobTitle'] = salaries['JobTitle'].apply(toCamelCase)



salaries[salaries['EmployeeName'].apply(isBernardFatooh)]
## STEP

# I noticed some vlaues in Benefits column are string, they should be float64



salaries['Benefits'] = salaries['Benefits'].apply(toFloat64)
# Now from info() we can see benefits column is cleaned now to be all in float64 datatype

salaries.info()
## Step

# We need to clean BasePay,OvertimePay, and OtherPay as well



for column_name in ['BasePay', 'OvertimePay', 'OtherPay']:

    salaries[column_name] = salaries[column_name].apply(toFloat64)
salaries.info()
## STEP

# I discovered that some names in EmployeeName column have more than 1 single whitespace between firstname and last name



def fixWhitespaceBetweenFirstnameAndLastname(name):

    return ' '.join(name.split())



salaries['EmployeeName'] = salaries['EmployeeName'].apply(fixWhitespaceBetweenFirstnameAndLastname)
salaries.head()