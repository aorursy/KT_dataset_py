# aqi = Air quality index ( Variable)

aqi_1 = 19                
aqi_2 = 22
# Addition by Syntax 

aqi_add = aqi_1 + aqi_2                 # aqi = Air quality index ( Variable)

print('Addition by Syntax = ',aqi_add)  # to see the output



# Addition by Function

import operator                         # import library

aqi_add1 = operator.add(aqi_1, aqi_2)

print('Addition by Function = ',aqi_add1)
# Subtraction by Syntax

aqi_subs = aqi_1 - aqi_2

print('Subtraction by Syntax = ',aqi_subs)


# Subtraction by Function

aqi_subs1 = operator.sub(aqi_1, aqi_2)

print('Subtraction by Function = ',aqi_subs1)
# Multiplication by Syntax

aqi_mul = aqi_1 * aqi_2 

print('Multiplication by Syntax = ',aqi_mul)


# Multiplication by Function

aqi_mul1 = operator.mul(aqi_1, aqi_2)

print('Multiplication by Function = ',aqi_mul1)
# division by Syntax
aqi_div = aqi_1 / aqi_2

print('division by Syntax = ',aqi_div)


# division by Function
aqi_div1 = operator.truediv(aqi_1, aqi_2)

print('division by Function = ',aqi_div1)
# Floor division (results into whole number) by Syntax

aqi_fldiv = aqi_1 // aqi_2

print('Floor division by Syntax  = ',aqi_fldiv)



# Floor division (results into whole number) by Function

aqi_fldiv1 = operator.floordiv(aqi_1, aqi_2)

print('Floor division by Function = ', aqi_fldiv1 )
# Modulus (remainder) by Syntax

aqi_mod = aqi_1 % aqi_2

print('Modulus by Syntax = ',aqi_mod)


# Modulus (remainder) by Function

aqi_mod1 = operator.mod (aqi_1, aqi_2)

print('Modulus by Function = ',aqi_mod1)
# Exponent (power) by Syntax
aqi_exp = aqi_1 ** aqi_2                 # a**b (5*5*5*5*5*5 = 15625)

print('Exponent by Syntax = ',aqi_exp)

# Exponent (power) by Function
aqi_exp1 = operator.pow(aqi_1, aqi_2)          # The pow() function returns the value of x to the power of y.
                                       # (5*5*5*5*5*5 = 15625)

print('Exponent by Function = ',aqi_exp1)
#Impoting library 

import pandas as pd
aqi_dataset = pd.read_csv('../input/computer-programming-and-data-analysis-with-python/Arithmetic operators _Class_1.csv')

#print(dataset)

california = aqi_dataset.california
texas = aqi_dataset.texas
florida = aqi_dataset.florida
mississippi = aqi_dataset.mississippi
vermont = aqi_dataset.vermont
    
# print('california = \n',california)
# print('texas = \n',texas)
# print('florida = \n',florida)
# print('vermont = \n',vermont)


aqi_dataset.head()
# calculate average Air Quality Index(AQI) by Syntax

avg_aqi = (california + texas + florida )/ 3

print('average Air Quality Index by Syntax = \n',avg_aqi)
