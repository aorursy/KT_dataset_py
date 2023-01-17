import math
def anvalue(word):

    return sum([ord(x)-64 for x in word if x!=' '])  #ASCII minus 64



print(anvalue('RIDDLER'))

print(anvalue('ONE'))

print(anvalue('TWO'))

print(anvalue('ONE THOUSAND FOUR HUNDRED SEVENTEEN'))

print(anvalue('THREE MILLION ONE HUNDRED FORTY THOUSAND TWO HUNDRED SEVENTY FIVE'))

components={1:'ONE',

           2:'TWO',

           3:'THREE',

           4:'FOUR',

           5:'FIVE',

           6:'SIX',

           7:'SEVEN',

           8:'EIGHT',

           9:'NINE',

           10:'TEN',

           11:'ELEVEN',

           12:'TWELVE',

           13:'THIRTEEN',

           14:'FOURTEEN',

           15:'FIFTEEN',

           16:'SIXTEEN',

           17:'SEVENTEEN',

           18:'EIGHTEEN',

           19:'NINETEEN',

           20:'TWENTY',

           30:'THIRTY',

           40:'FORTY',

           50:'FIFTY',

           60:'SIXTY',

           70:'SEVENTY',

           80:'EIGHTY',

           90:'NINETY',

           100:'HUNDRED'}

def num_to_english(num):

    english=''

    if num > 99:

        h_digit= math.trunc(num/100)

        english+=(components[h_digit] + ' ' + components[100] + ' ')  #e.g. 'FOUR' plus 'HUNDRED'

        num=num%100

    if num<20 and num!=0:

        english+=(components[num]+' ')  #single digits through the teens do not require a word for the tens place

    elif num!=0:  

        t_value=math.trunc(num/10)*10

        english+=(components[t_value]+' ')  #e.g. 'FIFTY' followed by a number

        if t_value!=num:

            num=num%10

            english+=(components[num])  #finally the ones place

    return(english)



print(num_to_english(1))

print(num_to_english(10))

print(num_to_english(100))

print(num_to_english(312))

print(num_to_english(337))

print(num_to_english(17))

print(num_to_english(20))

print(num_to_english(920))

        

        
for x in range(500):

    if x!=0:

        if anvalue(num_to_english(x))>x:  

            print(f"{x} {num_to_english(x)} alphanumeric value {anvalue(num_to_english(x))}")
values = list(range(500))

anvalues=[anvalue(num_to_english(x)) for x in values]

differences=[anvalue(num_to_english(x))-x for x in values]
# libraries

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

# Data

df=pd.DataFrame({'values': values, 'alphanumeric values': anvalues, 'differences': differences})

 

# multiple line plot

fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

plt.plot( 'values', 'alphanumeric values', data=df, marker='', color='skyblue', linewidth=2)

plt.plot( 'values', 'differences', data=df, marker='', color='red', linewidth=2)

plt.grid(True, which='both', axis='x')

plt.axhline(y=0, color='k')

plt.legend()


