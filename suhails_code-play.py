# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
a = 40

b = 0

space = 40

while b < a-1 and a> 0:

    print(' '*a+'*'+'*'*b)

    a -= 1

    b += 2

for _ in range(4):

    print(' '*(space-1)+'|||')

print(' '*(space-5), '\_@_@_@@/')
import random



r = random.randint(0, 10)

count = 0.00



while True:

   count = count + 1.00

   if r == 5:

       print("{:.4%}".format(1 / count))

       break

   # print(r)

   r = random.randint(0, 10)


input_str = '4,3'

dimensions = [int(x) for x in input_str.split(',')]

rowNum = dimensions[0]

colNum = dimensions[1]

multilist = [[0 for col in range(colNum)] for row in range(rowNum)]



for row in range(rowNum):

    for col in range(colNum):

        multilist[row][col] = row * col



print(multilist)
X = [[12, 7, 3],

     [4, 5, 6],

     [7, 8, 9]]



Y = [[5, 8, 1],

     [6, 7, 3],

     [4, 5, 9]]



result = [[0, 0, 0],

          [0, 0, 0],

          [0, 0, 0]]



for i in range(len(X)):

    # iterate through columns

    for j in range(len(X[0])):

        result[i][j] = X[i][j] + Y[i][j]



for r in result:

    print(r)
# Print nterms of Fibonacci Sequence starting 0



nterms = 10



n1 = 0

n2 = 1

count = 0



if nterms <= 0:

    print("Please enter a positive integer")

elif nterms == 1:

    print("Sequence upto", nterms, ":")

    print(n1)

else:

    print("Sequence upto", nterms, ":")

    while count < nterms:

        print(n1, end=' , ')

        nth = n1 + n2

        # update values

        n1 = n2

        n2 = nth

        count += 1
# Create a progoram which tells the date of last Friday



# Hints - use datetime and timedelta



from datetime import datetime, timedelta



weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',

            'Friday', 'Saturday', 'Sunday']





def get_previous_byday(dayname, start_date=None):

    if start_date is None:

        start_date = datetime.today()

    day_num = start_date.weekday()

    day_num_target = weekdays.index(dayname)

    days_ago = (7 + day_num - day_num_target) % 7

    if days_ago == 0:

        days_ago = 7

    target_date = (start_date - timedelta(days=days_ago)).strftime('%d-%m-%Y')

    #     target_date = (start_date - timedelta(days=days_ago)).__format__('%d%d-%m%m-%y%y%y%y')

    return target_date





if __name__ == '__main__':

    print(get_previous_byday('Friday'))

# Print all prime numbers between lower and upper range values



# lower = input('Enter lower range: ', )

# upper = input('Enter upper range: ', )



lower = int(1)

upper = int(45)



print("Prime numbers between", lower, "and", upper, "are:")



for num in range(lower, upper + 1):

    # prime numbers are greater than 1

    if num > 1:

        for i in range(2, num):

            if (num % i) == 0:

                break

        else:

            print(num)
# Create a program which takes a text string and returns every word of it in sorted order

# Like - today is great day

# Return answer -

# day

# great

# is

# today



# my_string = input('Enter a string: ')



my_string = 'today is a special day to try something new'



words = my_string.split()



words.sort()



print("The sorted words are:")

for word in words:

    print(word)