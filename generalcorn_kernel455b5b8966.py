# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import datetime







class workDays():

    def __init__(self, start_date, end_date, days_off=None):

        

        self.start_date = start_date

        self.end_date = end_date

        self.days_off = days_off

        if self.start_date > self.end_date:

            self.start_date, self.end_date = self.end_date, self.start_date

        if days_off is None:

            self.days_off = 5, 6

      

        self.days_work = [x for x in range(7) if x not in self.days_off]



    def workDays(self):

       

        # 还没排除法定节假日还有那些teacher trainingday...

        tag_date = self.start_date

        while True:

            if tag_date > self.end_date:

                break

            if tag_date.weekday() in self.days_work:

                yield tag_date

            tag_date += datetime.timedelta(days=1)



    def daysCount(self):

       

        return len(list(self.workDays()))



now = datetime.datetime.now()



era = datetime.date(2019,11,12)





print('\nTodays date is:')

print (now.strftime("%Y-%m-%d"))

dateinput = (now.strftime("%Y-%m-%d"))

dateinput = dateinput.split('-')

y = int(dateinput[0])

m = int(dateinput[1])

d = int(dateinput[2])



b = datetime.date(y,m,d)

work = workDays(era,b)



re = work.daysCount()%8

if re == 0 :

    print('\nToday is: DAY 8')

else:

    print('\nToday is: DAY', re)

    

classes = []

print("\nThe classes of today are:")

if re == 0:

    print("\nEnglish\nMath\nChinese\nScience")

elif re == 1:

    print("\nMath\nDesign\nEnglish\nI&S")

elif re == 2:

    print("\nPHE\nScience\nArt\nEnglish")

elif re == 3:

    print("\nPHE\nMath\nDrama\nChinese")

elif re == 4:

    print("\nMath\nDrama\nI&S\nEnglish")

elif re == 5:

    print("\nChinese\nI&S\nDesign\nPHE")

elif re == 6:

    print("\nScience\nPHE\nDesign\nI&S")

elif re == 7:

    print("\nChinese\nDrama\nScience\nDesign")

else:

    print("Error, please type in the right date!")