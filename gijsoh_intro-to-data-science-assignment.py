# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
alcohol = pd.read_csv('../input/student-mat.csv')
alcohol.head()
Consumption=["Dalc","Walc"]

Weekly_average = np.average(alcohol["Dalc"])

print("Gemiddelde Doordeweeks:",Weekly_average)

Weekend_average = np.average(alcohol["Walc"])

print("Gemiddelde Weekend:",Weekend_average)

plt.hist(alcohol['absences'])

plt.title('Histogram of absences')

plt.xlabel('Days of absence')

plt.ylabel('Amount of students')

plt.show()



avg_absences = np.average(alcohol['absences'])

print("The average days of absences in a year is:", round(avg_absences,2), "days.")
categories = [0, 1, 2, 3, 4]

plt.xticks(categories)

plt.hist(alcohol['failures'])

plt.title('Histogram of failures')

plt.xlabel('Number of failures')

plt.xticks(range(5))

plt.xlim([0, 4])

plt.ylabel('Amount of students')

plt.show()



avg_failures = np.average(alcohol['failures'])

print("The average failures of a student in a year is:", round(avg_failures, 2), "failures.")
Corr_abs_fail = np.corrcoef(alcohol['failures'], alcohol['absences'])

print("Corrrelation coefficient absenses & failures:")

print(Corr_abs_fail)

Corr_fail_Dalc = np.corrcoef(alcohol['failures'], alcohol['Dalc'])

print("Corrrelation coefficient failures & Dalc:")

print(Corr_fail_Dalc)

Corr_fail_Walc = np.corrcoef(alcohol['failures'], alcohol['Walc'])

print("Corrrelation coefficient failures & Walc:")

print(Corr_fail_Walc)

Corr_abs_Dalc = np.corrcoef(alcohol['absences'], alcohol['Dalc'])

print("Corrrelation coefficient absences & Dalc:")

print(Corr_abs_Dalc)

Corr_abs_Walc = np.corrcoef(alcohol['absences'], alcohol['Walc'])

print("Corrrelation coefficient absences & Walc:")

print(Corr_abs_Walc)

Weekly_Alcohol = alcohol['Dalc'] + alcohol['Walc']



plt.hist(Weekly_Alcohol, bins=9) # create a histogram of total alcohol consumption with 9 bins

plt.ylabel('Number of Students')

plt.xlabel('Weekly alcohol consumption')

plt.title('Overview of the weekly alcohol consumption of students')

plt.yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])

plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

plt.show()
Corr_fail_Weekly = np.corrcoef(alcohol['failures'], Weekly_Alcohol)

print("Corrrelation coefficient failures & Weekly_Alcohol:")

print(Corr_fail_Weekly)

Corr_abs_Weekly = np.corrcoef(alcohol['absences'], Weekly_Alcohol)

print("Corrrelation coefficient absences & Weekly_Alcohol:")

print(Corr_abs_Weekly)


Higher = alcohol['higher'] 

Highernum = pd.Series(np.where(Higher.values == 'yes', 1, 0), Higher.index) 



add_count = alcohol['higher'].value_counts()

print(add_count)



plt.hist(Highernum, bins=3) 

plt.ylabel('Number of Students')

plt.xlabel('Wants higher education?')

plt.title('Overview of the amount of students who strive for higher education')

plt.yticks([50, 100, 150, 200, 250, 300, 350, 400])

plt.xticks([0, 1], ['No', 'Yes'])

plt.show()
Corr_Higher_Dalc = np.corrcoef(Highernum, alcohol['Dalc'])

print("Corrrelation coefficient higher & Dalc:")

print(Corr_Higher_Dalc)

Corr_Higher_Walc = np.corrcoef(Highernum, alcohol['Walc'])

print("Corrrelation coefficient higher & Walc:")

print(Corr_Higher_Walc)

Corr_Higher_Weekly = np.corrcoef(Highernum, Weekly_Alcohol)

print("Corrrelation coefficient higher & Weekly_Alcohol:")

print(Corr_Higher_Weekly)