import numpy as np

import matplotlib.pyplot as plt

marks_in_exam = np.random.randint(10, 100, 50) # randomly generated dataset

print('Marks in the exam:', marks_in_exam) 
print('Mean of the Marks:',np.mean(marks_in_exam)) # calculate the mean

print()  

plt.boxplot(marks_in_exam) #, meanline=True, showmeans=True 

Q1 = np.percentile(marks_in_exam, 25)

print('The 25 percentile is:', Q1)

Q3 = np.percentile(marks_in_exam, 75)

print('The 75 percentile is:', Q3)