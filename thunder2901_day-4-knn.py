import pandas as pd

student = pd.read_csv("../input/student.csv")

student.shape
student = student.drop('Name',axis='columns')
student
test_data = [6.5 ,1]
import numpy as np



lstDist = []



for index in student.index:

    distance = np.sqrt((test_data[0] - student['Aptitude'][index])**2 + (test_data[1] - student['Communication'][index])**2)

    lstDist.append([distance, student['Class'][index]])

    

df = pd.DataFrame(lstDist,columns=['Distance','Class'])

df_sorted = df.sort_values('Distance')

n = 4

df_sorted.head(n)