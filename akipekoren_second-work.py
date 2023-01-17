# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
myDataFrame=pd.read_csv( "../input/StudentsPerformance.csv"
)
myDataFrame.head(10)
myDataFrame.info()
myDataFrame.columns=[each.split()[0]+ "_" +each.split()[1] if  (len(each.split())>1) else each for each in myDataFrame]

myDataFrame.columns





filter1=myDataFrame.math_score>90
filter2=myDataFrame.reading_score>90
filter3=myDataFrame.writing_score>90
successfullStudentData=myDataFrame[filter1&filter2&filter3]
successfullStudentData

successfullStudentData.info()
successfullStudentData.math_score.plot(kind="line", alpha=0.5, color="red" , label="math_score")
successfullStudentData.writing_score.plot( alpha=0.5 , color="blue" , label="writing_score")
successfullStudentData.reading_score.plot(alpha=0.5, color="green",label="reading_score")
plt.title("Line plot")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()








successfullStudentData

successfullStudentData.math_score=successfullStudentData.math_score - 5

        

successfullStudentData
def func1(x):
    return x**2
successfullStudentData.math_score.apply(func1)

successfullStudentData.math_score=successfullStudentData.math_score**(1/2.0)
successfullStudentData

z=zip(successfullStudentData.math_score , successfullStudentData.reading_score, successfullStudentData.writing_score)
z_list=list(z) # We can show all three scores in a list together by using zip function.
print(z_list)
successfullStudentData.reading_score

newReadingScore=[each-2 if each>95 else each+2  for each in successfullStudentData.reading_score]
successfullStudentData.reading_score=newReadingScore
successfullStudentData     #List compheresion example

successfullStudentData.reading_score.plot(kind="hist" , bins=50 , figsize=(8,8))
plt.show()
successfullStudentData.describe()
print(successfullStudentData["parental_level"].value_counts(dropna=False)) ## Check is there any NaN row on the data
successfullStudentData.boxplot(column="writing_score", by="test_preparation")
differentData={"gender" : "female","race/ethnicity" : "group C" , "parental_level" : "bachelor's degree",
              "lunch" : "free/reduced",  "test_preparation" : "completed", "math_score" : 9 ,"reading_score" : 95,
              "writing_score" : 95}
df=pd.DataFrame([differentData], columns=differentData.keys()) # convert dictionary to data frame

data1=successfullStudentData.head()
data2=successfullStudentData.tail()
conc_data_row=pd.concat([data1, data2, df] , axis=0 , ignore_index=True) #CONCATENATING THREE DATA
conc_data_row

successfullStudentData["math_score"]=successfullStudentData["math_score"].astype("int")
successfullStudentData
assert  successfullStudentData['gender'].notnull().all() #returns nothing, we don't have a missing value