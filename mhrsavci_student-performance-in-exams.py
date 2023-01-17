# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in :



import numpy as np # Linear Algebra

import pandas as pd # Data Processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data Visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls","../input"]).decode("utf-8"))



# Any results you write to the current directory are saved as output.

# StudentsPerformance.csv file defined as dataset



sperformance=pd.read_csv("../input/StudentsPerformance.csv")
# sperformance can study dataset which columns exist in the data table and what are the data types and numbers of these columns.



sperformance.info()
sperformance.describe()
# Return sperformance's column name and queue.



count=0

for item in sperformance.columns:

    count+=1

    print("{0}. Column Name -> {1}".format(count,item))
# The data types which include in columns of sperformance are returned. 



dataTypes=pd.DataFrame(sperformance.dtypes)

dataTypes
# Histogram Plot





# sperformance["math score"].plot(kind="hist",bins=50,grid=True)



sperformance["math score"].plot.hist(color="r",bins=50,figsize=(15,10))

plt.title("Math Score Graph")

plt.ylabel("Counts")

plt.xlabel("Math Score")

plt.grid(axis="y")

plt.text(40,45,r"Math Scores") # x=40 and y=45.

sperformance.head(10) # Return top 10 row in sperformance data frame.
sperformance.tail() # Return bottom 5 row in sperformance data frame.
# Line Plot







sperformance["reading score"].plot(kind="line",color="g",label="Reading Score",linestyle=":",figsize=(15,15))



plt.legend(loc="upper right")

plt.xlabel("Student Number")

plt.ylabel("Reading Score")

plt.title("Student Reading Score")

plt.show()

# Scatter Plot





sperformance.plot(kind="scatter",x="reading score",y="writing score",color="orange",figsize=(10,10))



plt.title("Reading & Writing Score Scatter Plot")

plt.show()

# List Filtering Numerical



highMathScore=sperformance["math score"]>85



highReadingScore=sperformance["reading score"]>85



highWritingScore=sperformance["writing score"]>85



highMath=sperformance[highMathScore].head(10) # Return high math scorer students.(top 10 row)



highmathANDreading=sperformance[highMathScore & highReadingScore].head(10) # Return high math AND reading scorer students. (top 10 row)



highmathORwriting=sperformance[highMathScore | highWritingScore].head(10) # Return high math OR writing scorer students. (top 10 row)



print(highmathANDreading)

# List Filtering For str





lunchFilter=sperformance["lunch"]=="standard"



sperformance[lunchFilter].head(20) # Apply lunchFilter on sperformance and return top 20 row.
# List Comprehension for math score



# del sperformance["Math Degree"] # Delete from sperformance column "Math Degree"



sperformance["Math Degree"] = ["Passed" if i>=50 else "FailInCourse" for i in sperformance["math score"]]



sperformance.head(10)

# List Comprehension for reading score



sperformance["Reading Degree"] = ["Passed" if i>=50 else "FailInCourse" for i in sperformance["reading score"]]



sperformance.head(10)

# List Comprehension for writing score



sperformance["Writing Degree"] = ["Passed" if i>=50 else "FailInCourse" for i in sperformance["writing score"]]



sperformance.head(10)

# Average Value for math, reading and writing score



averageMathScore=sperformance["math score"].mean()



averageReadingScore=sperformance["reading score"].mean()



averageWritingScore=sperformance["writing score"].mean()

# Average Value Columns



#del sperformance["Average Score"] # Delete Column in DataFrame

sperformance["Average Score"]=(sperformance["math score"]+sperformance["math score"]+sperformance["math score"])/3



sperformance.head()

# Concatenating Data



mathScore=sperformance["math score"]

readingScore=sperformance["reading score"]

writingScore=sperformance["writing score"]

averageScore=sperformance["Average Score"]



student_performance_scores=pd.concat([mathScore,readingScore,writingScore,averageScore],axis=1)



student_performance_scores.head()

# Box Plot 



sperformance.boxplot(column="math score") # Return average value in math scores with graphic.
# Count by value group in table's column.



# sperformance["gender"].value_counts() # Return group in  "gender" count by gender type.



ParentalEducation=sperformance["parental level of education"].value_counts()



ParentalEducation=pd.DataFrame(ParentalEducation)



ParentalEducation
# Apply Function on DataFrame Column



CoNumber=pd.DataFrame(sperformance["Average Score"].apply(lambda n:n/100))



CoNumberList=[]

for item in CoNumber["Average Score"]:

    CoNumberList.append(item)



CoNumberList = sorted(CoNumberList, key=float) # Sort the average values by float type.

sperformance.head()
# Grouping of column value number by "race/ethnicity"



sperformance["race/ethnicity"].value_counts()
# Grouping of column value number by "lunch"



sperformance["lunch"].value_counts()
# Grouping of column value number by "test preparation course"



sperformance["test preparation course"].value_counts()
# Doing some correction "test preparation course" column.



sperformance["test preparation course"]=sperformance["test preparation course"].replace("none","Not Completed")



sperformance["test preparation course"]=sperformance["test preparation course"].replace("completed","Completed")
# Get Index Column Name and Define Index Column Name



sperformance.index.name="id" # Define index name as "id" in "sperformance"



sperformance.head()



sperformance.columns=[item[0].upper()+item[1:] for item in sperformance.columns]



sperformance.head()
sp = sperformance.copy()
# Group by "Race/ethnicity"



sp.groupby("Race/ethnicity").mean()

# Group by "Parental level of education"



GroupParentalEdu=sp.groupby("Parental level of education").mean()

# Group Parental Education Line Plot



GroupParentalEdu["Math score"].plot(kind="line",label="Math Score",figsize=(25,25),linestyle=":")



GroupParentalEdu["Reading score"].plot(kind="line",label="Reading Score",figsize=(25,25),linestyle=":",color="g")



GroupParentalEdu["Writing score"].plot(kind="line",label="Writing Score",figsize=(25,25),linestyle=":",color="orange")



GroupParentalEdu["Average Score"].plot(kind="line",label="Average Score",figsize=(25,25),linestyle=":",color="r")



plt.xlabel("Race/Ethnicity")

plt.show()

# Group by "Lunch"



sp.groupby("Lunch").mean()

# Group by "Gender"



sp.groupby("Gender").mean()
sp.head()
# High School vs Master's Degree Parental Level of Education





EducationHighSchool=sp["Parental level of education"]=="high school"



EducationMasterDegree=sp["Parental level of education"]=="master's degree"



sp[EducationHighSchool]["Average Score"].plot(kind="line",linestyle=":",figsize=(20,10))

sp[EducationMasterDegree]["Average Score"].plot(linestyle="-.",color="r")



plt.show()


# Fail In Course Student has parental level of education







FailWritingStu=sp["Writing Degree"]=="FailInCourse"



FailReadingStu=sp["Reading Degree"]=="FailInCourse"



FailMathStu=sp["Math Degree"]=="FailInCourse"



FailInStudent=sp[FailWritingStu & FailReadingStu & FailMathStu]



FailStuParEdu=FailInStudent["parental level of education"].value_counts()



FailStuParEdu=pd.DataFrame(FailStuParEdu)



FailStuParEdu



sp.set_index(["race/ethnicity"]) # Set index column "race/ethnicity"