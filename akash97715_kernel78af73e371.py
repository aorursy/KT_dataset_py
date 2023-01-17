# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
df.head()
df['total']=df['math score']+df['reading score']+df['writing score']
df.head()
df['total'].head()
df.info()
df['parental level of education'].value_counts()
df[df['total']>270].head()
y=df['total']
num_bins = 5
n, bins, patches = plt.hist(y, num_bins, facecolor='blue', alpha=1.0)
plt.show()
df.set_index("parental_level_of_education")
def grade(key):
    if (key>270 and key<300):
        return "best"
    elif (key>200 and key<270):
        return "good"
    elif (key>150 and key<200):
        return "satisfactory"
    else:
        return "not good"
    
df['grade']=df['total'].apply(grade)
df['grade'].head()
df.columns = [column_name.replace(" ", "_") for column_name in df.columns]
df.head()
df['parental_level_of_education'].unique()
df.set_index('grade')
mask1=df['grade']=='satisfactory'
mask2=df['parental level of education']=='some college'
import seaborn as sns
plt.subplot(1, 3, 1)
sns.distplot(df['math score'])

plt.subplot(1, 3, 2)
sns.distplot(df['reading score'])

plt.subplot(1, 3, 3)
sns.distplot(df['writing score'])

plt.suptitle('Checking for Skewness', fontsize = 18)
plt.show()

!pip install dabl
import dabl
plt.rcParams['figure.figsize'] = (18, 6)
plt.style.use('tableau-colorblind10')
plt.style.available
dabl.plot(df, target_col = 'math score')
#these are the list of styles which we can use to display our graph
plt.style.available
df.head()
#probablity of students scoring more than 50 marks in maths
total_students=df.shape
a=total_students[0]
students_more_than_50marks=df[df['math score']>50]
g=students_more_than_50marks.shape[0]
print("probablity of students scoring more than 50 marks in maths",g/a*100,"%")
#let passing marks be 35 in maths
import math
students_less_then_35marks=df[df['math score']<35]
b=students_less_then_35marks.shape[0]
h=b/a*100
print("percentage of students failed the maths exam {0:.2f}%".format(h))
#percentage of students scored more than 90% in maths
student_more_than_90marks=df[df['math score']>90]
q=student_more_than_90marks.shape[0]
w=q/a*100
print("Number of students score more than 90 marks{0:.2f}%".format(w))
#probablity of students scoring more than 50 marks in Reading
total_students=df.shape
a=total_students[0]
students_more_than_50marks=df[df['reading score']>50]
e=students_more_than_50marks.shape[0]
print("probablity of students scoring more than 50 marks in reading score",e/a*100,"%")
#let passing marks be 35 in reading
import math
students_less_then_35marks=df[df['reading score']<35]
y=students_less_then_35marks.shape[0]
u=y/a*100
print("percentage of students failed the reading exam{0:.2f}%".format(u))
student_more_than_90marks=df[df['reading score']>90]
i=student_more_than_90marks.shape[0]
l=i/a*100
print("Percentage of students got more than 90per marks in reading {0:.2f}%".format(l))
#probablity of students scoring more than 50 marks in writing
total_students=df.shape
a=total_students[0]
students_more_than_50marks=df[df['writing score']>50]
k=students_more_than_50marks.shape[0]
print("probablity of students scoring more than 50 marks in writing score",k/a*100,"%")
#students failed in writing
student_les_than_35marks=df[df['writing score']<35]
v=student_les_than_35marks.shape[0]
x=v/a*100
print("students failed in writing {0:.2f}%".format(x))

#student got more than 90per in writing
student_more_than_90per=df[df['writing score']>90]
sd=student_more_than_90per.shape[0]
sf=sd/a*100
print("student having more than 90 is {0:.2f}%".format(sf))
df['total score']=df['math score']+df['reading score']+df['writing score']

#we will check if total score column get added to our dataset!
df.head()
#Now the total marks is 300. we need to calculate the performance of student overall
students_scored_150_marksoverall=df[df['total score']>150]
fg=students_scored_150_marksoverall.shape[0]
fd=fg/a*100
print("percentage of students who scored more than 50 overall are {0:.2f}%".format(fd))

#percentage of students who failed in overall subject
student_less_than35=df[df['total score']<105]
bn=student_less_than35.shape[0]
bv=bn/a*100
print("percentage of students who failed overall {0:.2f}%".format(bv))
#percentage of students who score more than 90per overall
students_more_than_90per=df[df['total score']>270]
bc=students_more_than_90per.shape[0]
bx=bc/a*100
print("percentage of students who scored more than 90 per overall {0:.2f}%".format(bx))
#to calculate the sample mean first we will define seed as the sample will be constant in each iteration.
#NOTE:you can take any numerical values as seed
np.random.seed(10)
#Lets take 100 sample values from 1000 samples
sample_math_score=np.random.choice(df['math score'], size=100)
print("sample mean of the maths score",sample_math_score.mean())
fl=df['math score'].mean()
print("population mean of the maths score {0:.2f}".format(fl))
#to calculate the sample mean first we will define seed as the sample will be constant in each iteration.
#NOTE:you can take any numerical values as seed
np.random.seed(10)
#Lets take 100 sample values from 1000 samples
sample_reading_score=np.random.choice(df['reading score'], size=100)
print("sample mean of the maths score",sample_reading_score.mean())
f2=df['reading score'].mean()
print("population mean of the reading score {0:.2f}".format(fl))
#to calculate the sample mean first we will define seed as the sample will be constant in each iteration.
#NOTE:you can take any numerical values as seed
np.random.seed(10)
#Lets take 100 sample values from 1000 samples
sample_writing_score=np.random.choice(df['writing score'], size=100)
print("sample mean of the writing score",sample_writing_score.mean())
fl=df['writing score'].mean()
print("population mean of the writing score {0:.2f}".format(fl))
#to calculate the sample mean first we will define seed as the sample will be constant in each iteration.
#NOTE:you can take any numerical values as seed
np.random.seed(10)
#Lets take 100 sample values from 1000 samples
sample_total_score=np.random.choice(df['total score'], size=100)
print("sample mean of the total score",sample_total_score.mean())
fl=df['total score'].mean()
print("population mean of the total score {0:.2f}".format(fl))
# lets import the scipy package
import scipy.stats as stats
import math

# lets seed the random values
np.random.seed(10)

# lets take a sample size
sample_size = 1000
sample = np.random.choice(df['math score'],
                          size = sample_size)
sample_mean = sample.mean()

# Get the z-critical value*
z_critical = stats.norm.ppf(q = 0.95)  

 # Check the z-critical value  
print("z-critical value: ",z_critical)                                

# Get the population standard deviation
pop_stdev = df['math score'].std()  
# checking the margin of error
margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size)) 

# defining our confidence interval
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

# lets print the results
print("Confidence interval:",end=" ")
print(confidence_interval)
print("True mean: {}".format(df['math score'].mean()))
# lets import the scipy package
import scipy.stats as stats
import math

# lets seed the random values
np.random.seed(10)

# lets take a sample size
sample_size = 1000
sample = np.random.choice(df['reading score'],
                          size = sample_size)
sample_mean = sample.mean()

# Get the z-critical value*
z_critical = stats.norm.ppf(q = 0.95)  

 # Check the z-critical value  
print("z-critical value: ",z_critical)                                

# Get the population standard deviation
pop_stdev = df['reading score'].std()  
# checking the margin of error
margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size)) 

# defining our confidence interval
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

# lets print the results
print("Confidence interval:",end=" ")
print(confidence_interval)
print("True mean: {}".format(df['reading score'].mean()))
# lets import the scipy package
import scipy.stats as stats
import math

# lets seed the random values
np.random.seed(10)

# lets take a sample size
sample_size = 1000
sample = np.random.choice(df['writing score'],
                          size = sample_size)
sample_mean = sample.mean()

# Get the z-critical value*
z_critical = stats.norm.ppf(q = 0.95)  

 # Check the z-critical value  
print("z-critical value: ",z_critical)                                

# Get the population standard deviation
pop_stdev = df['writing score'].std()  
# checking the margin of error
margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size)) 

# defining our confidence interval
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

# lets print the results
print("Confidence interval:",end=" ")
print(confidence_interval)
print("True mean: {}".format(df['writing score'].mean()))
df[(df['gender'] == 'female') &
     (df['math score'] > 90) & 
     (df['writing score'] > 90) &
     (df['reading score'] > 90)]
df.groupby(['gender']).agg(['min','median','max','mean'])
df[['gender','lunch','math score','reading score','writing score']].groupby(['lunch','gender']).agg('median')
df.info()
df[['math score','reading score','writing score','parental level of education','total score']].groupby(['parental level of education']).agg('median')
df.info()
df[['reading score','writing score','math score','test preparation course']].groupby(['test preparation course']).agg('median')
g=df['gender'].value_counts()
g.values[0]
df.set_index(['test preparation course'])