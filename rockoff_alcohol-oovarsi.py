# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Import matplotlib.pyplot with the alias plt

import seaborn as sns # Import seaborn with the alias sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Load csv file of Students learning Portuguese as a language 

por = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-por.csv")

por
#Load csv file Students learning Maths as a language 

mat = pd.read_csv("/kaggle/input/student-alcohol-consumption/student-mat.csv")

mat
#Merge students studying both portuguese and math

result = por.merge(mat[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]], how = 'inner',indicator=False)

result
#add a subject column

mat['subject'] = 'math'

por['subject'] = 'portugese'

result['subject'] = 'math/portugese'
#Combine everything in a single dataframe

frame = [mat, por, result]

df= pd.concat(frame)

df_no_dup = df.drop_duplicates(subset=list(por.columns[:-5]), keep='last')
def binary_counter(df, column):

    """

    Takes a dataframe and a column with count variables and shows the 

    proportions in a series in percent.

    """

    return df[column].value_counts(normalize=True).mul(100)



def filter_bar(df, column1, column2):

    """

    Takes 3 arguments :

    1-The dataframe to operate on

    2-The first column containing binary values like yes/no or M/F

    3-The second column containing more than 2 values 

    The function returns a stacked bar plot.

    """

    word_filter = list(df[column1].unique())[0]

    df_filter = df[column1].str.contains(word_filter)

    s1 = df.loc[df_filter, column2].value_counts()

    s2 = df.loc[~df_filter, column2].value_counts()

    new_df = pd.DataFrame([s1, s2])

    new_df.index = list(df[column1].unique())

    return new_df.plot.bar(stacked=True)



def plot_1 (col1, col2, df):

    fatherJobsList = list(df.col1.unique())

    studentDalc = []

    for each in fatherJobsList:

        x = result[result.col1 == each]

        studentDalc.append(sum(x.col2))

    return studentDalc
list_columns = df_no_dup.columns.values



# Prints columns and their unique values

for column_name in list_columns:

    print("Column", column_name)

    print(df_no_dup[column_name].unique())
filter_bar(mat, 'sex', 'Walc').set_title('Weekend consumption per gender')
#Weekly Consumed alcohol as per father's job

import matplotlib.pyplot as plt

import seaborn as sns

fatherJobsList = list(df_no_dup.Fjob.unique())

studentDalc = []

for each in fatherJobsList:

    x = result[result.Fjob == each]

    studentDalc.append(sum(x.Dalc))



#Visualitizon    

plt.figure(figsize=(15,5))

sns.barplot(x=studentDalc,y=fatherJobsList)

plt.xlabel("Weekly Alcohol Consumption")

plt.ylabel("Father Jobs")

plt.show()
#Age range consuming alcohol as per Mother's job

# x, y, hue: names of variables in data or vector data, optional

plt.figure(figsize=(15,8))

sns.countplot(x = 'Mjob', hue = 'age', data = result, palette = 'magma')
#Number of Students from low to high consumer on worday and weekend



#marker - big dot

l=[1,2,3,4,5] #Alcohol consumption levels from 1 - very low to 5 - very high

labels= "1-Very Low","2-Low","3-Medium","4-High","5-Very High"

plt.figure(figsize=(15,5))

plt.plot(labels,list(map(lambda l: list(result.Dalc).count(l),l)),color="red",linestyle="--",marker="o", markersize=10,label="Workday")

plt.plot(labels,list(map(lambda l: list(result.Walc).count(l),l)),color="green",linestyle="--",marker="o", markersize=10,label="Weekend")

plt.title("Student Alcohol Consumption")

plt.grid()

plt.ylabel("Number of Students")

plt.legend()

plt.show()
mat.corr()

f,ax=plt.subplots(figsize=(18,10))

sns.heatmap(mat.corr(),annot=True,linewidth=0.5,fmt='.3f',ax=ax)

plt.show() #Shows correlation between different variables

# Alcohol consumption shows a string correlation with students going outside with their friends
# Which bool variable has a higher impact on the notes ? 

plt.rcParams['figure.figsize'] = (15, 5)

# For each boolean value, calculate means (G1,G2,G3) in given DataFrame:

def grades_year_mean_binaries (binary_attribute,df) :

    binary_value = list(df[binary_attribute].unique())

    l1 = []

    if len(binary_value) == 2 :

        for i in binary_value :

            mean_i1 = df.loc[df[binary_attribute].str.contains(i),"G1"].mean()

            mean_i2 = df.loc[df[binary_attribute].str.contains(i),"G2"].mean()

            mean_i3 = df.loc[df[binary_attribute].str.contains(i),"G3"].mean()

            mean_i = (mean_i1+mean_i2+mean_i3)/3 # YEAR MEAN

            l1.append(mean_i) # to save the values

    return {binary_attribute: l1[0]-l1[1]} # difference in means 



# Calculate means for every bool colulmn : 

def all_columns (given_df):

    d1 = {}

    for j in ['school','sex','address','famsize','Pstatus', 'schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'] : # manually given, didn't find a better way (Andrés)

        d1.update(grades_year_mean_binaries(j,given_df))

        print (j,given_df[j].unique())

    return(d1)



plt.bar(all_columns(por).keys(), all_columns(por).values())

plt.title("Difference in Mean Grade for Binary Variables (Portuguese Grades)")

plt.yticks(np.arange(-1.5,3.5,0.5))

plt.grid(axis='y')



# This figure gives, from the binary variables, those who may have the higher impact on the grades (it shows tendencies, not rules) :

    # The higher the grade difference, the higher the impact. 

    # If math and portuguese graphs are consistent, the tendency is greater.



# Conclusion 1 : Students that want to do higher studies are 3.2 points above the others ! (positive value in graph = grade higher for the value on the left of the list)

# Conclusion 2 : Students following school support may have lower grades

# Conclusion 3 : 'school'=GP, 'address'=U and internet=yes seem to be correlated to higher grades (in a lesser extent)
plt.bar(all_columns(mat).keys(), all_columns(mat).values())

plt.title("Difference in Mean Grade for Binary Variables (Maths Grades)")

plt.yticks(np.arange(-1.5,3.5,0.5))

plt.grid(axis='y')
mineur = mat[(mat.age < 18)]

mineur.describe()
l=[1,2,3,4,5]

label= "1-Very Low","2-Low","3-Medium","4-High","5-Very High"

mineur_workday=list(map(lambda l: list(mineur.Dalc).count(l),l))

mineur_weekend=list(map(lambda l: list(mineur.Walc).count(l),l))

plt.figure(figsize=(10,5))

plt.plot(label,mineur_workday,color="red",linestyle="--",marker="o", markersize=10,label="Workday")

plt.plot(label,mineur_weekend,color="green",linestyle="--",marker="o", markersize=10,label="Weekend")

plt.title("Age less than 18 Student Alcohol Consumption")

plt.ylabel("Number of Students")

plt.legend()

plt.show()
majeur = mat[(mat.age >= 18)]

majeur.describe()
majeur_workday=list(map(lambda l: list(majeur.Dalc).count(l),l))

majeur_weekend=list(map(lambda l: list(majeur.Walc).count(l),l))

plt.figure(figsize=(10,5))

plt.plot(label,age15_workday,color="red",linestyle="--",marker="o", markersize=10,label="Workday")

plt.plot(label,age15_weekend,color="green",linestyle="--",marker="o", markersize=10,label="Weekend")

plt.title("Age more than 18 Student Alcohol Consumption")

plt.ylabel("Number of Students")

plt.legend()

plt.show()
filter_bar(mat, 'internet', 'Walc').set_title('Weekend consumption per Internet availability at home')
filter_bar(mat, 'romantic', 'Walc').set_title('Weekend consumption per romantic relationship status')