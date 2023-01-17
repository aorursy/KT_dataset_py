import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")







%matplotlib inline 
math_data = pd.read_csv("../input/student-mat.csv")

por_data = pd.read_csv("../input/student-por.csv")



math_data.head()

por_data.head()
math_data.isnull().sum()
por_data.isnull().sum()
def is_alchololic(dataframe):

    if(dataframe['Walc'] >= 3):

        return(True)

    else:

        return(False)
math_data['is_alchololic'] = math_data.apply(lambda row: is_alchololic(row), axis = 1)



por_data['is_alchololic'] = por_data.apply(lambda row: is_alchololic(row), axis = 1)
def grade_average(dataframe):

    return((dataframe['G1']+dataframe['G2']+dataframe['G3'])/3 )

    
math_data['average_grade'] = math_data.apply(lambda row: grade_average(row), axis = 1)

math_data['average_grade'].plot(kind = "density")
por_data['average_grade'] = math_data.apply(lambda row: grade_average(row), axis = 1)

por_data['average_grade'].plot(kind = "density")
def max_parenteducation(dataframe):

    return(max(dataframe['Medu'], dataframe['Fedu']))



math_data['maxparent_education'] = math_data.apply(lambda row: max_parenteducation(row), axis = 1)

math_data['maxparent_education'].plot(kind = "density")
#Lets consider a plot between the student grades to how well educated any of the parent is!

sns.barplot(x = "maxparent_education", y = "average_grade", data = math_data)
fig, axs = plt.subplots(ncols=2)

sns.barplot(x = "studytime", y = "average_grade", data = math_data, ax=axs[0])

sns.barplot(x = "studytime", y = "average_grade", data = por_data, ax=axs[1])
#Girls vs Boys!!

sns.barplot(x = "sex", y = "average_grade",data = math_data)
sns.barplot(x = "sex", y = "average_grade", hue = "famsup", data = math_data)
#will there be any difference if we add sex into picture? That is, is there any difference between boys and girls performacne?

fig, axs = plt.subplots(ncols=2)

sns.barplot(x = "studytime", y = "average_grade",hue = "sex", data = math_data, ax=axs[0])

sns.barplot(x = "studytime", y = "average_grade",hue = "sex", data = por_data, ax=axs[1])
#study time and failures!

fig, axs = plt.subplots(ncols=2)

sns.barplot(x = "failures", y = "studytime",hue = "sex", data = math_data, ax=axs[0])

sns.barplot(x = "failures", y = "studytime",hue = "sex", data = por_data, ax=axs[1])
#Will there be any pattern from students age to their grades? 



sns.regplot(x="age", y="average_grade", data=math_data)
#Let's see the relationship between the alcoholism and average grade. 

sns.barplot(x = "is_alchololic", y = "average_grade", hue = "sex", data = math_data)
#What if parents were highly educated. How would that effect students alcohol comsumption?

fig, axs = plt.subplots(ncols=2)

sns.barplot(x = "is_alchololic", y = "maxparent_education", data = math_data, ax = axs[0])

sns.barplot(x = "is_alchololic", y = "maxparent_education", data = math_data, hue = "sex", ax = axs[1])
#lets see health and failures. 

fig, axs = plt.subplots(ncols=2)

sns.barplot(x = "is_alchololic", y = "health", data = math_data, ax = axs[0])

sns.barplot(x = "is_alchololic", y = "health", data = math_data, hue = "sex", ax = axs[1])
fig, axs = plt.subplots(ncols=2)

sns.barplot(x = "is_alchololic", y = "failures", data = math_data, ax = axs[0])

sns.barplot(x = "is_alchololic", y = "failures", data = math_data, hue = "sex", ax = axs[1])
#Now lets divert our attention to the attribute 'romantic'!



math_data.groupby('sex')['romantic'].value_counts().to_frame()
##Is it possible that if a person has more time then he has more time to be romantic or other wise? Let's find out!

fig, axs = plt.subplots(ncols=2)

sns.barplot(x = "romantic", y = "freetime", data = math_data, ax = axs[0])

sns.barplot(x = "romantic", y = "freetime", hue = "sex", data = math_data, ax = axs[1])