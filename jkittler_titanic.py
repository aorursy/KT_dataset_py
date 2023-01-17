#Import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.stats.api as sms

import scipy

import scipy.stats
#Import the trainingset

df = pd.read_csv("../input/train.csv", sep=",")
#How does the dataset look like?

df.head()
#What are characteristics of the variables?

df.info()
#How many rows and columns do we have?

num_passengers = len(df.index)

print("num_rows =", num_passengers) # 891 rows

print("num_columns =",len(df.columns)) # 12 columns



#What are the column names?

column_names = df.columns.values

print("column_names = ", column_names)
#How many people survived?

num_survived = np.sum(df["Survived"])

print("num_survived = ",num_survived) #342 people survived



#How many people died?

num_died = len(df.index)-np.sum(df["Survived"])

print("num_died = ", num_died) #549 people died



#What is the baseline for people who died/survived (our classification models should do better than this)?

baseline_died = num_died/num_passengers

baseline_survived = num_survived/num_passengers

print("baseline_died = ", baseline_died) # 0.6161

print("baseline_survived = ", baseline_survived) #0.3838
#Can we identify any noticable differences in the quantitative variables

#just by observation?



df.ix[:, df.columns != 'PassengerId'].groupby("Survived").mean()



#Yes, it seems as if there is a difference between Pclass (survived>died), Age (survived<died), 

#Fare(survived>died), Number of parents/chidlren on board (survived>died), Number of siblings/spouses 

#on board (survived<died).
#Are these differences statistically significant?



# Create a subset with all people who died and with all people who survived:

df_survived = df[df["Survived"]==1]

df_died = df[df["Survived"]==0]



#Let us conduct an independent t-tests and a confidence interval for the variable "Pclass"



print("Pclass","\n")

#Indepdendent t-test

print("t-test: ",scipy.stats.ttest_ind(df_survived.Pclass, df_died.Pclass, equal_var=False))

# Ttest_indResult(statistic=-10.336953406118893, pvalue=2.9111554993758305e-23)

# Result is statistically significant with p<.001



#Confidence interval

ci_Pclass = sms.CompareMeans(sms.DescrStatsW(df_survived.Pclass),sms.DescrStatsW(df_died.Pclass))

print("CI: ",ci_Pclass.tconfint_diff(usevar='unequal'),"\n")

# We are 95% confident that people who survived had - on average - a PClass that is between 0.6921

# and 0.4711 smaller (better) than the PClass of the people who died.



#Now lets do the same thing for the remaining four variables Age, SibSp, Parch, Fare



remaining_variables = ['Age', 'SibSp', 'Parch', 'Fare']



for variable in remaining_variables:

    print(variable,"\n")

    print("t-test: ",scipy.stats.ttest_ind(df_survived[variable], df_died[variable], equal_var=False),"\n")

    variable = sms.CompareMeans(sms.DescrStatsW(df_survived[variable]),sms.DescrStatsW(df_died[variable]))

    print("CI: ",variable.tconfint_diff(usevar='unequal'),"\n")

    

#Conclusion:

    #All differences are statistically significant, except for SibSp and Age.

    #We will need to remove NAs for Age.
#How about the remaining, the qualitative, variables?



#Create subset with all remaining, qualitative variables

qualitative_variables = df.drop(['PassengerId','Pclass','Age', 'SibSp', 'Parch', 'Fare'], axis = 1, errors = 'ignore')



#How does the subset look like?

qualitative_variables.head()
# Can we identify any interesting pattern with respect to the variable "Names" by observation?



print(qualitative_variables['Name'])    # No! All names seem to be different at the first glance.

                                        # Maybe, one could extract "Miss"/"Mrs" later on and see if that

                                        # provides any relevant information. 
# Can we identify any interesting pattern with respect to the variable "Ticket" by observation?



print(qualitative_variables['Ticket']) # We don't know yet. This variable will have to be cleaned.

                                       # Maybe there is a relation between ticketnumber and the class.

                                       # However, some instances also have letters, wich we have to remove.
#What about "Sex"



df.groupby("Sex").count() #Looking at the PassengerId column, we know that there are 314 female and 

                            #577 male passengers



df_male = df[df["Sex"]=="male"]

df_female = df[df["Sex"]=="female"]



import numpy as np

import matplotlib.pyplot as plt



N = 5

menMeans = (20, 35, 30, 35, 27)

menStd = (2, 3, 4, 1, 2)



ind = np.arange(N)  # the x locations for the groups

width = 0.35       # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)



womenMeans = (25, 32, 34, 20, 25)

womenStd = (3, 5, 2, 3, 3)

rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)