import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt



normal_df = pd.read_csv('C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/NormalSample.csv')



#Write a Python program to calculate the density estimator of a histogram.  Use the field x in the NormalSample.csv file. 

#a)	(5 points) According to Izenman (1991) method, what is the recommended bin-width for the histogram of x?

q1 = np.percentile(normal_df.x, 25)

q3 = np.percentile(normal_df.x, 75)

IQR = q3-q1

print("Q1: ", q1)

print("Q3: ", q3)

print("IQR: ", IQR) 



#N =  total number of datapoints in the field x

N = len(normal_df)

N_hat = N**(-1/3)

print("N: ", N)

print("N_hat: ", N_hat)



bin_width = 2*IQR*N_hat

h = round(bin_width,1)

print("Recommended bin-width for the histogram of x",h)
#b)	(5 points) What are the minimum and the maximum values of the field x? 



x = list(normal_df.x)

print("Minimum Value is: ", min(x))

print("Maximum Value is: ", max(x))

#Minimum value : 30.40

#Maximum value : 35.40

#d)	(5 points) Use h = 0.1, minimum = a and maximum = b. List the coordinates of the density estimator.  

#Paste the histogram drawn using Python or your favorite graphing tools. 



#List the coordinates of the density estimator. 



df = pd.read_csv('C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/NormalSample.csv', header = 0)



x = list(df.x)



#Number of bins:

b = 36

a = 26

bin_no = (b-a)/0.1

print("Number of bins: ",bin_no) 



h = 0.1



mid01 = h/2

 

n = np.arange(26.0,36.01, mid01) 



mid = []

for i in range(0,len(n),1):

    if i%2 != 0:

        mid.append(n[i]) 



N = len(x)

Nh_01 = N*h

p = {}

pl = []



for m in mid:

    w = []

    for i in x:

        u = (i-m)/h

        if u > -0.5 and u <= 0.5:

            w.append(1)

    p[m] = sum(w)/Nh_01

    pl.append(sum(w)/Nh_01)



new_df01 = pd.DataFrame(list(zip(mid, pl)), 

               columns =['Midpoints', 'Density Estimate']) 



print("Desnsity  Estimators are: ")

print(new_df01)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

plt.step(mid,pl, where= 'mid', label = 'h=0.1')

plt.legend()

plt.grid(True)

plt.xticks(np.arange(26, 36, 0.5))

plt.xlabel('Mid-points')

plt.ylabel('Density Estimators')

plt.show()
#e)	(5 points) Use h = 0.5, minimum = a and maximum = b. List the coordinates of the density estimator.  

#Paste the histogram drawn using Python or your favorite graphing tools. 





h = 0.5

mid05 = h/2 



#Number of bins:

b = 36

a = 26

bin_no = (b-a)/h

print("Number of bins: ",bin_no) 

 

n = np.arange(26.0,36.01, mid05) 



mid = []

for i in range(0,len(n),1):

    if i%2 != 0:

        mid.append(n[i])

    

N = len(x)

Nh_05 = N*h

p = {}

pl = []



for m in mid:

    w = []

    for i in x:

        u = (i-m)/h

        if u > -0.5 and u <= 0.5:

            w.append(1)

    p[m] = sum(w)/Nh_05

    pl.append(sum(w)/Nh_05)





new_df05 = pd.DataFrame(list(zip(mid, pl)), 

               columns =['Midpoints', 'Density Estimate'])  



print("Desnsity  Estimators are: ")

print(new_df05)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

plt.step(mid,pl, where= 'mid', label = 'h=0.5')

plt.legend()

plt.grid(True)

plt.xticks(np.arange(26, 36, 0.5))

plt.xlabel('Mid-points')

plt.ylabel('Density Estimators')

plt.show() 



#f)	(5 points) Use h = 1, minimum = a and maximum = b. List the coordinates of the density estimator.

#Paste the histogram drawn using Python or your favorite graphing tools. 



h = 1

mid1 = h/2 



#Number of bins:

b = 36

a = 26

bin_no = (b-a)/h

print("Number of bins: ",bin_no) 

 

n = np.arange(26.0,36.01, mid1) 



mid = []

for i in range(0,len(n),1):

    if i%2 != 0:

        mid.append(n[i])

    

N = len(x)

Nh_1 = N*h

p = {}

pl = []



for m in mid:

    w = []

    for i in x:

        u = (i-m)/h

        if u > -0.5 and u <= 0.5:

            w.append(1)

    p[m] = sum(w)/Nh_1

    pl.append(sum(w)/Nh_1)





new_df1 = pd.DataFrame(list(zip(mid, pl)), 

               columns =['Midpoints', 'Density Estimate']) 

print("Desnsity  Estimators are: ")

print(new_df1)

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

plt.step(mid,pl, where= 'mid', label = 'h=1')

plt.legend()

plt.grid(True)

plt.xticks(np.arange(26, 36,1))

plt.xlabel('Mid-points')

plt.ylabel('Density Estimators')

plt.show() 

#g)	(5 points) Use h = 2, minimum = a and maximum = b. List the coordinates of the density estimator.

#Paste the histogram drawn using Python or your favorite graphing tools. 



h = 2

mid2 = h/2 



#Number of bins:

b = 36

a = 26

bin_no = (b-a)/h

print("Number of bins: ",bin_no) 



 

n = np.arange(26.0,36.01, mid2) 



mid = []

for i in range(0,len(n),1):

    if i%2 != 0:

        mid.append(n[i])

    

N = len(x)

Nh_2 = N*h

p = {}

pl = []



for m in mid:

    w = []

    for i in x:

        u = (i-m)/h

        if u > -0.5 and u <= 0.5:

            w.append(1)

    p[m] = sum(w)/Nh_2

    pl.append(sum(w)/Nh_2)





new_df2 = pd.DataFrame(list(zip(mid, pl)), 

               columns =['Midpoints', 'Density Estimate']) 

print("Desnsity  Estimators are: ")

print(new_df2)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))

plt.step(mid,pl, where= 'mid', label = 'h=2')

plt.legend()

plt.grid(True)

plt.xticks(np.arange(26, 36, 1))

plt.xlabel('Mid-points')

plt.ylabel('Density Estimators')

plt.show() 

#Question 2 (20 points)

#Use in the NormalSample.csv to generate box-plots for answering the following questions.

#a)	(5 points) What is the five-number summary of x?  What are the values of the 1.5 IQR whiskers? 



print(normal_df.x.describe())

 

q1 = np.percentile(normal_df.x, 25)

q3 = np.percentile(normal_df.x, 75)

IQR = q3-q1

print("Q1: ", q1)

print("Q3: ", q3)

print("IQR: ", IQR)  

Whisker_lower = q1 - 1.5*(IQR)

Whisker_upper = q3 + 1.5*(IQR)



print("Lower Whisker: ",Whisker_lower)

print("Upper Whisker: ",Whisker_upper)  
#b)	(5 points) What is the five-number summary of x for each category of the group? 

#What are the values of the 1.5 IQR whiskers for each category of the group?



#Answer: five-number summary of x for category = 0 of the group:

print("Five-number summary of x for category = 0 of the group:")

df_group0 = normal_df[normal_df.group == 0]

print(df_group0.x.describe()) 



print("1.5 IQR whiskers for x with category = 0: ") 

q10 = np.percentile(df_group0.x, 25)

q30 = np.percentile(df_group0.x, 75)

IQR0 = q30-q10

print("Q1: ", q10)

print("Q3: ", q30)

print("IQR: ", IQR0)  

Whisker_lower0 = q10 - 1.5*(IQR0)

Whisker_upper0 = q30 + 1.5*(IQR0)



print("Lower Whisker: ",Whisker_lower0)

print("Upper Whisker: ",Whisker_upper0)



#Answer: five-number summary of x for category = 1 of the group:

print("\n\nFive-number summary of x for category = 1 of the group:")

df_group1 = normal_df[normal_df.group == 1]

print(df_group1.x.describe()) 



print("1.5 IQR whiskers for x with category = 1: ") 

q11 = np.percentile(df_group1.x, 25)

q31 = np.percentile(df_group1.x, 75)

IQR1 = q31-q11

print("Q1: ", q11)

print("Q3: ", q31)

print("IQR: ", IQR1)  

Whisker_lower1 = q11 - 1.5*(IQR1)

Whisker_upper1 = q31 + 1.5*(IQR1)



print("Lower Whisker: ",Whisker_lower1)

print("Upper Whisker: ",Whisker_upper1)

#c)	(5 points) Draw a boxplot of x (without the group) using the Python boxplot function.  

#Can you tell if the Python’s boxplot has displayed the 1.5 IQR whiskers correctly? 



#Answer: from the boxplot it can be seen that the Python’s 

#boxplot has displayed the 1.5 IQR whiskers correctly.

#This can be verified from maually calculated values. 

#The package used to plot the box plot is seaborn and the whisker value taken by default is 1.5

#Whiskers manually calculated are:

#Lower Whisker:  27.599999999999994

#Upper Whisker:  32.400000000000006 

#These values are reflected from the boxplot.



import seaborn as sns  

x_plt = sns.boxplot(x="x" , data = df) 



#d)	(5 points) Draw a graph where it contains the boxplot of x, the boxplot of x for each category

#of Group (i.e., three boxplots within the same graph frame). Use the 1.5 IQR whiskers, identify 

#the outliers of x, if any, for the entire data and for each category of the group.

#Hint: Consider using the CONCAT function in the PANDA module to append observations. 



#Answer: Concat 3 subsets of dataframe: the boxplot of x, the boxplot of x for each category

#of Group 



import seaborn as sns 

from matplotlib import pyplot as plt

import seaborn as sns



x = df.x

df_group0 = normal_df[normal_df.group == 0].x

df_group1 = normal_df[normal_df.group == 1].x 



combined_x = pd.concat([x, df_group0, df_group1], axis = 1, keys=['x', 'df_group0', 'df_group1'])



sns.boxplot(data=combined_x) 



#Outliers of x,and entire dataframe for each category of the group.



q1 = np.percentile(normal_df.x, 25)

q3 = np.percentile(normal_df.x, 75)

IQR = q3-q1  

Whisker_lower = q1 - 1.5*(IQR)

Whisker_upper = q3 + 1.5*(IQR)



#Outliers of x are:

for xx in x:

    if xx > Whisker_upper:

        print("\nOutliers in Upper half of x are: ", xx) 

    else:

        pass

        

for xx in x:

    if xx < Whisker_lower:

        print("\nOutliers in Lower half of x are: ", xx)

    else:

        pass 

    
#Outliers of x,and entire dataframe for group = 0.

df_group0 = normal_df[normal_df.group == 0]

print("1.5 IQR whiskers for x with category = 0: ") 

q10 = np.percentile(df_group0.x, 25)

q30 = np.percentile(df_group0.x, 75)

IQR0 = q30-q10

print("Q1: ", q10)

print("Q3: ", q30)

print("IQR: ", IQR0)  

Whisker_lower0 = q10 - 1.5*(IQR0)

Whisker_upper0 = q30 + 1.5*(IQR0)



print("Lower Whisker: ",Whisker_lower0)

print("Upper Whisker: ",Whisker_upper0) 



# Outliers of x are:

for xx in df_group0.x:

    if xx > Whisker_upper0:

        print("\nOutliers in Upper half of x for group = 0 are: ", xx) 

    else:

        pass

        

for xx in df_group0.x:

    if xx < Whisker_lower0:

        print("\nOutliers in Lower half of x for group = 0 are: ", xx)

    else:

        pass 
#Outliers of x,and entire dataframe for group = 1.

df_group1 = normal_df[normal_df.group == 1]

 

print("1.5 IQR whiskers: ") 

q11 = np.percentile(df_group1.x, 25)

q31 = np.percentile(df_group1.x, 75)

IQR1 = q31-q11

print("Q1: ", q11)

print("Q3: ", q31)

print("IQR: ", IQR1)  

Whisker_lower1 = q11 - 1.5*(IQR1)

Whisker_upper1 = q31 + 1.5*(IQR1)



print("Lower Whisker: ",Whisker_lower1)

print("Upper Whisker: ",Whisker_upper1) 





#Outliers of x are:

for xx in df_group1.x:

    if xx > Whisker_upper1:

        print("\nOutliers in Upper half of x for group = 1 are: ", xx) 

    else:

        pass

        

for xx in df_group1.x:

    if xx < Whisker_lower1:

        print("\nOutliers in Lower half of x for group = 1 are: ", xx)

    else:

        pass

#a)	(5 points) What percent of investigations are found to be fraudulent?  

#Please give your answer up to 4 decimal places. 



Fraud_df = pd.read_csv('C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/Fraud.csv') 

tot = len(list(Fraud_df.FRAUD))

true_fraud = len(Fraud_df[Fraud_df.FRAUD == 1].FRAUD)

fraud_percent = round((true_fraud/tot)*100,4)

print("Percent of fraudulent data: ",fraud_percent)
#b)	(5 points) Use the BOXPLOT function to produce horizontal box-plots. 

#For each interval variable, one box-plot for the fraudulent observations, and 

#another box-plot for the non-fraudulent observations.  

#These two box-plots must appear in the same graph for each interval variable.



import seaborn as sns

sns.boxplot(data = Fraud_df, x = 'TOTAL_SPEND', y = 'FRAUD', orient = 'h')

plt.title("Total Spent")

plt.savefig("C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/TotalSpent.png")
import seaborn as sns

sns.boxplot(data = Fraud_df, x = 'DOCTOR_VISITS', y = 'FRAUD', orient = 'h')

plt.title("Doctor Visits")

plt.savefig("C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/DOCTOR_VISITS.png")
import seaborn as sns

sns.boxplot(data = Fraud_df, x = 'NUM_CLAIMS', y = 'FRAUD', orient = 'h')

plt.title("Num Claims")

plt.savefig("C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/NUM_CLAIMS.png")
import seaborn as sns

sns.boxplot(data = Fraud_df, x = 'MEMBER_DURATION', y = 'FRAUD', orient = 'h')

plt.title("Member Duration")

plt.savefig("C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/MEMBER_DURATION.png")
import seaborn as sns

sns.boxplot(data = Fraud_df, x = 'OPTOM_PRESC', y = 'FRAUD', orient = 'h')

plt.title("Optom Presc")

plt.savefig("C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/OPTOM_PRESC.png")
import seaborn as sns

sns.boxplot(data = Fraud_df, x = 'NUM_MEMBERS', y = 'FRAUD', orient = 'h')

plt.title("Num Members")

plt.savefig("C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/NUM_MEMBERS.png")

#c)	(10 points) Orthonormalize interval variables and use the resulting variables for

#the nearest neighbor analysis. Use only the dimensions whose corresponding eigenvalues are greater than one.

#i.	(5 points) How many dimensions are used?

#ii.	(5 points) Please provide the transformation matrix?  

#You must provide proof that the resulting variables are actually orthonormal. 



Fraud_df_drop = Fraud_df.drop(['CASE_ID','FRAUD'],axis = 1) 

x = np.matrix(Fraud_df_drop)

from numpy import linalg as LA

xtx = x.transpose() * x

print("t(x) * x = \n", xtx)



# Eigenvalue decomposition

evals, evecs = LA.eigh(xtx)

print("Eigenvalues of x = \n", evals)

print("Eigenvectors of x = \n",evecs)



# Here is the transformation matrix

transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)));

print("Transformation Matrix = \n", transf)



# Here is the transformed X

transf_x = x * transf;

print("The Transformed x = \n", transf_x) 



## Since the eigenvalues of all 6 variables are greater than 1, the dimension is = 6 



# Orthonormalize using the orth function 

import scipy

from scipy import linalg as LA2



orthx = LA2.orth(x)

print("The orthonormalize x = \n", orthx)



# Check columns of the ORTH function

check = orthx.transpose().dot(orthx)

print("Also Expect an Identity Matrix = \n", check) 



# Since the resulting matrix is an Identity Matrix, variables in dataset are orthonormal
#d)	(10 points) Use the NearestNeighbors module to execute the Nearest 

#Neighbors algorithm using exactly five neighbors and the resulting variables you have chosen in c).  

#The KNeighborsClassifier module has a score function.



#i.	(5 points) Run the score function, provide the function return value

#ii.	(5 points) Explain the meaning of the score function return value. 



from sklearn.neighbors import KNeighborsClassifier as Knn 

from sklearn import metrics



Fraud_data = pd.read_csv('C:/Users/KACHI/Desktop/ALL/1. SEM1/ML/Assignment1/Fraud.csv')

kNNSpec = Knn(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean') 

trainData = transf_x

target = Fraud_data.FRAUD

kNNSpec = kNNSpec.fit(trainData, target)

predict = kNNSpec.predict(trainData)

print(metrics.accuracy_score(target, predict))



#e)	(5 points) For the observation which has these input variable values: TOTAL_SPEND = 7500, DOCTOR_VISITS = 15, 

#    NUM_CLAIMS = 3, MEMBER_DURATION = 127, OPTOM_PRESC = 2, and NUM_MEMBERS = 2, find its five neighbors.  

#    Please list their input variable values and the target values. 

#    Reminder: transform the input observation using the results in c) before finding the neighbors. 





xt = [[7500,15,3,127,2,2]]* transf 

predict = kNNSpec.predict(xt) 

neigh = kNNSpec.kneighbors(xt, return_distance=False) 

print("The nearest five neighbors are: ",neigh)

print(Fraud_df.iloc[neigh[0][0:]])

 