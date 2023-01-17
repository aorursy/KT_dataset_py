#import necessary libaries 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline      
import seaborn as sns
sns.set_style("darkgrid")


#get the data and give it a good name  (df for DataFrame)
hr_df = pd.read_csv ("../input/attrition/HR_Employee_Attrition.csv")
hr_df.head() #good to get a first "feel" about the data, shows by default the first 5 lines
"""
first of all we have to twirk around a bit with the options we are seing 

this is important when we will later obsere our cleaned data 
"""

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
#we con not .loc everything we want as want after the chance of the .loc 

hr_df_firstpart = hr_df.loc[:, ["Age", "Attrition", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education", "EducationField"]]
#the second line was just there for me to check if the slicing has worked properly 
hr_df_firstpart

hr_df_secondpart = hr_df.loc[: , ["RelationshipSatisfaction"]]
hr_df_secondpart

hr_df_thirdpart = hr_df.loc [: , ["StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]]
hr_df_thirdpart


"""
we bring all datasets together again

NOTE: first takes all the DataFrames and than you have to pass on the axis (by default = 0 (== rows))
    you don´t want to sort anything so you are good advised to change the default of sort to False 
"""

sns.set_style("whitegrid")
#here the true concatenation takes place 
new_df = pd.concat ([hr_df_firstpart, hr_df_secondpart, hr_df_thirdpart], axis = 1, sort = False)
new_df.head()


#if ony two sets of data have to be put togther there is a simpler way. 
"""
I have shown you how to delete something manually. 
But, Python offers an easy solution 
    
    del df_name ["column_name"]
    
    "EmployeeNumber", "StandardHours
    
    
NOTE: if you start form a point after the line one, you get an error running this line 
    the reason for that is that you can not delete something twice! 
    
    
"""

#unfortunately, I have to show it to you in that way, as you can not 
#delete something that twice! (the hr_df.head() is only to check what I still have to delete)


del hr_df ["EmployeeCount"]
hr_df.head()




#for the other variables I show a nother version 
#note: the axis has to be passed on again 

#again: sorry but I can run it twice so with "#" that you see what I have done

hr_df = hr_df.drop ("EmployeeNumber", axis = 1)
hr_df.head()


#now if you don´t want to reasign the DataFrame, set inplace = True

hr_df.drop("StandardHours", axis = 1, inplace = True)

hr_df.head()



#inspect the data, to see if there are NaN or 0 values we have to replace
hr_df.info()
#general idea about the dataset 

hr_df.shape #no brackets needed as this value is already calculated


#OVERVIEW: we see an statistical overview


round (hr_df.describe())

dummy_gender = pd.get_dummies(hr_df ["Gender"])
dummy_gender.head()
#NOTE: we can always delete the default dummy, is it is perfectly correlated with the non-default dummy (for binary variables)

#again writne like that, because we already changed it before! 

hr_df ["Gender"] = dummy_gender ["Female"]

""" 
the unit8 means: we have changed the datatype sucessfully and the 
type is no a positive whole number between 0-255

"""

hr_df.dtypes

#convertion of the varialbes we want to have as dummies 
hr_dummy = pd.get_dummies(hr_df, columns=["Attrition", "JobRole", "Gender", "BusinessTravel", "Department", "EducationField","MaritalStatus","Over18", "OverTime"])
hr_dummy.head()

hr_dummy.info()
"""remove the variable of interst 
from the data and safe it in a new dataFrame

"""

col_of_interest = "Attrition_No"
first_col= hr_dummy.pop (col_of_interest)
first_col.head()
"""
now we insert the variable (on the first possition)

and we have a look if it worked! 
"""
#we have to write it like that because otherwise we get an error (that the attribute already exist) by running it aother time 
hr_dummy.insert(0, col_of_interest, first_col)
hr_dummy.head()
hr_dummy.shape
"""
note: you have to specify the axis to 1 because we are intrested in deleting only colums and all values under it
"""

hr_dummy = hr_dummy.drop (["Attrition_Yes", "Gender_1", "OverTime_Yes"], axis = 1)

hr_dummy.head()
#histogram for daily rate with seaborn as sns 

#shows histogram and KDE for univariate distribution in one step

fig =sns.distplot(a= hr_dummy["DailyRate"], kde= bool, color = "darkgreen", norm_hist = True, axlabel= "Daily Rate")
fig.set_title("Daily Rate Distribution", fontsize = 20)
fig.figure.set_size_inches (10,7)
plt.show()

"""

"""
plt.ylim(0, 1)
figure = sns.barplot(hr_dummy["YearsAtCompany"], hr_dummy ["Attrition_No"])

figure.figure.set_size_inches (13,7)
plt.savefig("Age_Attrition.png")
plt.show()


#we change the colors to have a bit of variety in it 

#hear we are using the dataFrame before we have assigned the dummy variables
plt.ylim(0, 1)
fig = sns.barplot(hr_dummy["Department_Sales"], hr_dummy ["Attrition_No"], palette = ["red", "purple", "pink"])
fig.figure.set_size_inches (10,7)
plt.show()

#we change the colors to have a bit of variety in it 

#here we have a color that is shaded of the categories 
plt.ylim(0, 1)
figure = sns.barplot(hr_dummy["Department_Sales"], hr_dummy ["Attrition_No"], palette = "Greens_d")
figure.figure.set_size_inches (10,7)
figure.set_title ("Attrition Difference \nBetween Departmet Sales and other Departments", fontsize = 20)
plt.show()

figure = sns.boxplot(hr_dummy["DailyRate"], color = "pink") 
figure.figure.set_size_inches (10,7)
figure.set_title("Daily Rate Distribution", fontsize =20)

plt.show()
#scatter plot of daily rate and age 


figure = sns.scatterplot(hr_df["DailyRate"], hr_df["Age"], color = "pink")

figure.figure.set_size_inches (10,7)
#interpretation: seamingly no realationship 


plt.show()
"""adding a regession to the data

regplot is very flexible (accepts different data types, )

x_bins: makes it more readable and basically puts all data into 
50 discrete bins, symontaniously it shows the confidence internal, the regression is still theorginal data"""  

plt.ylim(0, 1)
figure = sns.regplot(hr_dummy["DailyRate"], hr_dummy["Attrition_No"], x_bins = 50, color= "gold")

figure.figure.set_size_inches (10,7)
#persumably we dont knot the highest value we can insert a max () funcktin
plt.yticks ([0.4, 0.5, 0.5,0.6,0.7,0.8,0.9, max(hr_dummy["Attrition_No"]) ])

#though out I will demonstrate different possibilies to set the axis 
# >> this is by far my prefered option to set the ticks 
plt.xticks(np.arange (0,1800, step = 200))
#interpretation: almost no relationship and if very low (but positive) 

plt.show()
#what we see is a slight positiive correlation 
#we are showing the data with less bins 
#NOTE: the regression is made with the Unbinnded data 
plt.title ("Daily Rate and Attrition", fontsize= 20)

figure = sns.regplot(hr_dummy["DailyRate"], hr_dummy["Attrition_No"], x_estimator=np.mean, x_bins = 15, color = "darkred", ci = 80)
figure.figure.set_size_inches (10,7)
plt.xlabel("Daily Rate in USD", fontsize = 15)
plt.ylabel("Attrition", fontsize = 15)
plt.yticks ([0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0, 200,400,600,800,1000,1200,1400,1600])
plt.show()
"""just to demonstrate what happens if we are not binning at all: 
 -->> we can basically only observe that the data seams not have no influence with one another (no correlation)
 
 ci = confidence interval between 0 and 100
 
 color, can be choosen from all colors in python 
     for the color code, just google: color code python and you get a list 
    if you want to have a quick solution: you can basically use 
    all comon colors with dark or light before them and get a lot of nice colors 
    
    note: all liberies have a slightly different color code, but you can always include 
    the package in your google search :)
 """
plt.figure(figsize = (10,7))
plt.title ("Daily Rate and Age", fontsize = 20)

x= hr_df["DailyRate"]

y= hr_df["Age"]
regressions_graph=sns.regplot(x,y, color = "lime", ci= 90)
plt.yticks(np.arange(15, 65, step = 5))
plt.xticks(np.arange(0,1700,step=200))
regressions_graph.figure.set_size_inches (12,8)
plt.show()
"""
lets explore some other data viz types 

that is already a bit more intersting, we see that the big belly in the blue 
graphs is lower than the one in the orange no graph, that could show a slight indication 
that people who left the company morelikely a lower sallary (irgnoring how 
many people we have in the two categories)
"""

figure = sns.violinplot(hr_dummy["Attrition_No"],hr_dummy["DailyRate"], palette= ["pink", "lightblue"], hue = hr_dummy ["Gender_0"])
figure.figure.set_size_inches (10,7)
plt.show()
"""The same with a bar plot and truncated axis"""
plt.ylim(0, 1)
bins= [np.arange(15,65,step=10)]

figure = sns.barplot(hr_dummy["Age"],hr_dummy["Attrition_No"], palette= ["pink", "lightblue"], hue = hr_dummy ["Gender_0"])
figure.figure.set_size_inches (17,7)
plt.show()
"""
lets explore some other data viz types 

that is already a bit more intersting, we see that the big belly in the blue 
graphs is lower than the one in the orange no graph, that could show a slight indication 
that people who left the company morelikely a lower sallary (irgnoring how 
many people we have in the two categories)
"""

figure = sns.violinplot(hr_dummy["Attrition_No"],hr_dummy["DailyRate"], palette= ["pink", "lightblue"])
figure.figure.set_size_inches (10,7)
plt.show()
"""
An extremly related (less intuiteve but more intersting for statistical evaluations)
chart is the boxplot 
we clearly see here: the median daily rate of a person who left (attriton: yes) is lower 
than for a person who stayed 

further: we see that the spread of the two distributions is very similair
so wie have people erarning a lot and a litte per day in both categories 
"""

figure = sns.boxplot(hr_dummy["Attrition_No"],hr_dummy["DailyRate"], palette= ["pink", "lightblue"])
figure.figure.set_size_inches (10,7)
plt.show()
"""
An extremly related (less intuiteve but more intersting for statistical evaluations)
chart is the boxplot 
we clearly see here: the median daily rate of a person who left (attriton: yes) is lower 
than for a person who stayed 

further: we see that the spread of the two distributions is very similair
so wie have people erarning a lot and a litte per day in both categories 
"""

figure = sns.boxplot(hr_dummy["Attrition_No"],hr_dummy["DailyRate"], palette= ["pink", "lightblue"], hue =hr_dummy ["Gender_0"])
figure.figure.set_size_inches (10,7)
plt.show()
"""
that looks intersting, however, there is a huge catch! 
this graph by default is basically misleading and should not be used 
the y axis is only from 680-ish to 840-ish and ignores the values below
I show you how to change the axis and we see the a more similar picture 
than we have seen before ...
"""

figure = sns.lineplot(x = hr_dummy["Attrition_No"],y = hr_dummy["DailyRate"], color = "navy")
figure.figure.set_size_inches (10,7)
plt.show()
#we take a value (500 between the min daily rate and the python suggested one )

plt.ylim(500, 840)
figure = sns.set_style("whitegrid")
figure = sns.lineplot(x = hr_dummy["Attrition_No"],y = hr_dummy["DailyRate"])
figure.figure.set_size_inches (10,7)
plt.show()
#definition of the function for the headmap 

def halfheatmap (df, mirrow, title ): 
    corr = df.corr()
    fig, ax = plt.subplots(figsize = (12,12))
    colormap = sns.diverging_palette (220,10, as_cmap = True)
    ax.set_title(title, fontsize = 20)
    
    #mirrow basically means if we want to have every correlation in one or twice (we choose once)
    if mirrow == True:
        #Generate Heat Map, allow annotations and place floats in map
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
      #Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns);
      #Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
      #show plot
    
    else: 
        #drop selcorrelation
        dropself= np.zeros_like (corr)
        dropself [np.triu_indices_from(dropself)] = True
        colormap = sns.diverging_palette (200,10,as_cmap = True)
        sns.heatmap(corr, cmap=colormap, annot=False, fmt=".2g", mask=dropself, vmax = 1, vmin=-1)
         # Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns, rotation = 45);
        # Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
        
    # show plot and save it 
    plt.savefig("Happiness.Correlation.Heatmap.png")
    plt.show()
    
halfheatmap(df = hr_df, mirrow = False, title = "Correlation Heatmap")


#get insides to the regession line 


"""confidence interval by default 95%


#note: that does not work ! the reason for that is: 
we need to define the variables prior to calling them for the summary statistics 



we are now including more and more varibales and see that the R^2 increases 
"""

import statsmodels.api as sm 

x = hr_dummy[["DailyRate", "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear","WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "Gender_0", "JobRole_Healthcare Representative", "JobRole_Human Resources", "JobRole_Laboratory Technician", "JobRole_Manager", "JobRole_Manufacturing Director", "JobRole_Research Director", "JobRole_Research Scientist", "JobRole_Sales Executive", "JobRole_Sales Representative", "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely", "Department_Human Resources", "Department_Sales", "Department_Research & Development", "OverTime_No", "Over18_Y","MaritalStatus_Single", "MaritalStatus_Married", "MaritalStatus_Divorced" ]]
y = hr_dummy["Attrition_No"]


X = sm.add_constant(x)
model = sm.OLS(y, X)
est = model.fit()
print(est.summary())

"""
I got an error here and therefore, had to convert my int8 and int64
    to floats 
 
"""
hr_dummy = hr_dummy.astype (float)


x = np.column_stack ([hr_dummy.iloc [:,1:]])

y = hr_dummy["Attrition_No"]

X = sm.add_constant(x, prepend = True)
model = sm.OLS(y, X).fit()

print(est.summary())
