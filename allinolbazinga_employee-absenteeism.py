import pandas as pd                  #Importing Pandas library
import numpy as np                   #Importing numpy library
import matplotlib.pyplot as plt      # Import matplotlib library
import os
project=pd.read_excel('../input/Dataset used for the Project.xls') #Reading dataset
#Below code was found from here : https://stackoverflow.com/questions/26266362/how-to-count-the-nan-values-in-a-column-in-pandas-dataframe/39734251.
#This function was created by Nikos Tavoularis and shared on stackoverflow.com. Comments in below functions were added by Smit Patel

def missing_values_table(df):
        mis_val = df.isnull().sum()                                #Counts the number of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)        #Calculates the precentage of missing valyes
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)  #Concates the above variables
        mis_val_table_ren_columns = mis_val_table.rename(              #renames the column
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[              # Sort and round the values the column in ascending order
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(         
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")                     
        return mis_val_table_ren_columns  
missing_values_table(project)
new_project=project.loc[ : , ['ID','Age', 'Body mass index','Social drinker',
       'Social smoker','Son','Day of the week','Month of absence','Seasons','Service time','Absenteeism time in hours'] ] #Subsetting columns of interest
new_project.columns=['ID','Age', 'BMI', 'Social_drinker', 'Social_smoker', 'Son',
       'Day_of_the_week', 'Month_of_absence', 'Seasons', 'Service_time','Absenteeism_time_in_hours'] #Renaming column  names
new_project.drop(new_project[new_project.Absenteeism_time_in_hours==0].index,inplace=True) #Dropping data that contains 0 hours in absenteeism_time_in_hours columns
new_project.shape #Dimension of the dataset after cleaning and subsetting columns of interest
plt.rcParams['figure.figsize']=25,10               #Selecting size and width of the plot
new_project.hist()                                 #Choosing bar/histogram for visualization
plt.show()                                         #Display the visualization
new_project1=new_project #Copying dataframe into new dataframe. To avoid messing with original dataframe
#Calculation in this section is suggested by Lauren Foltz using Excel, However Coding in below section was developed and executed by Smit Patel
bins = [20,29,39,49,59]   #Creating bins
labels=['Adult20s','Adult30s','Adult40s','Adult50s'] #Labelling bins
new_project1['age_fact']=pd.cut(new_project1['Age'],bins=bins,labels=labels) #Creating new column with bins that are appropriate for each rows
import warnings                   #Import warnings library
warnings.filterwarnings('ignore')
hours_sum=new_project1.groupby(('ID','age_fact'),as_index=False)['Absenteeism_time_in_hours'].sum() #Sum number of hours missed by employees based on Unique ID and labeled bins
hours_sum=hours_sum.dropna() #Removing NA's that are generated during the process
age_hours_missed=round(hours_sum.groupby('age_fact')[['Absenteeism_time_in_hours']].mean(),2) #Calculating AVG hours missed by emplyees based on Age group and rounding to two decimals
age_hours_missed #Avg hours missed by employees based on Unique ID.
#Calculation in this section is suggested by Lauren Foltz using Excel, Coding developed and executed by Smit Patel
bin2=[19,24,29,38]          #Binning BMI 
labels2=['Normal','Overweight','Obese'] #labelling BMI's
new_project1['BMI_fact']=pd.cut(new_project1['BMI'],bins=bin2,labels=labels2) #Creating new column to represent BMI value assocaited with bins and labels
BMI_sum=new_project1.groupby(('ID','BMI_fact'),as_index=False)['Absenteeism_time_in_hours'].sum() #Sum number of hours missed by employees based on Unique ID and labeled bins
BMI_sum=BMI_sum.dropna() #Removing NA's that are generated during the process
BMI_hours_missed=round(BMI_sum.groupby('BMI_fact')[['Absenteeism_time_in_hours']].mean(),2) #Calculating AVG hours missed by emplyees based on BMI group and rounding to two decimals
BMI_hours_missed #Avg hours missed by employees based on Unique ID.
#Calculation in this section is suggested by Lauren Foltz using Excel, Coding developed and executed by Smit Patel
new_project1['Drinker_cat']=pd.cut(new_project1.Social_drinker,2,labels=['Drinker','Non-Drinker']) #Creating new column and assigning lables based on values present in Social_drinker column
Drinker_sum=new_project1.groupby(('ID','Drinker_cat'),as_index=False)['Absenteeism_time_in_hours'].sum() #Sum number of hours missed by employees based on Unique ID and labeled bins
Drinker_sum=Drinker_sum.dropna() #Removing NA's that are generated durig the process
Drinker_hours_missed=round(Drinker_sum.groupby('Drinker_cat')[['Absenteeism_time_in_hours']].mean(),2) #Calculating AVG hours missed by emplyees based on Drinker's group and rounding to two decimals
Drinker_hours_missed #Avg hours missed by employees based on Unique ID.
#Calculation in this section is suggested by Lauren Foltz using Excel, Coding developed and executed by Smit Patel
new_project1['Smoker_cat']=pd.cut(new_project1.Social_smoker,2,labels=['Smoker','Non-Smoker']) #Creating new column and assigning lables based on values present in Social_smoker column
Smoker_sum=new_project1.groupby(('ID','Smoker_cat'),as_index=False)['Absenteeism_time_in_hours'].sum() #Sum number of hours missed by employees based on Unique ID and labeled bins
Smoker_sum=Smoker_sum.dropna() #Removing NA's that are generated durig the process
Smoker_hours_missed=round(Smoker_sum.groupby('Smoker_cat')[['Absenteeism_time_in_hours']].mean(),2) #Calculating AVG hours missed by employees based on Smoker's group and rounding to 2 decimals
Smoker_hours_missed #Avg hours missed by employees based on Unique ID.
#Calculation in this section is suggested by Lauren Foltz using Excel, Coding developed and executed by Smit Patel
bin3=[-np.inf,0,np.inf] #Creting bins for 0-4 levels in "son" column
new_project1['son_fact']=pd.cut(new_project1.Son,bins=bin3,labels=['None','some']) #Creating new column and assigning lables based on values present in 'Son' column
Son_sum=new_project1.groupby(('ID','son_fact'),as_index=False)['Absenteeism_time_in_hours'].sum() #Sum number of hours missed by employees based on Unique ID and labeled bins
Son_sum=Son_sum.dropna() #Removing NA's that are generated durig the process
Son_hours_missed=round(Son_sum.groupby('son_fact')[['Absenteeism_time_in_hours']].mean(),2) #Calculating AVG hours missed by employees based on Son's group and rounding to 2 decimals
Son_hours_missed
#Calculation in this section is suggested by Lauren Foltz using Excel, Coding developed and executed by Smit Patel
from calendar import day_name     #Import days of the week library
from collections import deque
days = deque(day_name)          #Dequing days of the week 
days.rotate(2)                    # rotate days
days_map = dict(enumerate(days)) #Creating dictionary
new_project4=new_project1 #Copying dataset into new datafram to avoid overwriting
new_project4['Day_Factor'] = new_project4['Day_of_the_week'].map(days_map) #Mapping days of the week to a dataframe
Day_filter=new_project4.filter(['Day_Factor','Absenteeism_time_in_hours']) #Filtering columns of interest
Day_filter.groupby('Day_Factor').sum()[['Absenteeism_time_in_hours']].sort_values(['Absenteeism_time_in_hours'],ascending=False) #Grouping by Days of the week, summing and Sorting hours in descending order
import altair as alt
alt.renderers.enable('kaggle')#Rendering notebook
alt.Chart(new_project4).mark_bar().encode(                                                                                    #Selecting Bar chart for visualization
    alt.X('Day_Factor:N',axis=alt.Axis(title='Days of the week'),sort=['Monday','Tuesday','Wednesday','Thursday','Friday']),  #Assigning data to x-axis, adding title and sorting by Days of the week
    alt.Y('sum(Absenteeism_time_in_hours):Q',axis=alt.Axis(title='Absenteeism time in hours')),                               #Assigning data to Y-axis and adding title
    color=alt.Color('Day_Factor:N',title='Days of the Week',sort=['Monday','Tuesday','Wednesday','Thursday','Friday'])      #Assigning color to visualization, adding title and sorting by Days of the week
).properties(width=200,height=200)                                                                                          #Assigning height and width of the plot
#Calculation in this section is suggested by Lauren Foltz using Excel, Coding developed and executed by Smit Patel
month_filter=new_project1.filter(['Month_of_absence','Absenteeism_time_in_hours']) #Filtering columns of interest
month_group=month_filter.groupby(['Month_of_absence']).sum()[['Absenteeism_time_in_hours']] #Grouping and summing hours missed by employees
month_group #Output of Month and total hours missed by employees for each month
alt.Chart(new_project1).mark_bar().encode(                                                         #Selecting Bar chart for visualization
    alt.X('Month_of_absence:Q',axis=alt.Axis(title='Months',ticks=True),bin=alt.Bin(maxbins=30)),  #Assigning data to x-axis, adding title and adding maximum number of bins to x-axis
    alt.Y('sum(Absenteeism_time_in_hours):Q',axis=alt.Axis(title='Absenteeism time in hours')),    #Assigning data to Y-axis and adding title
    color=alt.Color('Month_of_absence:N',legend=None)                                              #Assigning color to visualization and adding title
).properties(width=300,height=200)                                                                 #Assigning height and width of the plot
#Calculation in this section is suggested by Lauren Foltz using Excel, Coding developed and executed by Smit Patel
new_project4['seasons_fact']=pd.cut(new_project4.Seasons,4,labels=['Summer','Autumn','Winter','Spring']) #Adding dummy column to covnert numerical data to categorical
season_filter=new_project4.filter(['seasons_fact','Absenteeism_time_in_hours']) #Filtering columns of interest
seasons_group=season_filter.groupby(['seasons_fact']).sum()[['Absenteeism_time_in_hours']] #Grouping by Seasons and adding number of hours missed by employees
seasons_group #Output of Seasons and number of hours missed by employees
alt.Chart(new_project4).mark_bar().encode(                                                     #Selecting Bar chart for visualization
    alt.X('seasons_fact:N',axis=alt.Axis(title='Seasons')),                                    #Assigning data to x-axis and adding title
    alt.Y('sum(Absenteeism_time_in_hours):Q',axis=alt.Axis(title='Absenteeism time in hours')), #Assigning data to Y-axis and adding title
    color=alt.Color('seasons_fact:N',title='Seasons')                                           #Assigning color to visualization and adding title
).properties(width=200,height=200)                                                              #Assigning height and width of the plot
