# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
#Default Stuff are above
import matplotlib.pyplot as plt                          #In order to use matplotlib I import it.
data=pd.read_csv("../input/StudentsPerformance.csv")     #.csv file dataset defined to dataframe as data. 
                                                         #Since I'm not going to work on more than one dataset it won't be a problem to write data.



data.info()
#Data.info() allows us to see what's inside of the dataset, once the code start, it will show the types of values in columns and their group name.

print("Solving Dataset Problems")
print("data.columns= data.columns.str.replace(' ', '_')")   #This code replaces space in columns with _
print("data.columns= data.columns.str.replace('/','_')")
data.columns= data.columns.str.replace(' ', '_')   #This code replaces space in columns with _
data.columns= data.columns.str.replace('/','_')
data.head()
data.tail()
print("Genders : ")
print(data.gender.unique())
print("")
data_gender_female=data[data.gender=="female"]   #Female subjects grouped in dataframe.
data_gender_male=data[data.gender=="male"]       #Male subjects grouped in dataframe.
#Gender Count gelecek.
print("Female Counts")
print("")
print(data_gender_female.gender.count())
print("")
print("Male Counts")
print("")
print(data_gender_male.gender.count())
print("")
#mean_female=data_gender_female.maths core.mean() #this gives an error as follows invalid synthax. In dataframe math score has to be written as math_score in order to avoid the error.
print(data_gender_female)
print("   ")
print(data_gender_male)

mean_female_math=data_gender_female.math_score.mean()
mean_male_math=data_gender_male.math_score.mean()

print("")
print("Female Students Math Score Mean : ")
print(mean_female_math)
print("")
print("Male Students Math Score Mean : ")
print(mean_male_math)
print("")
mean_dif_gender_math=abs(mean_female_math-mean_male_math)
print("Conclusion")
print("")
if (mean_male_math>mean_female_math):
    print("Males are better in Maths")
    print("")
elif(mean_male_math<mean_female_math):
    print("Females are better in Maths")
    print("")
else:
    print("No Difference")
    print("")
print("Difference between Genders : ")
print(mean_dif_gender_math)
mean_male_reading=data_gender_male.reading_score.mean()
mean_female_reading=data_gender_female.reading_score.mean()

print("")
print("Female Students Reading Score Mean : ")
print(mean_female_reading)
print("")
print("Male Students Reading Score Mean : ")
print(mean_male_reading)
print("")
mean_dif_gender_reading=abs(mean_female_reading-mean_male_reading)
print("Conclusion")
print("")
if (mean_male_reading>mean_female_reading):
    print("Males are better in Reading")
    print("")
elif(mean_male_reading<mean_female_reading):
    print("Females are better in Reading")
    print("")
else:
    print("No Difference")
    print("")
print("Difference between Genders : ")
print(mean_dif_gender_reading)
mean_male_writing=data_gender_male.writing_score.mean()
mean_female_writing=data_gender_female.writing_score.mean()

print("")
print("Female Students Writing Score Mean : ")
print(mean_female_writing)
print("")
print("Male Students Writing Score Mean : ")
print(mean_male_writing)
print("")
mean_dif_gender_writing=abs(mean_female_writing-mean_male_writing)
print("Conclusion")
print("")
if (mean_male_writing>mean_female_writing):
    print("Males are better in Writing")
    print("")
elif(mean_male_writing<mean_female_writing):
    print("Females are better in Writing")
    print("")
else:
    print("No Difference")
    print("")
print("Difference between Genders : ")
print(mean_dif_gender_writing)
mean_male_general=mean_male_math+mean_male_reading+mean_male_writing
mean_female_general=mean_female_math+mean_female_reading+mean_female_writing

print("In General")
print("")
print("Female Students General Score Mean : ")
print(mean_female_general)
print("")
print("Male Students General Score Mean : ")
print(mean_male_general)
print("")
mean_dif_gender_general=abs(mean_female_general-mean_male_general)
print("Conclusion")
print("")
if (mean_male_general>mean_female_general):
    print("Males are better in General")
    print("")
elif(mean_male_general<mean_female_general):
    print("Females are better in General")
    print("")
else:
    print("No Difference")
    print("")
print("Difference between Genders : ")
print(mean_dif_gender_writing)
print("Races/Ethnicities : ")
dre=sorted(data.race_ethnicity.unique())
print(dre)
print("")
print("Group A Ethnicity List : ")
data_re_gA=data[data.race_ethnicity=="group A"]
print(data_re_gA)
print("")
print("Group B Etnictiy List : ")
data_re_gB=data[data.race_ethnicity=="group B"]
print(data_re_gB)
print("")
print("Group C Etnictiy List : ")
data_re_gC=data[data.race_ethnicity=="group C"]
print(data_re_gC)
print("")
print("Group D Etnictiy List:")
data_re_gD=data[data.race_ethnicity=="group D"]
print(data_re_gD)
print("")
print("Group E Etnictiy List:")
data_re_gE=data[data.race_ethnicity=="group E"]
print(data_re_gE)
print("Gender Neutral Math Score Comparision Results")
data_re_gA_math_mean=data_re_gA.math_score.mean()
data_re_gB_math_mean=data_re_gB.math_score.mean()
data_re_gC_math_mean=data_re_gC.math_score.mean()
data_re_gD_math_mean=data_re_gD.math_score.mean()
data_re_gE_math_mean=data_re_gE.math_score.mean()
#Lines above could be updated with a code which will produce name data_re_gA_math_mean sort of stuff. Like data_re_x_y_mean. X and Y are variables. (WIP)
race_df_mean={"Group A":data_re_gA_math_mean,"Group B":data_re_gB_math_mean,"Group C":data_re_gC_math_mean,"Group D":data_re_gD_math_mean,"Group E":data_re_gE_math_mean}
print("")
i=0
c=0
each=0
print("Race Groups and Its Scores")
print("")
for key, value in race_df_mean.items():
      print ("For", key, "=", value)
      print("")
a={"":0}    
print("")
print("Most Successful Race")
a=max(race_df_mean.keys(), key=(lambda k: race_df_mean[k]))
print("")
print(a)
print("")
print("Least Successful Race")
print("")
b=min(race_df_mean.keys(), key=(lambda k: race_df_mean[k]))
print(b)
print("")
#This whole part could be reshaped with for or while loops in order to save time. Since this kernel is my first one I will not try harder. In the end of my course this parts will be updated.
drgAgF=data_re_gA[data.gender=="female"]
drgAgM=data_re_gA[data.gender=="male"]
drgBgF=data_re_gB[data.gender=="female"]
drgBgM=data_re_gB[data.gender=="male"]
drgCgF=data_re_gC[data.gender=="female"]
drgCgM=data_re_gC[data.gender=="male"]
drgDgF=data_re_gD[data.gender=="female"]
drgDgM=data_re_gD[data.gender=="male"]
drgEgF=data_re_gE[data.gender=="female"]
drgEgM=data_re_gE[data.gender=="male"]
print("Race Groups Female and Male Counts")
print("")                                                              #Starting from here,
print("Group A Female Count = ",drgAgF.gender.count())
print("Group A Male Count = ",drgAgM.gender.count())
print("")                                                              #To there, this part repeats, so it could be possible to use loops. This appears several times.
print("Group B Female Count = ",drgBgF.gender.count())
print("Group B Male Count = ",drgBgM.gender.count())
print("")
print("Group C Female Count = ",drgCgF.gender.count())
print("Group C Male Count = ",drgCgM.gender.count())
print("")
print("Group D Female Count = ",drgDgF.gender.count())
print("Group D Male Count = ",drgDgM.gender.count())
print("")
print("Group E Female Count = ",drgEgF.gender.count())
print("Group E Male Count = ",drgEgM.gender.count())
print("")
print("")
print("Race Math Mean Scores")
print("")
print("Group A")
drgAgF_mean=drgAgF.math_score.mean()
drgAgM_mean=drgAgM.math_score.mean()
print("Female Math Mean Score")
print(drgAgF_mean)
print("")
print("Male Math Mean Score")
print(drgAgM_mean)
print("")
print("Group B")
drgBgF_mean=drgBgF.math_score.mean()
drgBgM_mean=drgBgM.math_score.mean()
print("Female Math Mean Score")
print(drgBgF_mean)
print("")
print("Male Math Mean Score")
print(drgBgM_mean)
print("")
print("Group C")
drgCgF_mean=drgCgF.math_score.mean()
drgCgM_mean=drgCgM.math_score.mean()
print("Female Math Mean Score")
print(drgCgF_mean)
print("")
print("Male Math Mean Score")
print(drgCgM_mean)
print("")
print("Group D")
drgDgF_mean=drgDgF.math_score.mean()
drgDgM_mean=drgDgM.math_score.mean()
print("Female Math Mean Score")
print(drgDgF_mean)
print("")
print("Male Math Mean Score")
print(drgDgM_mean)
print("")
print("Group E")
drgEgF_mean=drgEgF.math_score.mean()
drgEgM_mean=drgEgM.math_score.mean()
print("Female Math Mean Score")
print(drgEgF_mean)
print("")
print("Male Math Mean Score")
print(drgEgM_mean)
print("")
print("")
print("Comparisions between Genders")
print("")
if (drgAgF_mean>drgAgM_mean):
    print("In Group A Female Math Scores are Better")
    print("")
elif(drgAgF_mean==drgAgM_mean):
    print("No Difference between Genders in Group A")
    print("")
else:
    print("In Group A Male Math Scores are Better")
    print("")
if (drgBgF_mean>drgBgM_mean):
    print("In Group B Female Math Scores are Better")
    print("")
elif(drgBgF_mean==drgBgM_mean):
    print("No Difference between Genders in Group B")
    print("")
else:
    print("In Group B Male Math Scores are Better")
    print("")
if (drgCgF_mean>drgCgM_mean):
    print("In Group C Female Math Scores are Better")
    print("")
elif(drgCgF_mean==drgCgM_mean):
    print("No Difference between Genders in Group C")
    print("")
else:
    print("In Group C Male Math Scores are Better")
    print("")
if (drgDgF_mean>drgDgM_mean):
    print("In Group D Female Math Scores are Better")
    print("")
elif(drgDgF_mean==drgDgM_mean):
    print("No Difference between Genders in Group D")
    print("")
else:
    print("In Group D Male Math Scores are Better")
    print("")
if (drgEgF_mean>drgEgM_mean):
    print("In Group E Female Math Scores are Better")
    print("")
elif(drgEgF_mean==drgEgM_mean):
    print("No Difference between Genders in Group E")
    print("")
else:
    print("In Group E Male Math Scores are Better")
    print("")
drggFM={"Group A Female":drgAgF_mean,"Group B Female":drgBgF_mean,"Group C Female":drgCgF_mean,"Group D Female":drgDgF_mean,"Group E Female":drgEgF_mean,}
drggMM={"Group A Male":drgAgM_mean,"Group B Male":drgBgM_mean,"Group C Male":drgCgM_mean,"Group D Male":drgDgM_mean,"Group E Male":drgEgM_mean,}
print("Most Successful Among Female and Its Score")
print("")
print(max(drggFM))
print("")
print("Least Successful Among Female and Its Score")
print("")
print(min(drggFM))
print("")
print("Most Successful Among Male and Its Score")
print("")
print(max(drggMM))
print("")
print("Least Successful Among Male and Its Score")
print("")
print(min(drggMM))
print("Gender Neutral Reading Score Comparision Results")
data_re_gA_read_mean=data_re_gA.reading_score.mean()
data_re_gB_read_mean=data_re_gB.reading_score.mean()
data_re_gC_read_mean=data_re_gC.reading_score.mean()
data_re_gD_read_mean=data_re_gD.reading_score.mean()
data_re_gE_read_mean=data_re_gE.reading_score.mean()
#Lines above could be updated with a code which will produce name data_re_gA_math_mean sort of stuff. Like data_re_x_y_mean. X and Y are variables. (WIP)
race_df_mean_R={"Group A":data_re_gA_read_mean,"Group B":data_re_gB_read_mean,"Group C":data_re_gC_read_mean,"Group D":data_re_gD_read_mean,"Group E":data_re_gE_read_mean}
print("")
i=0
c=0
each=0
print("Race Groups and Its Scores")
print("")
for key, value in race_df_mean_R.items():
      print ("For", key, "=", value)
      print("")
a={"":0}    
print("")
print("Most Successful Race")
a=max(race_df_mean_R.keys(), key=(lambda k: race_df_mean_R[k]))
print("")
print(a)
print("")
print("Least Successful Race")
print("")
b=min(race_df_mean_R.keys(), key=(lambda k: race_df_mean_R[k]))
print(b)
print("")
print("Race Reading Mean Scores")
print("")
print("Group A")
drgAgF_mean=drgAgF.reading_score.mean()
drgAgM_mean=drgAgM.reading_score.mean()
print("Female Reading Score")
print(drgAgF_mean)
print("")
print("Male Reading Mean Score")
print(drgAgM_mean)
print("")
print("Group B")
drgBgF_mean=drgBgF.reading_score.mean()
drgBgM_mean=drgBgM.reading_score.mean()
print("Female Reading Mean Score")
print(drgBgF_mean)
print("")
print("Male Reading Mean Score")
print(drgBgM_mean)
print("")
print("Group C")
drgCgF_mean=drgCgF.reading_score.mean()
drgCgM_mean=drgCgM.reading_score.mean()
print("Female Reading Score")
print(drgCgF_mean)
print("")
print("Male Reading Mean Score")
print(drgCgM_mean)
print("")
print("Group D")
drgDgF_mean=drgDgF.reading_score.mean()
drgDgM_mean=drgDgM.reading_score.mean()
print("Female Reading Score")
print(drgDgF_mean)
print("")
print("Male Reading Mean Score")
print(drgDgM_mean)
print("")
print("Group E")
drgEgF_mean=drgEgF.reading_score.mean()
drgEgM_mean=drgEgM.reading_score.mean()
print("Female Reading Score")
print(drgEgF_mean)
print("")
print("Male Reading Mean Score")
print(drgEgM_mean)
print("")
print("")
print("Comparisions between Genders")
print("")
if (drgAgF_mean>drgAgM_mean):
    print("In Group A Female Reading Scores are Better")
    print("")
elif(drgAgF_mean==drgAgM_mean):
    print("No Difference between Genders in Group A")
    print("")
else:
    print("In Group A Male Reading Scores are Better")
    print("")
if (drgBgF_mean>drgBgM_mean):
    print("In Group B Female Reading Scores are Better")
    print("")
elif(drgBgF_mean==drgBgM_mean):
    print("No Difference between Genders in Group B")
    print("")
else:
    print("In Group B Male Reading Scores are Better")
    print("")
if (drgCgF_mean>drgCgM_mean):
    print("In Group C Female Reading Scores are Better")
    print("")
elif(drgCgF_mean==drgCgM_mean):
    print("No Difference between Genders in Group C")
    print("")
else:
    print("In Group C Male Reading Scores are Better")
    print("")
if (drgDgF_mean>drgDgM_mean):
    print("In Group D Female Reading Scores are Better")
    print("")
elif(drgDgF_mean==drgDgM_mean):
    print("No Difference between Genders in Group D")
    print("")
else:
    print("In Group D Male Reading Scores are Better")
    print("")
if (drgEgF_mean>drgEgM_mean):
    print("In Group E Female Reading Scores are Better")
    print("")
elif(drgEgF_mean==drgEgM_mean):
    print("No Difference between Genders in Group E")
    print("")
else:
    print("In Group E Male Reading Scores are Better")
    print("")
drggFR={"Group A Female":drgAgF_mean,"Group B Female":drgBgF_mean,"Group C Female":drgCgF_mean,"Group D Female":drgDgF_mean,"Group E Female":drgEgF_mean,}
drggMR={"Group A Male":drgAgM_mean,"Group B Male":drgBgM_mean,"Group C Male":drgCgM_mean,"Group D Male":drgDgM_mean,"Group E Male":drgEgM_mean,}
print("Most Successful Among Female")
a=max(drggFR.keys(), key=(lambda k: drggFR[k]))
print("")
print(a)
print("")
print("Least Successful Among Female")
print("")
b=min(drggFR.keys(), key=(lambda k: drggFR[k]))
print(b)
print("")
print("Most Successful Among Male")
a=max(drggMR.keys(), key=(lambda k: drggMR[k]))
print("")
print(a)
print("")
print("Least Successful Among Male")
print("")
b=min(drggMR.keys(), key=(lambda k: drggMR[k]))
print(b)
print("")
print("Gender Neutral Writing Score Comparision Results")
data_re_gA_writing_mean=data_re_gA.writing_score.mean()
data_re_gB_writing_mean=data_re_gB.writing_score.mean()
data_re_gC_writing_mean=data_re_gC.writing_score.mean()
data_re_gD_writing_mean=data_re_gD.writing_score.mean()
data_re_gE_writing_mean=data_re_gE.writing_score.mean()
#Lines above could be updated with a code which will produce name data_re_gA_math_mean sort of stuff. Like data_re_x_y_mean. X and Y are variables. (WIP)
race_df_mean_W={"Group A":data_re_gA_writing_mean,"Group B":data_re_gB_writing_mean,"Group C":data_re_gC_writing_mean,"Group D":data_re_gD_writing_mean,"Group E":data_re_gE_writing_mean}
print("")
i=0
c=0
each=0
print("Race Groups and Its Scores")
print("")
for key, value in race_df_mean_W.items():
      print ("For", key, "=", value)
      print("")
a={"":0}    
print("")
print("Most Successful Race")
a=max(race_df_mean_W.keys(), key=(lambda k: race_df_mean_W[k]))
print("")
print(a)
print("")
print("Least Successful Race")
print("")
b=min(race_df_mean_W.keys(), key=(lambda k: race_df_mean_W[k]))
print(b)
print("")
print("Race Writing Mean Scores")
print("")
print("Group A")
drgAgF_mean=drgAgF.writing_score.mean()
drgAgM_mean=drgAgM.writing_score.mean()
print("Female Writing Score")
print(drgAgF_mean)
print("")
print("Male Writing Mean Score")
print(drgAgM_mean)
print("")
print("Group B")
drgBgF_mean=drgBgF.writing_score.mean()
drgBgM_mean=drgBgM.writing_score.mean()
print("Female Writing Mean Score")
print(drgBgF_mean)
print("")
print("Male Writing Mean Score")
print(drgBgM_mean)
print("")
print("Group C")
drgCgF_mean=drgCgF.writing_score.mean()
drgCgM_mean=drgCgM.writing_score.mean()
print("Female Writing Score")
print(drgCgF_mean)
print("")
print("Male Writing Mean Score")
print(drgCgM_mean)
print("")
print("Group D")
drgDgF_mean=drgDgF.writing_score.mean()
drgDgM_mean=drgDgM.writing_score.mean()
print("Female Writing Score")
print(drgDgF_mean)
print("")
print("Male Writing Mean Score")
print(drgDgM_mean)
print("")
print("Group E")
drgEgF_mean=drgEgF.writing_score.mean()
drgEgM_mean=drgEgM.writing_score.mean()
print("Female Writing Score")
print(drgEgF_mean)
print("")
print("Male Writing Mean Score")
print(drgEgM_mean)
print("")
print("")
print("Comparisions between Genders")
print("")
if (drgAgF_mean>drgAgM_mean):
    print("In Group A Female Writing Scores are Better")
    print("")
elif(drgAgF_mean==drgAgM_mean):
    print("No Difference between Genders in Group A")
    print("")
else:
    print("In Group A Male Writing Scores are Better")
    print("")
if (drgBgF_mean>drgBgM_mean):
    print("In Group B Female Writing Scores are Better")
    print("")
elif(drgBgF_mean==drgBgM_mean):
    print("No Difference between Genders in Group B")
    print("")
else:
    print("In Group B Male Writing Scores are Better")
    print("")
if (drgCgF_mean>drgCgM_mean):
    print("In Group C Female Writing Scores are Better")
    print("")
elif(drgCgF_mean==drgCgM_mean):
    print("No Difference between Genders in Group C")
    print("")
else:
    print("In Group C Male Writing Scores are Better")
    print("")
if (drgDgF_mean>drgDgM_mean):
    print("In Group D Female Writing Scores are Better")
    print("")
elif(drgDgF_mean==drgDgM_mean):
    print("No Difference between Genders in Group D")
    print("")
else:
    print("In Group D Male Writing Scores are Better")
    print("")
if (drgEgF_mean>drgEgM_mean):
    print("In Group E Female Writing Scores are Better")
    print("")
elif(drgEgF_mean==drgEgM_mean):
    print("No Difference between Genders in Group E")
    print("")
else:
    print("In Group E Male Writing Scores are Better")
    print("")
drggFW={"Group A Female":drgAgF_mean,"Group B Female":drgBgF_mean,"Group C Female":drgCgF_mean,"Group D Female":drgDgF_mean,"Group E Female":drgEgF_mean,}
drggMW={"Group A Male":drgAgM_mean,"Group B Male":drgBgM_mean,"Group C Male":drgCgM_mean,"Group D Male":drgDgM_mean,"Group E Male":drgEgM_mean,}
print("Most Successful Among Female")
a=max(drggFW.keys(), key=(lambda k: drggFW[k]))
print("")
print(a)
print("")
print("Least Successful Among Female")
print("")
b=min(drggFW.keys(), key=(lambda k: drggFW[k]))
print(b)
print("")
print("Most Successful Among Male")
a=max(drggMW.keys(), key=(lambda k: drggMW[k]))
print("")
print(a)
print("")
print("Least Successful Among Male")
print("")
b=min(drggMW.keys(), key=(lambda k: drggMW[k]))
print(b)
print("")
General_gA_mean=(data_re_gA_math_mean+data_re_gA_read_mean+data_re_gA_writing_mean)
General_gB_mean=(data_re_gB_math_mean+data_re_gB_read_mean+data_re_gB_writing_mean)
General_gC_mean=(data_re_gC_math_mean+data_re_gC_read_mean+data_re_gC_writing_mean)
General_gD_mean=(data_re_gD_math_mean+data_re_gD_read_mean+data_re_gD_writing_mean)
General_gE_mean=(data_re_gE_math_mean+data_re_gE_read_mean+data_re_gE_writing_mean)
General_g_means={"Group A":General_gA_mean,"Group B":General_gB_mean,"Group C":General_gC_mean,"Group D":General_gD_mean,"Group E":General_gE_mean}
print("Race Groups and Its General Scores")
print("")
for key, value in General_g_means.items():
      print ("For", key, "=", value)
      print("")
General_gA_mean_F=(drgAgF.math_score.mean()+drgAgF.reading_score.mean()+drgAgF.writing_score.mean())
General_gA_mean_M=(drgAgM.math_score.mean()+drgAgM.reading_score.mean()+drgAgM.writing_score.mean())
General_gB_mean_F=(drgBgF.math_score.mean()+drgBgF.reading_score.mean()+drgBgF.writing_score.mean())
General_gB_mean_M=(drgBgM.math_score.mean()+drgBgM.reading_score.mean()+drgBgM.writing_score.mean())
General_gC_mean_F=(drgCgF.math_score.mean()+drgCgF.reading_score.mean()+drgCgF.writing_score.mean())
General_gC_mean_M=(drgCgM.math_score.mean()+drgCgM.reading_score.mean()+drgCgM.writing_score.mean())
General_gD_mean_F=(drgDgF.math_score.mean()+drgDgF.reading_score.mean()+drgDgF.writing_score.mean())
General_gD_mean_M=(drgDgM.math_score.mean()+drgDgM.reading_score.mean()+drgDgM.writing_score.mean())
General_gE_mean_F=(drgEgF.math_score.mean()+drgEgF.reading_score.mean()+drgEgF.writing_score.mean())
General_gE_mean_M=(drgEgM.math_score.mean()+drgEgM.reading_score.mean()+drgEgM.writing_score.mean())

print("In General Succesful Gender in Groups")
print("")
if(General_gA_mean_F>General_gA_mean_M):
    print("Females are Better in Group A = ",General_gA_mean_F)
    print("")
elif(General_gA_mean_F==General_gA_mean_M):
    print("No Difference in Group A = ",General_gA_mean_F)
    print("")
else:
    print("Males are Better in Group A",General_gA_mean_M)
    print("")
if(General_gB_mean_F>General_gB_mean_M):
    print("Females are Better in Group B = ",General_gB_mean_F)
    print("")
elif(General_gB_mean_F==General_gB_mean_M):
    print("No Difference in Group B = ",General_gB_mean_F)
    print("")
else:
    print("Males are Better in Group B = ",General_gB_mean_M)
    print("")
if(General_gC_mean_F>General_gC_mean_M):
    print("Females are Better in Group C = ",General_gC_mean_F)
    print("")
elif(General_gC_mean_F==General_gC_mean_M):
    print("No Difference in Group C = ",General_gC_mean_F)
    print("")
else:
    print("Males are Better in Group C = ",General_gC_mean_M)
    print("")
if(General_gD_mean_F>General_gD_mean_M):
    print("Females are Better in Group D = ",General_gD_mean_F)
    print("")
elif(General_gD_mean_F==General_gD_mean_M):
    print("No Difference in Group D = ",General_gD_mean_F)
    print("")
else:
    print("Males are Better in Group D = ",General_gD_mean_M)
    print("")
if(General_gE_mean_F>General_gE_mean_M):
    print("Females are Better in Group E = ",General_gE_mean_F)
    print("")
elif(General_gE_mean_F==General_gE_mean_M):
    print("No Difference in Group E = ",General_gE_mean_F)
    print("")
else:
    print("Males are Better in Group E = ",General_gE_mean_M)
    print("")

drggFG={"Group A Female":General_gA_mean_F,"Group B Female":General_gB_mean_F,"Group C Female":General_gC_mean_F,"Group D Female":General_gD_mean_F,"Group E Female":General_gE_mean_F}
drggMG={"Group A Male":General_gA_mean_M,"Group B Male":General_gB_mean_M,"Group C Male":General_gC_mean_M,"Group D Male":General_gD_mean_M,"Group E Male":General_gE_mean_M}
print("Most Successful Among Female")
a=max(drggFG.keys(), key=(lambda k: drggFG[k]))
print("")
print(a)
print("")
print("Least Successful Among Female")
print("")
b=min(drggFG.keys(), key=(lambda k: drggFG[k]))
print(b)
print("")
print("Most Successful Among Male")
a=max(drggMG.keys(), key=(lambda k: drggMG[k]))
print("")
print(a)
print("")
print("Least Successful Among Male")
print("")
b=min(drggMG.keys(), key=(lambda k: drggMG[k]))
print(b)
print("")
print("Parental Level of Educations")
print("")
drploe=sorted(data.parental_level_of_education.unique())
print(drploe)

data_PLoE_As=data[data.parental_level_of_education=="associate's degree"]
data_PLoE_Ba=data[data.parental_level_of_education=="bachelor's degree"]
data_PLoE_Hs=data[data.parental_level_of_education=="high school"]
data_PLoE_Ma=data[data.parental_level_of_education=="master's degree"]
data_PLoE_Sc=data[data.parental_level_of_education=="some college"]
data_PLoE_Sh=data[data.parental_level_of_education=="some high school"]
print("Parental Level of Educations Group Lists")
print("")
print("Associate's Degree")
print(data_PLoE_As)
print("")
print("Bachelor's Degree")
print(data_PLoE_Ba)
print("")
print("High School")
print(data_PLoE_Hs)
print("")
print("Master's Degree")
print(data_PLoE_Ma)
print("")
print("Some College")
print(data_PLoE_Sc)
print("")
print("Some High School")
print(data_PLoE_Sh)
print("")

data_PLoE_As_C=data_PLoE_As.parental_level_of_education.count()
data_PLoE_Ba_C=data_PLoE_Ba.parental_level_of_education.count()
data_PLoE_Hs_C=data_PLoE_Hs.parental_level_of_education.count()
data_PLoE_Ma_C=data_PLoE_Ma.parental_level_of_education.count()
data_PLoE_Sc_C=data_PLoE_Sc.parental_level_of_education.count()
data_PLoE_Sh_C=data_PLoE_Sh.parental_level_of_education.count()
print("")
print("Associate's Degree")
print(data_PLoE_As_C)
print("")
print("Bachelor's Degree")
print(data_PLoE_Ba_C)
print("")
print("High School")
print(data_PLoE_Hs_C)
print("")
print("Master's Degree")
print(data_PLoE_Ma_C)
print("")
print("Some College")
print(data_PLoE_Sc_C)
print("")
print("Some High School")
print(data_PLoE_Sh_C)
print("")
dploe={"Associate's Degree":data_PLoE_As_C,"Bachelor's Degree":data_PLoE_Ba_C,"High School":data_PLoE_Hs_C,"Master's Degree":data_PLoE_Ma_C,"Some College":data_PLoE_Sc_C,"Some High School":data_PLoE_Sh_C}
print("")
print("Most Populated Group of PLoE")
print("")
a=max(dploe.keys(), key=(lambda k: dploe[k]))
print(a)
print("")
print("Least Populated Group of PLoE")
print("")
b=min(dploe.keys(), key=(lambda k: dploe[k]))
print(b)
data_PLoE_As_mM=data_PLoE_As.math_score.mean()
data_PLoE_As_rM=data_PLoE_As.reading_score.mean()
data_PLoE_As_wM=data_PLoE_As.writing_score.mean()
data_PLoE_As_gM=(data_PLoE_As_mM+data_PLoE_As_rM+data_PLoE_As_wM)
#
data_PLoE_Ba_mM=data_PLoE_Ba.math_score.mean()
data_PLoE_Ba_rM=data_PLoE_Ba.reading_score.mean()
data_PLoE_Ba_wM=data_PLoE_Ba.writing_score.mean()
data_PLoE_Ba_gM=(data_PLoE_Ba_mM+data_PLoE_Ba_rM+data_PLoE_Ba_wM)
#
data_PLoE_Hs_mM=data_PLoE_Hs.math_score.mean()
data_PLoE_Hs_rM=data_PLoE_Hs.reading_score.mean()
data_PLoE_Hs_wM=data_PLoE_Hs.writing_score.mean()
data_PLoE_Hs_gM=(data_PLoE_Hs_mM+data_PLoE_Ba_rM+data_PLoE_Ba_wM)   #Hs,Ma,Sc,Sh
#
data_PLoE_Ma_mM=data_PLoE_Ma.math_score.mean()
data_PLoE_Ma_rM=data_PLoE_Ma.reading_score.mean()
data_PLoE_Ma_wM=data_PLoE_Ma.writing_score.mean()
data_PLoE_Ma_gM=(data_PLoE_Ma_mM+data_PLoE_Ma_rM+data_PLoE_Ma_wM)
#
data_PLoE_Sc_mM=data_PLoE_Sc.math_score.mean()
data_PLoE_Sc_rM=data_PLoE_Sc.reading_score.mean()
data_PLoE_Sc_wM=data_PLoE_Sc.writing_score.mean()
data_PLoE_Sc_gM=(data_PLoE_Sc_mM+data_PLoE_Sc_rM+data_PLoE_Sc_wM)
#
data_PLoE_Sh_mM=data_PLoE_Sh.math_score.mean()
data_PLoE_Sh_rM=data_PLoE_Sh.reading_score.mean()
data_PLoE_Sh_wM=data_PLoE_Sh.writing_score.mean()
data_PLoE_Sh_gM=(data_PLoE_Sh_mM+data_PLoE_Sh_rM+data_PLoE_Sh_wM)
#
data_ploe_gM={"Associate's Degree":data_PLoE_As_gM,"Bachelor's Degree":data_PLoE_Ba_gM,"High School":data_PLoE_Hs_gM,"Master's Degree":data_PLoE_Ma_gM,"Some College":data_PLoE_Sc_gM,"Some High School":data_PLoE_Sh_gM}
print("General Scores of PLoE")
print("")
print("Most Successful Group of PLoE")
print("")
a=max(data_ploe_gM.keys(), key=(lambda k: data_ploe_gM[k]))
print(a)
print("")
print("Least Successful Group of PLoE")
print("")
b=min(data_ploe_gM.keys(), key=(lambda k: data_ploe_gM[k]))
print(b)
