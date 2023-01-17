# import the libraries here: 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import os

from collections import Counter

import seaborn as sns
# Load the data files in data-frames: 



files = {}

for filename in os.listdir('../input/kaggle-survey-2019/'):

    if filename.endswith('.csv'):

        files[str(filename[:-4])]    = pd.read_csv('../input/kaggle-survey-2019/'+filename,low_memory=False)

print("all the files have been read: ")        

for keys, value in files.items():

    print(keys)
countries = files["multiple_choice_responses"]

germany = countries['Q3'] == "Germany" # you can put other country here!

mcr_germany = files["multiple_choice_responses"][germany]

print('Shape of the subset is:', str(mcr_germany.shape))

print('Shape of the whole data is : ', str(countries.shape))

print('Number of Countries involved in the data is : ', len(set(countries['Q3'])))
#set(mcr_germany['Q10'].dropna())

target_salary = ['100,000-124,999','125,000-149,999','150,000-199,999',

                '200,000-249,999','250,000-299,999','300,000-500,000',

                '60,000-69,999','70,000-79,999','80,000-89,999',

                 '90,000-99,999','> $500,000','50,000-59,999']

mcr_final= mcr_germany[mcr_germany['Q10'].isin(target_salary)]

print('shape of the final subset is:', str(mcr_final.shape))
#Plot the salary category counts:

letter_counts = Counter(mcr_final["Q10"])

df = pd.DataFrame.from_dict(letter_counts, orient='index')

yy = df[0].values[:]

xx = df.index.values[:]

fig=plt.figure(figsize=(15, 5), dpi= 80, facecolor='w', edgecolor='w')

plt.bar(xx,yy, color='grey')

plt.xticks(xx, rotation=80,fontsize=12)

for i, v in enumerate(xx):

    plt.text(i-.25, 

              yy[i]+1, 

              yy[i], 

              fontsize=18, 

              color='k')

plt.ylim([0,65])   

plt.xlabel('Salary',fontsize=20)

plt.ylabel('Count',fontsize=20)

plt.title('Salary Category Counts', fontsize=25)

plt.show()
mcr_high_salary = mcr_germany[mcr_germany['Q10'].isin(['125,000-149,999','150,000-199,999',

                                                       '200,000-249,999','250,000-299,999',

                                                       '300,000-500,000','> $500,000'])]

# these rae high salary groups for my decision making
#function for plotting bar plots of counts of all available data in a loop :



def plot_bars_for_ds (target_group, lon=15, lat=5, color='red'):

    '''

    lon          = size of x axis in the plot

    lat          = size of y axis in the plot

    target_group = subset of the DataFrame

    color        = color of the bars in barplots

    '''

    

   

    

    

    

    for question in sorted(set(target_group.columns)): # loop over all the columns!

        if question in set(target_group.columns):      # Ceck if I plotted it before, I am droping the colomns which are plotted!



            if target_group[question].isnull().all():  # If all observations in a column are nan, then continue the loop!



                continue



            if len(question) > 5:



                if question[0] == "T":                 # Ignore time from start to finish!

                    continue

                if question[-10:] == "OTHER_TEXT" :    # OTHER_TEXT columns 





                    mcr_hs_part = []

                    if question[2] == "_":

                        mcr_hs_part = target_group[question[0:2]+"_OTHER_TEXT"] #1 to 9







                    else:

                        mcr_hs_part = target_group[question[0:3]+"_OTHER_TEXT"] # 10 to end









                    xx = mcr_hs_part.index.values[:]



                    # now plot it

                    fig=plt.figure(figsize=(lon, lat), dpi= 80, facecolor='w', edgecolor='w')

                    chart = sns.countplot(x=xx, data=mcr_hs_part,color=color )



                    if len(question) > 5:



                        if question[2] == "_":

                            plt.title(files["questions_only"][question[0:2]][0],fontsize=18)

                        else:

                            plt.title(files["questions_only"][question[0:3]][0],fontsize=18)

                    else:

                        plt.title(files["questions_only"][question][0],fontsize=18)

                    plt.ylabel('Number of People', fontsize = 16.0) # Y label

                    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

                    plt.show()



                else:



                    if question[0:3] == "Q14" : #Q14 has 5 parts only









                        mcr_hs_part = target_group["Q14_Part_1_TEXT"]

                        mcr_hs_part.columns = question[0:3]

                        mcr_hs_part.name = "Q14_Part_1_TEXT"

                        for prt in range(2,6):

                            if "Q14_Part_"+str(prt)+"_TEXT" in set(target_group.columns):





                                s = target_group["Q14_Part_"+str(prt)+"_TEXT"]

                                s.name = "Q14_Part_"+str(prt)+"_TEXT"



                                mcr_hs_part = mcr_hs_part.append( s)

                                target_group = target_group.drop(["Q14_Part_"+str(prt)+"_TEXT"], axis=1)

                        target_group = target_group.drop(["Q14_Part_1_TEXT"], axis=1)

                        xx = mcr_hs_part.index.values[:]





                        fig=plt.figure(figsize=(lon, lat), dpi= 80, facecolor='w', edgecolor='w')

                        chart = sns.countplot(x=xx, data=mcr_hs_part, color=color)



                        if len(question) > 5:



                            plt.title(files["questions_only"][question[0:3]][0],fontsize=18)

                        else:

                            plt.title(files["questions_only"][question][0],fontsize=18)

                        plt.ylabel('Number of People', fontsize = 16.0) # Y label



                        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

                        plt.show()

                        continue





                    mcr_hs_part = []

                    if question[2] == "_":

                        mcr_hs_part = target_group[question[0:2]+"_Part_1"]

                        mcr_hs_part.name = question[0:2]+"_Part_1"

                        mcr_hs_part.columns = question[0:2]

                        for prt in range(2,13):

                            if question[0:2]+"_Part_"+str(prt) in set(target_group.columns):





                                s = target_group[question[0:2]+"_Part_"+str(prt)]

                                s.name = question[0:2]+"_Part_"+str(prt)

                                mcr_hs_part = mcr_hs_part.append( s)

                                target_group = target_group.drop([question[0:2]+"_Part_"+str(prt)], axis=1)

                        target_group = target_group.drop([question[0:2]+"_Part_1"], axis=1)





                    else:               

                        mcr_hs_part = target_group[question[0:3]+"_Part_1"]

                        mcr_hs_part.name = question[0:3]+"_Part_1"

                        mcr_hs_part.columns = question[0:3]





                        for prt in range(2,13):

                            if question[0:3]+"_Part_"+str(prt) in set(target_group.columns):





                                s = target_group[question[0:3]+"_Part_"+str(prt)]

                                s.name = question[0:3]+"_Part_"+str(prt)

                                s.columns = question[0:3]

                                mcr_hs_part = mcr_hs_part.append(s)

                                target_group = target_group.drop([question[0:3]+"_Part_"+str(prt)], axis=1)

                        target_group = target_group.drop([question[0:3]+"_Part_1"], axis=1)



                    xx = mcr_hs_part.index.values[:]





                    fig=plt.figure(figsize=(lon, lat), dpi= 80, facecolor='w', edgecolor='w')

                    chart = sns.countplot(x=xx, data=mcr_hs_part , color = color)



                    if len(question) > 5: # there columns which are long including the ones with Part !



                        if question[2] == "_":

                            plt.title(files["questions_only"][question[0:2]][0],fontsize=18)

                        else:

                            plt.title(files["questions_only"][question[0:3]][0],fontsize=18)

                    else:

                        plt.title(files["questions_only"][question][0],fontsize=18)

                    plt.ylabel('Number of People', fontsize = 16.0) # Y label



                    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

                    plt.show()

            else: 



                fig=plt.figure(figsize=(lon, lat), dpi= 80, facecolor='w', edgecolor='w')

                xx = target_group[question].index.values[:]





                chart = sns.countplot(x=xx, data=target_group[question] , color=color)

                if len(question) > 5:



                    plt.title(files["questions_only"][question[0:3]][0],fontsize=18)

                else:

                    plt.title(files["questions_only"][question][0],fontsize=18)

                plt.ylabel('Number of People', fontsize = 16.0) # Y label

                chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

                plt.show()

                

            
# subset the data for high salaries in Germany: 

mcr_high_salary = mcr_germany[mcr_germany['Q10'].isin(['125,000-149,999','150,000-199,999',

                                                       '200,000-249,999','250,000-299,999',

                                                       '300,000-500,000','> $500,000'])]



plot_bars_for_ds (target_group=mcr_high_salary,lon=15, lat=6, color='salmon')
# subset the data for normal salaries in Germany: 

mcr_lower_salary = mcr_germany[mcr_germany['Q10'].isin(['60,000-69,999','70,000-79,999','80,000-89,999',

                                                        '90,000-99,999','50,000-59,999'])]



plot_bars_for_ds (target_group=mcr_lower_salary,lon=15, lat=6, color='skyblue')





