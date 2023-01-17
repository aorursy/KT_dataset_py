# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
age_data = pd.read_csv("../input/prison-in-india/Age_group.csv")

caste_data = pd.read_csv("../input/prison-in-india/Caste.csv")

death_sent_data = pd.read_csv("../input/prison-in-india/Death_sentence.csv")

domicile_data = pd.read_csv("../input/prison-in-india/Domicile.csv")

education_data= pd.read_csv("../input/prison-in-india/Education.csv")

education_fac_data = pd.read_csv("../input/prison-in-india/Education_facilities.csv")

convicted_data = pd.read_csv("../input/prison-in-india/IPC_crime_inmates_convicted.csv")

under_trial_data = pd.read_csv("../input/prison-in-india/IPC_crime_inmates_under_trial.csv")

inmate_death_data = pd.read_csv("../input/prison-in-india/Inmates_death.csv")

inmate_escape_data = pd.read_csv("../input/prison-in-india/Inmates_escapee.csv")

jail_population_data = pd.read_csv("../input/prison-in-india/Jail wise population of prison inmates.csv")

prison_details_data = pd.read_csv("../input/prison-in-india/Prison_details_2015.csv")





import seaborn as sns



state = caste_data.groupby(['state_name'])['convicts'].sum().sort_values(ascending= True)

state = state.tail(10)



state2 = caste_data.groupby(['state_name'])['under_trial'].sum().sort_values(ascending=True)

state2 = state2.tail(10)



fig, axarr = plt.subplots(2, 1, figsize=(15, 18))



state.plot.barh(

    ax=axarr[0] , fontsize=12, color='red'

)

axarr[0].set_title("Number of Convicts in Top 10 States", fontsize=18)

state2.plot.barh(

    ax=axarr[1], fontsize=12, color='green'

)

axarr[1].set_title("Number of Under Trial People in Top 10 States", fontsize=18)

#plt.subplots_adjust(hspace=.4)

#gender groups

gender = caste_data.groupby(['gender'])['convicts'].sum()



gender2 = caste_data.groupby(['gender'])['under_trial'].sum()





#caste groups

caste = caste_data.groupby(['caste'])['convicts'].sum()



caste2 = caste_data.groupby(['caste'])['under_trial'].sum()





explode = (0, 0.1, 0, 0) 

exploder = (0, 0.1)

fig, axarr = plt.subplots(2, 2, figsize=(20, 20))



gender.plot.pie(

    ax=axarr[0][0] , fontsize=12 , figsize=(16,16) ,startangle=90 ,autopct='%1.1f%%', shadow=True , explode=exploder

)

axarr[0][0].set_title("Gender of Convicts", fontsize=18)



gender2.plot.pie(

    ax=axarr[0][1], fontsize=12 ,figsize=(16,16) ,startangle=90 ,autopct='%1.1f%%' ,shadow=True, explode=exploder

)

axarr[0][1].set_title("Gender of Under Trial People", fontsize=18)



caste.plot.pie(

    ax=axarr[1][0] ,fontsize=12 ,figsize=(16,16) , startangle=90 ,autopct='%1.1f%%' , shadow=True, explode=explode

)

axarr[1][0].set_title("Caste distribution of Convicts", fontsize=18)



caste2.plot.pie(

    ax=axarr[1][1], fontsize=12, figsize=(16,16) ,startangle=90 ,autopct='%1.1f%%' ,shadow=True, explode=explode

)

axarr[1][1].set_title("Caste Distribution of Under Trial People", fontsize=18)







plt.subplots_adjust(hspace=.4)



year = caste_data.groupby(['year'])['convicts'].sum()

year = year/1000

year2 = caste_data.groupby(['year'])['under_trial'].sum()

year2 = year2/1000

fig, axarr = plt.subplots(2, 1, figsize=(15, 20))



year.plot.bar(

    ax=axarr[0] , fontsize=12, color='red'

)

axarr[0].set_title("Number of Convicts per Year", fontsize=18)

axarr[0].set_ylabel("number in thousends")

year2.plot.bar(

    ax=axarr[1], fontsize=12, color='blue'

)

axarr[1].set_title("Number of Under Trial People per Year", fontsize=18)

axarr[1].set_ylabel("number in thousends")

#plt.subplots_adjust(hspace=.8)
death_sent_data



punish_state = death_sent_data.groupby(['state_name'])['no_capital_punishment'].sum().sort_values(ascending=True)



punish_state = punish_state.tail(10)



lifeimp_state = death_sent_data.groupby(['state_name'])['no_life_imprisonment'].sum().sort_values(ascending=True)



lifeimp_state = lifeimp_state.tail(10)

fig , axarr = plt.subplots(2 , 1 , figsize =(15 , 20))



punish_state.plot.barh(

             ax = axarr[0] , color = "black"

)



axarr[0].set_title(" Capital Punishment in top 10 states", fontsize=18)

lifeimp_state.plot.barh(

             ax = axarr[1], color = "darkred"



)

axarr[1].set_title(" Life imprisonment in top 10 states", fontsize=18)





punish_state2 = death_sent_data.groupby(['year'])['no_capital_punishment'].sum()







lifeimp_state2 = death_sent_data.groupby(['year'])['no_life_imprisonment'].sum()





fig , axarr = plt.subplots(2 , 1 , figsize =(15 , 20))



punish_state2.plot.bar(

             ax = axarr[0] , color = "black"

)



axarr[0].set_title("Number of Capital Punishment per year", fontsize=18)

lifeimp_state2.plot.bar(

             ax = axarr[1], color = "darkred"



)

axarr[1].set_title("Number of Life imprisonment per year", fontsize=18)
#Religion Groups

religion = pd.read_csv("../input/prison-in-india/Religion.csv")

religion = religion.groupby(['gender'])['under_trial'].sum() #some problem with dataset religion is stored in gender column





#Education

education = education_data.groupby(['education'])['convicts'].sum()

explode = [0,0.1,0,0,0]

explode1 = [0.1,0,0,0,0,0]

fig, axarr = plt.subplots(1, 2, figsize=(25, 25))



religion.plot.pie(

    ax= axarr[0] , fontsize=12 ,figsize = (16,16) ,startangle=90 ,autopct='%1.1f%%', shadow=True , explode = explode , pctdistance=1.1, labeldistance=1.2

)

axarr[0].set_title("Religion distribution of Under Trial People", fontsize=18)



education.plot.pie(

    ax=axarr[1], fontsize=12 ,figsize = (16,16) ,startangle=90 ,autopct='%1.1f%%' ,shadow=True , explode = explode1

)

axarr[1].set_title(" Education of Convicts", fontsize=18)









crime_data = pd.read_csv("../input/prison-in-india/IPC_crime_inmates_convicted.csv")

crime = crime_data.groupby(['CRIME HEAD'])['Grand Total'].sum()







ax = crime.plot.pie(y = 'Grand Total',figsize = (10,10) ,startangle=90 ,labels = None ,autopct='%1.1f%%', shadow=True , pctdistance=1.1, labeldistance=1.2)

plt.legend(loc = 'best', labels = crime.index,fontsize = 12)

plt.tight_layout()

plt.title('Total Crime Distribution',size = 20)

ax.get_legend().set_bbox_to_anchor((1, 1)) #position of legend
crime_data['total'] = crime_data['Grand Total']

crime_data['crimes'] = crime_data['CRIME HEAD']

crimes2 =crime_data.groupby(['crimes', 'YEAR'])['total',].sum().reset_index() #adding comma at the end of total seems to work



arson = crimes2[crimes2.crimes== 'Arson']

attemptmurder = crimes2[crimes2.crimes== 'Attempt To Commit Murder']

burglary = crimes2[crimes2.crimes== 'Burglary']

culHomicide = crimes2[crimes2.crimes== 'C.H. Not Amounting To Murder'] #C.H is culpable Homicide not amounting to muerder

cheating = crimes2[crimes2.crimes== 'Cheating']

counterFeiting = crimes2[crimes2.crimes== 'Counter Feiting']

crimbot = crimes2[crimes2.crimes== 'Criminal Breach Of Trust']

cruelhusband = crimes2[crimes2.crimes== 'Cruelty By Husband Or Relative Of Husband']

dacoity = crimes2[crimes2.crimes== 'Dacoity']

dowry = crimes2[crimes2.crimes== 'Dowry Deaths']

eveteasing = crimes2[crimes2.crimes== 'Eve-Teasing']

extortion = crimes2[crimes2.crimes== 'Extortion']

kidnapping = crimes2[crimes2.crimes== 'Kidnapping And Abduction']

molestation = crimes2[crimes2.crimes== 'Molestation']

murder= crimes2[crimes2.crimes== 'Murder']

dowry = crimes2[crimes2.crimes== 'Dowry Deaths']

others = crimes2[crimes2.crimes== 'Other Ipc Crimes']

extortion = crimes2[crimes2.crimes== 'Extortion']

assedacoity = crimes2[crimes2.crimes== 'Prep. And Assembly For Dacoity']

rape = crimes2[crimes2.crimes== 'Rape']

riots = crimes2[crimes2.crimes== 'Riots']

robbery = crimes2[crimes2.crimes== 'Robbery']

theft = crimes2[crimes2.crimes== 'Thefts']







sns.set_style("darkgrid", {"axes.facecolor": ".9"})

sns.set_context("poster")



f, axes = plt.subplots(3, 2, figsize=(20, 20))

sns.lineplot(x = 'YEAR' , y = 'total',data = cruelhusband ,label='Cruelty by Husband/s Relatives' , marker = 'o',ax=axes[0 ,0])

axes[0,0].set_title('Cruelity by Husband / Husbands Relatives', size = 30)

sns.lineplot(x = 'YEAR' , y = 'total',data = eveteasing ,label= 'Eve Teasing' , marker = 'o',ax=axes[0 ,1])

axes[0,1].set_title('Eve Teasing', size = 30)

sns.lineplot(x = 'YEAR' , y = 'total',data = molestation ,label='Molestation', marker = 'o',ax=axes[1,0])

axes[1,0].set_title('Molestation', size = 30)

sns.lineplot(x = 'YEAR' , y = 'total',data = dowry ,label='Dowry' , marker = 'o',color = 'red' ,ax=axes[1 ,1])

axes[1,1].set_title('Dowry', size = 30)

sns.lineplot(x = 'YEAR' , y = 'total',data = rape ,label='Rape', marker = 'o',color = 'red' ,ax=axes[2 ,0])

axes[2,0].set_title('Rape', size = 30)

f.delaxes(axes[2,1])

plt.tight_layout()

plt.subplots_adjust(hspace=.5)


sns.set(style="darkgrid")

sns.set_context("talk")



f, axes = plt.subplots(3, 2, figsize=(20, 20))

sns.lineplot(x = 'YEAR' , y = 'total',data = burglary ,label='Burglary', marker = 'o',ax=axes[0][0])

axes[0,0].set_title('Burglary', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = dacoity ,label='Dacoity',color = 'red', marker = 'o',ax=axes[0][1])

axes[0,1].set_title('Dacoity', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = assedacoity ,label='Prep. and Assembly for Dacoity', marker = 'o',ax=axes[1][0])

axes[1,0].set_title('Prep and Assembly for Dacoity', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = robbery ,label='Robbery', marker = 'o',ax=axes[1][1])

axes[1,1].set_title('Robbery', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = theft ,label='Theft', marker = 'o',ax=axes[2][0])

axes[2,0].set_title('Theft', size = 20)

f.delaxes(axes[2,1])

plt.subplots_adjust(hspace=.8)

plt.tight_layout()


sns.set(style="darkgrid")

sns.set_context("talk")

f, axes = plt.subplots(2, 2, figsize=(20, 15))

sns.lineplot(x = 'YEAR' , y = 'total',data = cheating ,label='Cheating', marker = 'o',ax=axes[0,0], color = 'red')

axes[0,0].set_title('Cheating', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = counterFeiting ,label='Counter Feiting', marker = 'o',ax=axes[0 ,1])

axes[0,1].set_title('Counter Feiting', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = crimbot ,label='Criminal Breach of Trust', marker = 'o',ax=axes[1,0])

axes[1,0].set_title('Criminal Breach Of Trust', size = 20)

f.delaxes(axes[1,1])

plt.tight_layout()

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

sns.set_context("talk")

f, axes = plt.subplots(3, 2, figsize=(20, 15))

sns.lineplot(x = 'YEAR' , y = 'total',data = arson ,label='Arson', marker = 'o' , ax = axes[0][0])

axes[0,0].set_title('Arson', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = attemptmurder ,label='Attempt to Murder',color = 'red' ,marker = 'o', ax = axes[0][1])

axes[0,1].set_title('Attempt to Murder', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = culHomicide ,label='Culpable Homicide', marker = 'o', ax = axes[1][0])

axes[1,0].set_title('Culpable Homicide', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = kidnapping ,label='Kidnapping and Abduction', color = 'red' ,marker = 'o', ax = axes[1][1])

axes[1,1].set_title('Kidnapping and Abduction', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = others ,label='Others',marker = 'o', ax = axes[2][0])

axes[2,0].set_title('Others', size = 20)

sns.lineplot(x = 'YEAR' , y = 'total',data = riots ,label='Riots', marker = 'o', ax = axes[2][1])

axes[2,1].set_title('Riots', size = 20)



plt.tight_layout()

plt.figure(figsize=(20,15))

sns.set_context("poster")

sns.lineplot(x = 'YEAR' , y = 'total',data = murder ,label='Murder', marker = 'o', legend = False , color = 'red')



plt.legend(loc='upper left')

plt.title('Murder' , size = 30)
crime_data['states'] = crime_data['STATE/UT']

crimes3 =crime_data.groupby(['states', 'crimes'])['total',].sum().reset_index()

crimes3 = crimes3[crimes3.total>0]

crimes3 = crimes3[crimes3.states != 'Andman And Nicobar Islands']



arson2 = crimes3[crimes3.crimes== 'Arson']

arson2 = arson2.sort_values(by = 'total' ,ascending= False)

arson2 = arson2.head(10)



attemptmurder2 = crimes3[crimes3.crimes== 'Attempt To Commit Murder']

attemptmurder2 = attemptmurder2.sort_values(by = 'total' ,ascending= False)

attemptmurder2 = attemptmurder2.head(10)



burglary2 = crimes3[crimes3.crimes== 'Burglary']

burglary2 = burglary2.sort_values(by = 'total' ,ascending= False)

burglary2 = burglary2.head(10)



culHomicide2 = crimes3[crimes3.crimes== 'C.H. Not Amounting To Murder'] #C.H is culpable Homicide not amounting to muerder

culHomicide2= culHomicide2.sort_values(by = 'total' ,ascending= False)

culHomicide2 = culHomicide2.head(10)



cheating2 = crimes3[crimes3.crimes== 'Cheating']

cheating2 = cheating2.sort_values(by = 'total' ,ascending= False)

cheating2 = cheating2.head(10)



counterFeiting2 = crimes3[crimes3.crimes== 'Counter Feiting']

counterFeiting2 = counterFeiting2.sort_values(by = 'total' ,ascending= False)

counterFeiting2 = counterFeiting2.head(10)



crimbot2 = crimes3[crimes3.crimes== 'Criminal Breach Of Trust']

crimbot2 = crimbot2.sort_values(by = 'total' ,ascending= False)

crimbot2 = crimbot2.head(10)



cruelhusband2 = crimes3[crimes3.crimes== 'Cruelty By Husband Or Relative Of Husband']

cruelhusband2 = cruelhusband2.sort_values(by = 'total' ,ascending= False)

cruelhusband2 = cruelhusband2.head(10)



dacoity2 = crimes3[crimes3.crimes== 'Dacoity']

dacoity2 = dacoity2.sort_values(by = 'total' ,ascending= False)

dacoity2 = dacoity2.head(10)



dowry2 = crimes3[crimes3.crimes== 'Dowry Deaths']

dowry2 = dowry2.sort_values(by = 'total' ,ascending= False)

dowry2 = dowry2.head(10)



eveteasing2 = crimes3[crimes3.crimes== 'Eve-Teasing']

eveteasing2 = eveteasing2.sort_values(by = 'total' ,ascending= False)

eveteasing2 = eveteasing2.head(10)



extortion2 = crimes3[crimes3.crimes== 'Extortion']

extortion2 = extortion2.sort_values(by = 'total' ,ascending= False)

extortion2 = extortion2.head(10)



kidnapping2 = crimes3[crimes3.crimes== 'Kidnapping And Abduction']

kidnapping2 = kidnapping2.sort_values(by = 'total' ,ascending= False)

kidnapping2 = kidnapping2.head(10)



molestation2 = crimes3[crimes3.crimes== 'Molestation']

molestation2 = molestation2.sort_values(by = 'total' ,ascending= False)

molestation2 = molestation2.head(10)



murder2= crimes3[crimes3.crimes== 'Murder']

murder2 = murder2.sort_values(by = 'total' ,ascending= False)

murder2 = murder2.head(10)





others2 = crimes3[crimes3.crimes== 'Other Ipc Crimes']

others2 = others2.sort_values(by = 'total' ,ascending= False)

others2 = others2.head(10)



extortion2 = crimes3[crimes3.crimes== 'Extortion']

extortion2 = extortion2.sort_values(by = 'total' ,ascending= False)

extortion2 = extortion2.head(10)



assedacoity2 = crimes3[crimes3.crimes== 'Prep. And Assembly For Dacoity']

assedacoity2= assedacoity2.sort_values(by = 'total' ,ascending= False)

assedacoity2 = assedacoity2.head(10)



rape2 = crimes3[crimes3.crimes== 'Rape']

rape2 = rape2.sort_values(by = 'total' ,ascending= False)

rape2 = rape2.head(10)



riots2 = crimes3[crimes3.crimes== 'Riots']

riots2 = riots2.sort_values(by = 'total' ,ascending= False)

riots2 = riots2.head(10)



robbery2 = crimes3[crimes3.crimes== 'Robbery']

robbery2= robbery2.sort_values(by = 'total' ,ascending= False)

robbery2 = robbery2.head(10)



theft2 = crimes3[crimes3.crimes== 'Thefts']

theft2 = theft2.sort_values(by = 'total' ,ascending= False)

theft2 = theft2.head(10)

sns.set_style("darkgrid")

sns.set_context("talk")

f, axes = plt.subplots(2, 2, figsize=(20, 30))

sns.barplot(x = 'total' , y ='states',data = cheating2 ,label='Cheating',ax=axes[0,0])

axes[0,0].set_title('Cheating', size = 30)

sns.barplot(x = 'total' , y ='states',data = counterFeiting2 ,label='Counter Feiting', ax=axes[0 ,1])

axes[0,1].set_title('Counter Feiting', size = 30)

sns.barplot(x = 'total' , y ='states',data = crimbot2 ,label='Criminal Breach of Trust',ax=axes[1,0])

axes[1,0].set_title('Criminal Breach Of Trust', size = 30)

sns.barplot(x = 'total' , y ='states',data = murder2 ,label='Murder',color = 'red' ,ax = axes[1,1])

axes[1,1].set_title('Murder', size = 30)

plt.tight_layout()

sns.set_style("darkgrid")

sns.set_context("poster")

f, axes = plt.subplots(3, 2, figsize=(25, 30))

sns.barplot(x = 'total' , y ='states',data = arson2 , ax = axes[0][0])

axes[0,0].set_title('Arson', size = 30)

sns.barplot(x = 'total' , y ='states',data = attemptmurder2 ,label='Attempt to Murder',color = 'red' , ax = axes[0][1])

axes[0,1].set_title('Attempt to Murder', size = 30)

sns.barplot(x = 'total' , y ='states',data = culHomicide2 ,label='Culpable Homicide',  ax = axes[1][0])

axes[1,0].set_title('Culpable Homicide', size = 30)

sns.barplot(x = 'total' , y ='states',data = kidnapping2 ,label='Kidnapping and Abduction',  ax = axes[1][1])

axes[1,1].set_title('Kidnapping and Abduction', size = 30)

sns.barplot(x = 'total' , y ='states',data = others2 ,label='Others', ax = axes[2][0])

axes[2,0].set_title('Others', size = 30)

sns.barplot(x = 'total' , y ='states',data = riots2 ,label='Riots', ax = axes[2][1])

axes[2,1].set_title('Riots', size = 30)



plt.tight_layout()



sns.set(style="darkgrid")

sns.set_context("poster")



f, axes = plt.subplots(3, 2, figsize=(20, 30))

sns.barplot(x = 'total' , y ='states',data = burglary2 ,label='Burglary',ax=axes[0][0])

axes[0,0].set_title('Burglary', size = 30)

sns.barplot(x = 'total' , y ='states',data = dacoity2 ,label='Dacoity',color = 'red',ax=axes[0][1])

axes[0,1].set_title('Dacoity', size = 30)

sns.barplot(x = 'total' , y ='states',data = assedacoity2 ,label='Prep. and Assembly for Dacoity',ax=axes[1][0])

axes[1,0].set_title('Prep and Assembly for Dacoity', size = 30)

sns.barplot(x = 'total' , y ='states',data = robbery2 ,label='Robbery',ax=axes[1][1])

axes[1,1].set_title('Robbery', size = 30)

sns.barplot(x = 'total' , y ='states',data = theft2 ,label='Theft',ax=axes[2][0])

axes[2,0].set_title('Theft', size = 30)

f.delaxes(axes[2,1])

plt.subplots_adjust(hspace=.2)

plt.tight_layout()
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

sns.set_context("poster")



f, axes = plt.subplots(3, 2, figsize=(20, 30))

sns.barplot(x = 'total' , y ='states',data = cruelhusband2 ,label='Cruelty by Husband/s Relatives' ,ax=axes[0 ,0])

axes[0,0].set_title('Cruelity by Husband / Husbands Relatives', size = 30)

sns.barplot(x = 'total' , y ='states',data = eveteasing2 ,label= 'Eve Teasing' ,ax=axes[0 ,1])

axes[0,1].set_title('Eve Teasing', size = 30)

sns.barplot(x = 'total' , y ='states',data = molestation2 ,label='Molestation',ax=axes[1,0])

axes[1,0].set_title('Molestation', size = 30)

sns.barplot(x = 'total' , y ='states',data = dowry2 ,label='Dowry' ,ax=axes[1 ,1])

axes[1,1].set_title('Dowry', size = 30)

sns.barplot(x = 'total' , y ='states',data = rape2 ,label='Rape',color = 'red' ,ax=axes[2 ,0])

axes[2,0].set_title('Rape', size = 30)

f.delaxes(axes[2,1])

plt.tight_layout()

plt.subplots_adjust(hspace=.2)