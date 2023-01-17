import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/survey_2014.csv')

data.dropna()

data.shape
data.head()
data.nunique()
data['Gender'].unique()
gender_replace = {

    "M":"Male",

    "Male":"Male",

    "m":"Male",

    "Male-ish":"Male",

    "maile":"Male",

    "Mal":"Male",

    "Male (CIS)":"Male",

    "Make":"Male",

    "Guy (-ish) ^_^":"Male",

    "male leaning androgynous":"Male",

    "Male ":"Male",

    "Man":"Male",

    "msle":"Male",

    "Mail":"Male",

    "cis male":"Male",

    "Malr":"Male",

    "Cis Man":"Male",

    "ostensibly male, unsure what that really means":"Male",

    "male":"Male",

    "something kinda male?":"Male",

    "Cis Male":"Male",

    "Female":"Female",

    "female":"Female",

    "Trans-female":"Female",

    "Cis Female":"Female",

    "F":"Female",

    "f":"Female",

    "queer/she/they":"Female",

    'Woman':"Female",

    "woman":"Female",

    "Female ":"Female",

    "cis-female/femme":"Female",

    "Trans woman":"Female",

    "Female (trans)":"Female",

    "queer":"Female",

    "Female (cis)":"Female",

    "femail":"Female",

    "non-binary":"Unkown",

    "Femake":"Female",

    "Nah":"Unkown",

    "All":"Unkown",

    "Enby":"Unkown",

    "fluid":"Unkown",

    "Genderqueer":"Unkown",

    "Androgyne":"Unkown",

    "Agender":"Unkown",

    "A little about you":"Unkown",

    "p":"Unkown",

    "Unspecified":"Unkown",

    "Neuter":"Unkown"

    

}





data['Gender']=data['Gender'].apply(lambda i: gender_replace[i])
data['Gender'].value_counts()
data_survey = pd.read_csv('../input/subtitle_survey.csv', index_col='var')

data_survey
f1, ax1 = plt.subplots()

ax1.pie(list(data['Gender'].value_counts()), 

                   labels=['Male','Female','Unkown'],

                  autopct='%1.1f%%',startangle=90)

ax1.axis('equal')

ax1.set_title("Gender")

f2, ax2 = plt.subplots()

n, bins, patches = plt.hist(data['Age'],20,weights=np.ones_like(data['Age'].clip(10,80))/len(data['Age'].clip(10,80)),range=(10,80))

ax2.set_title("Age")

age=data['Age'].clip(10,80)

pd.DataFrame(data=age).describe()
f3, ax3= plt.subplots(nrows=2,ncols=2, figsize=(20,10))

ax3[0,0].pie(list(data['no_employees'].value_counts()), 

                   labels=['6-25', '26-100', '>1000','100-500','1-5','500-1000'],

                  autopct='%1.1f%%', startangle=90)

ax3[1,0].pie(list(data['remote_work'].value_counts()),

                                     labels=['No', 'Yes'],

                                     autopct='%1.1f%%', startangle=0)

ax3[0,1].pie(list(data['tech_company'].value_counts()),

                                     labels=['Yes','No'],

                                     autopct='%1.1f%%', startangle=0)

ax3[1,1].pie(list(data['self_employed'].value_counts()),

                                     labels=['No','Yes'],

                                     autopct='%1.1f%%', startangle=0)

ax3[0,0].axis('equal')

ax3[1,0].axis('equal')

ax3[1,1].axis('equal')

ax3[0,1].axis('equal')

ax3[0,0].set_title('Size') 

ax3[1,0].set_title('Remote Work >50% of the time?')

ax3[1,1].set_title('Self-employed?')

ax3[0,1].set_title('Tech Company?')

data['leave'].unique()
f4, ax4= plt.subplots(nrows=2,ncols=3, figsize=(20,10))

ax4[0,0].pie(list(data['benefits'].value_counts()), 

                   labels=['Yes', "Don't know", 'No'],

                  autopct='%1.1f%%', startangle=90)

ax4[1,0].pie(list(data['care_options'].value_counts()),

                                     labels=['Not sure', 'No', 'Yes'],

                                     autopct='%1.1f%%', startangle=0)

ax4[0,1].pie(list(data['wellness_program'].value_counts()),

                                     labels=['No', "Don't know", 'Yes'],

                                     autopct='%1.1f%%', startangle=0)

ax4[1,1].pie(list(data['seek_help'].value_counts()),

                                     labels=['Yes', "Don't know", 'No'],

                                     autopct='%1.1f%%', startangle=0)

ax4[0,2].pie(list(data['anonymity'].value_counts()),

                                     labels=['Yes', "Don't know", 'No'],

                                     autopct='%1.1f%%', startangle=0)

ax4[1,2].pie(list(data['leave'].value_counts()),

                                     labels=['Somewhat easy', "Don't know", 'Somewhat difficult',

       'Very difficult', 'Very easy'],

                                     autopct='%1.1f%%', startangle=0)

ax4[0,0].axis('equal')

ax4[1,0].axis('equal')

ax4[1,1].axis('equal')

ax4[0,1].axis('equal')

ax4[0,2].axis('equal')

ax4[1,2].axis('equal')





ax4[0,0].set_title('Does your employer provide mental health benefits?') 

ax4[1,0].set_title('Do you know options for mental health care \n your employer provides?')

ax4[1,1].set_title('Does your employer provide resources to seek help?')

ax4[0,1].set_title('Has your employer ever discussed \n mental health wellness program?')

ax4[0,2].set_title('Is your anonymity protected if you choose to \n take advantage of mental health resources?')

ax4[1,2].set_title('How easy is it for you to take medical leave \n for a mental health condition?')
f5, ax5= plt.subplots(nrows=2,ncols=3, figsize=(20,10))

ax5[0,0].pie(list(data['mental_health_consequence'].value_counts()), 

                   labels=['Yes', "Don't know", 'No'],

                  autopct='%1.1f%%', startangle=90)

ax5[1,0].pie(list(data['phys_health_consequence'].value_counts()),

                                     labels=['No', 'Yes', 'Maybe'],

                                     autopct='%1.1f%%', startangle=0)

ax5[0,1].pie(list(data['coworkers'].value_counts()),

                                     labels=['Some of them', 'No', 'Yes'],

                                     autopct='%1.1f%%', startangle=0)

ax5[1,1].pie(list(data['supervisor'].value_counts()),

                                     labels=['Yes', 'No', 'Some of them'],

                                     autopct='%1.1f%%', startangle=0)

ax5[0,2].pie(list(data['mental_health_interview'].value_counts()),

                                     labels=['No', 'Yes', 'Maybe'],

                                     autopct='%1.1f%%', startangle=0)

ax5[1,2].pie(list(data['phys_health_interview'].value_counts()),

                                     labels=['Maybe', 'No', 'Yes'],

                                     autopct='%1.1f%%', startangle=0)

ax5[0,0].axis('equal')

ax5[1,0].axis('equal')

ax5[1,1].axis('equal')

ax5[0,1].axis('equal')

ax5[0,2].axis('equal')

ax5[1,2].axis('equal')





ax5[0,0].set_title('Do you think that discussing a mental health issue \n with your employer would have negative consequences?') 

ax5[1,0].set_title('Do you think that discussing a physical health issue \n with your employer would have negative consequences?')

ax5[1,1].set_title('Would you be willing to discuss \n a mental health issue with your direct supervisor(s)?')

ax5[0,1].set_title('Would you be willing to discuss \n a mental health issue with your coworkers?')

ax5[0,2].set_title('Would you bring up a mental health issue \n with a potential employer in an interview?')

ax5[1,2].set_title('Would you bring up a physical health issue \n with a potential employer in an interview?')


company_policy = [ "benefits","care_options","wellness_program","seek_help","anonymity","leave"]

willingness=["mental_health_consequence",'phys_health_consequence','coworkers','supervisor','mental_health_interview','phys_health_interview']
def fac(x):

    for i in x:

        if i!='Yes'and i!='No':

            x.replace(i,0,inplace =True)

            break

    x.replace("Yes",1,inplace =True)

    x.replace("No",-1,inplace=True)
var_3=["benefits","care_options","wellness_program","seek_help","anonymity","mental_health_consequence",'phys_health_consequence','coworkers','supervisor','mental_health_interview','phys_health_interview']

#data = pd.read_csv('../input/survey_2014.csv')

for v in var_3:

    fac(data[v])



data['leave'].replace({'Somewhat easy':1,"Don't know":0,'Somewhat difficult':-1,'Very easy':2,'Very difficult':-2},inplace=True)



data.head()

data[company_policy+willingness].corr()