import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MultiResponses = pd.read_csv('../input/multipleChoiceResponses.csv', low_memory=False)
MultiResponses = MultiResponses.replace({'Q3': 'Republic of Korea'}, 'South Korea')
#Official name of South Korea is Republic of Korea, survey had two selections on one country
MultiResponses = MultiResponses.replace({'Q3': 'United States of America'}, 'USA')
MultiResponses = MultiResponses.replace({'Q3': 'United Kingdom of Great Britain and Northern Ireland'}, 'UK')
#Using Short Name
UserCountCountry = pd.DataFrame(MultiResponses['Q3']).rename(columns={'Q3' : 'Country'})
UserCountCountry = UserCountCountry.drop([0]).dropna()  
UserCountCountry = UserCountCountry[(~UserCountCountry['Country'].isin(['Other','I do not wish to disclose my location']))]

fig = plt.figure(figsize=(8,20))
fig.add_subplot(1,1,1)
ax = sns.countplot(y="Country",
                   data=UserCountCountry,
                   order = UserCountCountry['Country'].value_counts().index,
                   palette="GnBu_d")

UserViolin = pd.DataFrame(MultiResponses[['Q2','Q3','Q7','Q24','Q26']]).rename(columns={'Q2' : 'Age','Q3' : 'Country','Q7' : 'Job','Q24' : 'Code_Year','Q26' : 'Are_You_DS'})
UserViolin = UserViolin[(UserViolin['Country'].isin(['USA','UK','Russia', 'India','Germany', 'China', 'Brazil' ]))]
UserViolin = UserViolin.replace({'Age': '18-21'}, 19)
UserViolin = UserViolin.replace({'Age': '22-24'}, 23)
UserViolin = UserViolin.replace({'Age': '25-29'}, 27)
UserViolin = UserViolin.replace({'Age': '30-34'}, 32)
UserViolin = UserViolin.replace({'Age': '35-39'}, 37)
UserViolin = UserViolin.replace({'Age': '40-44'}, 42)
UserViolin = UserViolin.replace({'Age': '45-49'}, 47)
UserViolin = UserViolin.replace({'Age': '50-54'}, 52)
UserViolin = UserViolin.replace({'Age': '55-59'}, 57)
UserViolin = UserViolin.replace({'Age': '60-69'}, 64)
UserViolin = UserViolin.replace({'Age': '70-79'}, 74)
UserViolin = UserViolin.replace({'Age': '80+'}, 80)
UserViolin = UserViolin[((UserViolin['Job'] == 'I am a student') & (UserViolin['Code_Year'] != 'I have never written code and I do not want to learn'))
                        | ((UserViolin['Job'] != 'I am a student') & (UserViolin['Are_You_DS'].isin(['Definitely yes','Probably yes','Maybe'])))]
UserViolin = UserViolin.dropna()  
UserViolin['Marker'] = np.where(UserViolin['Job'] == 'I am a student', 'Student', 'Data Scientist')
UserViolin = UserViolin.sort_values(by='Marker', ascending=True).sort_values(by='Country', ascending=False)


sns.set()
f, ax = plt.subplots(figsize=(20, 10))
ax = sns.violinplot(x="Country", y="Age", hue="Marker", data=UserViolin, palette="muted", split=True, scale="count", inner="quartile", scale_hue=True, bw=.4).set_title('Age Distribution of Data Scientist / Student (Scaled)',fontsize=20)
sns.set()
f, ax = plt.subplots(figsize=(20, 10))
ax = sns.violinplot(x="Country", y="Age", hue="Marker", data=UserViolin, palette="muted", split=True, scale="count", inner="quartile", scale_hue=False, bw=.4).set_title('Age Distribution of Data Scientist / Student (Actual)',fontsize=20)
Student = pd.DataFrame(MultiResponses[['Q7','Q4','Q3']]).rename(columns={'Q7' : 'Job', 'Q4' : 'Education', 'Q3' : 'Country'})
Student = Student[(Student['Job'] == 'I am a student')]
Student = Student[(Student['Country'].isin(['USA','UK','Russia', 'India','Germany', 'China', 'Brazil']))]
Counter = 1
Student['#Student'] = Counter

sns.set()
Student = Student.pivot_table(values='Job', index='Education', columns='Country', aggfunc='count')
Student = Student[['USA','UK','Russia', 'India','Germany', 'China', 'Brazil']]
Student = Student.reindex(['I prefer not to answer',
                             'No formal education past high school',
                             'Some college/university study without earning a bachelor’s degree',
                             'Bachelor’s degree',
                             'Professional degree',
                             'Master’s degree',
                             'Doctoral degree'])

f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(Student, cmap="Blues", vmin=-50, vmax=500, annot=True, fmt=".0f", linewidths=.5, ax=ax).set_title('Degrees attained or plan to attain within the next 2 years in Student Group',fontsize=15)
UserIncome = pd.DataFrame(MultiResponses[['Q9','Q3','Q7','Q24','Q26']]).rename(columns={'Q9' : 'Compensation','Q3' : 'Country','Q7' : 'Job','Q24' : 'Code_Year','Q26' : 'Are_You_DS'})
UserIncome = UserIncome[(UserIncome['Country'].isin(['USA','UK','Russia', 'India','Germany', 'China', 'Brazil']))]
UserIncome = UserIncome.replace({'Compensation': '0-10,000'}, 5000)
UserIncome = UserIncome.replace({'Compensation': '10-20,000'}, 15000)
UserIncome = UserIncome.replace({'Compensation': '20-30,000'}, 25000)
UserIncome = UserIncome.replace({'Compensation': '30-40,000'}, 35000)
UserIncome = UserIncome.replace({'Compensation': '40-50,000'}, 45000)
UserIncome = UserIncome.replace({'Compensation': '50-60,000'}, 55000)
UserIncome = UserIncome.replace({'Compensation': '60-70,000'}, 65000)
UserIncome = UserIncome.replace({'Compensation': '70-80,000'}, 75000)
UserIncome = UserIncome.replace({'Compensation': '80-90,000'}, 85000)
UserIncome = UserIncome.replace({'Compensation': '90-100,000'}, 95000)
UserIncome = UserIncome.replace({'Compensation': '100-125,000'}, 112500)
UserIncome = UserIncome.replace({'Compensation': '125-150,000'}, 137500)
UserIncome = UserIncome.replace({'Compensation': '150-200,000'}, 187500)
UserIncome = UserIncome.replace({'Compensation': '200-250,000'}, 225000)
UserIncome = UserIncome.replace({'Compensation': '250-300,000'}, 275000)
UserIncome = UserIncome.replace({'Compensation': '300-400,000'}, 350000)
UserIncome = UserIncome.replace({'Compensation': '400-500,000'}, 450000)
UserIncome = UserIncome.replace({'Compensation': '500,000+'}, 500000)
UserIncome = UserIncome[(UserIncome['Compensation'] != 'I do not wish to disclose my approximate yearly compensation')]
UserIncome['Compensation'] = UserIncome['Compensation'].astype('float', copy=False)
UserIncome = UserIncome[((UserIncome['Job'] == 'I am a student') & (UserIncome['Code_Year'] != 'I have never written code and I do not want to learn'))
                        | ((UserIncome['Job'] != 'I am a student') & (UserIncome['Are_You_DS'].isin(['Definitely yes','Probably yes','Maybe'])))]
UserIncome = UserIncome.dropna()  
UserIncome['Marker'] = np.where(UserIncome['Job'] == 'I am a student', 'Student', 'Data Scientist')
UserIncome = UserIncome.sort_values(by='Marker', ascending=True).sort_values(by='Country', ascending=False)


sns.set()
f, ax = plt.subplots(figsize=(20, 10))
ax = sns.violinplot(x="Country", y="Compensation", hue="Marker", data=UserIncome, palette="muted", split=True, scale="count", inner="quartile", scale_hue=True, bw=.4).set_title('Compensation Distribution of Data Scientist / Student (Scaled)',fontsize=20)


sns.set()
f, ax = plt.subplots(figsize=(20, 10))
ax = sns.violinplot(x="Country", y="Compensation", hue="Marker", data=UserIncome, palette="muted", split=True, scale="count", inner="quartile", scale_hue=False, bw=.4).set_title('Compensation Distribution of Data Scientist / Student (Actual)',fontsize=20)
UserJobRatio = pd.DataFrame(MultiResponses[['Q3','Q7','Q6','Q24','Q26']]).rename(columns={'Q3' : 'Country','Q7' : 'Job','Q6' : 'Title','Q24' : 'Code_Year','Q26' : 'Are_You_DS'})
UserJobRatio = UserJobRatio[(UserJobRatio['Country'].isin(['USA','UK','Russia', 'India','Germany', 'China', 'Brazil' ]))]
UserJobRatio = UserJobRatio[((UserJobRatio['Job'] == 'I am a student') & (UserJobRatio['Code_Year'] != 'I have never written code and I do not want to learn'))
                        | ((UserJobRatio['Job'] != 'I am a student') & (UserJobRatio['Are_You_DS'].isin(['Definitely yes','Probably yes','Maybe'])))]
UserJobRatio = UserJobRatio.dropna()  
UserJobRatio['Marker'] = np.where(UserJobRatio['Job'] == 'I am a student', 'Student', 'Data Scientist')
UserJobRatio = UserJobRatio.sort_values(by='Marker', ascending=True).sort_values(by='Country', ascending=False)

UserJobRatio_Summary = pd.DataFrame(UserJobRatio[['Country','Marker']])
Counter = 1
UserJobRatio_Summary['#Counter'] = Counter
Pivot = pd.pivot_table(UserJobRatio_Summary, values='#Counter', index=['Country'], columns=['Marker'], aggfunc=np.sum)
Pivot['Ratio'] = Pivot['Data Scientist'] / Pivot['Student']
Pivot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

CodeTimeIncome = pd.DataFrame(MultiResponses[['Q9','Q3','Q7','Q8','Q26']]).rename(columns={'Q9' : 'Compensation','Q3' : 'Country','Q7' : 'Job','Q8' : 'Exp_Year','Q26' : 'Are_You_DS'})
CodeTimeIncome = CodeTimeIncome[(CodeTimeIncome['Country'].isin(['USA','UK','Russia', 'India']))]

CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '0-10,000'}, 5000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '10-20,000'}, 15000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '20-30,000'}, 25000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '30-40,000'}, 35000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '40-50,000'}, 45000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '50-60,000'}, 55000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '60-70,000'}, 65000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '70-80,000'}, 75000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '80-90,000'}, 85000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '90-100,000'}, 95000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '100-125,000'}, 112500)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '125-150,000'}, 137500)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '150-200,000'}, 187500)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '200-250,000'}, 225000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '250-300,000'}, 275000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '300-400,000'}, 350000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '400-500,000'}, 450000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '500,000+'}, 500000)
CodeTimeIncome = CodeTimeIncome[(CodeTimeIncome['Compensation'] != 'I do not wish to disclose my approximate yearly compensation')]
CodeTimeIncome['Compensation'] = CodeTimeIncome['Compensation'].astype('float', copy=False)

CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '0-1'}, 0.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '1-2'}, 1.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '2-3'}, 2.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '3-4'}, 3.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '4-5'}, 4.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '5-10'}, 7.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '10-15'}, 12.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '15-20'}, 17.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '20-25'}, 22.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '25-30'}, 27.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '30 +'}, 30.0)
CodeTimeIncome['Exp_Year'] = CodeTimeIncome['Exp_Year'].astype('float', copy=False)

CodeTimeIncome = CodeTimeIncome[((CodeTimeIncome['Job'] != 'I am a student') & (CodeTimeIncome['Are_You_DS'].isin(['Definitely yes','Probably yes','Maybe'])))]
CodeTimeIncome = CodeTimeIncome.dropna()

CodeTimeIncome = CodeTimeIncome.sort_values(by='Country', ascending=False)

sns.lmplot(x="Exp_Year", y="Compensation", col="Country", data=CodeTimeIncome, aspect=.5, height=8, fit_reg=True, x_estimator=np.mean);
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

CodeTimeIncome = pd.DataFrame(MultiResponses[['Q9','Q3','Q7','Q8','Q26']]).rename(columns={'Q9' : 'Compensation','Q3' : 'Country','Q7' : 'Job','Q8' : 'Exp_Year','Q26' : 'Are_You_DS'})
CodeTimeIncome = CodeTimeIncome[(CodeTimeIncome['Country'].isin(['Germany', 'China', 'Brazil']))]

CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '0-10,000'}, 5000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '10-20,000'}, 15000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '20-30,000'}, 25000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '30-40,000'}, 35000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '40-50,000'}, 45000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '50-60,000'}, 55000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '60-70,000'}, 65000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '70-80,000'}, 75000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '80-90,000'}, 85000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '90-100,000'}, 95000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '100-125,000'}, 112500)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '125-150,000'}, 137500)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '150-200,000'}, 187500)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '200-250,000'}, 225000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '250-300,000'}, 275000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '300-400,000'}, 350000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '400-500,000'}, 450000)
CodeTimeIncome = CodeTimeIncome.replace({'Compensation': '500,000+'}, 500000)
CodeTimeIncome = CodeTimeIncome[(CodeTimeIncome['Compensation'] != 'I do not wish to disclose my approximate yearly compensation')]
CodeTimeIncome['Compensation'] = CodeTimeIncome['Compensation'].astype('float', copy=False)

CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '0-1'}, 0.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '1-2'}, 1.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '2-3'}, 2.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '3-4'}, 3.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '4-5'}, 4.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '5-10'}, 7.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '10-15'}, 12.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '15-20'}, 17.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '20-25'}, 22.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '25-30'}, 27.5)
CodeTimeIncome = CodeTimeIncome.replace({'Exp_Year': '30 +'}, 30.0)
CodeTimeIncome['Exp_Year'] = CodeTimeIncome['Exp_Year'].astype('float', copy=False)

CodeTimeIncome = CodeTimeIncome[((CodeTimeIncome['Job'] != 'I am a student') & (CodeTimeIncome['Are_You_DS'].isin(['Definitely yes','Probably yes','Maybe'])))]
CodeTimeIncome = CodeTimeIncome.dropna()

CodeTimeIncome = CodeTimeIncome.sort_values(by='Country', ascending=False)

sns.lmplot(x="Exp_Year", y="Compensation", col="Country", data=CodeTimeIncome, aspect=.5, height=8, fit_reg=True, x_estimator=np.mean);
English = pd.DataFrame(
    {'USA' : [ '296603003' , '283160411', '95.46'],
     'UK' : [ '64000000' , '59600000', '97.74'],
     'Russia' : [ '138312535' , '7574303', '5.48'],
     'India' : ['1028737436' , '125344737', '12.18'],
     'Germany' : [ '80600000' , '45400000', '56.00'],
     'China' : [ '1210000000' , '10000000', '0.01'],
     'Brazil' : [ '205000000' , '10542000', '5.00']},
     index = ['Population','English Speaker','English Speaker%'])
English
UserIncome = pd.DataFrame(MultiResponses[['Q9','Q3','Q7','Q24','Q26']]).rename(columns={'Q9' : 'Compensation','Q3' : 'Country','Q7' : 'Job','Q24' : 'Code_Year','Q26' : 'Are_You_DS'})
UserIncome = UserIncome[(UserIncome['Country'].isin(['USA','Japan','Australia','China','South Korea','Singapore' ,'India']))]
UserIncome = UserIncome.replace({'Compensation': '0-10,000'}, 5000)
UserIncome = UserIncome.replace({'Compensation': '10-20,000'}, 15000)
UserIncome = UserIncome.replace({'Compensation': '20-30,000'}, 25000)
UserIncome = UserIncome.replace({'Compensation': '30-40,000'}, 35000)
UserIncome = UserIncome.replace({'Compensation': '40-50,000'}, 45000)
UserIncome = UserIncome.replace({'Compensation': '50-60,000'}, 55000)
UserIncome = UserIncome.replace({'Compensation': '60-70,000'}, 65000)
UserIncome = UserIncome.replace({'Compensation': '70-80,000'}, 75000)
UserIncome = UserIncome.replace({'Compensation': '80-90,000'}, 85000)
UserIncome = UserIncome.replace({'Compensation': '90-100,000'}, 95000)
UserIncome = UserIncome.replace({'Compensation': '100-125,000'}, 112500)
UserIncome = UserIncome.replace({'Compensation': '125-150,000'}, 137500)
UserIncome = UserIncome.replace({'Compensation': '150-200,000'}, 187500)
UserIncome = UserIncome.replace({'Compensation': '200-250,000'}, 225000)
UserIncome = UserIncome.replace({'Compensation': '250-300,000'}, 275000)
UserIncome = UserIncome.replace({'Compensation': '300-400,000'}, 350000)
UserIncome = UserIncome.replace({'Compensation': '400-500,000'}, 450000)
UserIncome = UserIncome.replace({'Compensation': '500,000+'}, 500000)
UserIncome = UserIncome[(UserIncome['Compensation'] != 'I do not wish to disclose my approximate yearly compensation')]
UserIncome['Compensation'] = UserIncome['Compensation'].astype('float', copy=False)
UserIncome = UserIncome[((UserIncome['Job'] == 'I am a student') & (UserIncome['Code_Year'] != 'I have never written code and I do not want to learn'))
                        | ((UserIncome['Job'] != 'I am a student') & (UserIncome['Are_You_DS'].isin(['Definitely yes','Probably yes','Maybe'])))]
UserIncome = UserIncome.dropna()  
UserIncome['Marker'] = np.where(UserIncome['Job'] == 'I am a student', 'Student', 'Data Scientist')
UserIncome = UserIncome.sort_values(by='Country', ascending=True).sort_values(by='Marker', ascending=True)


sns.set()
f, ax = plt.subplots(figsize=(20, 10))
ax = sns.violinplot(x="Country", y="Compensation", hue="Marker", data=UserIncome, palette="muted", split=True, scale="count", inner="quartile", scale_hue=True, bw=.4).set_title('Compensation Distribution of Data Scientist / Student (Scaled)',fontsize=20)
Employer_Chance = pd.DataFrame(MultiResponses[['Q10','Q3','Q7','Q26']]).rename(columns={'Q10' : 'ML_Status','Q3' : 'Country','Q7' : 'Job', 'Q26' : 'Are_You_DS'})
Employer_Chance = Employer_Chance[(Employer_Chance['Country'].isin(['USA','UK','Russia', 'India','Germany', 'China', 'Brazil','Japan','Australia','South Korea','Singapore']))]
Employer_Chance = Employer_Chance[((Employer_Chance['Job'] != 'I am a student') & (Employer_Chance['Are_You_DS'].isin(['Definitely yes','Probably yes','Maybe'])))]
Employer_Chance = Employer_Chance.dropna()

Counter = 1

Employer_Chance_Total = Employer_Chance
Employer_Chance_Total = pd.DataFrame(Employer_Chance_Total[['ML_Status', 'Country']])
Employer_Chance_Total['#Counter'] = Counter
Pivot_Total = pd.pivot_table(Employer_Chance_Total, values='#Counter', index=['ML_Status'], columns=['Country'], aggfunc=np.sum)
Pivot_Total = Pivot_Total[['USA','UK','Russia', 'India','Germany', 'China', 'Brazil','Japan','Australia','South Korea','Singapore']]
Pivot_Total = Pivot_Total.reindex(['I do not know',
         'No (we do not use ML methods)',
         'We are exploring ML methods (and may one day put a model into production)',
         'We use ML methods for generating insights (but do not put working models into production)',
         'We recently started using ML methods (i.e., models in production for less than 2 years)',
         'We have well established ML methods (i.e., models in production for more than 2 years)'  ])


f, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(Pivot_Total, cmap="Blues", vmin=-50, vmax=700, annot=True, fmt="d", linewidths=.5, ax=ax).set_title('Q10. Does your current employer incorporate machine learning methods into their business?',fontsize=15)
Student = pd.DataFrame(MultiResponses[['Q7','Q4','Q3']]).rename(columns={'Q7' : 'Job', 'Q4' : 'Education', 'Q3' : 'Country'})
Student = Student[(Student['Job'] == 'I am a student')]
Student = Student[(Student['Country'].isin(['USA','UK','Russia', 'India','Germany', 'China', 'Brazil','Japan','Australia','South Korea','Singapore']))]
Counter = 1
Student['#Student'] = Counter

sns.set()
Student = Student.pivot_table(values='Job', index='Education', columns='Country', aggfunc='count')
Student = Student[['USA','UK','Russia', 'India','Germany', 'China', 'Brazil','Japan','Australia','South Korea','Singapore']]
Student = Student.reindex(['I prefer not to answer',
                             'No formal education past high school',
                             'Some college/university study without earning a bachelor’s degree',
                             'Bachelor’s degree',
                             'Professional degree',
                             'Master’s degree',
                             'Doctoral degree'])

f, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(Student, cmap="Blues", vmin=-50, vmax=500, annot=True, fmt=".0f", linewidths=.5, ax=ax).set_title('Degrees attained or plan to attain within the next 2 years in Student Group',fontsize=15)