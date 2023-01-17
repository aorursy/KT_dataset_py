import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense



import matplotlib

import os

import matplotlib.pyplot as plt

import seaborn as sns

#%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

import seaborn as sns

plt.style.use('seaborn-white')
multiple_choice_responses = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
!pip install pywaffle

from pywaffle import Waffle
male_columns = multiple_choice_responses[multiple_choice_responses['Q2']=='Male'] # DataFrame containing male data



female_columns = multiple_choice_responses[multiple_choice_responses['Q2']=='Female'] # DataFrame containing female data



male_count = male_columns['Q2'].value_counts() # Counting the number of males



female_count = female_columns['Q2'].value_counts() # Counting the number of females



male_count = male_count[0]  # converts to number instead of Series



female_count = female_count[0]  # converts to number instead of Series



male_percent = int((male_count/(male_count+female_count))*100)      # Percentage of males

female_percent = int((female_count/(male_count+female_count))*100)  # Percentage of females





data = { 'Male'  : male_percent, 'Female' : female_percent}



fig = plt.figure(

    FigureClass=Waffle, 

    rows=8,

    values=data, 

    colors=("#232066", "#983D3D"),

    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},

    icons=['male','female'],

    icon_size=18,

    icon_legend=True

)



fig.set_tight_layout(False)
maleCountryCount = male_columns['Q3'].value_counts()[:59].reset_index() # variable for storing number of male respondents from each country





percentageCountry = [] # percentage of males from each country



for i in maleCountryCount.Q3:

  percentageCountry.append( (i / sum(maleCountryCount.Q3.values)) * 100 )



x_p = np.arange(len(maleCountryCount['index'].values)) # coordinates for labels and bars



plt.figure(figsize=(15, 7))

plt.bar(x_p, percentageCountry, align='center',color = 'blue')

plt.xticks(x_p, maleCountryCount['index'].values,rotation = 90)

plt.ylabel('Percentage of Males')

plt.title('Country of residence')



plt.show()
maleQualificationCount = male_columns['Q4'].value_counts().reset_index() # variable for storing qualification of  male respondents 



percentageQualification = [] # percentage of males from each country



for i in maleQualificationCount.Q4:

  percentageQualification.append( (i / sum(maleQualificationCount.Q4.values)) * 100 )



x_p = np.arange(len(maleQualificationCount['index'].values))



fig, ax = plt.subplots()



ax.barh(x_p, percentageQualification, align='center')  #used barh for making it look pulchritudinous

ax.set_yticks(x_p) 

ax.set_yticklabels(maleQualificationCount['index'].values)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Percentage Of Males')

ax.set_title('Qualifications of Males ')



plt.show()



maleEarningsCount = male_columns['Q10'].value_counts()[:59].reset_index() # variable for storing qualification of  male respondents 

maleEarningsCount = maleEarningsCount.reindex([0,7,18,19,20,10,15,1,9,8,12,3,4,5,6,11,16,17,2,13,14,21,24,23,22])



percentageEarnings_m = [] # percentage of males from each country



for i in maleEarningsCount.Q10:

  percentageEarnings_m.append( (i / sum(maleEarningsCount.Q10.values)) * 100 )



x_p = np.arange(len(maleEarningsCount['index'].values))





plt.figure(figsize=(15, 5))



plt.bar(x_p, percentageEarnings_m,color = 'blue',width = 0.8)

plt.title('Men In Each Income Bracket')

plt.xlabel('Incomes')  

plt.ylabel('Percentage Of Males')

plt.xticks(x_p,maleEarningsCount['index'].values,rotation=85)

plt.legend(['Male'])

plt.show()  

#### Drawing a simple line plot ####

#### Writing a function for storing the cumulative freqency values for each income bracket was too daunting for me so ###

#### I used my Casio fx-991 ES PLUS calculator instead ###

#### I double checked the values so rest assured ###

plt.figure(figsize=(15, 7))

plt.plot([1000, 2000, 3000, 4000,5000,7500,10000,15000,20000,25000,30000,40000,50000,60000,70000,80000,90000,100000,125000,150000,200000,250000,300000,500000,1000000],

         [1170,1640,1956,2206,2435,2880,3234,3936,4382,4839,5250,5885,6494,7100,7598,8038,8377,8700,9350,9756,10124,10271,10334,10398,10473])
femaleCountryCount = female_columns['Q3'].value_counts()[:59].reset_index() # variable for storing number of female respondents from each country





percentageCountry = [] # percentage of females from each country



for i in femaleCountryCount.Q3:

  percentageCountry.append( (i / sum(femaleCountryCount.Q3.values)) * 100 )



x_p = np.arange(len(femaleCountryCount['index'].values)) # coordinates for labels and bars



plt.figure(figsize=(15, 7))

plt.bar(x_p, percentageCountry, align='center',color = '#FF5106')

plt.xticks(x_p, femaleCountryCount['index'].values,rotation = 90)

plt.ylabel('Percentage of Females')

plt.title('Country of residence')



plt.show()



femaleEarningsCount = female_columns['Q10'].value_counts()[:59].reset_index() # variable for storing qualification of  male respondents 

femaleEarningsCount = femaleEarningsCount.reindex([0,2,11,20,17,7,19,1,8,14,13,4,3,5,10,9,15,18,6,12,16,21,23,22,24 ])

percentageEarnings_f = [] # percentage of females from each country



for i in femaleEarningsCount.Q10:

  percentageEarnings_f.append( (i / sum(femaleEarningsCount.Q10.values)) * 100 )



x_p = np.arange(len(femaleEarningsCount['index'].values))



plt.figure(figsize=(15, 5))



plt.bar(x_p, percentageEarnings_f,color = '#FF5106',width = 0.8)

plt.title('Women In Each Income Bracket')

plt.xlabel('Incomes')  

plt.ylabel('Percentage Of Females')

plt.xticks(x_p,maleEarningsCount['index'].values,rotation=85)

plt.legend(['Female'])

plt.show()  





    
### Again some casio magic. ###



plt.plot([1000, 2000, 3000, 4000,5000,7500,10000,15000,20000,25000,30000,40000,50000,60000,70000,80000,90000,100000,125000,150000,200000,250000,300000,500000,1000000],

         [318,436,505,555,611,695,745,863,939,1002,1066,1154,1253,1340,1411,1483,1541,1593,1680,1747,1803,1815,1817,1825,1827],color = '#FF5106')
femaleQualificationCount = female_columns['Q4'].value_counts().reset_index() # variable for storing qualification of  female respondents 



percentageQualification = [] # percentage of females from each country



for i in femaleQualificationCount.Q4:

  percentageQualification.append( (i / sum(femaleQualificationCount.Q4.values)) * 100 )



x_p = np.arange(len(femaleQualificationCount['index'].values))



fig, ax = plt.subplots()



ax.barh(x_p, percentageQualification, align='center',color = '#FF5106')  #used barh for making it look pulchritudinous

ax.set_yticks(x_p) 

ax.set_yticklabels(femaleQualificationCount['index'].values)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Percentage Of Females')

ax.set_title('Qualifications of Females ')



plt.show()

df = multiple_choice_responses[['Q2','Q10']]



df = df.iloc[1:,]  # dropping first row as it contains column labels



#df['Q10'].value_counts()



df['Q10'] = df.Q10.map({'$0-999' : 500,   # Converting the range to numbers for easier  calculations and graphs

'1,000-1,999': 1500,

'2,000-2,999' : 2500,

'3,000-3,999' : 3500,

'4,000-4,999': 4500,

'5,000-7,499' : 6250,

'7,500-9,999' : 8750,

'10,000-14,999' : 12500,

'15,000-19,999' : 17500,

'20,000-24,999' : 22500,

'25,000-29,999':  27500,

'30,000-39,999': 35000,

'40,000-49,999': 45000,

'50,000-59,999': 55000,

'60,000-69,999': 65000,

'70,000-79,999': 75000,

'80,000-89,999': 85000,

'90,000-99,999': 95000,

'100,000-124,999': 112500,

'125,000-149,999': 137500,

'150,000-199,999': 175000,

'200,000-249,999': 225000,

'250,000-299,999': 275000,

'300,000-500,000': 400000,

'> $500,000': 750000})



df.columns  = ['Gender','Income']  # naming columns



sns.catplot(data = (df.loc[((df.Gender == 'Female') | (df.Gender == 'Male')) & (df.Income.notnull())]), x='Gender', y='Income', kind='violin', split=True)

plt.title('Approximated Pay Distribution By Gender')
df = multiple_choice_responses

salary_mapping = {'$0-999':'low', '1,000-1,999':'low', 

                  '10,000-14,999':'low', '100,000-124,999':'high',

                  '125,000-149,999':'high', '15,000-19,999':'low', 

                  '150,000-199,999':'high', '2,000-2,999':'low',

                  '20,000-24,999':'low', '200,000-249,999':'high', 

                  '25,000-29,999':'medium', '250,000-299,999':'high',    # mapping the ranges to high,low and medium

                  '3,000-3,999':'low','30,000-39,999':'medium',

                  '300,000-500,000':'high', '4,000-4,999':'low',

                  '40,000-49,999':'medium', '5,000-7,499':'low', 

                  '50,000-59,999':'medium', '60,000-69,999':'medium',

                  '7,500-9,999':'low', '70,000-79,999':'medium', 

                  '80,000-89,999':'medium', '90,000-99,999':'medium',

                  '> $500,000':'high'}



# create new column for the income group and convert the old salary

df['income_group'] = df['Q10'].map(salary_mapping)



df = df[df.income_group.notnull()]

df = df[df.Q2 != 'Prefer not to say']

df = df[df.Q2 != 'Prefer to self-describe']



df = df[['Q2','income_group']]



males = df[df.Q2 == 'Male']

females = df[df.Q2 == 'Female']



malesHighRollers = males.groupby('income_group')['Q2'].count()[0]



femalesHighRollers = females.groupby('income_group')['Q2'].count()[0]



malesLowRollers = males.groupby('income_group')['Q2'].count()[1]



femalesLowRollers = females.groupby('income_group')['Q2'].count()[1]



malesMediumRollers = males.groupby('income_group')['Q2'].count()[2]



femalesMediumRollers = females.groupby('income_group')['Q2'].count()[2]



fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))



recipe = ['Male','Female']



data =  [malesHighRollers , femalesHighRollers]

ingredients = [x.split()[-1] for x in recipe]





def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%".format(pct, absolute)





wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),

                                  textprops=dict(color="w"))



ax.legend(wedges, ingredients,

          title="Genders",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))



plt.setp(autotexts, size=8, weight="bold")



ax.set_title("Males and Females in High Income Bracket")



plt.show()



fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))



recipe = ['Male','Female']



data =  [malesLowRollers , femalesLowRollers]

ingredients = [x.split()[-1] for x in recipe]





def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%".format(pct, absolute)





wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),

                                  textprops=dict(color="w"))



ax.legend(wedges, ingredients,

          title="Genders",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))



plt.setp(autotexts, size=8, weight="bold")



ax.set_title("Males and Females in Low Income Bracket")



plt.show()



fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))



recipe = ['Male','Female']



data =  [malesMediumRollers , femalesMediumRollers]

ingredients = [x.split()[-1] for x in recipe]





def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%".format(pct, absolute)





wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),

                                  textprops=dict(color="w"))



ax.legend(wedges, ingredients,

          title="Genders",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1))



plt.setp(autotexts, size=8, weight="bold")



ax.set_title("Males and Females in Medium Income Bracket")



plt.show()



maleJobCount = male_columns['Q5'].value_counts().reset_index() # variable for storing number of respondents in each job





percentageJob_m = [] # percentage of males in each job



for i in maleJobCount.Q5:

  percentageJob_m.append( (i / sum(maleJobCount.Q5.values)) * 100 )





femaleJobCount = female_columns['Q5'].value_counts().reset_index() # variable for storing number of respondents in each job





percentageJob_f = [] # percentage of females in each job



for i in femaleJobCount.Q5:

  percentageJob_f.append( (i / sum(femaleJobCount.Q5.values)) * 100 )



x_p = np.arange(len(maleJobCount['index'].values))



plt.figure(figsize=(15, 7))

plt.bar(x_p-0.2,percentageJob_m, width=0.4, label='Males', color='blue',alpha = 0.8)

plt.bar(x_p+0.2, percentageJob_f, width=0.4, label='Females', color = '#FF5106')

#give title

plt.title(' Percentage of Men and Women In Each Job')

plt.xticks(x_p,femaleJobCount['index'].values,rotation=90)

plt.xlabel('Job Title')

plt.ylabel('Percentage of Gender')

#plt.plot(ypos, Year) will remove the numbers along the y-axis

#shows legend

plt.legend()

#show to show the graph

plt.show()

labels = ('Males','Females')

sizes = [maleJobCount['Q5'][0],femaleJobCount['Q5'][1]]

explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Females')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title("Percentages of Data Scientists Who Are Males And Females", bbox={'facecolor':'0.8', 'pad':5})

plt.show() 
maleDataScientistColumns = male_columns[male_columns['Q5']=='Data Scientist'] # DataFrame containing male data



femaleDataScientistColumns = female_columns[female_columns['Q5']=='Data Scientist'] # DataFrame containing female data



maleEarningsCount = maleDataScientistColumns['Q10'].value_counts()[:59].reset_index() 

maleEarningsCount = maleEarningsCount.reindex([0, 11, 18 , 19, 21, 17 , 16, 3 , 14, 10, 12,  4, 2, 5, 8, 9, 15, 13, 1, 6, 7, 20 , 22, 24, 23])



percentageEarnings_m = [] # percentage of males 



for i in maleEarningsCount.Q10:

  percentageEarnings_m.append( (i / sum(maleEarningsCount.Q10.values)) * 100 )







femaleEarningsCount = femaleDataScientistColumns['Q10'].value_counts().reset_index()  

femaleEarningsCount = femaleEarningsCount.reindex([0, 8, 14 , 18 , 20 , 19 , 17, 7, 13, 16, 10, 4 ,3 ,9 ,6 , 12, 11, 15, 1,5 , 2 , 21, 24,22 , 23])

femaleEarningsCount.at[24,'index']= '250,000-299,999'

femaleEarningsCount.at[24,'Q10']= 0



femaleEarningsCount.at[23,'index']= '> 500,000'

femaleEarningsCount.at[23,'Q10']= 0



percentageEarnings_f = [] # percentage of females 



for i in femaleEarningsCount['Q10']:

  percentageEarnings_f.append( (i / sum(femaleEarningsCount['Q10'].values)) * 100 )



x_p = np.arange(len(femaleEarningsCount['index'].values))



x = x_p

new_x = x_p

plt.figure(figsize=(15, 5))

# the first call is as usual

plt.bar(new_x, percentageEarnings_m,color = 'blue')



# the second one is special to create stacked bar plots

plt.bar(new_x, percentageEarnings_f, bottom=percentageEarnings_m, color= '#FF5106')

plt.title('Percentage of Male and Female Data Scientists In Each Income Bracket')

plt.xlabel('Incomes')  

plt.ylabel('Number Of People')

plt.xticks(new_x,maleEarningsCount['index'].values,rotation=85)

plt.legend(['Male','Female'])

plt.show()  



# Here we calculate the average and plot it

incomes = [499.5, 1499.5 ,2499.5 ,3499.5, 4499.5, 6249.5, 8749.5, 12499.5, 17499.5, 22499.5, 27499.5, 34999.5, 44999.5, 54999.5, 64999.5, 74999.5, 84999.5, 94999.5, 112499.5, 137499.5, 174999.5,224999.5, 274999.5,400000,750000]





maleDataScientist_average = sum(np.multiply(maleEarningsCount['Q10'],incomes)) / sum(maleEarningsCount['Q10'])





femaleDataScientist_average = sum(np.multiply(femaleEarningsCount['Q10'],incomes)) / sum(femaleEarningsCount['Q10'])



objects = ('Male','Female')



plt.figure(figsize=(15, 10))

plt.bar(0,maleDataScientist_average , width=0.4, label='Males', color='blue')

plt.bar(1, femaleDataScientist_average, width=0.4, label='Females', color = '#FF5106')

plt.xticks([0,1], objects)

plt.ylabel('Average Earnings')

plt.title('Average Earnings Of Male and Female Data Scientists')



plt.show()





##############

# Et Voila ! #

##############



# Disclaimer : Please take everything with a grain of salt, it is very hard to convey sarcasm in a kaggle kernel.
# Here we prepare the data to feed the neural network with



df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

df = df[['Q2','Q3','Q4','Q10']]



df = df.iloc[1:,]



df['Q10'] = df.Q10.map({'$0-999' : 500,

'1,000-1,999': 1500,

'2,000-2,999' : 2500,

'3,000-3,999' : 3500,

'4,000-4,999': 4500,

'5,000-7,499' : 6250,

'7,500-9,999' : 8750,

'10,000-14,999' : 12500,

'15,000-19,999' : 17500,

'20,000-24,999' : 22500,

'25,000-29,999':  27500,

'30,000-39,999': 35000,

'40,000-49,999': 45000,

'50,000-59,999': 55000,

'60,000-69,999': 65000,

'70,000-79,999': 75000,

'80,000-89,999': 85000,

'90,000-99,999': 95000,

'100,000-124,999': 112500,

'125,000-149,999': 137500,

'150,000-199,999': 175000,

'200,000-249,999': 225000,

'250,000-299,999': 275000,

'300,000-500,000': 400000,

'> $500,000': 750000})



cleanup_nums = {"Q2": {"Male": 1, "Female": 2,"Prefer not to say" : 3,'Prefer to self-describe':4}} # label encoding

df.replace(cleanup_nums, inplace=True) # implementing the gender label encoding



# filling null values

df = df.fillna({"Q10": "0"})  

df = df.fillna({"Q2": "0"})

df = df.fillna({"Q3": "0"})

df = df.fillna({"Q4": "0"})



# label encoding other columns

df[["Q3"]] = df[["Q3"]].astype('category')  

df[["Q4"]] = df[["Q4"]].astype('category')



df["Q3"] = df["Q3"].cat.codes

df["Q4"] = df["Q4"].cat.codes





# main input and output x and y



x = df[['Q2','Q3','Q4']]

y = df['Q10']



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=40)  #splitting into test and train

#print(X_train.shape); print(X_test.shape)



# model formation

model = Sequential()

model.add(Dense(128, input_dim=3, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(1))

model.compile(optimizer= 'rmsprop',loss = "mse", metrics=['mae'])

model.fit(X_train,y_train, epochs=50, verbose=0)



test_males = X_test[X_test['Q2']==1]  # males in test set

test_females = X_test[X_test['Q2']==2]   #females in test set



#predictions

m = model.predict(test_males)  



f = model.predict(test_females)
average_males = sum(m) / len(m)



average_females = sum(f) / len(f)



objects = ('Male','Female')



plt.figure(figsize=(15, 10))

plt.bar(0,average_males , width=0.4, label='Males', color='blue')

plt.bar(1,average_females, width=0.4, label='Females', color = '#FF5106')

plt.xticks([0,1], objects)

plt.ylabel('Average Earnings')

plt.title('Average Earnings Predicted By The Model')



plt.show()

# this section is dedicated entirely to the scatter plot



men = X_test[X_test['Q2']==1]

women = X_test[X_test['Q2']==2]



men_pred = model.predict(men)

women_pred = model.predict(women)



men = men.drop(['Q3','Q4'],axis = 1)

women = women.drop(['Q3','Q4'],axis =1)



men.loc[men['Q2'] == 1, 'Q2'] = 'Male'

women.loc[women['Q2'] == 2, 'Q2'] = 'Female'



men['income'] = men_pred

women['income'] = women_pred



tips = pd.concat([men,women],axis = 0)



tips.columns = ['Gender','Incomes']



plt.style.use('seaborn-whitegrid')



sns.catplot(x='Gender', y='Incomes',data = tips)


# making box plot

sns.boxplot( x=tips["Gender"], y=tips["Incomes"] )