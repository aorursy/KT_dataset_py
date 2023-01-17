#Add all the import statements here

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import pandas as pd

import seaborn as sns







df = pd.read_csv('../input/mlgday1featureengineering/analytic_data2020.csv',skiprows=1)

df.head()
# Correlation Heatmap



#Adult smoking : v009_rawvalue

#Excessive drinking : v049_rawvalue

#Obesity : v011_rawvalue

#Physical Inactivity: v070_rawvalue

#Access to excercise :v132_rawvalue

#STD: v045_rawvalue





df_health = df[['v009_rawvalue','v049_rawvalue','v011_rawvalue','v070_rawvalue','v132_rawvalue','v045_rawvalue']].sample(200, random_state = 20)

df_health.dropna(inplace = True)

corr = df_health.corr()



xlabels = ['Smoking','Drinking','Obesity','Physical Inactivity','Access to excercise','STD']



# plot the heatmap

sns.heatmap(corr, 

        xticklabels=xlabels,

        yticklabels=xlabels)
#Plotting the relationship between smoking and death



df_sub_smokers_death = df[['v001_rawvalue','v009_rawvalue']].sample(200, random_state = 20)

df_sub_smokers_death.dropna(inplace = True)

ax = df_sub_smokers_death.plot(x = 'v009_rawvalue', y='v001_rawvalue', kind = 'scatter')

ax.set_xlabel("Smokers")

ax.set_ylabel("Deaths")



print("Correlation between smoking and dying")

df_sub_smokers_death['v009_rawvalue'].corr(df_sub_smokers_death['v001_rawvalue'])
# Correlation between Smoking and drinking alcohol



#Smokers Raw Value : v009_rawvalue

#Excessive Drinking Raw Value : v049_rawvalue



df_sub_smokers_alcoholics = df[['v009_rawvalue','v049_rawvalue']].sample(200, random_state = 20)

df_sub_smokers_alcoholics.dropna(inplace = True)

ax = df_sub_smokers_alcoholics.plot(x = 'v009_rawvalue', y='v049_rawvalue', kind = 'scatter')

ax.set_xlabel("Smokers")

ax.set_ylabel("Excessive Drinking")



print("Correlation between drinking excessive alcohol and smoking")

df_sub_smokers_alcoholics['v009_rawvalue'].corr(df_sub_smokers_alcoholics['v049_rawvalue'])
# Correlation between excessive drinking and alcohol impaired driving deaths



#Excessive Drinking Raw Value : v049_rawvalue

#Alcohol Impaired Driving Deaths : v134_rawvalue





df_sub_alcohol_drive = df[['v049_rawvalue','v134_rawvalue']].sample(200, random_state = 20)

df_sub_alcohol_drive.dropna(inplace = True)

ax = df_sub_alcohol_drive.plot(x = 'v049_rawvalue', y='v134_rawvalue', kind = 'scatter')

ax.set_xlabel("Excessive Drinking")

ax.set_ylabel("Drink & Drive deaths")



print("Correlation between drinking excessive alcohol and alcohol driving deaths")

df_sub_alcohol_drive['v049_rawvalue'].corr(df_sub_alcohol_drive['v134_rawvalue'])
#Correlation between excessive drinking and STD



#Excessive Drinking Raw Value : v049_rawvalue

#Sexually transmitted diseases : v045_rawvalue



df_sub_alcohol_STD = df[['v049_rawvalue','v045_rawvalue']].sample(200, random_state = 20)

df_sub_alcohol_STD.dropna(inplace = True)

ax = df_sub_alcohol_STD.plot(x = 'v049_rawvalue', y='v045_rawvalue', kind = 'scatter')

ax.set_xlabel("Excessive Drinking")

ax.set_ylabel("STD")



print("Correlation between drinking excessive alcohol and STD")

df_sub_alcohol_STD['v049_rawvalue'].corr(df_sub_alcohol_STD['v045_rawvalue'])

#Correlation between excessive drinking and Violent crimes



#Excessive Drinking Raw Value : v049_rawvalue

#Violent Crimes : v043_rawvalue



df_sub_alcohol_crime = df[['v049_rawvalue','v043_rawvalue']].sample(200, random_state = 20)

df_sub_alcohol_crime.dropna(inplace = True)

ax = df_sub_alcohol_crime.plot(x = 'v049_rawvalue', y='v043_rawvalue', kind = 'scatter')

ax.set_xlabel("Excessive Drinking")

ax.set_ylabel("Violent Crimes")



print("Correlation between drinking excessive alcohol and crimes")

df_sub_alcohol_crime['v049_rawvalue'].corr(df_sub_alcohol_crime['v043_rawvalue'])



# [12:30 AM] Gupta, Mallika

    

#Plotting the relationship between disp and income





df_sub_smokers_death = df[['v142_rawvalue','v063_rawvalue']].sample(200, random_state = 20)

df_sub_smokers_death.dropna(inplace = True)

ax = df_sub_smokers_death.plot(x = 'v063_rawvalue', y='v142_rawvalue', kind = 'scatter')

ax.set_xlabel("Disp")

ax.set_ylabel("Income")





print("Correlation between disp and income")

df_sub_smokers_death['v063_rawvalue'].corr(df_sub_smokers_death['v142_rawvalue'])





#Plotting the relationship between disp and food insec



 



df_sub_smokers_death = df[['v142_rawvalue','v139_rawvalue']].sample(200, random_state = 20)

df_sub_smokers_death.dropna(inplace = True)

ax = df_sub_smokers_death.plot(x = 'v139_rawvalue', y='v142_rawvalue', kind = 'scatter')

ax.set_xlabel("Disp")

ax.set_ylabel("food insec")



 



print("Correlation between disp and food insec")

df_sub_smokers_death['v139_rawvalue'].corr(df_sub_smokers_death['v142_rawvalue'])