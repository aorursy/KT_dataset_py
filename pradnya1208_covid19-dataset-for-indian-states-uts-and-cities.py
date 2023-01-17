import requests

import csv

import json

import pandas as pd

from pandas import DataFrame as df



import sys

import os



JSON_URL = 'https://api.covid19india.org/v4/min/timeseries.min.json'



DIR_NAME = 'Indian_States'



state_name  = []

dtcolNames =[]

covidData =[]

covidRec = []

covidDeath = []
req = requests.get(JSON_URL)





stateNames = df(req.json())

statesData = stateNames.T





for st in statesData.index:

    state_name.append(st)



unwanted_st = {'UN', 'Other State', 'Other Region', 'Other', 'Unknown'} 

state_name = [ele for ele in state_name if ele not in unwanted_st]





# # In case of fixed Dates

date_ = df(req.json()['TT'])

for dt in date_.index:

    dtcolNames.append(dt + ",")

    

dir_path = os.path.isdir(DIR_NAME)

if not dir_path:

    os.mkdir(DIR_NAME)

def name_correction(state):

        if state == 'TT':

            state = "India(Total cases)"

            return state

        

        if state == 'AN':

            state = "Andaman and Nikobar Islanda"

            return state

        

        if state == 'AP':

            state = "Andhra Pradesh"

            return state

        

        if state == 'AR':

            state = "Arunachal Pradesh"

            return state

        

        if state == 'AS':

            state = "Assam"

            return state

        

        if state == 'BR':

            state = "Bihar"

            return state

        

        if state == 'CH':

            state = "Chandigarh"

            return state

        

        if state == 'CT':

            state = "Chhattisgarh"    

            return state

        

        if state == 'DL':

            state = "Delhi"

            return state

        

        if state == 'DN':

            state = "Dadra and Nagar Haveli and Daman and Diu"

            return state

        

        if state == 'GA':

            state = "Goa"

            return state

        

        if state == 'GJ':

            state = "Gujarat"

            return state

        

        if state == 'HP':

            state = "Himachal Pradesh"

            return state

        

        if state == 'HR':

            state = "Haryana"

            return state

        

        if state == 'JH':

            state = "Jharkhand"

            return state

        

        if state == 'JK':

            state = "Jammu and Kashmir"

            return state

        

        if state == 'KA':

            state = "Karnataka"

            return state

        

        if state == 'KL':

            state = "Kerala"

            return state

        

        if state == 'LA':

            state = "Ladakh"

            return state

        

        if state == 'MH':

            state = "Maharashtra"

            return state

        

        if state == 'ML':

            state = "Meghalaya"

            return state

        

        if state == 'MN':

            state ="Manipur"

            return state

        

        if state == 'MP':

            state = "Madhya Pradesh"

            return state

        

        if state == 'MZ':

            state = "Mizoram"

            return state

            

        if state == 'NL':

            state = "Nagaland"

            return state

        

        if state == 'OR':

            state ="Odisha"

            return state

        

        if state == 'PB':

            state = "Punjab"

            return state

        

        if state == 'PY':

            state = "Puducherry"

            return state

        

        if state == 'RJ':

            state = "Rajasthan"

            return state

        

        if state == 'SK':

            state ="Sikkim"

            return state

        

        if state == 'TG':

            state ="Telangana"

            return state

        

        if state == 'TN':

            state ="Tamil Nadu"

            return state

        

        if state == 'TR':

            state = "Tripura"

            return state

        

        if state == 'UP':

            state ="Uttar Pradesh"

            return state

        

        if state == 'UT':

            state = "Uttarakhand"

            return state

        

        if state == 'WB':

            state = "West Bengal"

            return state

        
for state in state_name:   

    

    st_name = name_correction(state)

    print(st_name)

  

    i=0

    covidData.append('\n')

    covidData.append(st_name + "," + state + "," )

    

    covidRec.append('\n')

    covidRec.append(st_name + "," + state + "," )

    

    covidDeath.append('\n')

    covidDeath.append(st_name + "," + state + "," )

    

    covid = df(req.json()[state])



     

            

    for conf, dt in zip(covid.dates, covid.index):

        dt = dt + ","

        i+=1

        index = dtcolNames.index(dt) + 1

        if i!= index:

           #print(st + ":" + dist + ":" + dt + ":  Ind: " + str(index) + ":" +  "i :" + str(i))

           if i == 1:

               for n in range(index-1):

                   covidData.append("0,")

                   covidRec.append("0,")

                   covidDeath.append("0,")

           else:    

               for n in range(index-i+1):

                   covidData.append("0,")

                   covidRec.append("0,")

                   covidDeath.append("0,")

                   

           i = index             

            

        

        for t in conf.keys():

            if 'total' in t:

                if 'confirmed' in (conf['total'].keys()):

                    covidData.append(str(conf['total']['confirmed']) + ",")

                if not 'confirmed' in (conf['total'].keys()):

                    covidData.append("0,")

                   

                if 'recovered' in (conf['total'].keys()):

                    covidRec.append(str(conf['total']['recovered']) + ",")

                if not 'recovered' in (conf['total'].keys()):

                    covidRec.append("0,")

                   

                if 'deceased' in (conf['total'].keys()):

                    covidDeath.append(str(conf['total']['deceased']) + ",")

                if not 'deceased' in (conf['total'].keys()):

                    covidDeath.append("0,")

          

    if dt != dtcolNames[len(dtcolNames)-1]:

          diff = len(dtcolNames) - dtcolNames.index(dt)

          for n in range(diff):

              covidData.append(covidData[len(covidData)-1])

              covidDeath.append(covidDeath[len(covidRec)-1])

              covidRec.append(covidRec[len(covidRec)-1])         

# TODO: Delta and no of tests

                                



    

with open(DIR_NAME + '/Indian_States_total_confirmed_cases.csv', 'w') as f:

    f.writelines("State, State Code,")

    f.writelines(dtcolNames)

    f.writelines(covidData)

    

with open(DIR_NAME + '/Indian_States_total_recovered_cases.csv', 'w') as f:

    f.writelines("State, State Code,")

    f.writelines(dtcolNames)

    f.writelines(covidRec)



with open(DIR_NAME + '/Indian_States_total_Death_toll.csv', 'w') as f:

    f.writelines("State, State Code,")

    f.writelines(dtcolNames)

    f.writelines(covidDeath)





JSON_URL = 'https://api.covid19india.org/v4/min/timeseries-MH.min.json'

URL = 'https://api.covid19india.org/v4/min/timeseries-'

EXT = '.min.json'

JSON_INDIA = 'https://api.covid19india.org/v4/min/timeseries.min.json'





state_name = []

dtcolNames =[]

covidData =[]

covidRec = []

covidDeath = []
req_date = requests.get(JSON_URL)

req_India = requests.get(JSON_INDIA)



stateNames = df(req_India.json())

statesData = stateNames.T



for st in statesData.index:

     if not 'UN' in st:

         state_name.append(st)



unwanted_st = {'UN', 'TT', 'Other State', 'Other Region', 'Other', 'Unknown'} 

state_name = [ele for ele in state_name if ele not in unwanted_st]



state_name
dates = df(req_date.json()['MH']['districts']['Mumbai'])

for dt in dates.index:

    dtcolNames.append(dt + ",")
dtcolNames
isdir = os.path.isdir('Indian_Cities_Combined')

if not isdir:

    os.mkdir('Indian_Cities_Combined')
for st in state_name:

    st_name = name_correction(st)

    print ('Creating dataset for ' + st_name)

    dist_name  = []

  

    JSON_URL = URL + st + EXT

    req = requests.get(JSON_URL)    

    distNames = df(req.json())

    if 'districts' in distNames.index:

        distNames = df(req.json()[st]['districts'])

        distNames = distNames.T



        for dis in distNames.index:

            #print (st + " : " + dis)

            dist_name.append(dis)

 

  

        # items to be removed 

        unwanted_elem = {'Foreign Evacuees', 'Other State', 'Other Region', 'Others', 'Other', 'Unknown'} 

  

        dist_name = [ele for ele in dist_name if ele not in unwanted_elem]

            

        

                

        for dist  in dist_name:   

            

            i=0

            covidData.append('\n')

            covidData.append(st_name + "," + st + "," + dist + ",")

            

            covidRec.append('\n')

            covidRec.append(st_name + "," + st + "," + dist + ",")

            

            covidDeath.append('\n')

            covidDeath.append(st_name + "," + st + "," + dist + ",")

    

            covid = df(req.json()[st]['districts'][dist])

    

            

            for conf, dt in zip(covid.dates, covid.index):

                dt = dt + ","

                i+=1

                index = dtcolNames.index(dt) + 1

                if i!= index:

                    #print(st + ":" + dist + ":" + dt + ":  Ind: " + str(index) + ":" +  "i :" + str(i))

                    if i == 1:

                        for n in range(index-1):

                            covidData.append("0,")

                            covidRec.append("0,")

                            covidDeath.append("0,")

                    else:    

                        for n in range(index-i+1):

                            covidData.append("0,")

                            covidRec.append("0,")

                            covidDeath.append("0,")

                            

                    i = index

                

               

                    

                for t in conf.keys():

                    if 'total' in t:

                        if 'confirmed' in (conf['total'].keys()):

                            covidData.append(str(conf['total']['confirmed']) + ",")

                        if not 'confirmed' in (conf['total'].keys()):

                            covidData.append("0,")

                    

                        if 'recovered' in (conf['total'].keys()):

                            covidRec.append(str(conf['total']['recovered']) + ",")

                        if not 'recovered' in (conf['total'].keys()):

                            covidRec.append("0,")

                    

                        if 'deceased' in (conf['total'].keys()):

                            covidDeath.append(str(conf['total']['deceased']) + ",")

                        if not 'deceased' in (conf['total'].keys()):

                            covidDeath.append("0,")

                            

            if dt != dtcolNames[len(dtcolNames)-1]:

                diff = len(dtcolNames) - dtcolNames.index(dt)

                for n in range(diff):

                    covidData.append(covidData[len(covidData)-1])

                    covidDeath.append(covidDeath[len(covidRec)-1])

                    covidRec.append(covidRec[len(covidRec)-1])

                    

                

                

            

    else:

        print("State without sistrict's data : "+ st)



with open('Indian_Cities_Combined/Indian_Cities_total_confirmed_cases.csv', 'w') as f:

    f.writelines("State, State Code, Cities,")

    f.writelines(dtcolNames)

    f.writelines(covidData)

    

with open('Indian_Cities_Combined/Indian_Cities_total_recovered_cases.csv', 'w') as f:

    f.writelines("State, State Code, Cities,")

    f.writelines(dtcolNames)

    f.writelines(covidRec)



with open('Indian_Cities_Combined/Indian_Cities_total_Death_toll.csv', 'w') as f:

    f.writelines("State, State Code, Cities,")

    f.writelines(dtcolNames)

    f.writelines(covidDeath)
JSON_URL = 'https://api.covid19india.org/v4/min/timeseries-MH.min.json'

URL = 'https://api.covid19india.org/v4/min/timeseries-'

EXT = '.min.json'

JSON_INDIA = 'https://api.covid19india.org/v4/min/timeseries.min.json'

DIR_NAME = 'Indian_Cities_Stateswise'



state_name = []





req_date = requests.get(JSON_URL)

req_India = requests.get(JSON_INDIA)



stateNames = df(req_India.json())

statesData = stateNames.T



for st in statesData.index:

    state_name.append(st)



unwanted_st = {'UN', 'TT', 'Other State', 'Other Region', 'Other', 'Unknown'} 

state_name = [ele for ele in state_name if ele not in unwanted_st]





    



dtcolNames =[]





dates = df(req_date.json()['MH']['districts']['Mumbai'])

for dt in dates.index:

    dtcolNames.append(dt + ",")





dir_path = os.path.isdir(DIR_NAME)

if not dir_path:

    os.mkdir(DIR_NAME)



for st in state_name:

    st_name = name_correction(st)

    

    state_dir = st_name.replace(' ', '_')

    state_dir_path = DIR_NAME + "/" + state_dir

    isdir = os.path.isdir(state_dir_path)

    if not isdir:

        os.mkdir(state_dir_path)

        

   

    dist_name  = []

    covidData =[]

    covidRec = []

    covidDeath = []

  

    JSON_URL = URL + st + EXT

    req = requests.get(JSON_URL)

  

    distNames = df(req.json())

    if 'districts' in distNames.index:

        distNames = df(req.json()[st]['districts'])

        distNames = distNames.T



        for dis in distNames.index:

            #print (st + " : " + dis)

            dist_name.append(dis)

 

  

        # items to be removed 

        unwanted_elem = {'Foreign Evacuees', 'Other State', 'Other Region', 'Others', 'Other', 'Unknown'} 

  

        dist_name = [ele for ele in dist_name if ele not in unwanted_elem]

            

        file_t = DIR_NAME + "/" + state_dir + "/" + state_dir + '_total_confirmed_cases.csv'

        file_r = DIR_NAME + "/" + state_dir + "/" + state_dir + '_total_recovered_cases.csv'

        file_d = DIR_NAME + "/" + state_dir + "/" + state_dir + '_total_Death_toll.csv'

                        

               

                

         

        for dist  in dist_name:   

            

            i=0

            covidData.append('\n')

            covidData.append(st_name + "," + st + "," + dist + ",")

            

            covidRec.append('\n')

            covidRec.append(st_name + "," + st + "," + dist + ",")

            

            covidDeath.append('\n')

            covidDeath.append(st_name + "," + st + "," + dist + ",")

    

            covid = df(req.json()[st]['districts'][dist])

            

            

            for conf, dt in zip(covid.dates, covid.index):

                dt = dt + ","

                i+=1

                index = dtcolNames.index(dt) + 1

                if i!= index:

                    #print(st + ":" + dist + ":" + dt + ":  Ind: " + str(index) + ":" +  "i :" + str(i))

                    if i == 1:

                        for n in range(index-1):

                            covidData.append("0,")

                            covidRec.append("0,")

                            covidDeath.append("0,")

                    else:    

                        for n in range(index-i+1):

                            covidData.append("0,")

                            covidRec.append("0,")

                            covidDeath.append("0,")

                            

                    i = index

                

               

                    

                for t in conf.keys():

                    if 'total' in t:

                        if 'confirmed' in (conf['total'].keys()):

                            covidData.append(str(conf['total']['confirmed']) + ",")

                        if not 'confirmed' in (conf['total'].keys()):

                            covidData.append("0,")

                    

                        if 'recovered' in (conf['total'].keys()):

                            covidRec.append(str(conf['total']['recovered']) + ",")

                        if not 'recovered' in (conf['total'].keys()):

                            covidRec.append("0,")

                    

                        if 'deceased' in (conf['total'].keys()):

                            covidDeath.append(str(conf['total']['deceased']) + ",")

                        if not 'deceased' in (conf['total'].keys()):

                            covidDeath.append("0,")

            

                            

            # Data updatation for NODATA

          

            if dt != dtcolNames[len(dtcolNames)-1]:

                diff = len(dtcolNames) - dtcolNames.index(dt)

                for n in range(diff):

                    covidData.append(covidData[len(covidData)-1])

                    covidDeath.append(covidDeath[len(covidRec)-1])

                covidRec.append(covidRec[len(covidRec)-1])

            

       

            

            with open(file_t, 'w') as f:

                f.writelines("State,State Code,Cities,")

                f.writelines(dtcolNames)

                f.writelines(covidData)

        

            with open(file_r, 'w') as f:

                f.writelines("State,State Code,Cities,")

                f.writelines(dtcolNames)

                f.writelines(covidRec)

    

            with open(file_d, 'w') as f:

                f.writelines("State,State Code,Cities,")

                f.writelines(dtcolNames)

                f.writelines(covidDeath)            

        

    else:

        print("State without sistrict's data : "+ st)

    

        

df = pd.read_csv('Indian_States/Indian_States_total_confirmed_cases.csv')

df.head()
df = pd.read_csv('Indian_Cities_Stateswise/Karnataka/Karnataka_total_confirmed_cases.csv')

df.head()
df = pd.read_csv('Indian_Cities_Stateswise/Maharashtra/Maharashtra_total_confirmed_cases.csv')

df.head()
df = pd.read_csv('Indian_Cities_Stateswise/Nagaland/Nagaland_total_confirmed_cases.csv')

df.head()
df = pd.read_csv('Indian_Cities_Combined/Indian_Cities_total_confirmed_cases.csv')

df.head()
df = pd.read_csv('Indian_Cities_Combined/Indian_Cities_total_recovered_cases.csv')

df.head()
df = pd.read_csv('Indian_Cities_Combined/Indian_Cities_total_Death_toll.csv')

df.head()
#df.drop(df.columns[0], axis =1)

import pandas as pd



df = pd.read_csv('Indian_Cities_Stateswise/Maharashtra/Maharashtra_total_confirmed_cases.csv')

df_ = df.drop(df.columns[4:123], axis =1)

df_s = df_.drop(df.columns[3],axis =1)

df_s = df_s.rename(columns = {"State Code":"State_Code", "2020-08-24":"Confirmed_Cases"}) 

df_s = df_s.drop(df.columns[-1],axis =1)

df_s