#import modules

import numpy as np

import pandas as pd

from datetime import datetime
#read data, world Covid-19 datasets

df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
df_confirmed.head()
#select Canada data

df_confirmed_Canada = df_confirmed.loc[df_confirmed['Country/Region'] == 'Canada'].reset_index(drop = True)

df_deaths_Canada = df_deaths.loc[df_confirmed['Country/Region'] == 'Canada'].reset_index(drop = True)

df_recovered_Canada = df_recovered.loc[df_confirmed['Country/Region'] == 'Canada'].reset_index(drop = True)
#calculate sums in last row

#Canada_disease_trend = pd.DataFrame(np.zeros((1, df_confirmed_Canada.shape[1])), columns = df_confirmed_Canada.columns)

#Canada_disease_trend

df_confirmed_Canada = df_confirmed_Canada.append(df_confirmed_Canada.sum(axis=0).to_frame().transpose(), ignore_index = True)

df_deaths_Canada = df_deaths_Canada.append(df_deaths_Canada.sum(axis=0).to_frame().transpose(), ignore_index = True)

df_recovered_Canada = df_recovered_Canada.append(df_recovered_Canada.sum(axis=0).to_frame().transpose(), ignore_index = True)
#change last row name to total, 

df_confirmed_Canada['Province/State'][len(df_confirmed_Canada['Province/State'])-1] = 'total'

df_deaths_Canada['Province/State'][len(df_deaths_Canada['Province/State'])-1] = 'total'

df_recovered_Canada['Province/State'][len(df_recovered_Canada['Province/State'])-1] = 'total'



#drop useless columns

df_confirmed_Canada.drop(['Country/Region', 'Lat', 'Long'], axis = 1, inplace = True)

df_deaths_Canada.drop(['Country/Region', 'Lat', 'Long'], axis = 1, inplace = True)

df_recovered_Canada.drop(['Country/Region', 'Lat', 'Long'], axis = 1, inplace = True)



df_deaths_Canada
#last day's total data, like total death in last day

confirm_today = df_confirmed_Canada.iloc[-1,:][-1]

death_today = df_deaths_Canada.iloc[-1,:][-1]

recover_today = df_recovered_Canada.iloc[-1,:][-1]

print('Confirm, death, recover number in total: ', confirm_today, ',', death_today, ',', recover_today)
#2 weeks before last day

confirm_2wbefore = df_confirmed_Canada.iloc[-1,:][-14]

death_2wbefore = df_deaths_Canada.iloc[-1,:][-14]

recover_2wbefore = df_recovered_Canada.iloc[-1,:][-14]
#total num within nearby two weeks

confirm_2w = confirm_today - confirm_2wbefore

death_2w = death_today - death_2wbefore

recover_2w = recover_today - recover_2wbefore

print('Confirm, death, recover number in 2 weeks: ', confirm_2w, ',', death_2w, ',', recover_2w)
#total Canada population

#k1k1k1 = pd.read_csv('https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/cases.csv')

#k1k1k1.to_csv("../../kaggle/working/Canada_disease.csv", index=False)
#total Canada disease

Cases_Canada = pd.read_csv('https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/cases.csv')

Deaths_Canada = pd.read_csv('https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/mortality.csv')

test_Canada = pd.read_csv('https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/testing_cumulative.csv')

print(Cases_Canada.shape, Cases_Canada.columns.tolist())
Deaths_Canada.head()
#print(Cases_Canada.travel_history_country.unique())
#efficiency mask parameters %

homemade_mask_E = [58, (58+77)/2, 77]

surgical_mask_E = [72, (72+85)/2, 85]

N95_mask_E = [98, 98.5, 99]



#death rate VS efficiency

efficiency = [20, 50, 80]

death_rate = [5.4, 31.1, 64.5]
#function of efficiency and death rate

E_D_func = lambda efficiency: (77/18000 * efficiency ** 2 + 1003/1800 * efficiency - 671/90) / 100
type_mask_list = ['homemade', 'surgical', 'N95']

wear_mask_list_choice = ['seldom', 'normal', 'always', 'every time']

wear_mask_list = [.2, .5, .8, 1]

wear_mask_dic = {wear_mask_list_choice[i]: wear_mask_list[i] for i in range(len(wear_mask_list_choice))}

#print(wear_mask_dic)



solution_optimal_list = ['worst', 'normal', 'best']

solution_optimal_loc = [0,1,2]

solution_optimal_dic = {solution_optimal_list[i]: solution_optimal_loc[i] for i in range(len(solution_optimal_list))}

#print(solution_optimal_dic)



def mask_efficiency(type_mask, solution_optimal):

    

    #masks efficiency

    homemade_mask_E = [58, (58+77)/2, 77]

    surgical_mask_E = [72, (72+85)/2, 85]

    N95_mask_E = [98, 98.5, 99]



    if type_mask == 'homemade':

        return homemade_mask_E[solution_optimal_dic[solution_optimal]]



    if type_mask == 'surgical':

        return surgical_mask_E[solution_optimal_dic[solution_optimal]]

    

    if type_mask == 'N95':

        return N95_mask_E[solution_optimal_dic[solution_optimal]]

        

#print(mask_efficiency('homemade', 'normal'))



def wear_mask_func(wear_mask):

    if wear_mask == 'seldom':

        return wear_mask_list[0]



    if wear_mask == 'normal':

        return wear_mask_list[1]

    

    if wear_mask == 'always':

        return wear_mask_list[2]

    

    if wear_mask == 'every time':

        return wear_mask_list[3]



#function of our solution

#the third parameter is function of efficiency and death rate

def mask(type_mask, wear_mask, solution_optimal, E_D_func = E_D_func):

    # input: masks information and function 

    # input example: N95, 20%, best, E_D_func

    # output: solution to customers

    

    #error

    if solution_optimal not in solution_optimal_list or type_mask not in type_mask_list or wear_mask not in wear_mask_list_choice:

        print ('Raise error')

        return None

    

    #masks efficiency

    mask_eff = mask_efficiency(type_mask, solution_optimal)

    

    death_rate_reduction = E_D_func(mask_eff * wear_mask_func(wear_mask))

    #print('The reduction of death rate is {} for {} masks, for people who {} wear masks. This solution is the {} choice.'.format(round(death_rate,2), type_mask, wear_mask, solution_optimal))

    #print('By this solution, in Canada, there is {} reduction of people''s death due to the whole Covid-19, {} reduction of people''s death during the last two weeks.'.format(round(death_today * death_rate, 0), round(death_2w * death_rate, 0)))

    

    return death_rate_reduction



#testing

mask('N95', 'always', 'best')

#print(calculator('homemade', 'always', 'normal'))
social_distance_choice = ['Stay at home', 'Working full time', 'Close School', 'Distance Travelled from home']

social_distance_answer = [-.45, .85, -.60, .25]

social_distance_dic = {social_distance_choice[i]: social_distance_answer[i] for i in range(len(social_distance_choice))}



def social_distance(death_rate, choice_list = [False, False, False, False]):

    #input: death rate, choice(e.g: [True, True, False, True])

    #output: death rate after analysis

    for choice_pos in range(len(choice_list)):

        if choice_list[choice_pos] == True:

            death_rate = death_rate * (1 + social_distance_answer[choice_pos])

    return death_rate 

    

social_distance(1, [True, True, False, True])    
dt_factors = pd.read_csv('../input/canada-covid19-risk-factors/risk factor.csv')

dt_factors.head()
#correction of data

dt_factors['Unnamed: 0'][1] = 'Total'

dt_factors['Unnamed: 0'][0] = 'Column_name'
import re

pattern = re.compile(r'\d+')

result = int(''.join(pattern.findall('N=5,416')))

print(result)
for name in dt_factors.columns.tolist()[1:]:

    dt_factors[name][1] = int(''.join(pattern.findall(str(dt_factors[name][1]))))

dt_factors.head()
#fill NAN

dt_factors = dt_factors.fillna(0)

dt_factors
researched_name = ['Column_name', 'Total','Hypertension','Coronary Artery Disease', 'History of Stroke', 'Diabets', 'Obesity', 'Chronic Kidney Disease', 'Asthma ', 'COPD']

researched_data = [0,0,.23,.085,.026,.081,.64,.134,.81,.04]

researched_dic = {researched_name[i]: researched_data[i] for i in range(len(researched_name))}

dt_factors_researched = dt_factors.copy()



dt_factors_researched = dt_factors_researched.loc[dt_factors_researched['Unnamed: 0'].isin(researched_name)]

dt_factors_researched    
def other_factor(dt_factors_researched, disease = None, age = None, sex = None, race = None):

    if disease is None:

        return 1

    infect_rate_infected = int(dt_factors_researched.loc[dt_factors_researched['Unnamed: 0'] == disease]['Unnamed: 1']) / 5416

    #print('infect rate infected:', infect_rate_infected)

    infect_rate_notinfected = researched_dic[disease]

    #print('infect rate notinfected:', infect_rate_notinfected)

    infect_rate_increase_rate = infect_rate_infected / infect_rate_notinfected

    return infect_rate_increase_rate



for name in researched_name[2:]:

    print(other_factor(dt_factors_researched, name))
dt_factors_researched.loc[dt_factors_researched['Unnamed: 0'] == 'History of Stroke']['Unnamed: 1']
def calculator(type_mask, wear_mask, solution_optimal, choice_list = [False, False, False, False], dt_factors_researched = dt_factors_researched, E_D_func = E_D_func, disease = None):



    default_reduction = .5 #suppose currently, how much death rate reduction Canada already did

    death_rate_mask = mask(type_mask, wear_mask, solution_optimal, E_D_func = E_D_func)

    death_rate_SD = social_distance(death_rate_mask, choice_list)

    death_rate_final = death_rate_SD * other_factor(dt_factors_researched, disease) - default_reduction

    print('The reduction of death rate is {}% for {} masks, for people who {} wear masks. This solution is the {} choice.'.format(round(death_rate_final,2)*100, type_mask, wear_mask, solution_optimal))

    print('By this solution, in Canada, there is {} reduction of people''s death due to the whole Covid-19, {} reduction of people''s death during the last two weeks.'.format(round(death_today * death_rate_final, 0), round(death_2w * death_rate_final, 0)))

    

    return death_rate_final

    

    

calculator('homemade', 'always', 'normal', [True, True, False, True], disease = 'History of Stroke')    

calculator('homemade', 'seldom', 'worst')   