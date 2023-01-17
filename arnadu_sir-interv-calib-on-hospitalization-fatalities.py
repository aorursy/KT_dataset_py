# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

import seaborn as sns
import math
import datetime
from datetime import timedelta  


#formatting functions for charts
def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)

#formatting functions for charts
def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1fT' % (x * 1e-3)


#census populations
#add entries to this table in order to run simulations
Population = {
'AL':4908621,
'AK':734002,
'AZ':7378494,
'AR':3038999,
'CA':39937489,
'CO':5845526,
'CT':3563077,
'DE':982895,
'FL':21992985,
'GA':10736059,
'HI':1412687,
'ID':1826156,
'IL':12659682,
'IN':6745354,
'IA':3179849,
'KS':2910357,
'KY':4499692,
'LA':4645184,
'ME':1345790,
'MD':6083116,
'MA':6976597,
'MI':10045029,
'MN':5700671,
'MS':2989260,
'MO':6169270,
'MT':1086759,
'NE':1952570,
'NV':3139658,
'NH':1371246,
'NJ':8936574,
'NM':2096640,
'NY':19440469,
'NC':10611862,
'ND':761723,
'OH':11747694,
'OK':3954821,
'OR':4301089,
'PA':12820878,
'RI':1056161,
'SC':5210095,
'SD':903027,
'TN':6897576,
'TX':29472295,
'UT':3282115,
'VT':628061,
'VA':8626207,
'WA':7797095,
'WV':1778070,
'WI':5851754,
'WY':567025,
'DC':720687   
}

US_States_codes = {
'AL':'Alabama',
'AK':'Alaska',
'AZ':'Arizona',
'AR':'Arkansas',
'CA':'California',
'CO':'Colorado',
'CT':'Connecticut',
'DE':'Delaware',
'FL':'Florida',
'GA':'Georgia',
'HI':'Hawaii',
'ID':'Idaho',
'IL':'Illinois',
'IN':'Indiana',
'IA':'Iowa',
'KS':'Kansas',
'KY':'Kentucky',
'LA':'Louisiana',
'ME':'Maine',
'MD':'Maryland',
'MA':'Massachusetts',
'MI':'Michigan',
'MN':'Minnesota',
'MS':'Mississippi',
'MO':'Missouri',
'MT':'Montana',
'NE':'Nebraska',
'NV':'Nevada',
'NH':'New Hampshire',
'NJ':'New Jersey',
'NM':'New Mexico',
'NY':'New York',
'NC':'North Carolina',
'ND':'North Dakota',
'OH':'Ohio',
'OK':'Oklahoma',
'OR':'Oregon',
'PA':'Pennsylvania',
'RI':'Rhode Island',
'SC':'South Carolina',
'SD':'South Dakota',
'TN':'Tennessee',
'TX':'Texas',
'UT':'Utah',
'VT':'Vermont',
'VA':'Virginia',
'WA':'Washington',
'WV':'West Virginia',
'WI':'Wisconsin',
'WY':'Wyoming',
'DC':'District of Columbia',
'AS':'Samoa',
'GU':'Guam',
'PR':'Puerto Rico'
}
import urllib, json
url = 'https://covidtracking.com/api/states/daily'

import requests
r = requests.get(url)

data = pd.DataFrame(r.json())
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

d = data[data['state']=='NY']
fig, axs = plt.subplots(1,3, figsize=(12,4))

ax = plt.subplot(131)
plt.plot(d['date'], d['death'],label='death')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%B-%d'))
plt.legend()
plt.grid()

ax = plt.subplot(132)
plt.plot(d['date'], d['hospitalizedCurrently'],label='hospitalized')
plt.legend()
plt.grid()

ax = plt.subplot(133)
plt.plot(d['date'], d['positive'],label='positive')
plt.legend()
plt.grid()


fig.autofmt_xdate()
plt.show()

display(d.head())


data.columns
#######################################################
# SIR model with INTERVENTION
#------------------------------------------------------
# params:
#
# x                    : array of number of days since inception (not used except to size output); in the calibration below, inception starts on the first day reported fatalities reach a CUTOFF threshold
# i0                   : initial percentage of infected population, for 1 per million: i0 = 1e-6
# beta                 : initial daily rate of transmission by infected people to susceptible people, for R0=2.7 and gamma=1/21: beta=R0*gamma=2.7/21 
# gamma                : daily rate of recovery or death of infected people, for a 21 day speed of recovery or death: gamma = 1/21
# hospitalization_rate : daily rate of infected people needing hospitalization
# death_rate           : daily death rate of infected people (assuming 1% of infected people die about 3 weeks after infection: death_rate=0.01/21)
#
# intervention_day : number of days after inception for intervention to start to reduce the initial transmission rate (beta)
# intervention_lag : number of days it takes for intervention to reach full effect (linear interp)
# intervention_effect : percentage reduction of initial transmission  rate, 0.25 for 25% reduction of initial beta after full intervention takes effect
########################################################

#-------------------------------------------------------
#the number returned by this function will be multiplied with the initial beta in order to estimate the transmission rate each day of the simulation
def intervention(day, day0, lag=5, effect=0.25):
    if day>day0+lag:
        return 1.0 - effect
    if day>day0:
        return 1.0 - effect * (day-day0)/lag
    return 1.0

'''
days = np.arange(300)
effects = np.zeros(300)
for d in days:
    effects[d] = intervention(d, 200, 3, 0.75)
plt.plot(days, effects)
plt.show()
'''

#-------------------------------------------------------
#this function worsens the death_rate (higher) and hospitalization_rate (lower) as hospitals fill up
#inputs:
#-hospitalized : percentage of population current hospitalized
#-hospital_capacity : percentage of population at which hospitals would be full
#-death_rate : percentage of infected people that will die if there is room in hospitals
#-hospitalization_rate : percentage of infected people that will be admitted to hospitals
#-worst_case : maximum percentage increase/decrease of death_rate and hospitalization_rate because of rationing of hospital capacity
#outputs:
# (death_rate, hospitalization_rate) : effective rates given the level of hospitalization
def rationing(hospitalized,infectious, hospital_capacity, death_rate, hospitalization_rate, detection_rate, detection_capacity):
    
    deathr = death_rate + 0.5 * death_rate * (1 - math.exp(-hospitalized / hospital_capacity))
    
    hr = hospitalization_rate * math.exp(-hospitalized / hospital_capacity)   
    
    detectr = detection_rate * math.exp(-infectious / detection_capacity)   
    
    return deathr, hr, detectr


hospitalized = []
res = []
for i in range(0,800):
    h = i * 1e-6
    hospitalized.append(i)
    dr, hr, detectr = rationing(hospitalized=h, infectious=h, hospital_capacity=400e-6, death_rate=0.01, hospitalization_rate=0.20, detection_rate=0.3, detection_capacity=400e-6)
    res.append([dr,hr,detectr])
plt.plot(hospitalized, res)
plt.show()
#display(death_rate)


#-------------------------------------------------------

#-------------------------------------------------------
# basic daily integration of a classic SIR model with a time-variable beta parameter=beta*intervention(day)
# the function returns a numpy matrix, with a row per day and the following columns (cumulative results since day of inception)
cS   = 0  #percentage of Susceptible people
cE   = 1  #percentage of Exposed people who are incubating and not yet infectious
cI1  = 2  #percentage of Infectious people in the first phase (before hospitalization might be needed)
cI2  = 3  #percentage of Infectious people in the second phase (hospitalization not needed)
cH   = 4  #percentage of Hospitalized people
cR   = 5  #percentage of Recovered people (not including fatalities)
cF   = 6  #percentage of Fatalities
cP   = 7  #percentage of Positive cases
cRe  = 8  #effective rate of transmission
cNum = 9  #number of parameters
def SEIR(x, i0, beta, gamma_i1, gamma_i2, gamma_h, incubation_rate, hospitalization_rate, death_rate, intervention_day, intervention_lag, intervention_effect, detection_rate, hospital_capacity, detection_capacity):
    
    #start the simulation earlier than the calibration data to remove the initial oscillation due to the insertion of an intial exposed person
    early_start = 2 * math.ceil(1/incubation_rate)  
    n = early_start + x.size
    y = np.zeros((n,cNum))

    for i in range(0,n):
        
        if i==0:
            #initial conditions
            exposed      = i0  #incubating
            infectious1  = 0   #first phase after incubation: developing symptoms
            infectious2  = 0   #second phase after incubation: people do not need hospitalization
            hospitalized = 0   #second phase after incubation: people who are hospitalized
            positives    = 0   #detected positives
            fatalities   = 0   #death, from both stay-at-home and hospitalized categories
            recovered    = 0   #or recover
            susceptible  = 1.0 - i0 
            infectious   = 0
          
        else:
            #compute daily variations           
            
            infection_rate = beta * intervention(i-early_start, intervention_day, intervention_lag, intervention_effect)
            
            infectious = infectious1 + infectious2 + hospitalized
            
            dr, hr, detectr = rationing(hospitalized=hospitalized, infectious=infectious, hospital_capacity=hospital_capacity, death_rate=death_rate, 
                               hospitalization_rate=hospitalization_rate, detection_rate=detection_rate, detection_capacity=detection_capacity)
            
            newlyexposed = infection_rate * susceptible * infectious
            
            newlyinfectious1 = incubation_rate * exposed

            newlyinfectious2 = (1 - hr) * gamma_i1 * infectious1

            newlyhospitalized = hr * gamma_i1 * infectious1

            d_fatalities = dr * (gamma_i2 * infectious2 + gamma_h * hospitalized)
            
            d_recovered = (1 - dr) * (gamma_i2 * infectious2 + gamma_h * hospitalized)

            d_exposed = newlyexposed - newlyinfectious1  #these people are incubating, but not yet infectious; population is newly infected people less people finishing incubation

            d_positives = detectr * newlyinfectious1 #assume detection would happen at the end of incubation, when symptoms appear.
            
            d_infectious1 = newlyinfectious1 - newlyinfectious2 - newlyhospitalized
            
            d_infectious2 = newlyinfectious2 - gamma_i2 * infectious2
            
            d_hospitalized = newlyhospitalized - gamma_h * hospitalized   
            
            d_susceptible = - newlyexposed
            
            
            #integrate and store in result array
            susceptible += d_susceptible
            exposed += d_exposed
            infectious1 += d_infectious1
            infectious2 += d_infectious2
            hospitalized += d_hospitalized
            recovered += d_recovered
            fatalities += d_fatalities
            positives += d_positives
            
        y[i,cS]  = susceptible
        y[i,cE]  = exposed
        y[i,cI1] = infectious1
        y[i,cI2] = infectious2
        y[i,cH]  = hospitalized
        y[i,cR]  = recovered
        y[i,cF]  = fatalities
        y[i,cP]  = positives  #cumul of infected, does not come down on recovery. assuming all newly infected people are immediately detected

        #average number of people infected by an infectious person
        if infectious>0:
            y[i,cRe] = newlyexposed / infectious
        
            
    return y[early_start:,:]  #do not return the initialization period


x = np.arange(100)

#plot number of fatalities 
#in a population on 1 million people, with one person initially infected, 
#assuming 3 weeks recovery rate, intial R0=5=beta/gamma, 20% of cases require hospitalization, and 1% of infected people die
population = 1e6  
i0 = 1000e-6

gamma_i1 = 1/7                         #it takes 7 days for symptoms to reach a point where hospitalization may be needed
gamma_i2 = 1/7                         #if hospitalization is not needed, it takes another 7 days to recover while staying at home and stop being infectious, or to die
gamma_h  = 1/14                        #if hospitalization is needed, it takes 14 days to recover or die
beta = 2.5 * (gamma_i1+gamma_i2)       #beta = R0 * gamma
incubation_rate = 1/5                  #it takes 5 days after exposure for people to become infectious
hospitalization_rate = 0.20            #20% of infected people will require hospitalization
death_rate = 0.01                      #1% of infected people will die, either stay-at-home or hospitalized
intervention_day = 0        
intervention_lag = 0
intervention_effect = 0
detection_rate = 1                     #100% of newly infectious cases that are detected as positives
hospital_capacity = 1000e-6            #hospital capacity as percentage of total population
detection_capacity = 1000e-6           #capacity to detect infectious cases, as percentage of total population

#baseline: intervention has no effect in reducing initial transmission rate
y0 = population * SEIR(x, i0=i0, beta=beta, gamma_i1=gamma_i1, gamma_i2=gamma_i2, gamma_h=gamma_h, incubation_rate=incubation_rate, hospitalization_rate=hospitalization_rate, death_rate=death_rate, 
         intervention_day=intervention_day, intervention_lag=intervention_lag, intervention_effect=intervention_effect,
         detection_rate=detection_rate,
         hospital_capacity=hospital_capacity, detection_capacity=detection_capacity)

gamma_i1 = 1/7                         #it takes 7 days for symptoms to reach a point where hospitalization may be needed
gamma_i2 = 1/7                         #if hospitalization is not needed, it takes another 7 days to recover while staying at home and stop being infectious, or to die
gamma_h  = 1/14                        #if hospitalization is needed, it takes 14 days to recover or die
beta = 2.5 * (gamma_i1+gamma_i2)       #beta = R0 * gamma
incubation_rate = 1/5                  #it takes 5 days after exposure for people to become infectious
hospitalization_rate = 0.20            #20% of infected people will require hospitalization
death_rate = 0.01                      #1% of infected people will die, either stay-at-home or hospitalized
intervention_day = 0        
intervention_lag = 0
intervention_effect = 0
detection_rate = 1                     #100% of newly infectious cases that are detected as positives
hospital_capacity = 500e-6             #hospital capacity as percentage of total population
detection_capacity = 5000e-6           #capacity to detect infectious cases, as percentage of total population

y = population * SEIR(x, i0=i0, beta=beta, gamma_i1=gamma_i1, gamma_i2=gamma_i2, gamma_h=gamma_h, incubation_rate=incubation_rate, hospitalization_rate=hospitalization_rate, death_rate=death_rate, 
         intervention_day=intervention_day, intervention_lag=intervention_lag, intervention_effect=intervention_effect,
         detection_rate=detection_rate,
         hospital_capacity=hospital_capacity, detection_capacity=detection_capacity)

fig,ax = plt.subplots(figsize=[8,8])
plt.plot(x,  y[:,cH],'r--',label='')
plt.plot(x,  y[:,cF],'b-',label='')
plt.plot(x,  y[:,cP],'g-.',label='')
plt.legend()
plt.grid()
plt.show()



###############################################
####### CALIBRATION TO reported Hospitalization and Fatalities
####### with intervention
###############################################

#extract the data for the given region or state and prepare it for the calibration
def prep_data(data, state='NY', cutoff_on='death', cutoff=1, truncate=0):
    
    #filter the data and keep the given REGION and STATE only
    c = data[data['state']==state]
    c = c.sort_values(by='date',ascending=True)  #data pulled from web is sorted the other way...
    
    #cutoff_on = 'death'  | 'hospitalized': choose to start simul when hospitalized or death reach the cutoff number
    minDate = c[c[cutoff_on]>cutoff]['date'].min()

    s1 = c[c['date']>minDate].copy()  #keep only the records after the given number of fatalities have been reached
    if truncate>0:
        s1 = s1[:truncate].copy()  #keep only the given number of days

    #calculate the number of days since the first day fatalities exceeded the cutoff
    s1['Days'] = (s1['date'] - minDate) / np.timedelta64(1, 'D')
  
    x = s1['Days']
   
    #calibrate to hospitalization only, death only or both hospitalization and fatalities counts; these input vectors are appended (see SEIR_calib for processing)
    #by setting unwanted data to NaN
    
    calib_p = s1['positive'].copy()
    scale_p = calib_p.max()
    
    calib_h = s1['hospitalizedCurrently'].copy()
    scale_h = calib_h.max()
    
    calib_d = s1['death'].copy()
    scale_d = calib_d.max()
    
#    if calib_on=='hospitalized':
#        calib_d[:] = math.nan #blank out death data, it is not going to be used for calibration
    
    #normalize positive and hospital counts to the same order of magnitude as death counts to avoid biais in the calibration:
    
    scale_p = scale_d / scale_p
    calib_p *= scale_p 

    scale_h = scale_d / scale_h
    calib_h *= scale_h 
    
    z = calib_p.append(calib_h)   #calibration function works on a vector of positive data, followed by hospitalized data followed by death data
    z = z.append(calib_d)         

    #record where we do not have data, for use by SEIR_calib to ignore these points during the calibration
    missing_data = np.isnan(z)
    z = np.nan_to_num(z)
    
    
    return minDate, s1, x, z, missing_data, scale_p, scale_h


#Function called by scipy.curve_fit to calibrate the model parameters 
#This function calls the SIR model to simulate on the current guess parameters, and formats the results for use by curve_fit()
#note: SEIR() returns percentage of population, whereas reported data is absolute number of people, hence the need to use a total population number
#note: this function calibrates the final hospitaliation and death rates (eg 10% of infected people will be hospitalized people, rather than the instantaneous rates that need to be passed to SIR5() )
#note: the calibration is meant to be on both hospitalization and fatalities counts, so these two vectors are appended in the return; the x parameter needs to be a single length
def SEIR_calib(x, i0, beta, gamma_i1, gamma_i2, gamma_h, incubation_rate, hospitalization_rate, death_rate,
               intervention_day, intervention_lag, intervention_effect, 
               detection_rate,
               hospital_capacity,detection_capacity,
               population, missing_data, scale_p, scale_h):
    
    y = SEIR(x, i0=i0, beta=beta, gamma_i1=gamma_i1, gamma_i2=gamma_i2, gamma_h=gamma_h, incubation_rate=incubation_rate, hospitalization_rate=hospitalization_rate, death_rate=death_rate,
            intervention_day = intervention_day, intervention_lag = intervention_lag, intervention_effect=intervention_effect,
            detection_rate = detection_rate,
            hospital_capacity = hospital_capacity, detection_capacity=detection_capacity)

    #set calibration and simulation output to zero when calibration data is NaN. 
    #this is because hospitalization data tends to start later than fatalities data in the published numbers
    #use x as an array of booleans to indicate where this should be done
    
    ret = np.append(y[:,cP] * scale_p, y[:,cH] * scale_h)    #positive and hospitalization counts are rescaled to same order of magnitude as death count for calibration algorithm - see prep_data()
    ret = np.append(ret, y[:,cF])    
    ret = np.where(missing_data, 0, ret)

    return ret * population

#--------------------------
#This function calibrates and shows the results for one State
#data - DataFrame in the same format as train.csv but with Region and State columns added
#output - boolean, True to print results and charts
#
def calibrate(data, output=True, state='NY', cutoff_on='death', cutoff=1, incubation_rate=1/5):

    if output:
        print('-----------------')
        print(state)
        print('-----------------')
        print('')

    population = Population[state]

    #Bounds and initial guess for calibration algorithm scipy.curve_fit
    I0_min = 1e-6
    I0_max = 1000e-6
    Gamma_i1_min = 1/7
    Gamma_i1_max = 1/5
    Gamma_i2_min = 1/7
    Gamma_i2_max = 1/5
    Gamma_h_min = 1/14
    Gamma_h_max = 1/7
    Beta_min = 1.1 * 1/21
    Beta_max = 3 * 1/7
    HospitalizationRate_min = 0.10
    HospitalizationRate_max = 0.30
    DeathRate_min = 0.001
    DeathRate_max = 0.02
    InterventionDay_min = 0   
    InterventionDay_max = 50   #50 days after report of first death
    InterventionLag_min = 1
    InterventionLag_max = 2
    InterventionEffect_min = 0
    InterventionEffect_max = 0.99 
    DetectionRate_min = 0.1           #10% of infectious cases detected as positives
    DetectionRate_max = 1             #100% detection of infectious case reported as positives 
    HospitalCapacity_min = 100e-6     #1000 bed per million
    HospitalCapacity_max = 1000e-6    #1000 beds per million
    DetectionCapacity_min = 1e-6      #capacity to detect 1 per million
    DetectionCapacity_max = 1         #full detection

    initial_guess = [I0_min, 
                     Beta_min,
                     Gamma_i1_min,
                     Gamma_i2_min,
                     Gamma_h_min,
                     HospitalizationRate_min,
                     DeathRate_min,
                     InterventionDay_min,
                     InterventionLag_min,
                     InterventionEffect_max,
                     DetectionRate_max,
                     HospitalCapacity_min,
                     DetectionCapacity_min
                    ]

    bounds = ((I0_min, Beta_min, Gamma_i1_min, Gamma_i2_min, Gamma_h_min, HospitalizationRate_min, DeathRate_min, InterventionDay_min, InterventionLag_min, InterventionEffect_min, DetectionRate_min, HospitalCapacity_min, DetectionCapacity_min),
              (I0_max, Beta_max, Gamma_i1_max, Gamma_i2_max, Gamma_h_max, HospitalizationRate_max, DeathRate_max, InterventionDay_max, InterventionLag_max, InterventionEffect_max, DetectionRate_max, HospitalCapacity_max, DetectionCapacity_max))

    #prepare the calibration data
    #----------------------------

    minDate, s1, x, z, missing_data, scale_p, scale_h = prep_data(data, state=state, cutoff_on=cutoff_on, cutoff=cutoff, truncate=0)

    if output:
        print("Reported {} reached {} on {:%Y-%m-%d}".format(cutoff_on, cutoff, minDate))
        print('')
    
    
    
    #calibrate the model
    #use a lambda to pass the population, h0 and f0 when running a simulation on guess parametes, as these are not calibrated params but are needed by SIR5_calib()
    #-------------------
   
    popt, pcov = curve_fit(lambda x, i0, beta, gamma_i1, gamma_i2, gamma_h, hospitalization_rate, death_rate, intervention_day, intervention_lag, intervention_effect, detection_rate, hospital_capacity, detection_capacity :
                               SEIR_calib(x, i0, beta, gamma_i1, gamma_i2, gamma_h, incubation_rate, hospitalization_rate, death_rate, 
                                          intervention_day, intervention_lag, intervention_effect, 
                                          detection_rate = detection_rate,
                                          hospital_capacity = hospital_capacity,detection_capacity=detection_capacity,
                                          population = population, missing_data = missing_data, scale_p = scale_p, scale_h = scale_h),
                           x, z, bounds=bounds, p0=initial_guess)

    calib_I0                  = popt[0]
    calib_Beta                = popt[1]
    calib_GammaI1             = popt[2]
    calib_GammaI2             = popt[3]
    calib_GammaH              = popt[4]
    calib_HospitalizationRate = popt[5]
    calib_DeathRate           = popt[6]
    calib_InterventionDay     = popt[7]
    calib_InterventionLag     = popt[8]
    calib_InterventionEffect  = popt[9]
    calib_DetectionRate       = popt[10]
    calib_HospitalCapacity    = popt[11]
    calib_DetectionCapacity   = popt[12]

    if output:
        print("SIR model fit")
        print("-------------")
        print("Simulation starts on {:%Y-%m-%d}".format(minDate))
        print("{} has a population of {:,.0f}".format(state, population))
        print("current fatalities are {:,.0f}".format(s1['death'].iloc[-1]))
        print("Incubation = {:.3f}, or {:.1f} days to recover".format(incubation_rate, 1/incubation_rate))
        print('')
        print("I0 = {:,.0f} per million, or {:,.0f} persons infected as of {:%Y-%m-%d}".format(calib_I0*(1/incubation_rate+1)*1e6, calib_I0*population*(1/incubation_rate+1), minDate))
        print("BETA = {:.3f}".format(calib_Beta))
        print("GAMMA_I1 = {:.1f} days in first symptomatic phase".format(1/calib_GammaI1))
        print("GAMMA_I2 = {:.1f} days in second symptomatic phase".format(1/calib_GammaI2))
        print("GAMMA_H = {:.1f} days in hospital".format(1/calib_GammaH))
        print("HOSPITALIZATION RATE = {:.1%} cases require hospitalization".format(calib_HospitalizationRate))
        print("DEATH RATE = {:.1%} infected people will die".format(calib_DeathRate))
        print("Intervention Day = detected {:.0f} days after the cutoff, on {:%Y-%m-%d}".format(calib_InterventionDay, minDate+timedelta(days=calib_InterventionDay)))
        print("Intervention Lag = detected {:.0f} days for full intervention effect".format(calib_InterventionLag))
        print("Intervention Effect = detected {:.0%} reduction of initial transmission rate".format(calib_InterventionEffect))
        print("Detection Rate = {:.1%} infected are reported as positives".format(calib_DetectionRate))
        print("Hospital Capacity = {:.0f} beds per million".format(calib_HospitalCapacity*1e6))
        print("Detection Capacity = {:.0f} infectious cases per million".format(calib_DetectionCapacity*1e6))
        #display(popt)

    #compute model numbers for the calibration period
    #------------------------------------------------

    y = population * SEIR(x,i0=calib_I0, beta=calib_Beta, gamma_i1=calib_GammaI1, gamma_i2=calib_GammaI2, gamma_h=calib_GammaH, incubation_rate = incubation_rate, hospitalization_rate = calib_HospitalizationRate, death_rate=calib_DeathRate,
                         intervention_day = calib_InterventionDay, intervention_lag=calib_InterventionLag, intervention_effect = calib_InterventionEffect,
                         detection_rate = calib_DetectionRate,
                         hospital_capacity = calib_HospitalCapacity, detection_capacity = calib_DetectionCapacity)

    s1['fit Fatalities (SIR)'] = y[:,cF]  #reported stats are about new cases, they do not seem to account for people having recovered
    s1['fit NewFatalities (SIR)'] = s1['fit Fatalities (SIR)'].diff()

    s1['fit Hospitalized (SIR)'] = y[:,cH]  #reported positive stats are about new cases, they do not seem to account for people having recovered
    s1['fit NewHospitalized (SIR)'] = s1['fit Hospitalized (SIR)'].diff()

    s1['fit Positive (SIR)'] = y[:,cP]  #reported positive stats are about new cases, they do not seem to account for people having recovered
    s1['fit NewPositive (SIR)'] = s1['fit Positive (SIR)'].diff()

    s1['fit Infectious (SIR)'] = y[:,cI1]+y[:,cI2]+y[:,cH]  
    
    #display(s1.sort_values(by='Date',ascending=False))

    #plot the model in comparison with calibrating data
    #--------------------------------------------------

    if output:
        fig,axs = plt.subplots(nrows=3, ncols=2,figsize=[16,16])

        ax = plt.subplot(321)
        plt.title(state + ' Fatalities')
        plt.plot(s1['date'], s1['death'],'ko-',label='Actual')
        plt.plot(s1['date'], s1['fit Fatalities (SIR)'],'r-',label='SIR')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))
        plt.legend()
        plt.grid()
        #plt.yscale('log')

        ax = plt.subplot(322)
        plt.title(state + ' New Fatalities')
        plt.plot(s1['date'], s1['deathIncrease'],'ko-',label='Actual')
        plt.plot(s1['date'], s1['fit NewFatalities (SIR)'],'r-',label='SIR')
        plt.legend()
        plt.grid()
        #plt.yscale('log')

        ax = plt.subplot(323)
        plt.title(state + ' Hospitalized')
        #plt.plot(x, s1['hospitalized'],'ko-',label='Actual')
        plt.plot(s1['date'], s1['hospitalizedCurrently'],'ko-',label='Actual')
        plt.plot(s1['date'], s1['fit Hospitalized (SIR)'],'r-',label='SIR')
        plt.legend()
        plt.grid()
        #plt.yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))

        ax = plt.subplot(324)
        plt.title(state + ' New Hospitalized')
        #plt.plot(x, s1['hospitalizedIncrease'],'ko-',label='Actual')
        plt.plot(s1['date'], s1['hospitalizedCurrently'].diff(),'ko-',label='Actual')
        plt.plot(s1['date'], s1['fit NewHospitalized (SIR)'],'r-',label='SIR')
        plt.legend()
        plt.grid()
        #plt.yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))

        ax = plt.subplot(325)
        plt.title(state + ' Positives')
        #plt.plot(x, s1['hospitalized'],'ko-',label='Actual')
        plt.plot(s1['date'], s1['positive'],'ko-',label='Actual')
        plt.plot(s1['date'], s1['fit Positive (SIR)'],'r-',label='SIR')
        plt.legend()
        plt.grid()
        #plt.yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))

        ax = plt.subplot(326)
        plt.title(state + ' New Positives')
        #plt.plot(x, s1['hospitalizedIncrease'],'ko-',label='Actual')
        plt.plot(s1['date'], s1['positiveIncrease'],'ko-',label='Actual')
        plt.plot(s1['date'], s1['fit NewPositive (SIR)'],'b-',label='SIR')
        plt.legend()
        plt.grid()
        #plt.yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))

        fig.autofmt_xdate()
        plt.show()
        
        
    #long range forecast
    #-------------------

    xx = np.arange(365)
    y = population * SEIR(xx,i0=calib_I0, beta=calib_Beta, gamma_i1=calib_GammaI1, gamma_i2=calib_GammaI2, gamma_h=calib_GammaH, incubation_rate = incubation_rate, hospitalization_rate = calib_HospitalizationRate, death_rate=calib_DeathRate,
                         intervention_day = calib_InterventionDay, intervention_lag=calib_InterventionLag, intervention_effect = calib_InterventionEffect,
                         detection_rate = calib_DetectionRate,
                         hospital_capacity = calib_HospitalCapacity, detection_capacity = calib_DetectionCapacity)



    idx = np.argmax( np.diff(y[:,cF])).item()   #peak daily fatalities
    max_dailyfatalities_day = minDate+timedelta(days=idx)
    max_dailyfatalities_rate = np.diff(y[:,cF])[idx]
    total_fatalities = y[-1,cF]

    if output:
        print("")
        print("SIR long range forecast")
        print("-----------------------")
        print("Daily New Fatalities would peak day {:,}, on {:%Y-%m-%d}, at {:,.0f} fatalities per day".format(idx,max_dailyfatalities_day, max_dailyfatalities_rate))
        print("Cumulative Fatalities would reach {:,.0f} after one year".format(total_fatalities))

        fig,axs = plt.subplots(nrows=1, ncols=2,figsize=[12,6])
        
        ax = plt.subplot(121)
        plt.title(state + ' Forecast')
        plt.plot(xx, y[:,cH],'b-',label='Hospitalized')
        plt.plot(xx[1:], np.diff(y[:,cP]),'m-',label='New Positive')
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        plt.legend()
        plt.grid()
        #plt.yscale('log')

        ax = plt.subplot(122)
        plt.title(state + ' Forecast')
        lns2 = plt.plot(xx, y[:,cF],'c-',label='Fatalities (lhs)')
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))

        ax2 = ax.twinx() #instantiate second y axis, share same x axis
        lns3 = plt.plot(xx[1:], np.diff(y[:,cF]),'m-',label='Daily Fatalities (rhs)')
        ax2.yaxis.set_major_formatter(FuncFormatter(thousands))

        lns = lns2+lns3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        ax.grid()
        #plt.yscale('log')

        plt.show()
        
        #show death and hospitalized on same plot
        #########################################
        fig,axs = plt.subplots(nrows=2, ncols=3,figsize=[18,12])      

       
        deathRateCurve = []
        hospitalizationRateCurve = []
        interventionCurve = []
        detectionCurve = []
        day=0
        for idx, row in s1.iterrows():
            
            h = row['fit Hospitalized (SIR)'] / population
            infectious = row['fit Infectious (SIR)'] / population
            
            dr, hr, detectr = rationing(hospitalized=h, infectious=infectious, hospital_capacity=calib_HospitalCapacity, death_rate=calib_DeathRate, 
                               hospitalization_rate=calib_HospitalizationRate, detection_rate=calib_DetectionRate, detection_capacity=calib_DetectionCapacity)

            deathRateCurve.append(dr)
            hospitalizationRateCurve.append(hr)
            detectionCurve.append(detectr)
            
            
            e=intervention(day=day, day0=calib_InterventionDay, lag=calib_InterventionLag, effect=calib_InterventionEffect)
            interventionCurve.append(e)
            day += 1
            
        
        ax = plt.subplot(231)
        plt.title('{} - Fatalities'.format(state))

        lns1 = ax.plot(s1['date'], s1['death'],'ro-',label='actual')
        lns2 = ax.plot(s1['date'], s1['fit Fatalities (SIR)'],'r--',label='model')
        
        ax.legend()
        ax.grid()
        plt.yscale('log')      
        bottom, top = plt.ylim()  # return the current ylim to scale the other charts

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))

        ax = plt.subplot(234)
        plt.plot(s1['date'], deathRateCurve, 'k:', label='death rate')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(x, '.1%')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))
        plt.grid()
        plt.legend()
       
        
        ax = plt.subplot(232)
        plt.title('{} - Hospitalized'.format(state))
        
        #ax2 = ax1.twinx() #instantiate second y axis, share same x axis
        lns3 = plt.plot(s1['date'], s1['hospitalizedCurrently'],'k*-',label='actual)')
        lns4 = plt.plot(s1['date'], s1['fit Hospitalized (SIR)'],'k--',label='model')

        plt.legend()
        plt.grid()
        plt.yscale('log')
#        plt.ylim((10*bottom, 10*top))   # set the ylim to bottom, top

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))

        ax = plt.subplot(235)
        plt.plot(s1['date'], hospitalizationRateCurve, 'k:',label='hospitalization rate')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(x, '.1%')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))
        plt.grid()
        plt.legend()

        ax = plt.subplot(233)
        plt.title('{} - Positives'.format(state))
        
        lns5 = plt.plot(s1['date'], s1['positive'],'bo-',label='actual')
        lns6 = plt.plot(s1['date'], s1['fit Positive (SIR)'],'b--',label='model')

        plt.legend()
        plt.grid()
        plt.yscale('log')
#        plt.ylim((100*bottom, 100*top))   # set the ylim to bottom, top

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))

        ax = plt.subplot(236)
        plt.plot(s1['date'], interventionCurve, 'k-.',label='intervention effect')
        plt.plot(s1['date'], detectionCurve, 'k:', label='detection rate')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(x, '.1%')))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%B-%d'))
        plt.ylim((0, 1.1))   # set the ylim to 0%-110%
        plt.grid()
        plt.legend()

        #lns = lns1+lns2+lns3+lns4+lns5+lns6
        #labs = [l.get_label() for l in lns]
        #ax1.legend(lns, labs, loc=0)

        #ax1.yaxis.set_major_formatter(FuncFormatter(thousands))
        #ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        #ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        
        #ax1.grid()
        #plt.show()

        fig.autofmt_xdate()
        plt.show()

    result = {}
    result['state'] = state
    result['curr fatalities'] = s1['death'].iloc[-1]
    result['population'] = population
    result['cutoff'] = minDate
    result['i0/m'] = calib_I0 * 1e6
    result['gamma_i1/d'] = 1/calib_GammaI1
    result['gamma_i2/d'] = 1/calib_GammaI2
    result['gamma_h/d'] = 1/calib_GammaH
    result['incubation rate/d'] = 1/incubation_rate
    result['hospitalization rate'] = calib_HospitalizationRate
    result['death rate'] = calib_DeathRate
    result['interv'] = calib_InterventionDay
    result['lag'] = calib_InterventionLag
    result['effect'] = calib_InterventionEffect
    result['detection'] = calib_DetectionRate
    result['hosp capacity'] = calib_HospitalCapacity
    result['detect capacity'] = calib_DetectionCapacity
    result['peak day'] = max_dailyfatalities_day
    result['peak fatalities'] = max_dailyfatalities_rate
    result['cum fatalities'] = total_fatalities

    #print(result)
    
    return result
    
    
state = 'NY'    #NY CA
#display(data[data['state']==state])

#NY timeline of intervention
# March 13th: New Rochelle school closures within 1-mile radius of first case
# Wednesday March 18: state-wide school closures
# Sunday March 22: state-wide non-essential workers to stay home, non-essential gatherings prohibited
# Simulation starts on 2020-03-04

r = calibrate(data, output=True, state=state, cutoff_on='positive', cutoff=1, incubation_rate=1/5)    
    

state='NY'

cutoff_on='death'
cutoff = 1

calib_on='both'

IncubationRate = 1/5 #it takes five days for people to become infectious after exposure. this is not calibrated, the scientific evidence is pretty robust.

r = calibrate(data, output=False, state=state, calib_on=calib_on, incubation_rate=IncubationRate)
display(r)

population = r['population']
i0 = r['i0/m']*1e-6
gamma = 1/r['gamma/d']
beta = r['R0']*gamma
incubation_rate = IncubationRate
hospitalization_rate = r['hospitalization rate']*gamma
death_rate = gamma * r['death rate']
intervention_day = r['interv']
intervention_lag = r['lag']
intervention_effect = r['effect']
detection_rate = r['detection']
hospital_capacity = r['capacity']
worst_case = r['worst']


format_dict2 = {'population':'{:,.0f}', 
               'curr fatalities': '{:,.0f}',
               'cutoff':'{:%Y-%m-%d}',
               'i0/m': '{:.0f}',
               'gamma/d': '{:.0f}',
               'R0': '{:.1f}',
               'death rate': '{:.1%}',
               'interv': '{:.0f}',
               'lag': '{:.0f}',
               'effect': '{:.0%}',
               'detection': '{:.0%}',
               'capacity': '{:.000%}',
               'worst': '{:.1%}',
               'peak day':'{:%Y-%m-%d}',
               'peak fatalities': '{:,.0f}',
               'cum fatalities': '{:,.0f}',
              }

res = []
res.append(r)
res = pd.DataFrame(res)
display(res.style.format(format_dict2).hide_index()) 
#print(beta, gamma)

R0_min = 1.1
R0_max = 3.5

#Bounds and initial guess for calibration algorithm scipy.curve_fit
I0_min = 1e-6
I0_max = 1000e-6
Gamma_min = 1/14
Gamma_max = 1/5
Beta_min = 1.1 * Gamma_min
Beta_max = 3 * Gamma_max
HospitalizationRate_min = 0.01
HospitalizationRate_max = 0.30
DeathRate_min = 0.001
DeathRate_max = 0.05
InterventionDay_min = 0   
InterventionDay_max = 50   #50 days after report of first death
InterventionLag_min = 1
InterventionLag_max = 7
InterventionEffect_min = 0
InterventionEffect_max = 0.95 
DetectionRate_min = 0.1           #10% of infectious cases detected as positives
DetectionRate_max = 1             #100% detection of infectious case reported as positives 
HospitalCapacity_min = 1e-6       #1 bed per million
HospitalCapacity_max = 1000e-6    #1000 beds per million
WorstCase_min = 0                 #no rationing
WorstCase_max = 10                #10 times increase of death rate because of hospital rationing, 10 000% reduction of hospitalization rate


minDate, s1, x, z, missing_data, scale_h = prep_data(data=data, state=state, cutoff_on=cutoff_on, cutoff=cutoff, truncate=0, calib_on=calib_on)

from scipy.spatial import distance

N1 = 20
N2 = 20

#Y = np.linspace(R0_min, R0_max, num=N1)
#ylabel = 'R0'
Y = np.linspace(InterventionDay_min, InterventionDay_max, num=N1)
ylabel = 'intervention day'

X = np.linspace(Gamma_min, Gamma_max, num=N2)
X = 1/X
xlabel = 'gamma'

res = []
for i1 in range(N1):
    row = []
    for i2 in range(N2):
        
        gamma = Gamma_min + (Gamma_max-Gamma_min)/N2 * i2
        intervention_day = InterventionDay_min + (InterventionDay_max-InterventionDay_min)/N1 * i1
        #beta = gamma * R0_min + (R0_max-R0_min)/N1 * i1

        y = SEIR_calib(x, i0=i0, beta=beta, gamma=gamma, incubation_rate=incubation_rate, hospitalization_rate=hospitalization_rate, death_rate=death_rate,
                            intervention_day=intervention_day, intervention_lag=intervention_lag, intervention_effect=intervention_effect, 
                       detection_rate = detection_rate, hospital_capacity = hospital_capacity, worst_case = worst_case,
                       population=population, missing_data=missing_data, scale_h=scale_h)

        err = distance.euclidean(y,z)
        row.append(err)
    res.append(row)
        
#display(res)
plt.contour(X, Y, res, levels=5, cmap='cividis')#'coolwarm')
plt.contourf(X, Y, res, levels=100, cmap='cividis')#'coolwarm')
plt.colorbar()
plt.xlabel(xlabel)
plt.ylabel(ylabel)
#plt.plot(res['gamma'],res['err'])
plt.show()


state='NY'

cutoff_on='death'
cutoff = 1

calib_on='both'

IncubationRate = 1/5

minDate, s1, x, z, missing_data, scale_h = prep_data(data=data, state=state, cutoff_on=cutoff_on, cutoff=cutoff, truncate=0, calib_on=calib_on)

r = calibrate(data, output=False, state=state, calib_on=calib_on, incubation_rate=IncubationRate)
#display(r)

population = r['population']
i0 = r['i0/m']*1e-6
gamma = 1/r['gamma/d']
R0 = r['R0']
beta = R0*gamma
incubation_rate = IncubationRate
hospitalization_rate = r['hospitalization rate']
death_rate = r['death rate']
intervention_day = r['interv']
intervention_lag = r['lag']
intervention_effect = r['effect']
intervention_effect = r['effect']
detection_rate = r['detection']
hospital_capacity = r['capacity']


format_dict2 = {'population':'{:,.0f}', 
               'curr fatalities': '{:,.0f}',
               'cutoff':'{:%Y-%m-%d}',
               'i0/m': '{:.0f}',
               'gamma/d': '{:.0f}',
               'R0': '{:.1f}',
               'hospitalization rate': '{:.1%}',
               'death rate': '{:.1%}',
               'interv': '{:.0f}',
               'lag': '{:.0f}',
               'effect': '{:.0%}',
               'detection': '{:.0%}',
               'capacity': '{:.000%}',
               'worst': '{:.1%}',
               'peak day':'{:%Y-%m-%d}',
               'peak fatalities': '{:,.0f}',
               'cum fatalities': '{:,.0f}',
              }

res = []
res.append(r)
res = pd.DataFrame(res)
display(res.style.format(format_dict2).hide_index()) 


plt.subplots(nrows=1,ncols=3,figsize=(16,4))

xx = x #np.arange(50)

y = population * SEIR(xx, i0=i0, beta=beta, gamma=gamma, incubation_rate = incubation_rate, hospitalization_rate=gamma*hospitalization_rate, death_rate=gamma*death_rate,
                      intervention_day = intervention_day, intervention_lag=intervention_lag, intervention_effect = intervention_effect, detection_rate = detection_rate, hospital_capacity = hospital_capacity, worst_case=worst_case)


plt.subplot(131)
plt.title('hospitalized')
plt.plot(xx, y[:,cH],'ro-', label='calib')
plt.plot(xx, s1['hospitalizedCurrently'],'k*-', label='actual')
plt.yscale('log')
plt.grid()

plt.subplot(132)
plt.title('fatalities')
plt.plot(xx, y[:,cF],'ro-', label='calib')
plt.plot(xx, s1['death'],'k*-', label='actual')
plt.yscale('log')
plt.grid()

plt.subplot(133)
plt.title('positives')
plt.plot(xx, y[:,cP],'ro-', label='calib')
plt.plot(xx, s1['positive'],'k*-', label='actual')
plt.yscale('log')
plt.grid()

Gamma_min = 1/gamma - 3
for i in range(7):
    
    g = 1/(Gamma_min+i)
    b = g * R0
    
    y = population * SEIR(xx, i0=i0, beta=b, gamma=g, incubation_rate = incubation_rate, hospitalization_rate=g*hospitalization_rate, death_rate=g*death_rate,
                          intervention_day = intervention_day, intervention_lag=intervention_lag, intervention_effect = intervention_effect,
                          detection_rate = detection_rate, hospital_capacity = hospital_capacity, worst_case=worst_case)
    
    plt.subplot(131)
    plt.plot(xx, y[:,cH],label='gamma={:.1f}'.format(Gamma_min+i))

    plt.subplot(132)
    plt.plot(xx, y[:,cF],label='gamma={:.1f}'.format(Gamma_min+i))

    plt.subplot(133)
    plt.plot(xx, y[:,cP],label='gamma={:.1f}'.format(Gamma_min+i))
    
plt.legend()







