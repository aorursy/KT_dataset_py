import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#the people who test positive are only a fraction of all those who are infectious
positive_test_to_population_factor = 1
#prevalence -> week2 #infected in pennsylvania/ - test cases
#https://emcrit.org/ibcc/covid19/#general_prognosis


infectious_period_in_days = 14
# CDC recommends 14 day
# and 15 day quarantine

hospitalized_ratio = 1
#hospitalized_ratio goes into MEDSURG bed

days_projected_forward = 80
#the model starts on 3/14/20 and projects forward

"""
Parameters for the SIR-F Model
"""
N = 3000
# This is the susceptible population out of 1.223 million. Not everyone will be exposed to the virus

I0, R0, D0 = 48, 2, 1
# Initial number of infected and recovered and dead individuals, I0 and R0
# the models initial parameters are set at 3/23/20 when the interventions began

# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

mortality_rate =  .05
# The model mortality rate - Allegheny is abnormally low compared to Covid global rates
# .66% compared to global estimates of 3.4%

#time_to_recover = infectious_period_in_days = 5.2
gamma =  1./infectious_period_in_days
#gamma = transition rate from Infected to Recovered

Reproduction_rate0 = 2.8
#the average number of people that one infected person will infect

beta = gamma * Reproduction_rate0
#beta = transition rate from Susceptible to Infected
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4552173/

"""
Parameters for the overflow model - Allegheny Wide Model
"""
#time spent under each scenario, all fatal cases go to ICU and all ICU cases have a respirator
length_of_stay_MEDSURG = 6
length_of_stay_ICU = 3
length_of_stay_ICU_no_ventilator = 1

#Number of vents and beds
n_ventilators = 25
n_ICU_beds = 20
n_MEDSURG_beds = 100
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
Load the Data
"""
real_data = pd.read_excel('/kaggle/input/pittsburghcovid/Allegheny County COVID Daily Data.xlsx','Sheet2').dropna(how='all')  
real_data.head()
#print(real_data)
#print(real_data['Accu. Positive Cases'])
"""
Calculate Active Infectious from Accumulated Positive Cases
"""
ipid_rounded = round(int(infectious_period_in_days),0)
temp_estimated_infectious = np.zeros(len(real_data['Accu. Positive Cases']) + \
                                     ipid_rounded-1)
#for every day, generate x # of rows for each infected
for i in range(0,len(real_data['Accu. Positive Cases'])):
    if i == 0 :
        for z in range(0,ipid_rounded):
            temp_estimated_infectious[i+z] = round(temp_estimated_infectious[i+z] + \
            real_data['Accu. Positive Cases'][i] - 0,0)
    else:
        for z in range(0,ipid_rounded):
            temp_estimated_infectious[i+z]= round(temp_estimated_infectious[i+z] + \
            real_data['Accu. Positive Cases'][i] - real_data['Accu. Positive Cases'][i-1],0)

#delete the last three rows because they don't include 3 days of actual data, its runoff
for i in range(1,ipid_rounded-1):
    temp_estimated_infectious[-1*i] = np.nan

#print(real_data['Accu. Positive Cases'])
#print(temp_estimated_infectious)
temp_estimated_infectious = temp_estimated_infectious / positive_test_to_population_factor
real_data['Estimated Total Infectious'] = pd.Series(temp_estimated_infectious)
real_data[['Estimated Total Infectious','Accu. Positive Cases','Date']].plot(x='Date')



"""
Run the SIR-F Model, prefit to pittsburgh data
"""
# A grid of time points (in days)
t = np.linspace(0, days_projected_forward, days_projected_forward)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R, D = y
    #lost people to infection each time unit
    dSdt = -beta * (S/N) * I
    #gain people to infection but lose people to recovery each time unit
    dIdt =  beta * (S/N) * I - gamma * I
    #gain people that recover from their infection
    dRdt = (1-mortality_rate) * gamma * I
    dFdt = mortality_rate * gamma * I
    return dSdt, dIdt, dRdt, dFdt

# Initial conditions vector
y0 = S0, I0, R0, D0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R, D = ret.T



"""
Plot the SIRF Curve
"""
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.xaxis_date()
plt.xticks(rotation=90)

#SIR curve
date_labels = pd.date_range(start = "3/14/2020", periods = days_projected_forward).to_pydatetime().tolist()
#ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(date_labels, I, 'r', alpha=0.5, lw=2, label='Infected')
#ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.plot(date_labels, D, 'g', alpha=0.5, lw=2, label='Deaths')

#add estimated infectious
date_labels = pd.date_range(start = "3/14/2020", periods = len(real_data['Accu. Positive Cases'])).to_pydatetime().tolist()
ax.plot(date_labels,real_data['Accu. Positive Cases'],'k',label ='Estimated Actual Infectious')

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()


"""
Calculate Daily Projected Hospitalized from the SIR model Active Infected Count
"""
daily_projected_hospitalized = []
z0=0
#infected is the daily new arrivals
for z in I:
    #print(z - z0)
    if z > z0:
        daily_projected_hospitalized.append(round((z-z0) * hospitalized_ratio,0))
    else:
        daily_projected_hospitalized.append(0)
    z0 = z

"""
Calculate Daily Projected Deaths from the SIR model Fatalities
"""
daily_projected_fatalities = []
d0=0
for q in D:
    if q > d0:
        daily_projected_fatalities.append(round(q-d0,0))
    d0 = q


"""
Simple Usage Model
"""
#estimates put only 10% of total infected are being tested
#5% ventilators + subset of the 15% severe symptoms that come to the hospital

temp = np.zeros(len(daily_projected_hospitalized)+length_of_stay_MEDSURG)
for i in range(0,len(daily_projected_hospitalized)):
    for z in range(0,length_of_stay_MEDSURG):
        temp[i+z]= temp[i+z] + [daily_projected_hospitalized[i]]

temp2 = np.zeros(len(daily_projected_fatalities)+length_of_stay_ICU)
for i in range(0,len(daily_projected_fatalities)):
    for z in range(0,length_of_stay_ICU):
        temp2[i+z]= temp2[i+z] + [daily_projected_fatalities[i]]

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)

date_labels_C = pd.date_range(start = "3/14/2020", periods = 80+length_of_stay_MEDSURG).to_pydatetime().tolist()
date_labels_F = pd.date_range(start = "3/14/2020", periods = 80+length_of_stay_ICU).to_pydatetime().tolist()

ax.plot(date_labels_C, temp, 'r', alpha=0.5, lw=2, label='MEDSURG Beds Needed')
ax.plot(date_labels_F, temp2, 'b', alpha=0.5, lw=2, label='ICU Beds Needed')
ax.plot(date_labels_F, temp2, 'g', alpha=0.5, lw=2, label='Vents Beds Needed')

plt.xticks(rotation=90)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
"""
ICU / MEDSURG Overflow Model
"""
#input array with hospitalizations and deaths
hospitalizations = daily_projected_hospitalized
deaths =           daily_projected_fatalities


SEIR_Input = [deaths,hospitalizations]
#initiliaze settings
available_beds_ICU = n_ICU_beds
available_beds_MEDSURG = n_MEDSURG_beds
available_ventilators = n_ventilators

admit_discharge_history = []
usage_history = []
for t in range(0,days_projected_forward):
    #for each day
    #print('day: ' + str(t))
    
    #release discharged resources
    for i in admit_discharge_history:
        #if a patient is getting discharged today
        if i[2] == t:
            #release discharged ICU beds
            #release discharged MEDSURG beds
            if i[3] == 'ICU':
                #print('releasing bed')
                available_beds_ICU += 1
            elif i[3] == 'MEDSURG':
                #print('releasing bed')
                available_beds_MEDSURG += 1
            #release discharged ventilators
            if i[4] == 1:
                #print('releasing vent')
                available_ventilators += 1

    #run fatal patients first because they get first access to the best resources
    if len(SEIR_Input[0]) > t:
        #for every fatal patient admitted
        for i in range(0,int(SEIR_Input[0][t])):
            #iterator variables
            #reset them each loop
            i_patient_type =''
            i_bed_type = ''
            i_ventilators_used = 0
            i_admit_day = t
            i_final_day = 0

            #set the patient type
            i_patient_type = 'F'

            #set the bed type
            #is there an ICU bed free?
            #is there a hopsital bed free?
            #no to both questions? they are sent home and use no resources
            if available_beds_ICU > 0:
                i_bed_type = "ICU"
                available_beds_ICU = available_beds_ICU - 1

                #do you have a respirator available?
                if available_ventilators > 0:
                    i_ventilators_used = 1
                    available_ventilators = available_ventilators - 1
                    i_admit_day = t
                    i_final_day = t + length_of_stay_ICU
                else:
                    i_ventilators_used = 0
                    i_admit_day = t
                    i_final_day = t + length_of_stay_ICU_no_ventilator
            elif available_beds_MEDSURG > 0:
                i_bed_type = "MEDSURG"
                available_beds_MEDSURG = available_beds_MEDSURG - 1
                #do you have a respirator available?
                if available_ventilators > 0:
                    i_ventilators_used = 1
                    available_ventilators = available_ventilators - 1
                    i_admit_day = t
                    i_final_day = t + length_of_stay_ICU
                else:
                    i_ventilators_used = 0
                    i_admit_day = t
                    i_final_day = t + length_of_stay_ICU_no_ventilator
            else:
                i_bed_type = "NONE"
            #print(i_patient_type,i_admit_day,i_final_day,i_bed_type,i_ventilators_used)
            admit_discharge_history.append([i_patient_type,i_admit_day,i_final_day,i_bed_type,i_ventilators_used])

    #run MEDSURG patients second
    if len(SEIR_Input[1]) > t:
        #for every critical patient
        for i in range(0,int(SEIR_Input[1][t])):
            #iterator variables
            #reset them each loop
            i_patient_type =''
            i_bed_type = ''
            i_ventilators_used = 0
            i_admit_day = t
            i_final_day = 0

            #set the patient type
            i_patient_type = 'C'

            #set the bed type
            #is there a hopsital bed free?
            #no, they are sent home and use no resources
            if available_beds_MEDSURG > 0:
                i_bed_type = "MEDSURG"
                available_beds_MEDSURG = available_beds_MEDSURG - 1
                i_ventilators_used = 0
                i_admit_day = t
                i_final_day = t + length_of_stay_MEDSURG
            else:
                i_bed_type = "NONE"
            #print(i_patient_type,i_admit_day,i_final_day,i_bed_type,i_ventilators_used)
            admit_discharge_history.append([i_patient_type,i_admit_day,i_final_day,i_bed_type,i_ventilators_used])
    
    #print('available_beds_MEDSURG: ' + str(available_beds_MEDSURG) + ' available_beds_ICU: ' + str(available_beds_ICU) + ' available_ventilators: ' + str(available_ventilators) )
    usage_history.append( [available_beds_MEDSURG,available_beds_ICU, available_ventilators] )
    #how many incoming hospitalization patients today?
    #print(SEIR_Input[0][i-1])

#print(usage_history[:][:])
#process the list you generated into readable data

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
# multiple line plot
ax.xaxis_date()
date_labels = pd.date_range(start = "3/14/2020", periods = 80).to_pydatetime().tolist()

plt.plot( date_labels ,[col[0] for col in usage_history],color='green',label='MEDSURG_Available')
plt.plot( date_labels ,[col[1] for col in usage_history],color='red'  ,label='ICU')
plt.plot( date_labels ,[col[2] for col in usage_history],color='blue' ,label='Ventilators')
plt.legend()

plt.show()

#curve without intervention
#curve with SEIR starting date 3/23
#curve with intervention start and intervention finish
#curve with actual data
# march 23 as starting date
from ipywidgets import *
def update(w=0,h=0):
    print(h+w)
    #updategraph

interact(update, w= widgets.IntSlider(value=1, min=0, max=10, step=2) , 
                 h= widgets.IntSlider(value=1, min=0, max=7, step=1) );


#https://ipywidgets.readthedocs.io/en/stable/examples/Using%20Interact.html
#@interact(x=(0.0,20.0,0.5))
def f(x,y,z):
    return x+y+z
interact(f, x=(0.0,20.0,0.05),y=(0.0,20.0,0.05),z=(0.0,20.0,0.05));


#def h(x=5.5):
#    return x
def f(a):
    return float(a) * 1.2

interact(f, a='3.4');



print('hello')
#https://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html
