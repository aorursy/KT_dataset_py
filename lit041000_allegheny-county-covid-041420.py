import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

from scipy.integrate import odeint

import matplotlib.pyplot as plt



from ipywidgets import *



"""     

interact(interact_input, 

         Inf_Prd_Days = '9.40', 

         N = '3000',

         Reprod_rate = '3.10',

         Mortality = '0.05',

         LOS_MEDSURG = '3',

         LOS_ICU = '6',

         Hosp_rate = '0.20',

         ICU_rate = '0.50',

         Vent_rate = '0.80',

         max_hosp = '40',

         max_icu = '30',

         max_vents = '25'

        );

"""



def update(infectious_period_in_days, 

           N,

           reproduction_rate, 

           mortality_rate, 

           length_of_stay_MEDSURG, 

           length_of_stay_ICU, 

           hospitalization_ratio, 

           ICU_ratio, 

           ventilator_ratio,

           max_hosp_beds,

           max_icu_beds,

           max_ventilators

          ):

    import matplotlib.pyplot as plt

    

    positive_test_to_population_factor = 1

    #prevalence -> week2 #infected in pennsylvania/ - test cases

    #https://emcrit.org/ibcc/covid19/#general_prognosis



    #infectious_period_in_days = 14

    # CDC recommends 14 day

    # and 15 day quarantine

    days_projected_forward = 60

    #the model starts on 3/23/20 and projects forward



    """

    Parameters for the SIR-F Model

    """

    #N = 18000

    # This is the susceptible population out of 1.223 million. Not everyone will be exposed to the virus



    I0, R0, D0 = 6, 0, 1

    #I0, R0, D0 = 48, 0, 1

    # Initial number of infected and recovered and dead individuals, I0 and R0

    # the models initial parameters are set at 3/23/20 when the interventions began



    # Everyone else, S0, is susceptible to infection initially.

    S0 = N - I0 - R0



    #mortality_rate =  .05

    # The model mortality rate - Allegheny is abnormally low compared to Covid global rates

    # .66% compared to global estimates of 3.4%



    #time_to_recover = infectious_period_in_days = 5.2

    gamma =  1./infectious_period_in_days

    #gamma = transition rate from Infected to Recovered



    Reproduction_rate0 = reproduction_rate

    #the average number of people that one infected person will infect



    beta = gamma * Reproduction_rate0

    #beta = transition rate from Susceptible to Infected

    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4552173/





    """

    Load the Data

    """

    real_data = pd.read_excel('/kaggle/input/pittnews3/Allegheny County COVID Daily Data newtags2.xlsx','Sheet2').dropna(how='all')  

    #real_data = real_data[real_data['Date'] >= '3/23/2020']

    #filter from the first day of intervention

    #print(real_data.head())



    """

    Calculate Active Infectious from Accumulated Positive Cases

    """

    """

    ipid_rounded = round(int(infectious_period_in_days),0)

    temp_estimated_infectious = np.zeros(len(real_data['Accu. Positive Cases']) + \

                                         ipid_rounded-1)

    #for every day, generate x # of rows for each infected

    #print(real_data['Accu. Positive Cases'])

    

    #this is set to 9 because we start on 3/24/2020

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

    First Plot-------------------------------------------------------------------

    """

    #https://python4astronomers.github.io/plotting/advanced.html

    

    # Plot the data on three separate curves for S(t), I(t) and R(t)

    fig = plt.figure(figsize=(16, 8), dpi=200, facecolor='w', edgecolor='k')

    

    #fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8),dpi=200)

    ax1 = fig.add_subplot(121)

    

    #plt.xticks([0.2, 0.4, 0.6, 0.8, 1.],

    #       ["Jan\n2009", "Feb\n2009", "Mar\n2009", "Apr\n2009", "May\n2009"])

    #ax.xaxis_date()

    fig.autofmt_xdate(bottom=0.2, rotation=45, ha='right')

    #plt.xticks(rotation=90)

    #SIR curve

    #date_labels = pd.date_range(start = "3/23/2020", periods = days_projected_forward).to_pydatetime().tolist()

    date_labels = pd.date_range(start = "3/23/2020", periods = days_projected_forward).to_pydatetime().tolist()

    #ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')

    ax1.plot(date_labels, I, 'r', alpha=0.5, lw=2, label='Infected')

    #ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

    ax1.plot(date_labels, D, 'g', alpha=0.5, lw=2, label='Deaths')

    #add infectious actual data

    

    real_data_filtered = real_data[real_data['Date'] >= '3/23/2020']

    #date_labels = pd.date_range(start = "3/23/2020", periods = len(real_data_filtered['Accu. Positive Cases'])).to_pydatetime().tolist()

    #ax1.plot(date_labels,real_data_filtered['Accu. Positive Cases'],'k',label ='Accu. Positive Cases')

    date_labels = pd.date_range(start = "3/23/2020", periods = len(real_data_filtered['Daily Positive 7 Day Average'])).to_pydatetime().tolist()

    ax1.plot(date_labels,real_data_filtered['Daily Positive 7 Day Average'],'k',label ='Daily Positive 7 Day Average')

    #date_labels = pd.date_range(start = "3/23/2020", periods = len(real_data_filtered['Estimated Total Infectious'])).to_pydatetime().tolist()

    #ax1.plot(date_labels,real_data_filtered['Estimated Total Infectious'],'k',label ='Estimated Total Infectious')

    #print(real_data['Daily Positive 7 Day Average'])

    

    #ax.grid(b=True, which='major', c='w', lw=2, ls='-')

    #ax.yaxis.set_tick_params(length=1)

    #ax.xaxis.set_tick_params(length=1)

    legend = ax1.legend()

    legend.get_frame().set_alpha(0.5)

    #for spine in ('top', 'right', 'bottom', 'left'):

    #    ax.spines[spine].set_visible(False)

    ax1.set_title('Allegheny County Covid SIR Model')

        

    """

    Simple Usage Model - HIV

    """

    #SECOND GRAPH

    #print('I')

    #print(I)

    #print('hospitalization ratio')

    #print(hospitalization_ratio)

    daily_projected_hospitalized = []

    #z0=0.0

    #diff=0.0

    import math

    #infected is the daily new arrivals

    for z in I:

        #if z0 == 0.0:

        #    daily_projected_hospitalized.append(0.0)

        if z >= 1:

            #print('z')

            #print(z)

            #diff = round(z,0) - z0

            

            #diff = diff * hospitalization_ratio

            #daily_projected_hospitalized.append(round(diff,0))

            

            #changed b/c we are now modeling daily new instead of accumulated positive

            #everyone who is going to the hospital but not the ICU....

            daily_projected_hospitalized.append(round(z*hospitalization_ratio*(1-ICU_ratio),0))

        else:

            daily_projected_hospitalized.append(0.0)

        #z0 = z

    #print('daily_hosp')

    #print(daily_projected_hospitalized)

    

    daily_projected_ICU = []

    #infected is the daily new arrivals

    for z in I:

        if z >= 1:

            daily_projected_ICU.append(round(z * hospitalization_ratio * ICU_ratio,0))

            #print(type(z * hospitalization_ratio * ICU_ratio))

        else:

            #print(z)

            #print(z * hospitalization_ratio)

            #print(z * hospitalization_ratio * ICU_ratio)

            #print(round(z * hospitalization_ratio * ICU_ratio,0))

            daily_projected_ICU.append(0)

    

    daily_projected_ventilators = []

    #infected is the daily new arrivals

    for z in I:

        if z >= 1:

            daily_projected_ventilators.append(round(z * hospitalization_ratio * ICU_ratio * ventilator_ratio,0))

        else:

            daily_projected_ventilators.append(0)



    tempH=[]

    tempH = np.zeros(len(daily_projected_hospitalized)+length_of_stay_MEDSURG)

    for i in range(0,len(daily_projected_hospitalized)):

        for z in range(0,length_of_stay_MEDSURG):

            tempH[i+z]= tempH[i+z] + [daily_projected_hospitalized[i]]

    #print('tempH')

    #print(tempH)

    

    tempICU=[]

    tempICU = np.zeros(len(daily_projected_ICU)+length_of_stay_ICU)

    for i in range(0,len(daily_projected_ICU)):

        for z in range(0,length_of_stay_ICU):

            tempICU[i+z]= tempICU[i+z] + [daily_projected_ICU[i]]



    tempV=[]

    tempV = np.zeros(len(daily_projected_ventilators)+length_of_stay_ICU)

    for i in range(0,len(daily_projected_ventilators)):

        for z in range(0,length_of_stay_ICU):

            tempV[i+z]= tempV[i+z] + [daily_projected_ventilators[i]]

    

        ax2 = fig.add_subplot(122)

    fig.autofmt_xdate(bottom=0.2, rotation=45, ha='right')

    

    axis_length = int(days_projected_forward/2)

    date_labels_C = pd.date_range(start = "3/23/2020", periods = axis_length).to_pydatetime().tolist()

    date_labels_F = pd.date_range(start = "3/23/2020", periods = axis_length).to_pydatetime().tolist()

    #ax2.plot(date_labels_C, daily_projected_hospitalized[:axis_length], 'r', alpha=0.5, lw=2, label='Daily MEDSURG Beds Needed')

    #ax2.plot(date_labels_F, daily_projected_ICU[:axis_length], 'b', alpha=0.5, lw=2, label='Daily ICU Beds Needed')

    #ax2.plot(date_labels_F, daily_projected_ventilators[:axis_length], 'g', alpha=0.5, lw=2, label='Daily Vents Needed')

    

    '''

    real_data_filtered = real_data[real_data['Date'] >= '3/23/2020']

    date_labels = pd.date_range(start = "3/23/2020", periods = len(real_data_filtered['Delta_H'])).to_pydatetime().tolist()

    ax2.plot(date_labels,real_data_filtered['Delta_H'],'k',label ='Delta_H')

    

    

    real_data_filtered = real_data[real_data['Date'] >= '3/23/2020']

    date_labels = pd.date_range(start = "3/23/2020", periods = len(real_data_filtered['Delta_ICU'])).to_pydatetime().tolist()

    ax2.plot(date_labels,real_data_filtered['Delta_ICU'],'m',label ='Delta_ICU')

    

    

    real_data_filtered = real_data[real_data['Date'] >= '3/23/2020']

    date_labels = pd.date_range(start = "3/23/2020", periods = len(real_data_filtered['Delta_V'])).to_pydatetime().tolist()

    ax2.plot(date_labels,real_data_filtered['Delta_V'],'y',label ='Delta_V')

    '''

    

    real_data_filtered = real_data[real_data['Date'] >= '3/23/2020']

    date_labels = pd.date_range(start = "3/23/2020", periods = len(real_data_filtered['New_Hospitalized'])).to_pydatetime().tolist()

    ax2.plot(date_labels,real_data_filtered['New_Hospitalized'],'k',label ='New_Hospitalized')

    

    

    real_data_filtered = real_data[real_data['Date'] >= '3/23/2020']

    date_labels = pd.date_range(start = "3/23/2020", periods = len(real_data_filtered['ICU'])).to_pydatetime().tolist()

    ax2.plot(date_labels,real_data_filtered['ICU'],'m',label ='ICU')

    

    

    real_data_filtered = real_data[real_data['Date'] >= '3/23/2020']

    date_labels = pd.date_range(start = "3/23/2020", periods = len(real_data_filtered['Ventilatorstotal'])).to_pydatetime().tolist()

    ax2.plot(date_labels,real_data_filtered['Ventilatorstotal'],'y',label ='Ventilatorstotal')

    

    

    

    ax2.axhline(y=max_hosp_beds, color='r', linestyle=':')

    ax2.axhline(y=max_icu_beds, color='b', linestyle=':')

    ax2.axhline(y=max_ventilators, color='g', linestyle=':')

    #plt.xticks(rotation=90)

    legend = ax2.legend()

    legend.get_frame().set_alpha(0.5)

    #for spine in ('top', 'right', 'bottom', 'left'):

    #    ax.spines[spine].set_visible(False)

    ax2.set_title('Daily Beds & Vents Needed')

    plt.show()

    

    #print(tempH)

    #print('tempICU')

    #print(tempICU)

    #print('vents')

    #print(tempV)

    #print(length_of_stay_ICU)

    #print(length_of_stay_MEDSURG)

    #print(len(daily_projected_hospitalized))

    #print(len(daily_projected_ICU))

    

    """Second Plot-----------------------------------------------------------"""

    

    """

    ax2 = fig.add_subplot(122)

    fig.autofmt_xdate(bottom=0.2, rotation=45, ha='right')

    

    axis_length = int(days_projected_forward/2)

    date_labels_C = pd.date_range(start = "3/23/2020", periods = axis_length).to_pydatetime().tolist()

    date_labels_F = pd.date_range(start = "3/23/2020", periods = axis_length).to_pydatetime().tolist()

    ax2.plot(date_labels_C, tempH[:axis_length], 'r', alpha=0.5, lw=2, label='MEDSURG Beds Needed')

    ax2.plot(date_labels_F, tempICU[:axis_length], 'b', alpha=0.5, lw=2, label='ICU Beds Needed')

    ax2.plot(date_labels_F, tempV[:axis_length], 'g', alpha=0.5, lw=2, label='Vents Needed')

    

    ax2.axhline(y=max_hosp_beds, color='r', linestyle=':')

    ax2.axhline(y=max_icu_beds, color='b', linestyle=':')

    ax2.axhline(y=max_ventilators, color='g', linestyle=':')

    #plt.xticks(rotation=90)

    legend = ax2.legend()

    legend.get_frame().set_alpha(0.5)

    #for spine in ('top', 'right', 'bottom', 'left'):

    #    ax.spines[spine].set_visible(False)

    ax2.set_title('Beds & Vents Needed')

    plt.show()

    """

    

    """

    #THIRD GRAPH

    #input array with hospitalizations and deaths

    hospitalizations = daily_projected_hospitalized

    deaths = daily_projected_fatalities



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

    """



#convert the strings to floats

def interact_input( 

    Inf_Prd_Days , 

    N ,

    Reprod_rate ,

    Mortality ,

    LOS_MEDSURG ,

    LOS_ICU ,

    Hosp_rate ,

    ICU_rate ,

    Vent_rate,

    max_hosp ,

    max_icu ,

    max_vents):

    

    update(float(Inf_Prd_Days),float(N),float(Reprod_rate),float(Mortality),int(LOS_MEDSURG), \

          int(LOS_ICU),float(Hosp_rate),float(ICU_rate),float(Vent_rate),int(max_hosp), \

          int(max_icu),int(max_vents))



#you can also change the defaults here

interact(interact_input, 

         Inf_Prd_Days = '4.10', 

         N = '300',

         Reprod_rate = '2.05',

         Mortality = '0.05',

         LOS_MEDSURG = '3',

         LOS_ICU = '6',

         Hosp_rate = '0.20',

         ICU_rate = '0.50',

         Vent_rate = '0.80',

         max_hosp = '15',

         max_icu = '15',

         max_vents = '15');

#https://ipywidgets.readthedocs.io/en/stable/examples/Using%20Interact.html

"""

interact(interact_input, 

         Inf_Prd_Days = '4.10', 

         N = '300',

         Reprod_rate = '2.05',

         Mortality = '0.05',

         LOS_MEDSURG = '3',

         LOS_ICU = '6',

         Hosp_rate = '0.20',

         ICU_rate = '0.50',

         Vent_rate = '0.80',

         max_hosp = '40',

         max_icu = '30',

         max_vents = '25'

        );

        

SIR-F Parameters and Definitions         

         Inf_Prd_Days = '4.10',  Infectious period in days for SIR

         N = '300', Number of Susceptible 

         Reprod_rate = '2.05', Reproduction Rate for SIR

         Mortality = '0.05', Mortality Rate for SIR-F



Utilization Model Parameters

         LOS_MEDSURG = '3', Number of days a normal patient stays in the hospital before getting discharged

         LOS_ICU = '6', Number of days a ICU patient stays before getting discharged

         Hosp_rate = '0.20', Percentage of Infected that become hospitalized (Normal + ICU)

         ICU_rate = '0.50', Percentage of Hospitalized that end up in the ICU

         Vent_rate = '0.80', Percentage of ICU Patients that end up using a ventilator

         max_hosp = '40', Max number of hospital beds (its the straight line at the top of the chart)

         max_icu = '30', Max number of ICU beds (its the straight line at the top of the chart)

         max_vents = '25', Max number of ventilators (its the straight line at the top of the chart)

"""


