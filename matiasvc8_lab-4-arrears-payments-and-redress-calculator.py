#Define the core mortgage inputs

amount_borrowed = 100000.00

periods = 180

monthly_rate = 0.06 / 12



#Create one data point per period

list_period = [x for x in range(periods)]

list_monthly_rate = [0.005 for x in range(periods)]

list_remaining_months = [periods - x for x in range(periods)]

list_remaining_months = [periods - x for x in range(periods)]



trigger_flag = [0 for x in list_period]

missed_flag = [0 for x in list_period]
import pandas as pd

#Create a dataframe with the mortgage timeline

data = {'period' : list_period,

        'remaining_months' : list_remaining_months,

        'monthly_rate' : list_monthly_rate,

        'amount_borrowed' : amount_borrowed,

        'trigger_flag' : trigger_flag,

        'missed_flag' : missed_flag}



df_original = pd.DataFrame(data).set_index('period')

df_original.head()
def cmi_calculator(rate, nper, fv):

    cmi = fv * ((rate * (1 + rate) ** nper) / ((1 + rate) ** nper - 1))

    return cmi
def balance_profile_recalculator(df, scenario_option='scenario_1', pay_arrears_period = None):

    '''

    Defines the balance of a mortgage for every period, according to some initial inputs stored in a DataFrame.

    The columns in the DataFrame input need to include:

        amount_borrowed: Original principal amount

        remaining_months: Decreasing monthly periods for the mortgage term

        monthly_rate: Interest rate effective at respective period

        trigger_flags: 1 or 0; 1 if CMI needs to be recalculated on that period

        missed_flags: 1 or 0; 1 if the customer missed their whole CMI payment on that period

    The outputs are:

        A dataframe with a breakdown of CMI and Arrears components, as well as Closing Amount to Settle.



    Optional Parameters:

        Four parameters available, each change the definition of Amount to Settle.

        Base definition is, for each period: 

            Opening Amount to Settle = Opening Remaining Principal (i.e. previous period closing balance)

            CMI of Amount to Settle = CMI of Principal

            Closing Amount to Settle = Opening Remaining Principal - CMI of Principal



        scenario_1: [Default] maintains base definition

        scenario_2: Includes all missed CMI payments into Amount to Settle (i.e. Amount to Settle includes total Arrears)

        scenario_3: Includes missed 'CMI of Interest' payments into Amount to Settle (i.e. includes Arrears of interest)

        scenario_4: Includes missed 'CMI of Amount to Settle' payments into Amount to Settle (i.e. includes Arrears of amount to settle)

    '''

    #Make copy of original df to avoid overwrites

    df = df.copy()



    #Define placeholders for the CMI values and respective components

    

    cmi_breakdown = (0.00, 0.00, 0.00, 0.00)

    df['cmi_total'] = cmi_breakdown[0]

    df['cmi_interest'] = cmi_breakdown[1]

    df['cmi_settle'] = cmi_breakdown[2]

    df['cmi_adjusted'] = cmi_breakdown[3]

    

    #Set default value for remaining principal

    df['closing_amount_to_settle'] = df['amount_borrowed']

    

    #Set default value for arrears balance and respective components

    arrears_breakdown = (0.00, 0.00, 0.00, 0.00)

    df['arrears_total'] = arrears_breakdown[0]

    df['arrears_interest'] = arrears_breakdown[1]

    df['arrears_settle'] = arrears_breakdown[2]

    df['arrears_adjusted'] = arrears_breakdown[3]

    

    #Iterate through every row in the dataframe

    for index, row in df.iterrows():    

        

        #Define variables at time 0

        if index == 0:

            closing_amount_to_settle = row['closing_amount_to_settle']

            arrears_total, arrears_interest, arrears_settle, arrears_adjusted = arrears_breakdown

            missed_cmi_breakdown = cmi_breakdown

        

        #Set up scenarios to calculate CMI Adjusted

        scenario_dict = {'scenario_1' : 0,

                         'scenario_2' : missed_cmi_breakdown[0],

                         'scenario_3' : missed_cmi_breakdown[1],

                         'scenario_4' : missed_cmi_breakdown[2]} 

        

        scenario_arrears_dict = {'scenario_1' : 0,

                         'scenario_2' : arrears_breakdown[0],

                         'scenario_3' : arrears_breakdown[1],

                         'scenario_4' : arrears_breakdown[2]} 

        

        

        #Functionality to allow full payment of arrears

        if index == pay_arrears_period:

            arrears_breakdown = (0, 0, 0, arrears_adjusted - arrears_total)

            arrears_total, arrears_interest, arrears_settle, arrears_adjusted = arrears_breakdown

            

            #Define opening amount to settle after payment of arrears

            opening_amount_to_settle = closing_amount_to_settle - scenario_arrears_dict[scenario_option]

            

        else:

            #Define opening amount to settle

            opening_amount_to_settle = closing_amount_to_settle

        

        #Run CMI calculation and recalculations

        if row['trigger_flag'] == 1:

            #Define input values for CMI calculation

            rate = row['monthly_rate']

            nper = row['remaining_months']

            cmi_total = cmi_calculator(rate, nper, opening_amount_to_settle)

            

            '''--------------Exercise Part 1 Section begins----------------------------------------'''

            cmi_adjusted = None

            '''--------------Exercise Part 1 Section ends------------------------------------------'''

            

        else:

            cmi_total = cmi_total



        #Define CMI Components

        cmi_interest = opening_amount_to_settle * rate

        cmi_settle = cmi_total - cmi_interest

        cmi_adjusted = cmi_adjusted

        cmi_breakdown = (cmi_total, cmi_interest, cmi_settle, cmi_adjusted)

        

        #Define missed payments and respective CMI Components

        missed_cmi_breakdown = tuple(i * row['missed_flag'] for i in cmi_breakdown)

         

        #Update scenario configuration

        scenario_dict = {'scenario_1' : 0,

                         'scenario_2' : missed_cmi_breakdown[0],

                         'scenario_3' : missed_cmi_breakdown[1],

                         'scenario_4' : missed_cmi_breakdown[2]} 

        

        #Update closing_amount_to_settle

        closing_amount_to_settle = opening_amount_to_settle - cmi_settle + scenario_dict[scenario_option]

        

        #Update Arrears values

        arrears_total = arrears_total + missed_cmi_breakdown[0]

        arrears_interest = arrears_interest + missed_cmi_breakdown[1]

        arrears_settle = arrears_settle + missed_cmi_breakdown[2]

        '''--------------Exercise Part 2 Section begins----------------------------------------'''

        arrears_adjusted = None

        '''--------------Exercise Part 2 Section ends------------------------------------------'''

        arrears_breakdown = (arrears_total, arrears_interest, arrears_settle, arrears_adjusted)

        

        #Update new values to the dataframe

        df.at[index, 'cmi_total'] = cmi_breakdown[0]

        df.at[index, 'cmi_interest'] = cmi_breakdown[1]

        df.at[index, 'cmi_settle'] = cmi_breakdown[2]  

        df.at[index, 'cmi_adjusted'] = cmi_breakdown[3]

        

        df.at[index, 'closing_amount_to_settle'] = round(closing_amount_to_settle,6)

        

        df.at[index, 'arrears_total'] = arrears_breakdown[0]

        df.at[index, 'arrears_interest'] = arrears_breakdown[1]

        df.at[index, 'arrears_settle'] = arrears_breakdown[2]  

        df.at[index, 'arrears_adjusted'] = arrears_breakdown[3]

        

    return df
def redress_calculator(df):

    df['detriment_flag'] = round(df['arrears_adjusted'],2) <= 0.00

    df['DOF'] = df['cmi_adjusted'] * df['detriment_flag']

    df['DOM'] = df['DOF'] * (0.08/12) * df['remaining_months']

    return df
df_test = df_original.copy()



df_test.at[3:, 'monthly_rate'] = 0.01

df_test.at[[0,2], 'trigger_flag'] = 1

df_test.at[[0,1,4], 'missed_flag'] = 1



df_test = balance_profile_recalculator(df_test, 'scenario_2', 3)

display(df_test.head())



df_test = redress_calculator(df_test)

display(df_test.head())

display(df_test.tail())
'''

This function stores the different scenario outputs of the recalculator, 

and performs some summary statistics on results.



The inputs required:

    A DataFrame that meets criteria to run balance_profile_recalculator

    

The outputs:

    A dictionary of DataFrames for scenarios 1 to 4 of the balance_profile_recalculator

    A dictionary of Summary statistics on each scenario that can be converted into DataFrame

'''



def run_scenarios(df, pay_arrears_period=None):

    

    #Define dictionary to store output DataFrames

    df_dict = { 'scenario_1' : None,

                'scenario_2' : None,

                'scenario_3' : None,

                'scenario_4' : None} 

    

    #Define summary dictionary for each of the output DataFrames

    summary = {}

    

    #Iterate through each of the scenario labels

    for key in df_dict.keys():

        df_scenario = df.copy()

        

        #get outputs from recalculator and store on dictionary

        df_scenario = balance_profile_recalculator(df_scenario, key, pay_arrears_period)

        df_scenario = redress_calculator(df_scenario)

        df_dict[key] = df_scenario

        

        

        #produce summary statistics

        cmi_paid = df_scenario[df_scenario['missed_flag'] == 0]

        total_paid = cmi_paid['cmi_total'].sum()

        total_interest = cmi_paid['cmi_interest'].sum()

        closing_to_settle = list(cmi_paid['closing_amount_to_settle'])[-1]

        closing_arrears = list(cmi_paid['arrears_total'])[-1]

        cmi_adjusted = cmi_paid['cmi_adjusted'].sum()

        dof = df_scenario['DOF'].sum()

        dom = df_scenario['DOM'].sum()

        

        #labels dictionary to easily iterate and print summary statistics

        labels_dict = {'Scenario' : key,

                       'Total Paid' : total_paid,

                       'Total Interest' : total_interest,

                       'Closing Amount to Settle' : closing_to_settle,

                       'Closing Arrears' : closing_arrears,

                       'CMI - Adjusted' : cmi_adjusted,

                       'DOF' : dof,

                       'DOM' : dom

                      }

        

        for key in labels_dict.keys():

            value = labels_dict[key]

            

            #print summary statistic

            if isinstance(value, float):

                print(f'{key} : {value:,.2f}')

            else:

                print(f'{key} : {value}')

            

            #store summary statistic in dictionary

            if key in summary.keys():

                summary[key].append(value)

            else:

                summary[key] = [value]



        display(df_scenario.head())

        display(df_scenario.tail())

    return df_dict, summary
#Define df_config DataFrame and change inputs

df_config = df_original.copy()



df_config.at[3:, 'monthly_rate'] = 0.01

df_config.at[[0,2], 'trigger_flag'] = 1

df_config.at[[0,1], 'missed_flag'] = 1



display(df_config.head())

display(df_config.tail())
pay_arrears_period = 3

df_scenarios_dict, scenarios_summary = run_scenarios(df_config, pay_arrears_period)
df_summary = pd.DataFrame(scenarios_summary).set_index('Scenario')

df_summary