# Things to use

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Some constants

data_dir = "../input/santa-workshop-tour-2019/"

family_file = "family_data.csv"



NDAYS = 100

NFAMS = 5000

MAX_PPL = 300

MIN_PPL = 125



# The family preference cost parameters

PENALTY_CONST = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]

PENALTY_PPL = [0, 0, 9, 9, 9, 18, 18, 36, 36, 199+36, 398+36]



# The seed is set once here at beginning of notebook.

RANDOM_SEED = 127

np.random.seed(RANDOM_SEED)
# Define the family_cost function

def family_cost(ichoice, nppl):

    global PENALTY_CONST, PENALTY_PPL

    return PENALTY_CONST[ichoice] + nppl*PENALTY_PPL[ichoice]
# Show the cost-per-person in matrix form

# Note that higher choice values can give lower per-person cost.

# Also created a dictionary mapping the choice-nppl tuple to cost_pp.

cost_pp_dict = {}

print("    Cost per Person")

print("\n       nppl= 2       3       4       5       6       7       8\nichoice")

# choices are 0 to 10

for ichoice in range(11):

    # numbers of people in a family are 2 to 8:

    choice_str = str(ichoice).rjust(5)+":"

    for nppl in range(2,9):

        cost_pp = family_cost(ichoice, nppl)/nppl

        cost_pp_dict[(ichoice,nppl)] = cost_pp

        choice_str = choice_str + str(int(cost_pp)).rjust(8)

    print(choice_str)
# Can use the cost_pp_dict to go through the ichoice, nppl combinations,

# in order from least to greatest cost-per-person, if that's useful.

# (Didn't use this, put values in by hand below.)

if False:

    sorted_cost_pp = sorted(cost_pp_dict.items(), key = 

             lambda kv:(kv[1], kv[0]))

    for ich_nppl in sorted_cost_pp:

        ichoice = ich_nppl[0][0]

        nppl = ich_nppl[0][1]

        print(ichoice,nppl)
# Define the accounting cost function

def accounting_cost(people_count):

    # people_count[iday] is an array of the number of people each day,

    # valid for iday=1 to NDAYS (iday=0 not used).

    day_costs = np.zeros(NDAYS+1)

    total_cost = 0.0

    ppl_yester = people_count[NDAYS]

    for iday in range(NDAYS,0,-1):

        ppl_today = people_count[iday]

        ppl_delta = np.abs(ppl_today - ppl_yester)

        day_cost = (ppl_today - 125)*(ppl_today**(0.5+ppl_delta/50.0))/400.0

        day_costs[iday] = day_cost

        total_cost += day_cost

        ##print("Day {}: delta = {}, $ {}".format(iday, ppl_delta, int(day_cost)))

        # save for tomorrow

        ppl_yester = people_count[iday]

    print("Total accounting cost: {:.2f}.  Ave costs:  {:.2f}/day,  {:.2f}/family".format(

        total_cost,total_cost/NDAYS,total_cost/NFAMS))

    return total_cost, day_costs



# Test it with several test cases

# Constant 210, cost should be 3.08/day

##people_count = 210 + np.zeros(101)

# Alternate 210 +/-25 = 185, 235. Should be ~ 674/day

##people_count = (210 + 25*np.cos(np.pi*np.arange(0,102))).astype(int)

# Alternate 210 +/-50 = 160, 260. Should be ~ 194,000/day

##people_count = (210 + 50*np.cos(np.pi*np.arange(0,102))).astype(int)

#

##accounting = accounting_cost(people_count)
# Define the Gift Shop Metric and Profit Factor

gsp_profit_factor = 25.0

def gsp_metric(diff_value):

    return 126.0/(diff_value**3 + 125.0)
# Read in the data

df_family = pd.read_csv(data_dir+family_file)

# The "choice_" column headings use a lot of room, change to "ch_"

the_columns = df_family.columns.values

for ich in range(10):

    the_columns[ich+1] = "ch_"+str(ich)

df_family.columns = the_columns
df_family.head(5)
# Total number of people

total_people  = df_family['n_people'].sum()

# and average per day:

ave_ppl_day = int(total_people / NDAYS)

print("Total number of people visiting is {}, about {} per day".format(total_people, ave_ppl_day))
# Add an assigned day column, inititalize it to -1 (not assigned)

df_family['assigned_day'] = -1

# 

df_family.tail(5)
# As the results of v1-v3 showed, there are certain days that are less subscribed than others.



# Show the distribution of each choice's days

##for ichoice in range(10):

df_family[['ch_0','ch_1','ch_2','ch_3',

           'ch_4','ch_5','ch_6','ch_7','ch_8','ch_9']].hist(bins=100,figsize=(12,8),

                                sharex=True,sharey=True)

plt.show()



# Yes, all choices have similar behavior and 6 low regions above day 60 will be hardest to fill.
# Create a buddy_id column to record any buddy family pairs,

# also save the day that the buddies share.

df_family['buddy_id'] = -1

df_family['buddy_day'] = -1



# Go through the days, look for buddies pairs.

# Criteria for being a buddy (besides on the same day)

#   family_ids within:

max_buddy_diff = 3     # using 3,  or 0 for no buddies

#   and the same choice value.



for iday in range(1,NDAYS+1):

    # Select families to consider buddying-up

    select_fams = df_family[(df_family['buddy_day'] < 0) & 

                (  ((df_family['ch_1'] == iday) & (df_family['n_people'] >= 2)) |

                ((df_family['ch_2'] == iday) & (df_family['n_people'] >= 3)) |

             ((df_family['ch_3'] == iday) & (df_family['n_people'] >= 6))  ) ]

    

    num_fams = len(select_fams)

    ##print("Day {}: {} families".format(iday,num_fams))

    # Look for ones that are close in family_id:

    fam_ids = select_fams['family_id'].values

    # the differences between ids:

    diff_ids = -1 + 0*fam_ids

    diff_ids[1:] = fam_ids[1:] - fam_ids[0:-1]

    diff_ids[0] = fam_ids[0] + NFAMS - fam_ids[-1]

    ##print(fam_ids, "\n", diff_ids)

    # Find ones that have the desired difference:

    for idiff, diff in enumerate(diff_ids):

        if diff <= max_buddy_diff:

            ##print(diff, fam_ids[idiff], fam_ids[idiff]-diff)

            # Assign these as each others buddy

            # watch for wrap-around of the lower buddy id

            buddy_low = fam_ids[idiff]-diff

            if buddy_low < 0:

                buddy_low += NFAMS

            df_family.loc[fam_ids[idiff],'buddy_id'] = buddy_low

            df_family.loc[buddy_low,'buddy_id'] = fam_ids[idiff]

            # and the day

            df_family.loc[fam_ids[idiff],'buddy_day'] = iday

            df_family.loc[buddy_low,'buddy_day'] = iday
print("Total familes that are buddied-up: {}".format(sum(df_family['buddy_id'] >= 0)))
# (v1) Random scheduling

if False:

    sched_method = "Random"

    # Select a random day, 1 to 100, for each family

    df_family['assigned_day'] = np.random.choice(1+np.arange(NDAYS), size=NFAMS, replace=True)

    
# (v7) As v6 but assign (any) buddy familes together.

if True:

    sched_method = 'MsClausBuddies'

    

    # Reset the assignements and the people_count_m1 array:

    df_family['assigned_day'] = -1

    people_count_m1 = np.zeros(NDAYS)

    

    # Output columns when printing buddy information

    buddy_cols = ['family_id','ch_0','ch_1','ch_2','n_people','assigned_day','buddy_id','buddy_day']
if True:    

    print("\n\nFilling the above-55 low-request ('mid-week') days ...\n")

    # First, assign the lower-requested days.

    # The low-people days are every 4 out of 7.

    # The 6 low regions above day 60 are:

    lower_days = [55,56,57,58, 62,63,64,65, 69,70,71,72, 76,77,78,79, 83,84,85,86, 90,91,92,93, 97,98,99,100]

    

    max_ppl_day = 126

    

    # Set the desired cost-per-person limit by specifying:

    # i) specific choice to use, and ii) a minimum number of people (inclusive)

    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4] #,5,4,5,6, 5,4,3,6,5,7,4,6,7] #,8,7,4,8,5,6,7,8,6,7,8,8,9,9]

    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6] #,7,5,6,8, 5,4,0,6,4,8,3,5,7] #,7,5,0,6,0,3,3,4,0,0,3,0,7,0]

    # for printing out info

    for icost in range(len(ichs)):

        ich = ichs[icost]

        ich_str = 'ch_'+str(ich)

        nppl_min = nppl_mins[icost]

        

        print(" - - - - - - - Doing ",ich_str,"  nppl >=",nppl_min," - - - - - - -")

        # Go though the families and assign ones that meet the criteria

        for ifam in df_family.index:

            day_ich = df_family.loc[ifam,ich_str]

            nppl = df_family.loc[ifam,'n_people']

            if ((df_family.loc[ifam,'assigned_day'] < 0) and

                    (day_ich in lower_days) and (nppl >= nppl_min) and

                    (people_count_m1[day_ich-1] < max_ppl_day)):

                ##print(ifam,day_ich,nppl,sum(people_count_m1))

                # OK, got one. Assign it:

                df_family.loc[ifam,'assigned_day'] = day_ich

                # and keep track of the people count:

                people_count_m1[day_ich-1]  += df_family.loc[ifam,'n_people']

                #

                # - Does this family have a Buddy family? and

                # - Is this day the buddy day? and

                # - Is the buddy family not already assigned?

                # That's a lot of ands ;-)

                if ((df_family.loc[ifam,'buddy_id'] >= 0) and

                        (df_family.loc[ifam,'buddy_day'] == day_ich)):

                    # assign the buddy family here if not already assigned

                    buddy_fam = df_family.loc[ifam,'buddy_id']

                    if (df_family.loc[buddy_fam,'assigned_day'] < 0):

                        ##print("\nAssigning Buddy!")                    

                        df_family.loc[buddy_fam,'assigned_day'] = day_ich

                        people_count_m1[day_ich-1]  += df_family.loc[buddy_fam,'n_people']

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                    else:

                        pass

                        ##print("\nBuddy, but not assigned.")

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                        

        print("Total assigned families = ",sum(df_family['assigned_day'] > 0),

             " and people =",sum(people_count_m1),"\n")
if True:

    print("\n\nFilling the above-55 TRANSITION low-request days ...\n")

    # The "transition" days are the beginning and end of the 4-day low-demand regions

    lower_days = [55,58, 62,65, 69,72, 76,79, 83,86, 90,93, 97,100]

    # Fill these to a higher level:

    

    max_ppl_day = 126 + 32

    

    # Set the desired cost-per-person limit by specifying:

    # i) specific choice to use, and ii) a minimum number of people (inclusive)

    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4] #,5,4,5,6, 5,4,3,6,5,7,4,6,7] #,8,7,4,8,5,6,7,8,6,7,8,8,9,9]

    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6] #,7,5,6,8, 5,4,0,6,4,8,3,5,7] #,7,5,0,6,0,3,3,4,0,0,3,0,7,0]

    for icost in range(len(ichs)):

        ich = ichs[icost]

        ich_str = 'ch_'+str(ich)

        nppl_min = nppl_mins[icost]

        

        print(" - - - - - - - Doing ",ich_str,"  nppl >=",nppl_min," - - - - - - -")

        # Go though the families and assign ones that meet the criteria

        for ifam in df_family.index:

            day_ich = df_family.loc[ifam,ich_str]

            nppl = df_family.loc[ifam,'n_people']

            if ((df_family.loc[ifam,'assigned_day'] < 0) and

                    (day_ich in lower_days) and (nppl >= nppl_min) and

                    (people_count_m1[day_ich-1] < max_ppl_day)):

                ##print(ifam,day_ich,nppl,sum(people_count_m1))

                # OK, got one. Assign it:

                df_family.loc[ifam,'assigned_day'] = day_ich

                # and keep track of the people count:

                people_count_m1[day_ich-1]  += df_family.loc[ifam,'n_people']

                #

                # - Does this family have a Buddy family? and

                # - Is this day the buddy day? and

                # - Is the buddy family not already assigned?

                # That's a lot of ands ;-)

                if ((df_family.loc[ifam,'buddy_id'] >= 0) and

                        (df_family.loc[ifam,'buddy_day'] == day_ich)):

                    # assign the buddy family here if not already assigned

                    buddy_fam = df_family.loc[ifam,'buddy_id']

                    if (df_family.loc[buddy_fam,'assigned_day'] < 0):

                        ##print("\nAssigning Buddy!")                    

                        df_family.loc[buddy_fam,'assigned_day'] = day_ich

                        people_count_m1[day_ich-1]  += df_family.loc[buddy_fam,'n_people']

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                    else:

                        pass

                        ##print("\nBuddy, but not assigned.")

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                

        print("Total assigned families = ",sum(df_family['assigned_day'] > 0),

             " and people =",sum(people_count_m1),"\n")
if True:    

    print("\n\nFilling the below-59 low-request ('mid-week') days ...\n")

    # First, assign the lower-requested days.

    # The low-people days are every 4 out of 7.

    # The below-51 low-request days:

    lower_days = [2, 6,7,8,9, 13,14,15,16, 20,21,22,23, 27,28,29,30, 

                               34,35,36,37, 41,42,43,44, 48,49,50,51, 55,56,57,58]

    # will fill these to some minimum:

    

    max_ppl_day = 210 - 15

    

    # Set the desired cost-per-person limit by specifying:

    # i) specific choice to use, and ii) a minimum number of people (inclusive)

    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4] #,5,4,5,6, 5,4,3,6,5,7,4,6,7] #,8,7,4,8,5,6,7,8,6,7,8,8,9,9]

    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6] #,7,5,6,8, 5,4,0,6,4,8,3,5,7] #,7,5,0,6,0,3,3,4,0,0,3,0,7,0]

    for icost in range(len(ichs)):

        ich = ichs[icost]

        ich_str = 'ch_'+str(ich)

        nppl_min = nppl_mins[icost]

        

        print(" - - - - - - - Doing ",ich_str,"  nppl >=",nppl_min," - - - - - - -")

        # Go though the families and assign ones that meet the criteria

        for ifam in df_family.index:

            day_ich = df_family.loc[ifam,ich_str]

            nppl = df_family.loc[ifam,'n_people']

            if ((df_family.loc[ifam,'assigned_day'] < 0) and

                    (day_ich in lower_days) and (nppl >= nppl_min) and

                    (people_count_m1[day_ich-1] < max_ppl_day)):

                ##print(ifam,day_ich,nppl,sum(people_count_m1))

                # OK, got one. Assign it:

                df_family.loc[ifam,'assigned_day'] = day_ich

                # and keep track of the people count:

                people_count_m1[day_ich-1]  += df_family.loc[ifam,'n_people']

                #

                # - Does this family have a Buddy family? and

                # - Is this day the buddy day? and

                # - Is the buddy family not already assigned?

                # That's a lot of ands ;-)

                if ((df_family.loc[ifam,'buddy_id'] >= 0) and

                        (df_family.loc[ifam,'buddy_day'] == day_ich)):

                    # assign the buddy family here if not already assigned

                    buddy_fam = df_family.loc[ifam,'buddy_id']

                    if (df_family.loc[buddy_fam,'assigned_day'] < 0):

                        ##print("\nAssigning Buddy!")                    

                        df_family.loc[buddy_fam,'assigned_day'] = day_ich

                        people_count_m1[day_ich-1]  += df_family.loc[buddy_fam,'n_people']

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                    else:

                        pass

                        ##print("\nBuddy, but not assigned.")

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])    

            

        print("Total assigned families = ",sum(df_family['assigned_day'] > 0),

             " and people =",sum(people_count_m1),"\n")
if True:

    print("\n\nFilling the below-51 TRANSITION low-request days ...\n")

    # The "transition" days are the beginning and end of the 4-day low-demand regions

    lower_days = [2, 6,9, 13,16, 20,23, 27,30, 34,37, 41,44, 48,51, 55,58]   #, 55,58]

    # Fill these to a higher level than the middle 2 days:

    

    max_ppl_day = 210 + 20

    # Special case for day 2 since day 1 and 3 are always large

    max_ppl_day2 = 210 + 35

    

    # Set the desired cost-per-person limit by specifying:

    # i) specific choice to use, and ii) a minimum number of people (inclusive)

    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4] #,5,4,5,6, 5,4,3,6,5,7,4,6,7] #,8,7,4,8,5,6,7,8,6,7,8,8,9,9]

    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6] #,7,5,6,8, 5,4,0,6,4,8,3,5,7] #,7,5,0,6,0,3,3,4,0,0,3,0,7,0]

    for icost in range(len(ichs)):

        ich = ichs[icost]

        ich_str = 'ch_'+str(ich)

        nppl_min = nppl_mins[icost]

        

        print(" - - - - - - - Doing ",ich_str,"  nppl >=",nppl_min," - - - - - - -")

        # Go though the families and assign ones that meet the criteria

        for ifam in df_family.index:

            day_ich = df_family.loc[ifam,ich_str]

            nppl = df_family.loc[ifam,'n_people']

            if day_ich == 2:

                ppl_limit = max_ppl_day2

            else:

                ppl_limit = max_ppl_day

            if ((df_family.loc[ifam,'assigned_day'] < 0) and

                    (day_ich in lower_days) and (nppl >= nppl_min) and

                    (people_count_m1[day_ich-1] < ppl_limit)):

                ##print(ifam,day_ich,nppl,sum(people_count_m1))

                # OK, got one. Assign it:

                df_family.loc[ifam,'assigned_day'] = day_ich

                # and keep track of the people count:

                people_count_m1[day_ich-1]  += df_family.loc[ifam,'n_people']

                #

                # - Does this family have a Buddy family? and

                # - Is this day the buddy day? and

                # - Is the buddy family not already assigned?

                # That's a lot of ands ;-)

                if ((df_family.loc[ifam,'buddy_id'] >= 0) and

                        (df_family.loc[ifam,'buddy_day'] == day_ich)):

                    # assign the buddy family here if not already assigned

                    buddy_fam = df_family.loc[ifam,'buddy_id']

                    if (df_family.loc[buddy_fam,'assigned_day'] < 0):

                        ##print("\nAssigning Buddy!")                    

                        df_family.loc[buddy_fam,'assigned_day'] = day_ich

                        people_count_m1[day_ich-1]  += df_family.loc[buddy_fam,'n_people']

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                    else:

                        pass

                        ##print("\nBuddy, but not assigned.")

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                

        print("Total assigned families = ",sum(df_family['assigned_day'] > 0),

             " and people =",sum(people_count_m1),"\n")
if True:    

    print("\n\nFilling the High-Demand days ...\n")

    # Now fill in the "high-demand" days to different amount above/beow day 59:

    

    max_ppl_day = 215

    max_ppl_above = 165

    

    # don't add to the low-demand days (but adding to 2 is OK):

    lower_days = [6,7,8,9, 13,14,15,16, 20,21,22,23, 27,28,29,30, 

                               34,35,36,37, 41,42,43,44, 48,49,50,51]

    lower_days += [55,56,57,58, 62,63,64,65, 69,70,71,72, 76,77,78,79, 83,84,85,86, 90,91,92,93, 97,98,99,100]

    # Set the desired cost-per-person limit by specifying:

    # i) specific choice to use, and ii) a minimum number of people (inclusive)

    # These look like enough to get 125 in each of the low

    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4] #,5,4,5,6, 5,4,3,6,5,7,4,6,7] #,8,7,4,8,5,6,7,8,6,7,8,8,9,9]

    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6] #,7,5,6,8, 5,4,0,6,4,8,3,5,7] #,7,5,0,6,0,3,3,4,0,0,3,0,7,0]

    for icost in range(len(ichs)):

        ich = ichs[icost]

        ich_str = 'ch_'+str(ich)

        nppl_min = nppl_mins[icost]

        

        print(" - - - - - - - Doing ",ich_str,"  nppl >=",nppl_min," - - - - - - -")

        # Go though the families and assign ones that meet the criteria

        for ifam in df_family.index:

            day_ich = df_family.loc[ifam,ich_str]

            nppl = df_family.loc[ifam,'n_people']

            if day_ich < 59:

                ppl_limit = max_ppl_day

            else:

                ppl_limit = max_ppl_above

            if ((df_family.loc[ifam,'assigned_day'] < 0) and

                    not(day_ich in lower_days) and (nppl >= nppl_min) and

                    (people_count_m1[day_ich-1] < ppl_limit)):

                ##print(ifam,day_ich,nppl,sum(people_count_m1))

                # OK, got one. Assign it:

                df_family.loc[ifam,'assigned_day'] = day_ich

                # and keep track of the people count:

                people_count_m1[day_ich-1]  += df_family.loc[ifam,'n_people']

                #

                # - Does this family have a Buddy family? and

                # - Is this day the buddy day? and

                # - Is the buddy family not already assigned?

                # That's a lot of ands ;-)

                if ((df_family.loc[ifam,'buddy_id'] >= 0) and

                        (df_family.loc[ifam,'buddy_day'] == day_ich)):

                    # assign the buddy family here if not already assigned

                    buddy_fam = df_family.loc[ifam,'buddy_id']

                    if (df_family.loc[buddy_fam,'assigned_day'] < 0):

                        ##print("\nAssigning Buddy!")                    

                        df_family.loc[buddy_fam,'assigned_day'] = day_ich

                        people_count_m1[day_ich-1]  += df_family.loc[buddy_fam,'n_people']

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                    else:

                        pass

                        ##print("\nBuddy, but not assigned.")

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])



        print("Total assigned families = ",sum(df_family['assigned_day'] > 0),

             " and people =",sum(people_count_m1),"\n")

if True:

    # Finally, the remaining families can go anywhere their choice allows,

    # this adds to both low and high-demand days, with different limits above/below day 59.

    print("\n\nPut these last few anywhere ...\n")

    

    max_ppl_day = 270

    max_ppl_above = 200

    

    # Set the desired cost-per-person limit by specifying:

    # i) specific choice to use, and ii) a minimum number of people (inclusive)

    # These look like enough to get 125 in each of the low

    ichs =      [0,1,2,1,2,3,2,3,3,1,2,4,3,4] #,5,4,5,6, 5,4,3,6,5,7,4,6,7] #,8,7,4,8,5,6,7,8,6,7,8,8,9,9]

    nppl_mins = [0,4,7,3,4,7,3,6,4,0,0,7,3,6] #,7,5,6,8, 5,4,0,6,4,8,3,5,7] #,7,5,0,6,0,3,3,4,0,0,3,0,7,0]

    for icost in range(len(ichs)):

        ich = ichs[icost]

        ich_str = 'ch_'+str(ich)

        nppl_min = nppl_mins[icost]

        

        print(" - - - - - - - Doing ",ich_str,"  nppl >=",nppl_min," - - - - - - -")

        # Go though the families and assign ones that meet the criteria

        for ifam in df_family.index:

            day_ich = df_family.loc[ifam,ich_str]

            nppl = df_family.loc[ifam,'n_people']

            if day_ich < 59:

                ppl_limit = max_ppl_day

            else:

                ppl_limit = max_ppl_above

            if ((df_family.loc[ifam,'assigned_day'] < 0) and

                    (nppl >= nppl_min) and

                    (people_count_m1[day_ich-1] < ppl_limit)):

                ##print(ifam,day_ich,nppl,sum(people_count_m1))

                # OK, got one. Assign it:

                df_family.loc[ifam,'assigned_day'] = day_ich

                # and keep track of the people count:

                people_count_m1[day_ich-1]  += df_family.loc[ifam,'n_people']

                #

                # - Does this family have a Buddy family? and

                # - Is this day the buddy day? and

                # - Is the buddy family not already assigned?

                # That's a lot of ands ;-)

                if ((df_family.loc[ifam,'buddy_id'] >= 0) and

                        (df_family.loc[ifam,'buddy_day'] == day_ich)):

                    # assign the buddy family here if not already assigned

                    buddy_fam = df_family.loc[ifam,'buddy_id']

                    if (df_family.loc[buddy_fam,'assigned_day'] < 0):

                        ##print("\nAssigning Buddy!")                    

                        df_family.loc[buddy_fam,'assigned_day'] = day_ich

                        people_count_m1[day_ich-1]  += df_family.loc[buddy_fam,'n_people']

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                    else:

                        pass

                        ##print("\nBuddy, but not assigned.")

                        ##print(df_family.loc[[ifam, buddy_fam],buddy_cols])

                

        num_assigned = sum(df_family['assigned_day'] > 0)

        print("Total assigned families = ",num_assigned,

             " and people =",sum(people_count_m1),"\n")

        # If all NFAMS are assigned then we're done:

        if num_assigned >= NFAMS:

            break

# And really finally, if there are some families left, just put them at their 2nd choices.

if True:

    # Did we assign them all?

    # If not, put the (few?) remaining ones at their 2nd choices, ignoring people count

    if (num_assigned < NFAMS):

        print("Putting the remaining {} families at their 2nd choice.".format(NFAMS-num_assigned))

        not_assigned = (df_family['assigned_day'] < 0)

        df_family.loc[not_assigned, 'assigned_day'] = df_family.loc[not_assigned, 'ch_2']
# Show the dataframe

df_family.head(10)
# Check for any not-assigned families

if df_family['assigned_day'].min() < 0:

    print("Ooops!  Some families did not get days assigned!")

    print("Number assigned = {}".format(sum(df_family['assigned_day'] > 0)))

    # show the not-assigned families:

    print(df_family[df_family['assigned_day'] < 0].head(20))

    halt_on_this_routine()

else:

    print("\nAll Families have an assigned day.")
# Show the distribution of Families assigned per day

df_family['assigned_day'].plot.hist(bins=100,figsize=(10,5))

plt.title("Histogram of Number of Families each day")

plt.show()
# Create the people count for each day from the family assignements

people_count = np.zeros(NDAYS+1)

for ifam in df_family.index:

    their_day = df_family.loc[ifam,'assigned_day']

    # add their number to this days people:

    if their_day > 0:

        people_count[their_day] += df_family.loc[ifam,'n_people']

    

ppl_max_count = max(people_count)

ppl_min_count = min(people_count[1:])



print("\nPeople_count has min = {}, and a max = {}".format(ppl_min_count, ppl_max_count))
# Calculate the Accounting Cost for this people count

account_cost, day_costs = accounting_cost(people_count)



# Check if the people limits are exceeded

if (ppl_max_count >= MAX_PPL) or (ppl_min_count <= MIN_PPL):

    print("\n       "+10*"*"+" W A R N I N G - People limits exceeded "+15*"*")

    ppl_color = 'red'

else:

    ppl_color = 'blue'



plt.figure(figsize=(10,5))

plt.plot(1+np.arange(1,NDAYS+1), people_count[1:],marker='o',color=ppl_color)

# show the average and max min levels too

plt.plot([1,NDAYS],[MIN_PPL,MIN_PPL],color='gray')

plt.plot([1,NDAYS],[ave_ppl_day,ave_ppl_day],color='gray')

plt.plot([1,NDAYS],[MAX_PPL,MAX_PPL],color='gray')

plt.title("Number of People each day")

plt.xlabel("Day")

plt.ylabel("Number of People")

plt.show()



# Show the Accounting Cost contribution for each day:

plt.figure(figsize=(10,5))

plt.plot(np.arange(1,NDAYS+1), day_costs[1:] ,marker='o',color=ppl_color)

# show the average and max min levels too

plt.plot([1,NDAYS],[MIN_PPL,MIN_PPL],color='gray')

plt.plot([1,NDAYS],[ave_ppl_day,ave_ppl_day],color='gray')

plt.plot([1,NDAYS],[MAX_PPL,MAX_PPL],color='gray')

plt.title("Accounting Cost for each day")

plt.xlabel("Day")

plt.ylabel("Accounting Cost")

plt.show()
# Determine the "ichoice" value, 0-10, for each family

# (do this more efficiently with an apply() ?)

df_family['ichoice'] = -1   # not defined

for ifam in df_family.index:

    # default: didn't get any of the 0-9 specified

    ichoice = 10

    # see if the assigned day is any of the preferences

    assigned_day = df_family.loc[ifam,'assigned_day']

    for ipref in range(10):

        if (df_family.loc[ifam,'ch_'+str(ipref)] == assigned_day):

            ichoice = ipref

            break;

    df_family.loc[ifam,'ichoice'] = ichoice

            
# Show the distribution of the ichoice values

df_family['ichoice'].plot.hist(bins=21,figsize=(10,5))

plt.title("Histogram of the Number of Families vs their assigned choice")

plt.show()
# Now get the cost for each family

# (do this more efficiently with an apply() ?)

df_family['cost'] = -1   # not calculated

for ifam in df_family.index:

    df_family.loc[ifam,'cost'] = family_cost(

        df_family.loc[ifam,'ichoice'], df_family.loc[ifam,'n_people'])
df_family.head(10)
# Scatter plot of the cost per family vs assigned_day

##hist_vs = 'ichoice'

hist_vs = 'assigned_day'



df_family.plot.scatter(hist_vs,'cost',figsize=(10,5),alpha=0.3)

plt.title("Scatter plot of the  Cost-per-family vs their "+hist_vs)

plt.show()
# Scatter plot of the Cost per Person vs assigned_day

hist_vs = 'assigned_day'



# add the cost-per-person

df_family['cost_pp'] = df_family['cost']/df_family['n_people']

df_family.plot.scatter(hist_vs,'cost_pp',figsize=(10,5),alpha=0.3)

plt.title("Scatter plot of the  Cost-per-Person vs their "+hist_vs)

plt.show()
# Show the histogram of family costs

# This is a "y-axis projection" of the previous scatter plot.

fig, ax = plt.subplots(figsize=(10,5))

df_family['cost'].hist(ax=ax, bins=100, bottom=1.0)

ax.set_yscale('log')

plt.title("Histogram of the Number of Families vs their Cost")

plt.show()
if True:

    # Ghostskipper has a data dataframe, so

    data = df_family

    

    # From Ghostskipper:

    

    import seaborn as sns

    

    # Now lets make columns that contain the number of people for each choice

    for c in range(10):

        data[f'n_people_{c}'] = np.where(data[f'ch_{c}'] == data['assigned_day'], data['n_people'], 0)

    # dd: I have an ichoice column, so add the choice = 10 ones too:

    data['n_people_10'] = np.where(data['ichoice'] == 10, data['n_people'], 0)

    

    # This is a trick to make a stacked bar plot in seaborn

    for c in range(1, 11):   # dd: to 11

        d = c -1

        data[f'n_people_{c}'] = data[f'n_people_{d}'] + data[f'n_people_{c}']

    

    # We need to aggregate the data into number of people per choice per day.

    # Luckily pandas can do this for us using groupby.

    agg_data = data.groupby(by=['assigned_day'])['n_people', 'n_people_0', 'n_people_1', 'n_people_2', 

                        'n_people_3', 'n_people_4', 'n_people_5', 'n_people_6', 'n_people_7', 

                        'n_people_8', 'n_people_9', 'n_people_10'].sum().reset_index()

    

    # Now for the plot!

    f, ax = plt.subplots(figsize=(12, 20))

    sns.set_color_codes("pastel")

    sns.barplot(x='n_people_10', y='assigned_day', data=agg_data, label='choice_10', orient='h', color='red')

    sns.barplot(x='n_people_9', y='assigned_day', data=agg_data, label='choice_9', orient='h', color='k')

    sns.barplot(x='n_people_8', y='assigned_day', data=agg_data, label='choice_8', orient='h', color='k')

    sns.barplot(x='n_people_7', y='assigned_day', data=agg_data, label='choice_7', orient='h', color='k')

    sns.barplot(x='n_people_6', y='assigned_day', data=agg_data, label='choice_6', orient='h', color='k')

    sns.barplot(x='n_people_5', y='assigned_day', data=agg_data, label='choice_5', orient='h', color='k')

    sns.barplot(x='n_people_4', y='assigned_day', data=agg_data, label='choice_4', orient='h', color='r')

    sns.barplot(x='n_people_3', y='assigned_day', data=agg_data, label='choice_3', orient='h', color='y')

    sns.barplot(x='n_people_2', y='assigned_day', data=agg_data, label='choice_2', orient='h', color='g')

    sns.barplot(x='n_people_1', y='assigned_day', data=agg_data, label='choice_1', orient='h', color='c')

    sns.barplot(x='n_people_0', y='assigned_day', data=agg_data, label='choice_0', orient='h', color='b')

    ax.axvline(125, color="k", clip_on=False)

    ax.axvline(300, color="k", clip_on=False)

    ax.legend(ncol=2, loc="lower right", frameon=True)

    ax.set(xlabel="Occupancy")

    
# The GSP metric function and the gsp_profit_factor were defined at the beginning of this file.

if True:

    # Plot the metric value vs family_id gap size

    gap_vals = np.arange(1,11)

    plt.plot(gap_vals, gsp_metric(gap_vals), marker='o')

    plt.ylim(0,)

    plt.show()
# Try this metric as the basis for a Gift Shop Profit

# Assumes dataframe df_family has columns of: assigned_day, family_id

# (see stand-alone version below)

if True:

    # Will evaluate/save a cluster metric for each day

    cluster_metrics = []

    # Show detailed output for some select days

    show_days = [1]

    for iday in range(1,NDAYS+1):

        # Get the family_ids for guests on this day - the ids will be in order

        fam_vals = df_family[df_family['assigned_day'] == iday]['family_id'].values

        num_fams_today = len(fam_vals)

        #

        if iday in show_days:

            print("Day "+str(iday)+":\n",fam_vals)

        

        # calculate the adjacent differences between family ids, i.e., size of the gaps

        diff_vals = np.zeros(len(fam_vals))

        diff_vals[1:] = fam_vals[1:] - fam_vals[0:-1]

        # and calculate the difference for the first one by wrapping it around and subtracting the last

        diff_vals[0] = fam_vals[0]+NFAMS - fam_vals[-1]

        #

        if iday in show_days:

            plt.plot(diff_vals,marker='.')

            plt.plot([0.0,num_fams_today-1],[0.0,0.0],color='gray')

            plt.title("Day "+str(iday)+" - Difference values for each Family")

            plt.show()

            

        # Use something like 1/gap_length as the main variable to measure the clustering.

        # Use 3rd power plus an offset to limit profit to close family_ids

        cluster_metric = gsp_metric(diff_vals)

        #

        if iday in show_days:

            plt.plot(cluster_metric,marker=".")

            plt.plot([0.0,num_fams_today-1],[0.0,0.0],color='gray')

            plt.title("Day "+str(iday)+" - cluster metric values for each Family"

                 +"  Total = {:.2f}".format(sum(cluster_metric)))

            plt.show()

        # save this day's metric

        cluster_metrics.append(sum(cluster_metric))        

if True: 

    # Define the Gift Shop profits, each day, as

    gift_profits = gsp_profit_factor * np.array(cluster_metrics)

    total_profits = sum(gift_profits)



    print("\nTotal Gift Shop Profit: $ {:.2f}\n".format(total_profits))



    # Look at the histogram of cluster_metric for the days

    plt.hist(gift_profits,bins=50)

    plt.title("Histogram of the daily Gift Shop Profits")

    plt.xlabel("Day's Profits (\$)")

    plt.ylabel("Number of Days")

    plt.show()

    

    # Plot the cluster_metrics values for the days

    plt.plot(1.0+np.arange(NDAYS),gift_profits,marker=".")

    plt.ylim(0,)

    plt.title("Gift Shop Profits vs Day   Total = {:.2f}".format(sum(gift_profits)))

    plt.show()
# Show the two contributions and the Total Cost:

print("   Families cost: $ {:>10}".format(int(df_family['cost'].sum())))

print(" Accounting cost: $ {:>10}".format(int(account_cost)))

print("      TOTAL cost: $ {:>10}".format(int(account_cost + df_family['cost'].sum())))



print("\n   less:")

print("Gift Shop Profit: $ {:>10}".format(int(sum(gift_profits))))

print("   gives:")

print("        NET cost: $ {:>10}".format(int(account_cost + df_family['cost'].sum() -

                                              sum(gift_profits))))



# Note if the people limits are exceeded

if (ppl_max_count >= MAX_PPL) or (ppl_min_count <= MIN_PPL):

    print("\n       "+10*"*"+" W A R N I N G - People limits exceeded "+15*"*")
# Notes on version 7



# Evaluating GSP using:  "25, 5^3" , i.e.:

#   def gsp_metric(diff_value):

#       return 126.0/(diff_value**3 + 125.0)

# with Profit Factor:

#      gsp_profit_factor = 25.0





# Results with no Buddies:  (Same as v6 except GSP is changed.)



#   Families cost: $      88418

# Accounting cost: $       8725

#      TOTAL cost: $      97143

#

#   less:

#Gift Shop Profit: $       7734

#   gives:

#        NET cost: $      89409





# Using some Buddies, With max_buddy_diff = 3

# and finding buddies by doing:

#    # Select families to consider buddying-up

#    select_fams = df_family[(df_family['buddy_day'] < 0) & 

#                (  ((df_family['ch_1'] == iday) & (df_family['n_people'] >= 2)) |

#                ((df_family['ch_2'] == iday) & (df_family['n_people'] >= 3)) |

#             ((df_family['ch_3'] == iday) & (df_family['n_people'] >= 6))  ) ]



# Results with Buddies:



#   Families cost: $      88852   +434

# Accounting cost: $       8082   -643

#      TOTAL cost: $      96934        -209

#

#   less:

#Gift Shop Profit: $       8058   +324

#   gives:

#        NET cost: $      88875



# Hmmm... A very slight effect on the gift shop profit

# which does not quite balance the increase in the Families cost. 

# The decrease in Accounting cost (why?) saves the day.



# It's possible that since I'm going through the families in family order,

# that I've already built-in a reasonable amount of buddying up.

# Will 'shelve' this for now ;-)

# Write out the submission file:

df_family[['family_id','assigned_day']].to_csv("submission.csv", index=False)
# Look at the head, tail of the file:



!head -10 submission.csv
!tail -10 submission.csv
# Gift Shop Profit calculated from the submission file

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# The submission file that was just created above:

sub_name = "submission.csv"

df_submit = pd.read_csv("submission.csv")



# Read some other submission files grabbed from notebooks:

##sub_name = "hengzheng_santa-s-seed-seeker_72150"

##sub_name = "vipito_santa-ip_72398"

##sub_name = "jazivxt_using-a-baseline_76101"

##sub_name = "xhlulu_santa-s-2019-stochastic-product-search_76114"

##sub_name = "ilu000_greedy-dual-and-tripple-shuffle-with-fast-scoring_78350"

##sub_name = "ghostskipper_visualising-results_79913"

##sub_name = "pulkitmehtawork1985_fast-jonker-volgenant-algorithm_84433"

##sub_name = "dan3dewey_santa-s-simple-scheduler_96934"

##sub_name = "pavelvod_pytorch-starter-solution_117334"

##sub_name = "zzy990106_better-initialization_127711"

##sub_name = "kaushal2896_random-assignment-benchmark_10552936"

#

##df_submit = pd.read_csv("../input/"+sub_name+".csv")



NDAYS = 100

NFAMS = 5000



# Define the metric as a function of the difference in family id values

# Use something like 1/gap_length as the main variable to measure the clustering.

# Use 3rd power plus an offset to limit profit to near-by family_ids

def gsp_metric(diff_value):

    return 126.0/(diff_value**3 + 125.0)



# And the Profit Factor:

gsp_profit_factor = 25.0



if True:

    # Plot the metric value vs family_id gap size

    gap_vals = np.arange(1,11)

    plt.plot(gap_vals, gsp_metric(gap_vals), marker='o')

    plt.ylim(0,)

    plt.show()

    

# Will evaluate/save a cluster metric for each day

cluster_metrics = []

# Show detailed output for some select days

show_days = [1]

for iday in range(1,NDAYS+1):

    # Get the family_ids for guests on this day - the ids will be in order

    fam_vals = df_submit[df_submit['assigned_day'] == iday]['family_id'].values

    num_fams_today = len(fam_vals)

    if iday in show_days:

        print("Day "+str(iday)+"  Family IDs:\n",fam_vals)

        

    # calculate the adjacent differences between family ids, i.e., size of the gaps

    diff_vals = np.zeros(len(fam_vals))

    diff_vals[1:] = fam_vals[1:] - fam_vals[0:-1]

    # and calculate the difference for the first one by wrapping it around and subtracting the last

    diff_vals[0] = fam_vals[0]+NFAMS - fam_vals[-1]

    if iday in show_days:

        plt.plot(diff_vals,marker='.')

        plt.plot([0.0,num_fams_today-1],[0.0,0.0],color='gray')

        plt.title("Day "+str(iday)+" - Differences of successive Family IDs")

        plt.show()

            

    # The metric is defined with a simple function (above)

    cluster_metric = gsp_metric(diff_vals)

    if iday in show_days:

        plt.plot(cluster_metric,marker=".")

        plt.plot([0.0,num_fams_today-1],[0.0,0.0],color='gray')

        plt.title("Day "+str(iday)+" - cluster metric values for each Family"

                 +"  Total = {:.2f}".format(sum(cluster_metric)))

        plt.show()

    # save this day's metric

    cluster_metrics.append(sum(cluster_metric))   

        

# Define the Gift Shop profits, each day, as

gift_profits = gsp_profit_factor * np.array(cluster_metrics)

total_profits = sum(gift_profits)



# Look at the histogram of cluster_metric for the days

plt.hist(gift_profits,bins=50)

plt.title("Histogram of the daily Gift Shop Profits")

plt.xlabel("Day's Profits (\$)")

plt.ylabel("Number of Days")

plt.show()

    

# Plot the cluster_metrics values for the days

plt.plot(1.0+np.arange(NDAYS),gift_profits,marker=".")

plt.ylim(0,)

plt.title("Gift Shop Profits vs Day   Total = {:.2f}".format(sum(gift_profits)))

plt.show()



print("\nTotal Gift Shop Profit: $ {:.2f} for {}\n".format(total_profits,sub_name))

# Total Gift Shop Profit for some public submissions:  (v7)

#

#  $ 7617.87 for hengzheng_santa-s-seed-seeker_72150

#  $ 7571.1 for vipito_santa-ip_72398

#  $ 7603.67 for jazivxt_using-a-baseline_76101

#  $ 7613.56 for xhlulu_santa-s-2019-stochastic-product-search_76114

#  $ 7467.86 for ilu000_greedy-dual-and-tripple-shuffle-with-fast-scoring_78350

#  $ 7510.47 for ghostskipper_visualising-results_79913

#  $ 7512.49 for pulkitmehtawork1985_fast-jonker-volgenant-algorithm_84433

#  $ 8058.83 for dan3dewey_santa-s-simple-scheduler_96934 (v7)

#  $ 6813.30 for pavelvod_pytorch-starter-solution_117334

#  $ 7363.77 for zzy990106_better-initialization_127711

#  $ 6497.01 for kaushal2896_random-assignment-benchmark_10552936
