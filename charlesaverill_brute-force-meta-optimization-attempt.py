import pandas as pd

from random import randint



data = pd.read_csv("../input/santa-workshop-tour-2019/family_data.csv")



family_size_dict = data[['n_people']].to_dict()['n_people']



cols = [f'choice_{i}' for i in range(10)]

choice_dict = data[cols].to_dict()



N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



days = list(range(N_DAYS,0,-1))



def cost_function(prediction):



    penalty = 0



    # We'll use this to count the number of people scheduled each day

    daily_occupancy = {k:0 for k in days}



    # Looping over each family; d is the day for each family f

    for f, d in enumerate(prediction):



        # Using our lookup dictionaries to make simpler variable names

        n = family_size_dict[f]

        choice_0 = choice_dict['choice_0'][f]

        choice_1 = choice_dict['choice_1'][f]

        choice_2 = choice_dict['choice_2'][f]

        choice_3 = choice_dict['choice_3'][f]

        choice_4 = choice_dict['choice_4'][f]

        choice_5 = choice_dict['choice_5'][f]

        choice_6 = choice_dict['choice_6'][f]

        choice_7 = choice_dict['choice_7'][f]

        choice_8 = choice_dict['choice_8'][f]

        choice_9 = choice_dict['choice_9'][f]



        # add the family member count to the daily occupancy

        daily_occupancy[d] += n



        # Calculate the penalty for not getting top preference

        if d == choice_0:

            penalty += 0

        elif d == choice_1:

            penalty += 50

        elif d == choice_2:

            penalty += 50 + 9 * n

        elif d == choice_3:

            penalty += 100 + 9 * n

        elif d == choice_4:

            penalty += 200 + 9 * n

        elif d == choice_5:

            penalty += 200 + 18 * n

        elif d == choice_6:

            penalty += 300 + 18 * n

        elif d == choice_7:

            penalty += 300 + 36 * n

        elif d == choice_8:

            penalty += 400 + 36 * n

        elif d == choice_9:

            penalty += 500 + 36 * n + 199 * n

        else:

            penalty += 500 + 36 * n + 398 * n



    # for each date, check total occupancy

    #  (using soft constraints instead of hard constraints)

    for _, v in daily_occupancy.items():

        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):

            penalty += 100000000



    # Calculate the accounting cost

    # The first day (day 100) is treated special

    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)

    # using the max function because the soft constraints might allow occupancy to dip below 125

    accounting_cost = max(0, accounting_cost)



    # Loop over the rest of the days, keeping track of previous count

    yesterday_count = daily_occupancy[days[0]]

    for day in days[1:]:

        today_count = daily_occupancy[day]

        diff = abs(today_count - yesterday_count)

        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))

        yesterday_count = today_count



    penalty += accounting_cost



    return penalty


def swap(df, i1, i2):

    values = df.values



    temp = values[i1].copy()

    values[i1] = values[i2]

    values[i2] = temp



    df['assigned_day'] = values



    return df



def changeval(df, index, val):

    values = df.values



    values[index] = val



    df['assigned_day'] = values



    return df
csv = pd.read_csv("../input/submission-file/submission_76169.41944832797.csv", index_col=0)

init_cost = cost_function(csv['assigned_day'].tolist())

current_cost = init_cost

lowest_cost = 1000000000



print(cost_function(changeval(csv, 0, 3)['assigned_day'].tolist()))



offset = 0



count = 0
print("Step 0")

print("---------")

"""

while(True):

    for index in range(offset, 4999):

        if(not index == 0):

            print("Step", str(index))

            print("Lowest Cost:", lowest_cost)

            print("Initial Cost:", init_cost)

            print("---------")

        for val in range(1, 100):

            swap = csv.copy()

            swap = changeval(swap, index, val)



            current_cost = cost_function(swap['assigned_day'].values)



            if(current_cost < lowest_cost and not current_cost == init_cost):

                lowest_cost = current_cost



            if(lowest_cost < init_cost):

                csv = temp.copy()



                print("********")

                print("Low score!")

                print("Current cost is:", str(lowest_cost))

                print("I1:", i1)

                print("I2:", i2)

                print("********")



                csv.to_csv(f'submission_{lowest_cost}.csv')



                init_cost = lowest_cost

"""