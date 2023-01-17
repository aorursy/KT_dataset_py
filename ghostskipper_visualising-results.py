import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from numba import njit



pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 150)
data = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')

sub = pd.read_csv("../input/submission-79913/79913_submission.csv", index_col='family_id', dtype=np.uint16)



# I'll add the submission assigned day to the family dataset

data['assigned_day'] = sub['assigned_day']
# Constants

N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



family_size = data.n_people.values

days_array = np.arange(N_DAYS, 0, -1)

choice_dict = data.loc[:, 'choice_0': 'choice_9'].T.to_dict()



choice_array_num = np.full((data.shape[0], N_DAYS + 1), -1)

for i, choice in enumerate(data.loc[:, 'choice_0': 'choice_9'].values):

    for d, day in enumerate(choice):

        choice_array_num[i, day] = d

        

penalties_array = np.array([

    [

        0,

        50,

        50 + 9 * n,

        100 + 9 * n,

        200 + 9 * n,

        200 + 18 * n,

        300 + 18 * n,

        300 + 36 * n,

        400 + 36 * n,

        500 + 36 * n + 199 * n,

        500 + 36 * n + 398 * n

    ]

    for n in range(family_size.max() + 1)

])
@njit(fastmath=True) # fast math makes it a bit quicker, but less accurate

def cost_function_detailed(prediction, penalties_array, family_size, days):

    penalty = 0



    # We'll use this to count the number of people scheduled each day

    daily_occupancy = np.zeros((len(days)+1))

    N = family_size.shape[0]

    

    # Looping over each family; d is the day, n is size of that family, 

    # and choice is their top choices

    for i in range(N):

        # add the family member count to the daily occupancy

        n = family_size[i]

        d = prediction[i]

        choice = choice_array_num[i]

        

        daily_occupancy[d] += n



        # Calculate the penalty for not getting top preference

        penalty += penalties_array[n, choice[d]]



    choice_cost = penalty

        

    # for each date, check total occupancy

    #  (using soft constraints instead of hard constraints)

    relevant_occupancy = daily_occupancy[1:]

    incorrect_occupancy = np.any(

        (relevant_occupancy > MAX_OCCUPANCY) | 

        (relevant_occupancy < MIN_OCCUPANCY)

    )

    

    if incorrect_occupancy:

        penalty += 100000000



    # Calculate the accounting cost

    # The first day (day 100) is treated special

    init_occupancy = daily_occupancy[days[0]]

    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)

    # using the max function because the soft constraints might allow occupancy to dip below 125

    accounting_cost = max(0, accounting_cost)

    

    # Loop over the rest of the days, keeping track of previous count

    yesterday_count = init_occupancy

    for day in days[1:]:

        today_count = daily_occupancy[day]

        diff = np.abs(today_count - yesterday_count)

        accounting_cost += max(0, (today_count - 125.0) / 400.0 * today_count**(0.5 + diff / 50.0))

        yesterday_count = today_count



    penalty += accounting_cost



    return penalty, choice_cost, accounting_cost
def calculate_df(original_df, pred):

    new_df = original_df.copy()

    new_df['assigned_day'] = pred

    new_df['choice'] = 0

    for c in range(10):

        new_df.loc[new_df[f'choice_{c}'] == new_df['assigned_day'], 'choice'] = c

    

    new_df['choice_cost'] = new_df.apply(lambda x: penalties_array[x['n_people']][x['choice']], axis=1)

    

    for c in range(10):

        new_df[f'n_people_{c}'] = np.where(new_df[f'choice_{c}'] == new_df['assigned_day'], new_df['n_people'], 0)

        

    for c in range(1, 10):

        d = c -1

        new_df[f'n_people_{c}'] = new_df[f'n_people_{d}'] + new_df[f'n_people_{c}']

        

    aggdata = new_df.groupby(by=['assigned_day'])['n_people', 'n_people_0', 'n_people_1', 'n_people_2', 'n_people_3', 'n_people_4', 'n_people_5', 'n_people_6', 'n_people_7', 'n_people_8', 'n_people_9', 'choice_cost'].sum().reset_index()

    

    daily_occupancy = aggdata['n_people'].values

    accounting_cost_daily = np.zeros(len(daily_occupancy))

    for day in range(N_DAYS-1, -1, -1):

        if day == 99:

            n_next = 125.0

        else:

            n_next = daily_occupancy[day + 1]

        n = daily_occupancy[day]

        diff = abs(n - n_next)

        accounting_cost_daily[day] = max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))



    aggdata['accounting_cost'] = accounting_cost_daily

    aggdata['total_cost'] = aggdata['choice_cost'] + aggdata['accounting_cost']

    aggdata['accounting_cost']=aggdata['accounting_cost'].astype(int)

    aggdata['total_cost']=aggdata['total_cost'].astype(int)

   

    new_df = pd.merge(left=new_df, right=aggdata[['assigned_day', 'n_people']].rename(columns={'n_people': 'n_people_per_day'}), on='assigned_day')

    

    return new_df, aggdata
# You cann choose your ranges here

MIN_OCCUPANCY = 0

MAX_OCCUPANCY = 300

MIN_COST = 0

MAX_COST = 2000
def plot_both(data, pred):

    _, adata3 = calculate_df(data.copy(), pred.copy())



    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 18), sharey=True)

    sns.set_color_codes("pastel")



    sns.barplot(x='n_people_9', y='assigned_day', data=adata3, label='choice_9', orient='h', color='m', ax=ax1)

    sns.barplot(x='n_people_8', y='assigned_day', data=adata3, label='choice_8', orient='h', color='grey', ax=ax1)

    sns.barplot(x='n_people_7', y='assigned_day', data=adata3, label='choice_7', orient='h', color='orange', ax=ax1)

    sns.barplot(x='n_people_6', y='assigned_day', data=adata3, label='choice_6', orient='h', color='olive', ax=ax1)

    sns.barplot(x='n_people_5', y='assigned_day', data=adata3, label='choice_5', orient='h', color='k', ax=ax1)

    sns.barplot(x='n_people_4', y='assigned_day', data=adata3, label='choice_4', orient='h', color='r', ax=ax1)

    sns.barplot(x='n_people_3', y='assigned_day', data=adata3, label='choice_3', orient='h', color='y', ax=ax1)

    sns.barplot(x='n_people_2', y='assigned_day', data=adata3, label='choice_2', orient='h', color='g', ax=ax1)

    sns.barplot(x='n_people_1', y='assigned_day', data=adata3, label='choice_1', orient='h', color='c', ax=ax1)

    sns.barplot(x='n_people_0', y='assigned_day', data=adata3, label='choice_0', orient='h', color='b', ax=ax1)

    ax1.axvline(125, color="k", clip_on=False)

    ax1.axvline(300, color="k", clip_on=False)

    ax1.axvline(210, color="k", clip_on=False, linestyle='--')

    ax1.legend(ncol=2, loc="lower right", frameon=True)

    ax1.set(xlabel="Occupancy")

    ax1.set_xlim(MIN_OCCUPANCY, MAX_OCCUPANCY)



    total_cost, choice_cost, acc_cost = cost_function_detailed(pred, penalties_array, family_size, days_array)

    

    sns.set_color_codes("deep")

    sns.barplot(x='total_cost', y='assigned_day', data=adata3, label='total_cost', orient='h', color='k', ax=ax2)

    sns.barplot(x='choice_cost', y='assigned_day', data=adata3, label='choice_cost', orient='h', color='r', ax=ax2)

    sns.barplot(x='accounting_cost', y='assigned_day', data=adata3, label='accounting_cost', orient='h', color='y', ax=ax2)

    ax2.legend(ncol=2, loc="lower right", frameon=True)

    ax2.set(xlabel=f"Costs: {choice_cost:.0f} + {acc_cost:.0f} = {total_cost:.0f}")

    ax2.set_xlim(MIN_COST, MAX_COST)
plot_both(data.copy(), data['assigned_day'].values)