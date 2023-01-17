import pandas as pd
data = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')
def choice_cost(choice, n_people):

    if choice == 0:

        return 0

    elif choice == 1:

        return 50

    elif choice == 2:

        return 50 + (9*n_people)

    elif choice == 3:

        return 100 + (9*n_people)

    elif choice == 4:

        return 200 + (9*n_people)

    elif choice == 5:

        return 200 + (18*n_people)

    elif choice == 6:

        return 300 + (18*n_people)

    elif choice == 7:

        return 300 + (36*n_people)

    elif choice == 8:

        return 400 + (36*n_people)

    elif choice == 9:

        return 500 + ((36+199)*n_people)

    else:

        return 500 + ((36+398)*n_people)
data.n_people.value_counts() / data.n_people.sum()
new_data = data[data.family_id == 1]

new_data
fams = list(data.family_id)
type(new_data)
days = [i for i in range(100,0,-1)]
fam = {'family_id':fams,'assigned_day':days*50}
fam_df = pd.DataFrame.from_dict(fam)
def check_family_cost(family_id = 1, assigned_day = 1, data = data):

    new_data = data[data.family_id == family_id]

    n_people = new_data.n_people

    choice = -1

    choices = [int(new_data['choice_0']), int(new_data['choice_1']),

                   int(new_data['choice_2']), int(new_data['choice_3']),

                   int(new_data['choice_4']), int(new_data['choice_5']),

                   int(new_data['choice_6']), int(new_data['choice_7']),

                   int(new_data['choice_8']), int(new_data['choice_9'])]

    for i in choices:

        if assigned_day == i:

            choice = i

            break

        else:

            choice = 10

    return int(choice_cost(choice, n_people))

    
check_family_cost(1,1)
x = 0

for index, row in fam_df.iterrows():

    x += check_family_cost(row[0], row[1])

print(x)
data = data.sort_values(by=['choice_6','choice_5','choice_4','choice_3','choice_2','choice_1','choice_0'])

data
fam_df_n = pd.merge(data.iloc[:,[0,11]],fam_df, on='family_id', how='inner')
fam_df_n.groupby('assigned_day')['n_people'].sum()
def cost_improvement(family_id, assigned_day, fam_data = data, assign_data = fam_df_n):

    x_data = fam_data[fam_data.family_id == family_id]

    choices = [int(x_data['choice_0']), int(x_data['choice_1']),

               int(x_data['choice_2']), int(x_data['choice_3']),

               int(x_data['choice_4']), int(x_data['choice_5']),

               int(x_data['choice_6']), int(x_data['choice_7']),

               int(x_data['choice_8']), int(x_data['choice_9'])]

    n_people = int(x_data.n_people)

    assignments = assign_data.groupby('assigned_day')['n_people'].sum()

    best_choice = assigned_day

    for choice in choices:

        if check_family_cost(family_id, choice) < check_family_cost(family_id, assigned_day) and (assignments[assigned_day] >= 125 + n_people  and assignments[choice] <= 300 - n_people):

            best_choice = choice

    assign_data.assigned_day[assign_data.family_id == family_id] = best_choice

    return assign_data
new_assign = fam_df_n.sort_values(by=['n_people'], ascending=False)

for index, row in new_assign.iterrows():

    new_assign = cost_improvement(row.family_id, row.assigned_day, assign_data= new_assign)
x = 0

for index, row in new_assign.iterrows():

    x += check_family_cost(row[0], row[1])

print(x)
days = new_assign.groupby('assigned_day')['n_people'].sum()
days[days < 125]
sub = new_assign[['family_id', 'assigned_day']]
sub.to_csv('sub.csv',index=False)