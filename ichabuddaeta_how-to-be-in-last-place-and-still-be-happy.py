import pandas as pd

import random as rnd
data = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')
data
d = rnd.sample(range(0,5000),5000)

day_dict = {}

fam_dict = {}

df = data

for i in range(100,0,-1):

    day_dict[i] = 0
def get_choice_for_family(df, family_id, day_dict,fam_dict):

    c=1

    row = df[df['family_id'] == family_id]

    while c < 11:

        if day_dict[row.iloc[0,c]] >289:

            c += 1

        else:

            day_dict[row.iloc[0,c]] += int(row.n_people)

            break

    fam_dict[family_id] = [row.iloc[0,c],c-1]

    return day_dict, fam_dict
for i in d:

    x = get_choice_for_family(df, i, day_dict, fam_dict)

    day_dict = x[0]

    fam_dict = x[1]
min(day_dict.values())
len(fam_dict)
fam_id = list(fam_dict.keys())
assigned_day = list(fam_dict.values())
assigned_days = [i[0] for i in assigned_day]
final = {'family_id':fam_id,'assigned_day':assigned_days}
sub = pd.DataFrame.from_dict(final)
sub = sub.sort_values(by='family_id')
sub.to_csv('sub.csv', index=False)