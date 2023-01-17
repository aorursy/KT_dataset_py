import pandas as pd
people = pd.read_csv("../input/people.csv", header = 0)

activity = pd.read_csv("../input/act_train.csv", header = 0)

train = pd.merge(left = people, 

                 right = activity,

                 how = 'inner',

                 on = 'people_id')
print("People fields: " + str(people.columns))

print("Activity fields: " + str(activity.columns))

print("Training fields: " + str(train.columns))
