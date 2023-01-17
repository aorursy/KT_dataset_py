import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data
train_data.describe(include='all')
women = train_data[train_data['Sex'] == 'female']['Survived']

rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)
men = train_data[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)
# alternative way of computing the above

train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
# alternative way of computing the above

train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean()


train_data[['Ticket', 'Survived']].groupby(['Ticket']).mean()
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data.corr()


891*(1-1502/2224)
#computes the probability that someone with this characteristic survived

# p_lived = {"Sex":      {"female": 0.742038,

#                         "male": 0.188908},

#            "Pclass":   {1: 0.629630,

#                         2: 0.472826,

#                         3: 0.242363},

#            "SibSp":    {0: 0.345395,

#                         1: 0.535885,

#                         2: 0.464286,

#                         3: 0.250000,

#                         4: 0.166667,

#                         5: 0.000000,

#                         6: 0.000000,

#                         8: 0.000000},   

#            "Age":      {0: 0.612903,

#                         10: .4100000,

#                         20: 0.360975,

#                         30: 0.443661,

#                         40: 0.368421,

#                         50: 0.394736,

#                         60: 0.266666,

#                         70: 0.000000,

#                         80: 0.000000},

#       }



# new_lst = []



# for idx, row in test_data.iterrows():

#     lst_row = row.tolist()

    

#     try:

#         sex = p_lived["Sex"][lst_row[4]]

#         pclass = p_lived["Pclass"][lst_row[2]]

#         sibsp = p_lived["SibSp"][lst_row[6]]

#         age = p_lived["Age"][int(lst_row[2]/10)*10]

#     except:

#         sex = 0

#         pclass = 0

#         sibsp = 0

#         age = 0

    

#     probability = sex + pclass + sibsp + age

#     lst_row.append(probability)

#     new_lst.append(lst_row)



# sorted_lst = sorted(new_lst, key=lambda l:l[-1], reverse=True)



# for idx, row in test_data.iterrows():

#     if idx < 289:

#         sorted_lst[1] = 1

#     else:

#         sorted_lst[1] = 0

        

# sorted_lst = sorted(new_lst, key=lambda l:l[0])



# predictions = [sorted_lst[i][1] for i in range(len(sorted_lst))]

predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    if row['Sex'] == 'female':

        if(row['Pclass'] == 1):

            predictions.append(1)

        elif(row['Pclass'] == 2):

            predictions.append(1)

        elif(row['Pclass'] == 3):

            if(row['Age'] <= 1):

                predictions.append(0)

            else:

                predictions.append(1)

        else:

            predictions.append(0)

    else:

        if(row['Pclass'] == 1):

            if(row['Age'] <= 18):

                predictions.append(1)

            else:

                predictions.append(0)

        elif(row['Pclass'] == 2):

            if(row['Age'] <= 16):

                predictions.append(1)

            else:

                predictions.append(0)

        else:

            predictions.append(0)





# for idx, row in train_data.iterrows():

#     if row['Sex'] == 'female' and row['Age'] > 5:

#         predictions.append(1)

#     else:

#         predictions.append(0)



# total = 0

# correct = 0

# for idx, row in train_data.iterrows():

#     total += 1

#     if predictions[idx] == row['Survived']:

#         correct += 1

# score = correct/total

# print(count, correct, score*100)
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)