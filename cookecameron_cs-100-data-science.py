import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:10]
train_data.describe(include='all')
women = train_data[train_data['Sex'] == 'female']['Survived']

rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)
men = train_data[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)
# alternative way of computing the above

people = train_data[['Age', 'Survived']].groupby(['Age'])

#for person in people:

    #print(person)

menPclass = train_data[train_data.Sex == 'male']['Pclass']

#type(menPclass)

print(menPclass)
train_data[['Pclass' , 'Survived']].groupby(['Pclass']).mean()



#men

menPclass = train_data[train_data.Sex == 'male']['Pclass']



Pclass1 = []

Pclass2 = []

Pclass3 = []



indexes = pd.Index(menPclass)



i = 0

while i < 891:

    if i in menPclass:

        if menPclass[i] == 1:

            Pclass1.append(men[i])

        elif menPclass[i] == 2:

            Pclass2.append(men[i])

        else:

            Pclass3.append(men[i])

    

    i += 1



print(sum(Pclass1)/len(Pclass1))

print(sum(Pclass2)/len(Pclass2))

print(sum(Pclass3)/len(Pclass3))



#women

womenPclass = train_data[train_data.Sex == 'female']['Pclass']



wPclass1 = []

wPclass2 = []

wPclass3 = []



indexes = pd.Index(womenPclass)



i = 0

while i < 891:

    if i in womenPclass:

        if womenPclass[i] == 1:

            wPclass1.append(women[i])

        elif womenPclass[i] == 2:

            wPclass2.append(women[i])

        else:

            wPclass3.append(women[i])

    

    i += 1



print(sum(wPclass1)/len(wPclass1))

print(sum(wPclass2)/len(wPclass2))

print(sum(wPclass3)/len(wPclass3))
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data.corr()
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        women_count += 1

        if row['Survived'] == 1:

            women_survived_count += 1

women_survived_count / women_count
predictions = []

for idx, row in test_data.iterrows():

    # make your changes in this cell!

    prob = 0

    if row['Sex'] == 'female':

        prob = 0.7420382165605095

        if row['Pclass'] == 1:

            prob = 0.9680851063829787

        elif row['Pclass'] == 2:

            prob = 0.9210526315789473

        else:

            prob = 0.5

    else:

        prob = 0.18890814558058924

        if row['Pclass'] == 1:

            prob = 0.36885245901639346

        elif row['Pclass'] == 2:

            prob = 0.1574074074074074

        else:

            prob = 0.13544668587896252

            

    if prob >= 0.5:

        predictions.append(1)

    else:

        predictions.append(0)

    #if row['Age'] < 10:

    #    predictions.append(1)

    #elif row['Sex'] == 'female':

    #    predictions.append(1)

    #else:

    #    predictions.append(0)
assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)