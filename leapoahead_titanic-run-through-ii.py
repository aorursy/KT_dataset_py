import csv

import numpy as np



csv_data = csv.reader(open('../input/train.csv', 'r'))

header = next(csv_data) # The first row is headers



data = [row for row in csv_data]

data = np.array(data)



print('Dataset size: {0}\nHeaders:'.format(len(data)))

print(header)
def c(col_name):

    for idx, name in enumerate(header):

        if name == col_name:

            return idx

    raise Error('Column name not found')
number_passengers = np.size(data[0::, c('PassengerId')].astype(np.float))

number_survived = np.sum(data[0::, c('PassengerId')].astype(np.float))



proportion_survived = number_survived / number_passengers

print('So {0:.2f}% of passengers survived.'.format(proportion_survived))
women_only_stats = data[0::, c('Sex')] == 'female'

men_only_stats = data[0::, c('Sex')] == 'male'



women_onboard = data[women_only_stats, 1].astype(np.float)

men_onboard = data[men_only_stats, 1].astype(np.float)



proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)

proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)



print('{0:.2f}% of women survived while {1:.2f}% of men survived.'.format(

    proportion_women_survived,

    proportion_men_survived,

))
test_file = open('../input/test.csv', 'r')

test_csv_data = csv.reader(test_file)

header = next(test_csv_data)



prediction_file = open('genderbasedmodel.csv', 'w')

prediction_csv_data = csv.writer(prediction_file)



prediction_csv_data.writerow(['PassengerId', 'Survived'])

for row in test_csv_data:

    if row[c('Sex')] == 'female':

        prediction_csv_data.writerow([row[c('PassengerId')], '1'])

    else:

        prediction_csv_data.writerow([row[c('PassengerId')], '0'])



test_file.close()

prediction_file.close()