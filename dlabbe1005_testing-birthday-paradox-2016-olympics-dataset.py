# import packages

import pandas as pd

import warnings

warnings.filterwarnings("ignore")
# import and convert the birthday to date

athletes = pd.read_csv('../input/athletes.csv')

athletes['birthday'] = pd.to_datetime(athletes['dob'], format='%m/%d/%y', errors='ignore')

athletes = athletes[~athletes['birthday'].isnull()]
# set the number of trials

num_trials = 100



# define the sample sizes that will be tested and its associated probability

sample_size = [[23,50.7], [30,70.6], [40,89.1], [50,97.0], [60,99.4]]



flag_print = False

count = []
# function to sample the data in the parameterized size

def sample(sample_size):

    return athletes.sample(sample_size).reset_index()



# compare if two birthdays are the same

def isSameMonthDay(date1, date2):

    if str(date1.month) + '-' + str(date1.day) == str(date2.month) + '-' + str(date2.day):

        return True

    else:

        return False



# print the pair of matches

def printPair(pair):

    text = ' - {}({}) was born in {}, and {}({}) born in {}.'

    print(text.format(athletes[athletes.id == pair[0]].name.values[0],

                      athletes[athletes.id == pair[0]].nationality.values[0],

                      athletes[athletes.id == pair[0]].dob.values[0],

                      athletes[athletes.id == pair[1]].name.values[0],

                      athletes[athletes.id == pair[1]].nationality.values[0],

                      athletes[athletes.id == pair[1]].dob.values[0]))



# print the all pairs of matches

def printMatch(sample_size, match_list):

    text = 'With sample size of {}, there are {} pairs of athletes who were born at the same day and month. The probability of having at least 1 pair is {}%:'

    print (text.format(sample_size[0], len(match_list), sample_size[1]))

    

    for i in range(len(match_list)):

        printPair(match_list[i])



# run on round of the trials

def oneRound():

    match_by_sample = []

    count_by_round = []

    for s in sample_size:

        df = sample(s[0])

        match = []

        for i in range(len(df)):

            for j in range(len(df)):

                if i != j:

                    if isSameMonthDay (df.ix[i].birthday, df.ix[j].birthday):

                        match.append([df.ix[i].id, df.ix[j].id])

        match_by_sample.append(match)

        count_by_round.append(len(match))

    if flag_print:

        for i in range(len(sample_size)):

            printMatch(sample_size[i], match_by_sample[i])

    count.append(count_by_round)        



# print the summary of the trials

def printResults():

    text = 'The probability of at least two birthdays same day and month with the sample size of {} is {}%. The converging probability of this trial is {}%'

    for j in range(len(sample_size)):

        zeros = 0

        for i in range(len(count)):

            if count[i][j] == 0:

                zeros = zeros +1

        print (text.format(sample_size[j][0],sample_size[j][1],(1 - zeros/num_trials) * 100.0))
# run the trials

for i in range(num_trials):

    oneRound()
# print the summary

printResults()
# enable the printing details and run one round

flag_print = True

oneRound()

flag_print = False