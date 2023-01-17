import random

import operator
def BinaryToDecimal(string):

    length = len(string)

    ret = 0

    power = 0

    for i in range(length - 1, 0, -1):

        if string[i] == '1':

            ret += 2**power

        power += 1

    

    if string[0] == '1':

        ret = -ret

    return ret
def func(x):

    return x * x * ( - 1)  + 5
class Population:

    def __init__(self, chromosome, fit_value, decimal_value):

        self.chromosome = chromosome

        self.fit_value = fit_value

        self.decimal_value = decimal_value

    def fitness_calculate(self):

        self.decimal_value = BinaryToDecimal(self.chromosome)

        self.fit_value = func(self.decimal_value)
random.seed()

populations = []

for i in range(4):

    string = ''

    for j in range(6):

        digit = random.randint(0,1)

        string += str(digit)

    val = BinaryToDecimal(string)

    temp = Population(string, func(val), val)

    populations.append(temp)



populations = sorted(populations, key=operator.attrgetter('fit_value'), reverse = True)

def change(string, idx, val):

    list1 = list(string)

    list1[idx] = val

    string = ''.join(list1)

    return string
iteration = 1000

for it in range(iteration):

    off1 = populations[0].chromosome

    off2 = populations[1].chromosome

    idx = random.randint(0, 5)

    for i in range(idx, 6, 1):

        val1 = off1[i]

        val2 = off2[i]

        off1 = change(off1, i, val2)

        off2 = change(off2, i, val1)

    

    populations[2].chromosome = off1

    populations[3].chromosome = off2

    

    mutation_probablity = random.randint(1,50)

    if mutation_probablity == 20:

        m_idx = random.randint(0, 5)

        m_pop = random.randint(0, 3)

        c = int(populations[m_pop].chromosome[m_idx])

        c = 1 - c;

        populations[m_pop].chromosome = change(populations[m_pop].chromosome, m_idx, str(c))

    

    populations = sorted(populations, key=operator.attrgetter('fit_value'), reverse = True)

    for i in range(4):

        populations[i].fitness_calculate()

        print(populations[i].chromosome, populations[i].fit_value, populations[i].decimal_value)

    print("After iteration ", it + 1, " : The best value is ", populations[0].fit_value," and the chromosome is ", populations[0].chromosome)

    







