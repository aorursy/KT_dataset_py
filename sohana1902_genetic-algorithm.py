# -*- coding: utf-8 -*-



import random

chromosomes=["010101","011101","010011","001101"]



def fitnessFunction(x):

    return -x*x+5



def binToSignedInt(bin):

	base=1

	integer=0

	for i in range(len(bin)-1,0,-1):

		integer=integer+base*int(bin[i])

		base=base*2

	if(bin[0]=='1'):

            print("Bin: ",bin," Dec: ",-integer)

            return -integer

	else: 

            print("Bin: ",bin," Dec: ",integer)            

            return integer



def crossover():

    pair=[]

    print(chromosomes)

    for i in range(len(chromosomes)):

        pair.append((fitnessFunction(binToSignedInt(chromosomes[i])),chromosomes[i]))

    pair=sorted(pair,reverse=True)

    print(pair)

    chromosomes.clear()

    maxm=-1000000000000000

    for t1,t2 in pair:

        chromosomes.append(t2)

        if maxm<int(t1):

            maxm=int(t1)

    print("maxm so far: ",maxm)            

    chromosomes.pop()

    chromosomes.pop()

    crossoverStartsAt=random.randint(0,5)

    chromosomes.append(chromosomes[0][:crossoverStartsAt]+chromosomes[1][crossoverStartsAt:])

    chromosomes.append(chromosomes[1][:crossoverStartsAt]+chromosomes[0][crossoverStartsAt:])



def mutation(seed):

    if(random.randint(0,50)==seed):

        print(chromosomes)

        selectChromosome=random.randint(0,3)

        selectMutationPoint=random.randint(0,5)

        toBeMutated=list(chromosomes.pop(selectChromosome))

        toBeMutated[selectMutationPoint]=str(1-int(toBeMutated[selectMutationPoint]))

        string=""

        for i in toBeMutated:

            string+=i

        chromosomes.append(string)



loop=10000

for itr in range(loop):

    print('Iteration ',itr+1)

    crossover()

    mutation(31)