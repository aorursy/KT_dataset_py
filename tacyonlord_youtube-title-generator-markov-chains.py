def Read_file(FileName):

    with open(FileName,'r') as file:

        for line in file: #The delimiter for a file is \n by default

            for word in line.replace('\n','').split(' '): #Splits according to space and replaces '\n' by nothing

                yield word #Returns a sequnce/ series of words
Words = Read_file('../input/Title_Data.txt')
from collections import defaultdict

def assemble_chain(Words):

    chain = defaultdict(list) #create a dictionary which maps word to list

    try:

        word,next_word = next(Words),next(Words) #from iterables

        while True:

            chain[word].append(next_word)

            word,next_word = next_word, next(Words)

    except StopIteration:  #Error which arises when no words left

        return chain
Chain = assemble_chain(Words)

#print(Chain)
#Create the random word generator

from random import choice



def random_word(sample): #Could be any list of words cos Im using it only for beginning

    return choice(list(sample)) #Converting the dictionary to a list and then passing to choice

    

#Create a chain based on the first state



def random_title(Sample):

    word = random_word(['I','My'])#random_word(Sample)

    i = 0.0

    while True:

        yield word

        if word in Sample:

            word = random_word(Sample[word]) #Cos we only need the words which have a chance of coming after the given word

            

        else:

            i = i+1

            word = random_word(Sample)

        if(word[-1] =='.' or word[-1]=='!' or word[-1]=='?'):

            yield word

            #print(i)

            break
Title = random_title(Chain)

T_l = list(Title)

print(*T_l, sep=' ')