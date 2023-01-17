import numpy as np
data = open("../input/speech/Speech.txt", "r")
reading= [x.strip() for x in data.readlines()]
text = ''

for i in reading:

    text = text + i

print(text)
def generateTable(data,k=4):

    

    T = {}

    for i in range(len(data)-k):

        X = data[i:i+k]

        Y = data[i+k]

        #print("X  %s and Y %s  "%(X,Y))

        

        if T.get(X) is None:

            T[X] = {}

            T[X][Y] = 1

        else:

            if T[X].get(Y) is None:

                T[X][Y] = 1

            else:

                T[X][Y] += 1

    

    return T
def convertFreqIntoProb(T):     

    for kx in T.keys():

        s = float(sum(T[kx].values()))

        for k in T[kx].keys():

            T[kx][k] = T[kx][k]/s

                

    return T
def trainMarkovChain(text,k=4):

    

    T = generateTable(text,k)

    T = convertFreqIntoProb(T)

    

    return T
model = trainMarkovChain(text)
def sample_next(ctx,T,k):

    ctx = ctx[-k:]

    if T.get(ctx) is None:

        return " "

    possible_Chars = list(T[ctx].keys())

    possible_values = list(T[ctx].values())

    

    #print(possible_Chars)

    #print(possible_values)

    

    return np.random.choice(possible_Chars,p=possible_values)
def generateText(starting_sent,k=4,maxLen=1000):

    

    sentence = starting_sent

    ctx = starting_sent[-k:]

    

    for ix in range(maxLen):

        next_prediction = sample_next(ctx,model,k)

        sentence += next_prediction

        ctx = sentence[-k:]

    return sentence
text = generateText("dear",k=4,maxLen=2000)

print(text)