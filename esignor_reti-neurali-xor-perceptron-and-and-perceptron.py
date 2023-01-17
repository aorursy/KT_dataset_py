training_set =  [([0,0,0,0], 0), 

                ([1,0,1,0], 0),

                ([1,1,1,1], 1),

                ([0,0,0,0], 0), 

                ([1,1,0,0], 0),

                ([1,0,0,1], 0), 

                ([1,1,1,0], 0),

                ([0,0,0,1], 0),

                ([0,0,1,1], 0),

                ([1,0,1,1], 0),

                ([0,1,0,0], 0),

                ([1,1,1,1], 1)]
import numpy as np



weight= np.random.rand(4)

learning_rate = 0.2

epochs = 250
# Elaborazione del risultato finale

def final_output(result):

    if result < 0:

        return 0

    else:

        return 1
# Estrazione di una riga random dal training set per fare apprendimento

def choose_train():

    n = np.random.randint(0, len(training_set) - 1)

    x = training_set[n][0]

    expected = training_set[n][1]

    return x,expected
# Esegue l'apprendimento del Perceptron

def train(weight):

    print("tasso di apprendimento: " + str(learning_rate))

    print("numero di iterazioni sullo stesso esempio: " + str(epochs))

    for i in range(0,epochs): # ripeto l'apprendimento epochs volte

        x, expected = choose_train() # restituisco un elemento casuale del training set, memorizzando in x l'array di input e in expected il risultato dell'AND logico

        wx = np.dot(weight, x) # fa il prodotto scalare tra weight e x

        error = expected - final_output(wx) # calcolo l'errore tra il valore aspettato e quello risultante normalizzato

        for j in range (0,len(weight)):

            weight[j]+= learning_rate * error * x[j] # aggiorno il vettore dei pesi

    return weight

        

def prediction(weight, validation_set):

    return final_output(np.dot(weight, validation_set))
from sklearn.metrics import accuracy_score # per calcolare l'accuratezza delle previsioni



weight = train(weight = weight)



# predizione sui dati di validazione

validation_set = [[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],[0,1,1,1],[1,1,1,1],[0,1,1,0],[1,0,0,1],[0,0,1,0],[1,0,0,0],[1,0,1,0],[0,0,1,1]]

expected_set = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

result_set = [ ] # deve contenere tutti output risultanti dal validation_set

# stampa output del set di validazione

for i in range(0,len(validation_set)):

    pred = prediction(weight, validation_set[i])

    print((str(validation_set[i]) + ": -> " + str(pred)))

    result_set.append(pred)

# calcolo dell'accuratezza della previsione

print("La previsione ha un'accuratezza di: ", round(accuracy_score(expected_set,result_set)* 100), "%")

import matplotlib.pyplot as plt # plot



# disegno plot



plt.rcParams['figure.figsize'] = [10, 5] # ridimensiono l'area di stampa del plot



plt.xlabel('sample validation')

plt.ylabel('output')

plt.plot(range(0,12), expected_set, 'r--', color='orange')

plt.plot(range(0,12), result_set, '.', color='purple')

plt.legend(['output atteso', 'output predetto'], numpoints=1)

plt.show()
list_learning_rate = [0.4, 0.6, 1.0]

list_epochs = [50, 150, 250]

for i in range(0,3):

    learning_rate = list_learning_rate[i]

    for j in range (0,3):

        result_set = []

        epochs = list_epochs[j]

        weight= np.random.rand(4) # pesi nuovamente random

        weight = train(weight = weight)

    

        # stampa output del set di validazione

        for i in range(0,len(validation_set)):

            pred = prediction(weight, validation_set[i])

            result_set.append(pred)

        # calcolo dell'accuratezza della previsione

        print("La previsione ha un'accuratezza di: ", round(accuracy_score(expected_set,result_set)* 100), "%")

        print("---------------------------------------------------------------")   
# classe che contiene i singoli neuroni. Uno per ogni funzione logica

class perceptron:

    def __init__(self, x, w, b):

        self.x = x

        self.w = w

        self.b = b



    def calculate_output(self):

        result = np.dot(self.w, self.x) + self.b

        if result < 0:

            return 0

        else:

            return 1

        

    # perceptron OR 

    def percept_OR(self):

    # w = [1,1,1,1], b = - 0.5

        self.w = [1,1]

        self.b = - 0.5

        return self.calculate_output()



     # perceptron AND

    def percept_AND(self):

    # w = [1,1,1,1], b = len(w) + 0.5

        self.w = [1,1]

        self.b = - len(self.w) + 0.5

        return self.calculate_output()

    

     # perceptron NOT

    def percept_NOT(self):

    # w = -1, b = 0.5

        self.w = [-1]

        self.b = 0.5

        return self.calculate_output()



    # concatenazione di perceptron per implementare lo XOR

    def XOR(self):

        out_and = self.percept_AND()

        out_or = self.percept_OR()

        self.x = [out_and]

        out_not = self.percept_NOT()

        self.x = [out_not, out_or]

        return self.percept_AND()

    

    

    

test = [[0,0,0,1],[1,1,1,1],[1,1,0,1],[1,0,1,0],[0,0,0,0],[0,0,1,1],[0,1,0,0],[1,1,1,0],[1,0,0,0],[1,1,0,0],[1,0,1,1],[0,1,0,1]]

expected = [1,0,1,0,0,0,1,1,1,0,1,0]



result = []

for i in test:

    entry = i

    print(str(entry) + ": ->", end = " ")

    for j in range(0,len(entry)-1):

        x = [entry[j],entry[j+1]]

        perc = perceptron(x,0,0)

        entry[j+1] = perc.XOR()

    result.append(entry[len(entry)-1])

    print(entry[len(entry)-1])



# calcolo dell'accuratezza della previsione

print("La previsione ha un'accuratezza di: ", round(accuracy_score(expected,result)* 100), "%")
# disegno plot



plt.xlabel('sample validation')

plt.ylabel('output')

plt.plot(range(0,12), expected, 'r--', color='pink')

plt.plot(range(0,12), result,'.', color='blue')

plt.legend(['output atteso', 'output predetto'], numpoints=1)

plt.show()