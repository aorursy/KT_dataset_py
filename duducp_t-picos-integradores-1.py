#dados para treinamento
pig1 = [1,1,0]
pig2 = [1,1,0]
pig3 = [1,1,0]
dog1 = [1,1,1]
dog2 = [0,1,1]
dog3 = [0,1,1]
data = [pig1, pig2, pig3, dog1, dog2, dog3]

#especifica o que é porco e o que é cachorro e faz as marcações para treinamento
is_pig = 1
is_dog = -1
markations = [is_pig, is_pig, is_pig, is_dog, is_dog, is_dog]

#importa as bibliotecas para treinamento
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

#Faz o treinamento dos dados e da marcação
model.fit(data, markations)

#passando individuos para prever de acordo com o treinamento
test1 = [1,1,1] #cachorro
test2 = [1,0,0] #porco
test3 = [0,0,1] #cachorro
test = [test1, test2, test3]

#preve os dados que eu quero testar
prediction = model.predict(test)
#print(prediction)

array = []
for item in prediction:
    animal = ''
    if item == 1:
        animal = 'pig'
    else:
        animal = 'dog'
    array.append(animal)
    
print(array)
#dados para treinamento
pig1 = [1,1,0]
pig2 = [1,1,0]
pig3 = [1,1,0]
dog1 = [1,1,1]
dog2 = [0,1,1]
dog3 = [0,1,1]
data = [pig1, pig2, pig3, dog1, dog2, dog3]

#especifica o que é porco e o que é cachorro e faz as marcações para treinamento
is_pig = 1
is_dog = -1
markations = [is_pig, is_pig, is_pig, is_dog, is_dog, is_dog]

#importa as bibliotecas para treinamento
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

#Faz o treinamento dos dados e da marcação
model.fit(data, markations)

#passando individuos para prever de acordo com o treinamento
test1 = [1,1,1] #cachorro
test2 = [1,0,0] #porco
test3 = [0,0,1] #cachorro
test = [test1, test2, test3]

test_markations = [-1, 1, -1] #saida esperada após o treinamento
result = model.predict(test) #preve os dados que eu quero testar
differences = test_markations - result

success = [d for d in differences if d == 0]
total_success = len(success)
total_elements = len(test)

#percentual de acertos e erros
success_rate = 100 * (total_success/total_elements)
error_rate = 100 - success_rate
print(success_rate)
print(error_rate)