guests = ['Matheus', 'Luana', 'Lucas', 'Alfredo', 'Jéssica']
print(guests)

first_guest = guests[0]

second_guest = guests[1]



print('O primeiro convidado é: %s' % (first_guest))

print('O segundo convidado é: %s' % (second_guest))
print(guests)

last_guest = guests[-1]

last_but_one_guest = guests[-2]



print('O último convidado é: %s' % (last_guest))

print('O penúltimo convidado é: %s' % (last_but_one_guest))
print(guests)

print(f'Os dois primeiros itens são: {guests[0:2]}')

print(f'O 3º, 4º e 5º itens são: {guests[2:5]}')
print(guests)

print(f'Os dois primeiros itens são: {guests[-5:-3]}')

print(f'O 3º e 4º itens são: {guests[-3:-1]}')
print(guests[:])

print(f'Os dois primeiros itens são: {guests[:2]}')

print(f'Os três últimos itens são: {guests[-3:]}')
guests = ['Matheus', 'Luana', 'Lucas', 'Alfredo', 'Jéssica']



print(guests)

del guests[1] # Exclui o segundo elemento da lista guests

print(guests)

del guests    # Exclui a lista guests

print(guests)
numbers = [ 3, 94,  2, 94, 72, 58, 72, 52]

print(type(numbers))

print(type(numbers) == list)

print(len(numbers))

print(max(numbers))

print(min(numbers))
guests = ['Matheus', 'Luana', 'Lucas', 'Alfredo', 'Jéssica', 'Rodrigo', 'Ely', 'João']



guests.append('Carol')

print(f'convidados: {guests}')



guests.insert(1, 'Adriana')

print(f'convidados: {guests}')



guests.remove('Lucas')

print(f'convidados: {guests}')



name = guests.pop()

print(f'nome: {name}, convidados: {guests}')



name = guests.pop(3)

print(f'nome: {name}, convidados: {guests}')



index = guests.index('Luana')

print(f'indice: {index}')



guests[-1] = 'Alfredo'

times = guests.count('Alfredo')

print(f'vezes: {times}')



guests.sort()

print(f'convidados: {guests}')



guests.reverse()

print(f'convidados: {guests}')



other_guests = guests.copy()

print(f'outros convidados: {other_guests}, convidados: {guests}')



guests2 = guests

guests.clear()

print(f'outros convidados: {other_guests}, convidados 2: {guests2}, guests: {guests}')



guests.append('Franciele')

guests.append('Fabio')

print(guests)

guests.extend(other_guests)

print(f'outros convidados: {other_guests}, convidados: {guests}')
import math



numbers = [ 3, 94,  2, 39, 72, 58, 75, 52]



square_roots = list(map(math.sqrt, numbers))

print(square_roots)
def is_even(x):

    return (x % 2) == 0



numbers = [ 3, 94,  2, 39, 72, 58, 75, 52]



even = list(filter(is_even, numbers))

print(even)
# Criação de tuplas

tlp1 = (82,'edge',2+1j, True)

tlp2 = (1,)

var1 = (1) # Não é uma tupla

tlp3 = tuple([1,2])



# Tamanho da tupla

len(tlp1)



# Acessar os valores a partir dos índeces

print(tlp1[2])

print(tlp1[-2])



# - Slice

print(tlp1[1:3])



# - Casting

list1 = list(tlp1)

set1 = set(tlp1)



# Juntar tuplas

print( tlp1 + tlp2 )
# Criação de set

set1 = set([1,2,3,2])

set2 = {1,2,3,2}

set3 = set()

var1 = {} # Não é um set



# Tamanho de um set

len(set1)



# Verificar se um item está no set

print( 1 in set1 )

print( 1 not in set1 )



# - Adição de item(s)

set1.add(6)

set2.update([5,4,7,10])

set1.union(set2) # Este método só funciona para set



# Remoção do primeiro item

set1.pop() # Remove o primeiro

set1.remove(2)

set1.clear() # Remove todos os itens
# Criação de um dicionário Create a dictionary

dic1 = {}

dic2 = dict()

dic3 = {

    1: 'abc',

    'a': 'd'

}

dic4 = {

	1: 1,

	1: 'a'

}

# Adição de uma nova chave

dic3[10] = '10'



# Tamanho

len(dic3)



# Acessar os valores a partir das chaves

print(dic3[1])



# Remoção de uma chave

del dic3[10]



# Verifica se a chave está no dicionário

print(1 in dic3)



# Pegar todas as chaves

print(dic3.keys())



# Pegar todas os valores

print(dic3.values())



# Pegar todas as chave/valor

print(dic3.items())
object_iterable = [1,2,3,4,5,6]



for v in object_iterable:

    print(v)

    

length = len(object_iterable)

idx = 0



while ( idx < length ):

    print(object_iterable[idx])

    idx = idx + 1
element = 0

while element < 5:

    element += 1

    print(element-1)

    pass # Neste caso o pass é opcinal



for element in range(5): # range(5) constrói uma lista de 5 elementos da seguinte maneira: [0,1,2,3,4] 

    pass # Neste caaso o pass é obrigatório
element = 0

while element < 5:

    element += 1

    if element == 3:

        continue 

    # Quando element for igual a 3 essa parte do código não será executada

    print(element-1)

print(element)
element = 0

while element < 5:

    element += 1

    if element == 2:

        break

    # Quando element for igual a 2, as linhas dentro do while não serão executas e sairá do while

    print(element-1)

print(element)
element = 0

while element < 5:

    element += 1

    if element == 2:

        break

    print(element-1)

else :

    print("[1] Executou o else.")

print(element)



element = 0

while element < 5:

    element += 1

    if element == 10:

        break

    print(element-1)

else :

    print("[2] Executou o else.")

print(element)