import numpy as np # importa a biblioteca usada para trabalhar com vetores de matrizes.
# criando um vetor
vet = np.array( [1,2,3,4,5] )

#você pode imprimir usando a função print ou a função display
print(vet)
display(vet)
vet_seq1 = np.arange(8)

display(vet_seq1)
vet_seq1 = np.arange(5,11)

display(vet_seq1)
vet_seq1 = np.arange(5,20,2)
print('Sequencia de 5 até 20, variando de 2 em 2:', vet_seq1)

vet_seq1 = np.arange(5,20,5)
print('Sequencia de 5 até 20, variando de 5 em 5:', vet_seq1)

vet_seq1 = np.arange(5,20,4)
print('Sequencia de 5 até 20, variando de 4 em 4:', vet_seq1)
for i in np.arange(2,11):
    b = 0.0512
    c = 'software'
    print('%d %0.2f %s' %(i,b,c))
    
print('qualquer coisa')
#criando um vetor com 10-zeros
vet_zeros = np.zeros(10)

#criando um vetor com 10-uns
vet_ones = np.ones(10)

print('Vetor de zeros: ', vet_zeros)
print('\nVetor de valores um: ', vet_ones)
# criando uma matriz
A = np.array( [[1,2,3,4,5],[6,7,8,9,10]] )

display(A)
#criando um array de valores zeros com dimensão 2x10
array_zeros = np.zeros( [2,10] )

#criando um array de valores um com dimensão 2x10
array_ones = np.ones( [2,10] )

print('Vetor de zeros: ')
display(array_zeros)

print('\nVetor de valores um: ')
display(array_ones)
# criando um vetor
vetA = np.array(['a','b','c','d','e','f','g','h','i'])

print('Três primeiros elementos de vetA: ')
print( vetA[0:3] )

print('\ntodos os valores após o Quinto elementos de vetA: ')
print( vetA[5:] )

print('\nOs três ultimos valores de vetA: ')
print( vetA[-3:] )

print('\nOs valores de vetA entre o 5 elemento até o penúltimo elemento: ')
print( vetA[4:-2], 'ou', vetA[4:7] )
# criando um vetor
arrayA = np.array( [['1a','1b','1c','1d','1e','1f','1g','1h','1i'],
                    ['2a','2b','2c','2d','2e','2f','2g','2h','2i'],
                    ['3a','3b','3c','3d','3e','3f','3g','3h','3i'],
                    ['4a','4b','4c','4d','4e','4f','4g', '4h','4i']])

print('Matriz inteira: ')
print( arrayA )

print('\nTodos os elementos da coluna 3: ')
print( arrayA[:,2] )

print('\nTodos os elementos da linha 2: ')
print( arrayA[1,:] )

print('\nTodos os elementos das 2 primeiras colunas: ')
print( arrayA[:,0:2] )

print('\nTodos os elementos das 2 primeiras linhas: ')
print( arrayA[0:2,:] )

print('\nApenas os elementos das 2 primeiras linhas e das 2 primeiras colunas: ')
print( arrayA[0:2,0:2] )

print('\nApenas os elementos das 2 últimas linhas e das 4 últimas colunas: ')
print( arrayA[-2:,-4:] )

print('\nApenas os elementos das linhas 2 até 4 e das colunas 4 até 6: ')
print( arrayA[1:3,3:6] )
def soma(a,b):
    
    soma_valores = a+b
    
    return soma_valores

resultado = soma(10,6)
print('10 + 6 = %d' %resultado)
A = np.array( [1,2,3] )
B = np.array( [4,5,6] )
print('A:', A);
print('B:', B);

print('\nA+B: ', A+B )
print('A-B: ', A-B )
X1 = np.array( [[1,2,3],
                [4,5,6]] )

X2 = np.array( [[4,5,6],
                [7,8,9]])

print('X1: \n', X1);
print('\nX2: \n', X2);

print('\nX1+X2:')
display(X1+X2)

print('\nX1-X2:')
display(X1-X2)
A = np.array( ([[1,2],[3,4],[5,6]]) )
B = np.array( ([[1,2,3,4],[5,6,7,8]]) )

print('A: ')
display(A)

print('B: ')
display(B)

print('A*B: ')
display(np.dot(A,B)) 
print('B: ')
display(B)

# média das linhas de B
media1 = np.mean(B, axis = 1)
print('\nMédia das linhas de B: ')
display(media1)

# média das colunas de B
media2 = np.mean(B, axis = 0)
print('\nMédia das colunas de B: ')
display(media2)

# média de todos os valores de B 
media3 = np.mean(B)
print('\nMédia de todos os valores de B: ')
display(media3)
print('B: ')
display(B)

# média das linhas de B
std1 = np.std(B, axis = 1, ddof=1)
print('\nDesvio padrão das linhas de B: ')
display(std1)

# média das colunas de B
std2 = np.std(B, axis = 0, ddof=1)
print('\nDesvio padrão das colunas de B: ')
display(std2)

# média de todos os valores de B 
std3 = np.std(B, ddof=1)
print('\nDesvio padrão de todos os valores de B: ')
display(std3)
ExA = np.array( [[12,9,4,1],[11,5,8,1],[1,2,3,1]] )
ExB = np.array( [[1,5],[1,7],[1,9],[1,1]] )

display(ExA)
display(ExB)
ExC = np.dot(ExA,ExB)
display(ExC)
def media_desvio(matriz):
    
    media1 = np.mean(matriz, axis = 1)
    print('\nMédia das linhas da matriz: ')
    display(media1)
    
    desvio1 = np.std(matriz, axis = 1, ddof=1)
    print('\nDesvio padrão das linhas da matriz: ')
    display(desvio1)
    
    media2 = np.mean(matriz, axis = 0)
    print('\nMédia das colunas da matriz: ')
    display(media2)
    
    desvio2 = np.std(matriz, axis = 0, ddof=1)
    print('\nDesvio padrão das colunas da matriz: ')
    display(desvio2)
    
media_desvio(ExC)
ExD = ExA[:,2:]
media = np.mean(ExD)
print('\nMédia da matriz ExD: ')
display(media)
ExE = ExA[0:2,0:2]
print(ExE)
ExF = np.zeros([5,2])
print(ExF)
ExG = np.ones([4,3])
print(ExG)