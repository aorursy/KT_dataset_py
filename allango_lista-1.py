#Operadores Aritméticos em Python



#Adição (+)

print("Adição: 2+2 = ",2+2)

#Subtração (-)

print("Subtração: 2-2 = ",2-2)

#Multiplicação (*)

print("Multiplicação: 2*2 = ",2*2)

#Divisão (/)

#veja que a divisão, ao imprimir na tela (print), o resultado sai um número real e não inteiro, isso porque a divisão pode não ser inteira. 

#Para retornar apenas a parte inteira da divisão, há outro perador mostrado mais abaixo

print("Divisão: 3/2 = ",3/2)



#Exponenciação (**)

print("Exponenciação 2 elevado a 2: ", 2**2)

#Extrair a parte inteira da divisão de dois números (//)

print("Parte inteira da divisão de 3 por 2: ", 3//2)

#Resto da divisão inteira (%)

print ("Resto da divisão de 3 por 2: ", 3 % 2)
#Operadores lógicos em Python



# >  maior

if (5 > 3):

    print ("5 é maior que 3?  \n\t Reposta:", 5>3)

# >=  maior ou igual

if (5 >= 5):

    print ("5 é maior ou igual a 5?  \n\t Resposta: ", 5 >=5)

# <=  menor ou igual

if (5 <= 6):

    print("5 é menor ou igual a 6?  \n\t Resposta: ", 5 <= 6)

# ==  igual

if( 5 == 5):

    print("5 é igual a 5?  \n\t Resposta: ", 5 == 5)

# !=  diferente

if(5 != 4):

    print("5 é diferente de 4?  \n\t Reposta: ", 5 != 4)



# not  Operador lógico que representa a negação (inverso) da variável atual. Se ela for verdade, torna-se falsa, e vice-versa.

if( not 5 != 5):

    print("5 NÃO é diferente de 5?  \n\t Resposta: ", not 5!=5)



# and  Operador lógico onde a resposta da operação é verdade se ambas as variáveis de entrada forem verdade.

if (5 == 5 and 4 != 5):

    print("5 é igual a 5 e 4 é diferente de 5? \n\t Resposta: ", 5 == 5 and 4 != 5)

# or  Operador lógico onde a resposta da operação é verdade se e somente se pelo menos uma das variáveis de entrada for verdade.

if (5 == 5 or 4 == 5):

    print("5 é igual a 5 OU 4 é igual a 5?  \n\t Resposta: ",5 == 5 or 4 == 5)

i = 1

while(i <= 1001):

    if((i % 2 == 0) and ((i % 3 != 0) or (i % 5 != 0))):

       print(i)

    i = i+1
a = float(input("informe o priemiro seguimento"))

b = float (input("informe o segundo seguimento"))

c = float (input("informe o terceiro seguimento"))

'''

if (abs(b - c) < a < (b + c)):

    if (abs(a - c) < b < ( a + c)):

        if( abs(a - b) < c < (a + b)):

            if (a == b and b == c):

                print("É um triângulo equilátero")

            else:

                if (a == b or b == c or a == c):

                    print("É um triângulo isósceles")

                else:

                    print("É um triangulo qualquer")

            

        else:

            print("Não é um triângulo")

    else:

        print("Não é um triângulo")

else:

    print("Não é um triângulo")

    

'''

if (abs(b - c) < a < (b + c)) and (abs(a - c) < b < ( a + c)) and ( abs(a - b) < c < (a + b)):

    if (a == b and b == c):

        print("É um triângulo equilátero")

    else:

        if (a == b or b == c or a == c):

            print("É um triângulo isósceles")

        else:

            print("É um triangulo qualquer")

else:

    print("Não é um triângulo")
a = 4 

b = 5

c = 1



delta = (b**2) - 4*a*c

print("Delta: ", delta)

if (delta < 0):

    print("Sem raízes reais")

else:

    if(a == 0):

        print("Apenas uma raiz real")

        print("A raiz é: ", (-1)*(b/c))

    else:

        raizdelta = (delta ** 0.5)

        x1 = (-b + raizdelta)/(2*a)

        x2 = (-b - raizdelta)/(2*a)

        print("Raiz x1: ",x1)

        print("Raiz x2: ",x2)

        

    
par = 0

impar = 1

while (par <= 12):

    soma = par + impar

    print("Soma: ", soma)

    par = par + 2

    impar = impar + 2


#bin = int(input())

bin = 101

dec = 0

exp = 0

while (bin != 0):

    dec = dec + ((bin % 10)*(2**exp))

    bin = bin // 10    

    exp = exp + 1

print(dec)
#iniciei a idade com 1 para forçar a leitura da primeira idade dentro do while, uma vez que a condição de parada do while é uma idade igual a Zero. 

#Veja que logo após iniciar o while, uma idade é lida e ela só é considerada se for maior que Zero



idade = -1

soma = 0

contador = 0



#while que lê idades indeterminadas até que uma idade igual a Zero seja digitada

# nesse exercício, é obvio que se uma idade negativa for digitada o while não irá parar, mas também não é considerada no calculo por causa do IF, 

#deixei a condição de Diferente de Zero no while para não ficar diferente do enunciado do exercício, mas pode substituir a comparação para Maior que Zero

while (idade != 0):

    idade = int(input())

    if (idade > 0):

        soma = soma+idade

        contador = contador + 1





print("A média das idades digitadas é: %d" %(soma/contador))



from IPython.display import clear_output

print('Teste')

clear_output()



print("Teste2")