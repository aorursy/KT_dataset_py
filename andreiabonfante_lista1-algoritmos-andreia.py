custo_de_fabrica = float(input('Entre com valor do custo de fabrica do carro: '))

custo_consumidor = custo_de_fabrica + (custo_de_fabrica*(12+45)/100)

print("O custo final para o consumidor é: ", custo_consumidor)
numero = 1

while (numero <= 1001):

    if (numero % 2 == 0):

        if (numero % 3 != 0) and (numero % 5 != 0):

            print("O número ", numero, " é par e não é multiplo de 3 nem 5")

    numero = numero + 1
import math



print("Entre com os cinco valores reais para o cálculo do desvio padrão")

x1 = float(input())

x2 = float(input())

x3 = float(input())

x4 = float(input())

x5 = float(input())



media = (x1+x2+x3+x4+x5)/5

desvio = math.sqrt((1/4) * ((x1-media)**2)+ ((x2-media)**2)+((x3-media)**2)+((x4-media)**2)+((x5-media)**2))

print(" O desvio padrão calculado desses valores é: ",desvio)
import math

S1 = float(input())

S2 = float(input())

S3 = float(input())

T = (S1+S2+S3)/2

print("Valor de T: ",T)

prod = T*(T-S1)*(T-S2)*(T-S3)

area = math.sqrt(abs(prod))

print(" A área desde triângulo é: ", area)
print("Entre com um número de 3 algarismos")

numero = input()



while len(numero) != 3:  #fica na repeticao enquanto numero tem tamanho diferente de 3

    print("Entre com um número de 3 algarismos")

    numero = input()



print("algarismo 1: ",numero[0])

print("algarismo 2: ",numero[1])

print("algarismo 3: ",numero[2])



controle = str( (int(numero[0]) + (int(numero[1])*3) + (int(numero[2])*5)) % 7)

print(" Digito de Controle: ",controle)



novo_numero = numero + controle  # esse operador + quando executado com string (cadeia de caracteres) provoca a concatenação das duas cadeias

print("Novo numero: ", novo_numero)
# do Allan

bin = int(input())

print("Numero binario recebido: ", bin)

dec = 0

exp = 0

while (bin != 0):

    dec = dec + ((bin % 10)*(2**exp))

    bin = bin // 10    

    exp = exp + 1

print(dec)
print("Entre com valor: ")

valor = int(input())   

print(" Valor entrado: ",valor) 

anterior = -1000000      # criei a variavel anterior para conseguir comparar com o valor atual. Coloquei um valor que supostamente não havera entrada menor

quant = 1

soma = valor



while (anterior < valor):

    if (anterior != -1000000):

        soma = soma + valor

    media = soma / quant

    print(" Média até agora: ",media)

    anterior = valor

    print("Entre com novo valor: ")

    valor = int(input())   

    print(" Valor entrado: ",valor) 

    quant = quant + 1
# Allan

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
