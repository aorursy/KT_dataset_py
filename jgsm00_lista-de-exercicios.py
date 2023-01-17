custoFab = float(input("Digite o custo de fabricação do veículo: R$"))

porDist = 0.45

impostos = 0.12



custoTotal = custoFab + (custoFab * porDist) + (custoFab * impostos)



print("O custo total para o consumidor é: R$", custoTotal)
n = 1

for i in range (1, 1002):

    if (n % 3 != 0) and (n % 5 != 0):

        print (n)

    n += 1
import math



x1 = float(input("Digite o valor de X1: "))

x2 = float(input("Digite o valor de X2: "))

x3 = float(input("Digite o valor de X3: "))

x4 = float(input("Digite o valor de X4: "))

x5 = float(input("Digite o valor de X5: "))



media = (x1 + x2 + x3 + x4 + x5) / 5

somatorio = (x1 - media)**2 + (x2 - media)**2 + (x3 - media)**2 + (x4 - media)**2 + (x5 - media)**2 

total = math.sqrt(0.25 * somatorio)



print("Resultado: ", total)
import math



S1 = float(input("Digite o valor de S1: "))

S2 = float(input("Digite o valor de S2: "))

S3 = float(input("Digite o valor de S3: "))



T = (S1 + S2 + S3) / 2

areaTriangulo = math.sqrt(T * (T - S1) * (T - S2) * (T - S3))



print("Area do triângulo: ", areaTriangulo)
maior = int(input("Digite o maior valor: "))

menor = int(input("Digite o menor valor: "))



quociente = maior / menor

resto = maior % menor

print("resto: ", resto, " | quo: ", quociente)



while (resto > 0):

    maior = menor

    menor = resto

    resto = maior % menor

    quociente = maior / menor

    print("Resto: ", resto, " | Quociente: ", quociente)

    

print("Resultado: ", quociente)
num = int(input("Digite um valor: "))



digito1 = int(num/100)

digito2 = int(num/10) - (digito1 * 10)

digito3 = num - int(num/10) * 10



digito4 = (digito1 + (digito2 * 3) + (digito3 * 5)) % 7



print("Digitos separados:", digito1, digito2, digito3)

print("Novo digito:", digito4)



print("Novo número:", ((digito1 * 1000) + (digito2 * 100) + (digito3 * 10) + digito4))

ladoA = float(input("Digite o tamanho do lado A do triângulo: "))

ladoB = float(input("Digite o tamanho do lado B do triângulo: "))

ladoC = float(input("Digite o tamanho do lado C do triângulo: "))



triangulo = 0

tipo = ""



if (abs(ladoB - ladoC ) < ladoA) and (ladoA < (ladoB + ladoC)):

    triangulo = 1

elif (abs(ladoA - ladoC) < ladoB) and (ladoB < (ladoA + ladoC)):

    triangulo = 1

elif (abs(ladoA - ladoB) < ladoC) and (ladoC < (ladoA + ladoB)):

    triangulo = 1

else:

    triangulo = 0

            

if (triangulo == 1):

    if ((ladoA == ladoB) and (ladoB == ladoC)):

        tipo = "equilátero"

    elif((ladoA == ladoB) or (ladoB == ladoC) or (ladoC == ladoA)):

        tipo = "isósceles"

    else:

        tipo = "escaleno"

    

    print("É um triângulo", tipo, ".")

else:

    print("Não é um triângulo.")
import math



numRaiz = 0



coA = float(input("Digite o valor do coeficiente A: "))

coB = float(input("Digite o valor do coeficiente B: "))

coC = float(input("Digite o valor do coeficiente C: "))



delta = (coB**2) - (4 * coA * coC)



if (delta >= 0):

    x1 = ((0 - coB) + math.sqrt(delta))/(2 * coA)

    x2 = ((0 - coB) - math.sqrt(delta))/(2 * coA)

    

    if ((x1 >= 0) and (x2 >= 0)):

        print("Duas raizes reais distintas. X1 =", x1, "/ X2 =", x2)

    elif ((x1 >= 0) or (x2 >= 0)):

        if(x1 >= 0):

            print("Apenas uma raiz real. X1 =", x1)

        elif(x2 >= 0):

            print("Apenas uma raiz real. X2 =", x2)

    else:

        print("Sem raizes reais.")

else:

    print("Sem raizes reais.")