#trans (star,id1,srr,ide,id2,rtr,r1,r0,dlc,data1,srr)

quadro = []





return quadro



start [1]

id1 [1,1,1,0,0,1,1,0,0,1,0]

srr [1]

ide [1]

id2 [1,0,1,0,1,1,1,1,0,1,0,0,0,1,1,1,0,0,1]

rtr [1]

r1 [0]

r0 [0]

dlc [0,1,0]

data_byte1 [0,1,0,1,1,1,0,0]

data_byte2 [0,1,1,1,0,1,0,1]

data_byte3 [1,1,1,1,0,0,1,0]

data_byte4 [1,1,0,0,0,0,1,1]



num_bytes=0

vet1_num_bytes=[0,0,1]

vet2_num_bytes=[0,1,0]

vet3_num_bytes=[0,1,1]

vet4_num_bytes=[1,0,0]



if(dlc==vet1_num_bytes):

    num_bytes=1

else:

    if(dlc==vet_num_bytes):

        num_bytes=2

    else:

        if(dlc==vet3_num_bytes):

            num_bytes=3

        else:

            if(dlc==vet4_num_bytes):

                num_bytes=4

            else:

                print("O número binário é maior do que o permitido ou é zero")



data1 = []

if(num_bytes==1):

    data1.extend (data_byte1)

else:

        if(num_bytes==2):

            data1.extend (data_byte1)

            data1.extend (data_byte2)

        else:

            if(num_bytes==3):

                data1.extend (data_byte1)

                data1.extend (data_byte2)

                data1.extend (data_byte3)

            else:

                if(num_bytes==4):

                    data1.extend (data_byte1)

                    data1.extend (data_byte2)

                    data1.extend (data_byte3)

                    data1.extend (data_byte4)

                else:

                    print("O número de bytes pedido esta incorreto")

                    

quadro.extend (start)

quadro.extend (id1)

quadro.extend (srr)

quadro.extend (ide)

quadro.extend (id2)

quadro.extend (rtr)

quadro.extend (r1)

quadro.extend (r0)

quadro.extend (dlc)

quadro.extend (data1)

quadro.extend (srr)                    



posicoes_zero = []

posicoes_um = []



for idx in range(len(data1)):

    if idx >= 5:

        s = sum(data1[idx - 5:idx])

        if s == 0:

            posicoes_um.append(idx)

        elif s == 5:

            posicoes_zero.append(idx)

                

            

z=0

y=0

for idz in range(len(posicoes_zero)):

    z = posicoes_zero [idz]

    data1[z].append(0) 



for idz in range(len(posicoes_um)):

    y = posicoes_zero [idz]

    data1[y].append(1) 
#CRC

#dados de entrada : 

#gerador do can 2.0b :

dados_entrada = [1,1,0,0,0,0,1,0,1,0,1,0,0,1,0] # acrescentar 15 "0"  exemplo = crc ([1,1,0,0,0,0,1,0], 8, []) #acrescentar 8 zeros no crc

gerador = [1,1,0,0,0,1,0,1,1,0,0,1,1,0,0,1]

#as duas tem que tem o mesmo tamanho

resultado = []

#list(zip(dados_entrada,gerador))

#x = [1,0,0,1]

#y = [1,0,1,0]

#resultado = []

#for i,j in zip(x,y):

 #   print(f"primeiro elemento : {i} - segundo elemento: {j}")

#def xor (a,b) :

 #   resultado = []

    

  #  for i,j in zip (a,b) : 

   #     resultado.append(i ^ j)

    #return resultado    

#COMO FAZER MAIS OU MENOS

def crc (bits, n, divisor) : #crc(dados_entrada,15,gerador) 

    dividendo = bits.copy()

    dividendo.extend([0]*n)

    

    pos_inicial = dividendo.index(1)

    pos_final = pos_inicial + (n + 1)

    

    novo_dividendo = xor(dividendo[pos_inicial : pos_final], divisor)

    

    fim = False

    while (not fim) :

        pos_inicial = novo_dividendo.index(1)

        

        pedaco_novo_dividendo = novo_dividendo[novo_dividendo.index(1) :]

        pedaco_antingo_dividendo = dividendo[pos_final :]

        

        novo_dividendo = xor(pedaco_novo_dividendo + pedaco_antigo_dividendo, divisor)

        

        pos_final = pos_inicial + (n + 1)

        

        if pos_final >= len(dividendo) : 

            fim = True

    

    

 
#CRC.  Vamos ter o vetor Data1 como mensagem do código e o código gerador que vai ser usado o do slide.

#Para fazer o crc precisa criar um outro vetor que vai ser composto pelo data1 eum vetor com uma quantidade

#de zeros equivalente a quantidade de casas do cod gerador -1.

#Para fazer isso usa la um for que percorre todo o cod geradore em cada casa va somando 1 em uma variável

#e no final subtraia 1. Depois disso cria um vetor Datacrc que vai ser o Data1 com esse vetor com uma 

#quantidade dezeros dita pela variavel que tem a soma.

#No começo do CRCfazemos um XOR com as primeiras casas do dara1 equivalente ao num de casas do cod gerador,

#vamos obiter um resultado. Neste resultado fazemos um for para percorrer ele até que o num na casa seja 

#1, em quanto isso uma variável vai somando quantas casas se passaram. Guardada essa variável nós vamos 

#fazer um extended no resultado com um vetor que pega dentro desse Datacrc as posições equivalentes à

#quantidade de num de casas do cod gerador mais essa variavel percorrida. No final disso soma esa variavel

#percorrida na variavel que tem guardada o num de casas do cod gerador, pois no proximo xor vai precisar das

#casas dali pra frente.

#Fazendo esse extended faz outro XOR com o cod gerador e repete isso até acabar os num para adicionar.

#Antes disso faz um for para percorrer o Datacrc para saber quantas casas ele tem. Depois subtrai o num de 

#casas do cod gerador e guarda em uma variavel. Todo o processo falado antes desse paragrafo vai acabar 

#quando o valor da variavel que tem quantas casas ja adicionou isso se faz com um while.

#Para saber quantas casas ja foram adicionadas é só ter uma variável total que vai somando.

#No receptor é só fazer a mesma coisa so que em vez de adicionar ao data 1 o vator de zeros adiciona o 

#resultado final do crc ja feito e fazer tudo de novo com o mesmo cod gerador. No final do segundo crc o 

#resultado tem que dar um vetor de zeros.