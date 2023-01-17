"""
Entrada

A entrada é dada em duas linhas. 
A primeira tem dois inteiros positivos P e N, 
a altura do pulo do sapo e o número de canos (1 ≤ P ≤ 5 e 2 ≤ N ≤ 100). 
A segunda linha tem N inteiros positivos que indicam as alturas dos canos ordenados da esquerda para a direita. 
Não há altura maior do que 10.

Saída

A saída é dada em uma única linha. 
Se o sapo pode chegar no cano mais à direita, escreva "YOU WIN". 
Se o sapo não consegue, escreva "GAME OVER".
"""
#Total de operações dentro do programa é
# total = total3 + total 8
# total = (8+n) + (n*12) - 5
# total = (13*n + 3)

pulo, n = input().split() # atribiu a altura do pulo e a quantidade de pulos e separa # 3 op
pulo = int(pulo) # transforma em int e atribui # 2 op
n = int(n) # 2 op
h = input().split() # atribui o h conforme qtde de pulos e separa # (n+1) op
############# toral3 = 7 + (n+1)  = (8+n) op

if(n==len(h)): # contagem pelo len e comparação # 2 op ########## total8 = total7 + total5 + total4 + total2
                                                       ########## total8 = 2 + (n*3) + ((n-1)*6) + (n-1)*3 + 2op
                                                       ########## total8 = 4 + (n*3) + (6*n-6) + (3*n-3)
                                                       ########## total8 = 4 + (n*12) -6 -3
                                                       ########## total8 = (n*12) - 5

    h_diff=[] #diferença de h (altura) # define como vazio # 1 op 
    cont=0 # atribui zero à variavel # 1 op 
    h_int = [int(a) for a in h] # atribui a a e transforma em int e atribui a h_int # n*3 op 
    ######################## total7 = 2 + (n*3) = 
    
    for i in range(len(h_int)-1): #calcula qtde de alturas e subtrai 1 todas as vezes que realizar o for ######### total5 = (n-1)*total6 = (n-1)*6
        # 2 op
        diff = abs(h_int[i]-(h_int[i+1])) # calcula a diferença, transforma em absoluto e atribui # 3 op
        h_diff.append(diff) # 1 op 
        ######### total6 = 6
        
    for j in h_diff: # (n-1) op ###### total4 = (n-1)*total1 = (n-1)*3
        if j>pulo: # se o pulo for menor que a diferença de h, ocorre a contagem # 1 op
            cont+=1 # 2 op
        ###### total1 = 3 op
            
    if(cont>0): # 1 op ####### total2 = 2op (comparação e print)
        print("GAME OVER") ########## 1 op
    else:
        print("YOU WIN") ########1 op