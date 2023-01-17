"""
Entrada

A entrada é dada em duas linhas. 
A primeira tem o número N de medidas da paisagem (1 < N ≤ 100). 
A segunda linha tem N inteiros: a altura Hi de cada medida (-10000 ≤ Hi ≤ 10000, para todo Hi, tal que 1 ≤ i ≤ N). 
Uma medida é considerada um pico se é maior que a medida anterior. 
Uma medida é considerada um vale se é menor que a medida anterior.

Saída

A saída é dada em uma única linha. 
Caso a paisagem tenha o mesmo padrão da Nlogônia, deve ser mostrado o número 1. 
Caso contrário, mostra-se o número 0.
"""
#Total de operações dentro do programa é
# total = total6 + total3 + total7
# total = n+7 + 2 + (n*8) + 2 
# total = 11 + (n*9) 


n = int(input()) # quantidade de picos e vales # atribui e transforma em int # 2op
seq = input().split() #altura da sequencia # atribui e separa # (n+1) op
int_seq = [int(i) for i in seq] # atribui o seq em i e transforma em int e atribui o valor final no int_seq # 3 op
nok=0 # atribui zero ao nok # 1 op
########### total6 = 6 + (n+1) = n+7

#quando quantidade de picos e de vales é ímpar

#Após a contagem, obeservamos que o pior caso entre if e else é o if com total3 = 2 + (n*8) op
if(n%2==1): #divide e compara # 2 op ######## para concluir no if ocorrem 2 + total2 = 2 + (n*8) op
    
    # o valor do meio a ser comparado inicia à partir do segundo elemento, contando de 2 em 2 
    for i in range(1,n,2):# n op ########### total2 = n*total1 = n*8 
        
        #verifica se o número do meio é maior que a anteiror e posterior ou menos que a anterior e a posterior        
        if(int_seq[i]<int_seq[i-1] and int_seq[i]<int_seq[i+1]): #realiza ambas as comparações e verifica se atende ambas # 3 op
            pass # 1op
        elif (int_seq[i]>int_seq[i-1] and int_seq[i]>int_seq[i+1]):# 3 op
            pass # 1op
        # se não se enquadrar en nenhuma das condições, não obedece a regra da sequência
        else:
            nok+=1 # soma e atribui # 2 op
        ############## total1 =    3(if) + 3(elif) + 2(else) Seria o pior caso
            
#quando quantidade de picos e de vales é par 

else:# Considerando o pior caso como elif tendo (2+(n*5)) op entre as próximas condições...
    
    # se iniciar com o maior
    if(int_seq[0]>int_seq[1]): # 1 op ############# para concluir no if ocorrem (1(próprio if) + total5) op = 1+(n*5) op
        
        # a contagem inicia do terceiro elemento e ela deve ser maior que o anterior e maior que o posterior, contando de 2 em 2
        for i in range(2,n,2): # n op  ############# total5 = n*5
            if(int_seq[i]>int_seq[i-1] and int_seq[i]>int_seq[i+1]): # 3 op
                pass # 1 op
            else:# para concluir no else(pior caso) ocorrem 5 op
                nok+=1 # 2 op
                
    # se iniciar com o maior
    elif (int_seq[0]<int_seq[1]): # 1 op ################ para concluir no elif ocorrem (1(if anterior)+ 1(próprio elif) + total4) op = 2+(n*5) op
        # a contagem inicia do terceiro elemento e ela deve ser menor que o anterior e menor que o posterior, contando de 2 em 2
        
        for i in range(2,n,2): # n op ############ total4 = n*5
            if (int_seq[i]<int_seq[i-1] and int_seq[i]<int_seq[i+1]): # 3 op
                pass # 1 op
            else:# para concluir no else(pior caso) ocorrem 5 op
                nok+=1 # 2 op
        
    else: # para concluir no else ocorrem 4 op (1(if anterior) + 1(próprio elif) + 2(nok))
        nok+=1 # 2 op
        
if(nok==0): #compara e print ################# total7 = 2 op
    print('1')
else:
    print('0')
