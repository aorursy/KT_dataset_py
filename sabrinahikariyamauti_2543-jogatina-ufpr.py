"""
Entrada

A entrada contém vários casos de teste. 
A primeira linha de cada caso contém dois inteiros N e I (1 ≤ N ≤ 104, 1000 ≤ I ≤ 9999), 
o número de gameplays publicados na página e o seu identificador na universidade, respectivamente.
As próximas N linhas descrevem os gameplays publicados. 
Cada gameplay é descrito por dois inteiros i e j (1000 ≤ i ≤ 9999, j=0 ou 1), 
onde i é o identificador na universidade do autor do gameplay, 
e j=0 se o gameplay é de Contra-Strike, ou j=1 se é de Liga of Legendas.

A entrada termina com fim-de-arquivo (EOF).

Saída

Para cada caso de teste, imprima uma única linha com um número indicando quantos gameplays seus de Contra-Strike foram publicados na página.

"""
#Total de operações no programa desconsiderando arquivo EOF é 
# total3 + total4 + 1op(print) = 8+(n*12)+1 = (9 + n*12) operações

while True:
  try:
    
    n,iden = input().split() #numero de gameplays e identificação e separa # 3 op
    n=int(n) # transforma em int e atribiu à variável # 2 op
    iden=int(iden) # transforma em int e atribiu à variável # 2 op
    cont=0 # 1 op
    ############### total3 = 8 op
    
    for i in range(n): # n operaçãoes ############# total4 = n*((total1)+(total2)) = n*(12)
        
        proc, gameplay = input().split() #  identidicação procurada e gameplay e separa #3 op
        proc=int(proc) #2 op
        gameplay=int(gameplay) #2 op
        # total1 = 7 op
        
        if(proc==iden and gameplay==0): #realiza duas comparações e verifica se atende ambas as condições #3 op
            cont+=1 # 2 op identificação igual com o gameplay procurado
            # total2 = 5 op
        else:
            pass # 1 op
        
    print(cont) ################1 op
  except EOFError:
    break
