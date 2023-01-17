"""
Entrada

A entrada contém vários casos de teste. 
A primeira linha de cada caso contém uma string de exatamente 26 letras maiúsculas 
contendo todas as letras do alfabeto inglês. 
A primeira letra da string está associada à lâmpada 1; 
a segunda letra está associada à lâmpada 2; e assim por diante. 
A próxima linha contém um inteiro N (1 ≤ N ≤ 104), o número de lâmpadas que foram piscadas. 
A terceira linha contém N inteiros li (1 ≤ li ≤ 26), indicando as lâmpadas que foram piscadas, em ordem.
A entrada termina com fim-de-arquivo (EOF).

Saída

Para cada caso de teste, imprima uma única linha contendo a mensagem enviada por Will.
"""
import string

# Total de operações desconsiderando o arquivo EOF é:
#total = 2*(total1 + total3 + 2) 
# total = 2*( 4 + (4*n) + 3*n + 2)
# total = 2*(6 + (7*n))
# total = 12 + 14*n

while True:
  try:
      for i in range(2): #2 op ############# total = 2*(total1 + total3 + 2)    
        alfabeto = input() # 1 op
        tamanho=int(input()) #qtde de palavras # 2 op
        posicoes = input().split() #selecionar pos das palavras e separa # (n + 1)op
        indices = [int(a) for a in posicoes] #atribui string ao "a" e transforma em int e atribui o int no indice # n*3  op
        frase=[]
        ############# total1 = 1 + 2 + (n+1) + (n*3) = 4 + (4*n)

        for f in indices: #n op ############ total3 = 3*n
            f-=1 # 2 op
            frase.append(alfabeto[f])  # 1 op
            #total2 = 3op 
            
        print(''.join(frase)) # junta as frases e mostra na tela ###### 2 op      
        
  except EOFError:
    break
    
