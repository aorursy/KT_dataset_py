%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np
# Gerando 10 pontos igualmente espaçados entre 0 e 9 para o vetor x e entre 0 e -90 para y:

x = np.linspace(0, 9, 10)

y = np.linspace(0, -90, 10)



# Gerando o gráfico com os pares de pontos x e y criados acima. Aqui já podemos criar uma legenda para nossos

# dados.

plt.plot(x, y, label='Dados gerados pelo Numpy')



# Inserimos a legenda descrita no campo label.

plt.legend()



# E finalmente plotamos o gráfico

plt.show()
plt.scatter(x, y, label='Dados gerados pelo Numpy')



# Inserimos a legenda descrita no campo label.

plt.legend()



# E finalmente plotamos o gráfico

plt.show()
# Primeiramente, vamos definir o tamanho de nossa imagem para 15x10

plt.figure(figsize=(15,10))



# Vamos gerar a curva ligando os pontos, mas dessa vez na cor azul clara e com espessura de 3 pontos.

plt.plot(x, y, label='Curva',color='lightblue', linewidth=3)



# Os pontos agora serão verde escuro e com tamanho de 40.

plt.scatter(x, y, label='Dados',color='darkgreen',s=40)



# Inserimos a legenda descrita no campo label.

plt.legend()



# E finalmente plotamos o gráfico

plt.show()
# Primeiramente, vamos definir o tamanho de nossa imagem para 15x10

plt.figure(figsize=(15,10))



# Vamos gerar a curva ligando os pontos, mas dessa vez na cor azul clara e com espessura de 3 pontos.

plt.plot(x, y, label='Curva',color='lightblue', linewidth=3)



# Os pontos agora serão verde escuro e com tamanho de 60.

plt.scatter(x, y, label='Dados',color='darkgreen',s=60)



# Inserimos a legenda descrita no campo label.

plt.legend()



# Definindo o intervalo do eixo x para [0,9] e y para [-90,0]

plt.axis([0, 9, -90, 0])



# Criando um grid para facilitar a leitura

plt.grid(True)



# Nome do eixo X:

plt.xlabel('Dados entre 0 e 9', size=16)



# Nome do eixo Y:

plt.ylabel('Dados entre -90 e 0', size=16)



#Título do gráfico:

plt.title('Gráfico básico do Matplotlib', size=24)



# E finalmente plotamos o gráfico

plt.show()
lista_tamanho = np.linspace(20, 2000, 10)

lista_cores = ['grey','red','green','yellow','blue','magenta','cyan','lightblue','darkgreen']



# Vamos definir o tamanho de nossa imagem para 15x10

plt.figure(figsize=(15,10))



# Vamos gerar a curva ligando os pontos, mas dessa vez na cor azul clara e com espessura de 3 pontos.

plt.plot(x, y, label='Curva',color='lightblue', linewidth=3)



# Os pontos agora serão verde escuro e com tamanho de 60.

plt.scatter(x, y, label='Dados',color=lista_cores,s=lista_tamanho)



# Definindo o intervalo do eixo x para [0,9] e y para [-90,0]

plt.axis([-0.5, 9.5, -95, 0.5])



# Criando um grid para facilitar a leitura

plt.grid(True)



# Nome do eixo X:

plt.xlabel('Dados entre 0 e 9', size=16)



# Nome do eixo Y:

plt.ylabel('Dados entre -90 e 0', size=16)



#Título do gráfico:

plt.title('Gráfico básico do Matplotlib', size=24)



# E finalmente plotamos o gráfico

plt.show()
# Vamos primeiro criar um objeto figura, onde os subplots estarão

fig = plt.figure(figsize=(15,10))



# Criando um título para a figura toda

fig.suptitle("Exemplo de Subfigure", fontsize=32)



# Vamos agora adicionar 2 subplots com 2 colunas e uma linha (a estrutura é: linhas, colunas e plot_num)

sub1 = fig.add_subplot(1, 2, 1)

sub2 = fig.add_subplot(1, 2, 2)



#Vamos inserir os dados do primeiro subplot 1 do mesmo modo que fizemos anteriormente

sub1.plot(x, y, label='Curva',color='lightblue', linewidth=3)

sub1.scatter(x, y, label='Dados',color='darkgreen',s=60)

sub1.legend()

sub1.axis([0, 9, -90, 0])

sub1.grid(True)

#Uma atenção especial aos lables e titles!

sub1.set_xlabel('Dados entre 0 e 9', size=14)

sub1.set_ylabel('Dados entre -90 e 0', size=16)

sub1.set_title('Gráfico básico do Matplotlib', size=20)



#Vamos inserir os dados do primeiro subplot 2

sub2.plot(x, y, label='Curva',color='lightblue', linewidth=3)

sub2.scatter(x, y, label='Dados',color='darkgreen',s=60)

sub2.legend()

sub2.axis([0, 9, 0, -90])

sub2.grid(True)

sub2.set_xlabel('Dados entre 0 e 9', size=14)

sub2.set_ylabel('Dados entre 0 e -90', size=16)

sub2.set_title('Gráfico básico do Matplotlib 2', size=20)





# E finalmente plotamos o gráfico

plt.show()
# Essa é uma outra forma de criar os subplots

# Nessa estrutura, os parâmetros são: número de gráficos, compartilhamento de eixo e tamanho.

fig, (sub1, sub2) = plt.subplots(2, sharex=True, figsize=(15,10))



# Criando um título para a figura toda

fig.suptitle("Exemplo de Subfigure com a mesma escala no eixo X", fontsize=32)



# Vamos inserir os dados do primeiro subplot 1

sub1.plot(x, y, label='Curva',color='lightblue', linewidth=3)

sub1.scatter(x, y, label='Dados',color='darkgreen',s=60)

sub1.legend()

sub1.axis([0, 9, -90, 0])

sub1.grid(True)

#Uma atenção especial aos lables e titles!

sub1.set_ylabel('Dados entre -90 e 0', size=16)

sub1.set_title('Gráfico básico do Matplotlib', size=20)



# Vamos inserir os dados do primeiro subplot 2

sub2.plot(x, y, label='Curva',color='lightblue', linewidth=3)

sub2.scatter(x, y, label='Dados',color='darkgreen',s=60)

sub2.legend()

sub2.axis([0, 9, 0, -90])

sub2.grid(True)

sub2.set_xlabel('Dados entre 0 e 9', size=14)

sub2.set_ylabel('Dados entre 0 e -90', size=16)

sub2.set_title('Gráfico básico do Matplotlib 2', size=20)





# E finalmente plotamos o gráfico

plt.show()
fig = plt.figure(figsize=(15,10))

sub1 = fig.add_subplot(2, 2, 1)

sub2 = fig.add_subplot(2, 2, 2)

sub3 = fig.add_subplot(2, 2, 3)

sub4 = fig.add_subplot(2, 2, 4)

sub1.text(0.5, 0.5, "1", va="center", ha="center",size=60)

sub2.text(0.5, 0.5, "2", va="center", ha="center",size=60)

sub3.text(0.5, 0.5, "3", va="center", ha="center",size=60)

sub4.text(0.5, 0.5, "4", va="center", ha="center",size=60)

sub1.set_xticks([])

sub1.set_yticks([])

sub2.set_xticks([])

sub2.set_yticks([])

sub3.set_xticks([])

sub3.set_yticks([])

sub4.set_xticks([])

sub4.set_yticks([])

plt.show()
fig = plt.figure(figsize=(15,10))

for i in range (1,11):

    sub = fig.add_subplot(2, 5, i)

    sub.text(0.5, 0.5, i, va="center", ha="center",size=60)

    sub.set_xticks([])

    sub.set_yticks([])

plt.show()
fig = plt.figure(figsize=(15,10))

sub1 = fig.add_subplot(2, 2, 2)

sub2 = fig.add_subplot(2, 5, 6)

sub3 = fig.add_subplot(2, 5, 10)

sub1.text(0.5, 0.5, "2", va="center", ha="center",size=60)

sub2.text(0.5, 0.5, "6", va="center", ha="center",size=60)

sub3.text(0.5, 0.5, "10", va="center", ha="center",size=60)

sub1.set_xticks([])

sub1.set_yticks([])

sub2.set_xticks([])

sub2.set_yticks([])

sub3.set_xticks([])

sub3.set_yticks([])



plt.show()
fig = plt.figure(figsize=(15,6))



# Removendo os espaços entre os canvas

fig.subplots_adjust(wspace=0, hspace=0)



# Desenhando as teclas brancas (7 por oitava)

for i in range (1,15):

    sub = fig.add_subplot(1, 16, i)

    sub.set_xticks([])

    sub.set_yticks([])



# Desenhando as teclas pretas (5 sustenidos por oitava)

for i in [1,2,4,5,6,8,9,11,12,13]:

    sub = fig.add_subplot(2, 48, 3*i)

    sub.set_facecolor('black')

    sub.set_xticks([])

    sub.set_yticks([])

    sub = fig.add_subplot(2, 48, 3*i+1)

    sub.set_facecolor('black')

    sub.set_xticks([])

    sub.set_yticks([])



plt.show()
# Vamos travar a semente aleatória para sempre obter o mesmo resultado.

np.random.seed(42)



# Agora vamos sortear um vetor de 10000 números com média 1000 e um sigma de 20

media = 1000

sigma = 20

x = media + sigma * np.random.randn(10000)
# Se tudo correu como o esperado, devemos ter obtido um vetor com 10000 posições e valores próximos de 1000.

print('Valores:  ' + str(x) +'\n')

print('Número de posições do vetor: ' + str(x.shape[0]))
# Vamos criar agora um histograma com o vetor x

fig = plt.figure(figsize=(15,8))



# Observe que a função hist retorna, além do gráfico, 3 variáveis que salvaremos para discutir logo mais

n, bins, patches = plt.hist(x, 50, normed=1, facecolor='blue', alpha=0.5)



# As demais funções usadas são as mesmas já discutidas anteriormente:



# Criando um grid para facilitar a leitura

plt.grid(True)



# Nome do eixo X:

plt.xlabel('Valor', size=16)



# Nome do eixo Y:

plt.ylabel('Probabilidade', size=16)



#Título do gráfico:

plt.title('Histograma básico do Matplotlib', size=24);



plt.show()
# Vamos criar agora um histograma com o vetor x

fig = plt.figure(figsize=(15,8))



# Observe que a função hist retorna, além do gráfico, 3 variáveis que salvaremos para discutir logo mais

n, bins, patches = plt.hist(x, 50, normed=1, facecolor='blue', alpha=0.5)



# As demais funções usadas são as mesmas já discutidas anteriormente:



# Criando um grid para facilitar a leitura

plt.grid(True)



# Nome do eixo X:

plt.xlabel('Valor', size=16)



# Nome do eixo Y:

plt.ylabel('Probabilidade', size=16)



#Título do gráfico:

plt.title('Histograma básico do Matplotlib', size=24);

plt.text(940, .015, 'Média = 1000 \n $\sigma$ = 15', size=20, color='red')



plt.show()
# n é um vetor contendo o valor de cada um dos 50 "bins" do histograma

print(n)
# bins é um vetor contendo a coordenada do início de cada "bin"

print(bins)
# Vamos primeiro criar um objeto figura, onde os subplots estarão

fig = plt.figure(figsize=(15,8))



# Criando um título para a figura toda

fig.suptitle("Dois exemplos de gráfico de barras", fontsize=32)



# Ajustando o espaço entre os dois plots

fig.subplots_adjust(wspace=0.6)



# Criando os subplots

sub1 = fig.add_subplot(1, 2, 1)

sub2 = fig.add_subplot(1, 2, 2)



# Gerando os valores para nosso gráfico de barras e a largura da barra

grupos = [0,2,4]

nome_grupos = ['Grupo 1', 'Grupo 2', 'Grupo 3']

valores = [4,5,8]

largura = 2



# Vamos criar um gráfico de barras verticais, com os labels nas posições 0.5 e 2.5

sub1.bar(grupos, valores, largura, color='blue')

sub1.set_xticks([0.5,2.5,4.5])

sub1.set_xticklabels(nome_grupos, size =24)



# Usando exatamente os mesmos valores, vamos agora gerar um gráfico de barras horizontais

sub2.barh(grupos, valores, largura, color='red')

sub2.set_yticks([0.5,2.5,4.5])

sub2.set_yticklabels(nome_grupos, size =24)



plt.show()
# Por último, vamos criar um gráfico pizza (pie plot) com os mesmo dados do gráfico de barras.



fig = plt.figure(figsize=(10,9))



patches, texts, autotexts = plt.pie(valores, explode=(0,0,0.08), labels=nome_grupos, shadow=True, autopct='%1.1f%%', colors = ['red','gray','blue'])



# A variável text possui as informações relativas aos labels, logo: 

texts[0].set_fontsize(20)

texts[0].set_color('red')

texts[1].set_fontsize(20)

texts[1].set_color('gray')

texts[2].set_fontsize(20)

texts[2].set_color('blue')



# A variável autotext possui as informações relativas aos textos gerados automaticamente.

# No nosso exemplo, ela contem as informações dos percentuais

autotexts[0].set_fontsize(30)

autotexts[1].set_fontsize(30)

autotexts[2].set_fontsize(30)

autotexts[2].set_color('white')



plt.title('Exemplo de gráfico Pizza', size=24);



plt.show()