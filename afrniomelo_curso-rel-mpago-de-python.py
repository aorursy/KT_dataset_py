import numpy as np

import scipy.integrate

import matplotlib.pyplot as plt

%matplotlib inline



# definindo o sistema de EDO's

def rigid (y, t):

    return [ y[1]*y[2], -y[0]*y[2], -0.51*y[0]*y[1] ]



# faixa de tempo

t = np.arange(0.0, 12.1, 0.1)



# condição inicial

x0 = [0.0, 1.0, 1.0]



# resolvendo!!

x = scipy.integrate.odeint(rigid, x0, t) 



# plotando

plt.plot(t,x)

plt.xlabel ("Tempo, t (s)")

plt.ylabel ("Posição, x (m)");
obj = 2
obj = 3
id(obj)
type(obj)
novo_nome = 3
id(obj)
id(novo_nome)
dir()
# Este é um comentário. Comentários são linhas que começam com hashtag (#)

# Comentários são ignorados pelo interpretador do Python.

# Eles só servem para o programador se comunicar com eventuais 

# leitores do código, como vc, caro amigo.



# abaixo temos comandos para realizar as operações aritméticas



a = 3

b = 4

c = a + b

d = a*b

e = c/d

f = b**a
print(c)

print(d)

print(e)

print(f)

print(c+d)
type(c)
type(d)
type(e)
oi = 2.5

type(oi)
ai = 1.0 - 2.0j

type(ai)
ei = complex(2.0,3.0)

type(ei)
print(ai+ei)
import numpy as np
valor1 = np.cos(0)

valor2 = np.sin(np.pi)



print(valor1)

print(valor2)
lista_bolada = [1,2,3]
elemento1 = lista_bolada[0]

elemento2 = lista_bolada[1]

elemento3 = lista_bolada[2]
print(elemento1)

print(elemento2)

print(elemento3)
ult = lista_bolada[-1]

penult = lista_bolada[-2]

antepenult = lista_bolada[-3]



print(ult)

print(penult)

print(antepenult)
A = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]



B = A[1:8:2]



print(B)
print(A[1: :2])
print(A[0:4])
print(A)
A[0:9:2] = [1, 3, 5, 7, 9]



print(A)
lista_sagaz = [1.0, 2.0-3j, np.cos]



x = lista_sagaz[2](np.pi)



print(x)
tupla_boladona = 1,2,3
tupla_mandada = (1,2,3)
a, b, c = 1, 2, 3
print(a)

print(b)

print(c)
a, b = b, a
print(a)

print(b)
tupla_guardada = (100,200,300,400,500)



print(tupla_guardada[0:3:2])
s1 = 'Oi! Eu sou uma string!'

type(s1)
s2 = "Olá. Eu também sou uma string."

type(s2)
print(s1,s2)
delta_S = 2.0

delta_t = 3.0



v = delta_S/delta_t



print("A velocidade média vale: ",v, "m/s.")
S = 'computador'



print(S[0:3], S[7:])
# dicionário armazenando massas molares de três substâncias

# os índices dos objetos são strings contendo as fórmulas das substâncias

massas_molares = {'H20': 18, 'CO2': 44, 'H2': 2}



print(massas_molares['CO2'])
# dicionário armazenando os vencedores da copa do mundo fifa, de 1990 a 2014

# os índices dos objetos são números inteiros representando os anos das copas



vencedores_copas = {1990: 'Alemanha', 1994: 'Brasil', 1998: 'França',\

                    2002: 'Brasil', 2006: 'Itália', 2010: 'Espanha',\

                    2014: 'Alemanha', 2018: 'França'}



print(vencedores_copas[2014])
A = np.array([1,2,3,4])



print(A)
B = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])



print(B)
print(B[0,2])
B[0:2,1:4]
B[:,3]
B[0,:]
A = np.arange(0.0, 5.0, 0.5)

print(A)
B = np.linspace(0.0, 5.0, 3)

print(B)
C = np.zeros(3)

print(C)
D = np.zeros((2,3))

print(D)
E = np.ones((4,5))

print(E)
F = np.zeros_like(E)

print(F)
G = np.random.standard_normal((2,3))

print(G)
A = np.array([[1,2],[3,4]])

print(A)
B = np.array([[5,6],[7,8]])

print(B)
C = A@B

print(C)
D = A*B

print(D)
E = D.T

print(E)
F = np.linalg.inv(A)

print(F)
G = np.linalg.norm(B)

print(G)
H = np.linalg.eigvals(A)

print(H)
import matplotlib.pyplot as plt
x = np.linspace(-2*np.pi,2*np.pi,50)
y = np.cos(x)
plt.plot(x,y);
plt.plot(x,y,'*r');
help(plt.plot)
# definindo pontos do eixo x

x = np.arange(0.0, 1.0, 0.001)



# plotando quatro curvas

plt.plot(x, x**2, label ='$y = x^2$')

plt.plot(x, x**3, label = '$y = x^3$')

plt.plot(x, x**(1/2), label = '$y = \sqrt{x}$')

plt.plot(x, x**(1/3), label='$y = \sqrt[3]{x}$')



# limites dos eixos: função axis

# o argumento deve ser uma lista na forma [xmin, xmax, ymin, ymax]

plt.axis([0, 1, 0, 1])



# título do gráfico: função title

plt.title('Gráficos boladões')



# legendas dos eixos: funções xlabel e ylabel

plt.xlabel('Eixo x boladão')

plt.ylabel('Eixo y boladão')



# legendas das curvas: função legend

plt.legend();
from mpl_toolkits.mplot3d import Axes3D
x = y = np.linspace(-4, 4, 50)

X, Y = np.meshgrid(x, y)
Z = (X**2 + Y**2)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_surface(X, Y, Z);
T = float(input("Insira uma temperatura em graus Celsius, por favor: "))



if T<-273.15:

    print('Temperatura abaixo do zero absoluto!')

elif T>-273.15 and T<0:

    print('Água no estado sólido.')

elif T==0:

    print('Água no equilíbrio sólido-líquido.')

elif T>0 and T<100:

    print('Água no estado líquido.')

elif T==100:

    print('Água no equilibrio líquido-vapor.')

else:

    print('Água no estado vapor.')
if (100>2000):

    print('se esta string for impressa, significa que 100 é maior que 2000!')

    print('se o python imprimir essa string, ele ficou doido!')

print('essa string vai ser impressa pq não faz parte do if.')
i = 10



while i>0:

    print(i)

    i = i-1

    

print('acabou!')
sequencia = [1,2,3,4,5]



for i in sequencia:

    print(i)

    

print('acabou!')
palavra = 'ain'



for cada_letrinha in palavra:

    print(cada_letrinha)
a = range(1,11,2)



print(a)

print(a[0])

print(a[1])

print(a[2])

print(a[3])

print(a[4])
L = [1,2,3,4,5]

soma = 0



for i in range(len(L)):

    soma = soma+ L[i]**2



print(soma)
str = 'Nao gosto da letra a. Nao gosto que imprimam a letra a'



for s in str:

    if s == 'a':

        continue

    print(s,end='')
str = 'Nao gosto da letra a. Nao gosto que imprimam a letra a'



for s in str:

    if s == 'a':

        break

    print(s,end='')
a = [i for i in range(10)]

a
b = [i for i in range(10) if i%2!=0]

b
def somar2 (x):

    y = x+2

    return y
somar2(4)
k = 20

z = somar2(k)

print(z)
y = np.zeros((4,3))

x = somar2(y)

print(x)
def funcao_bolada (arg1, arg2):

    

    arg1 = 1.0

    

    for i in range(len(arg2)):

        arg2[i] = 1.0

        

arg1 = 0.0

arg2 = np.zeros((2,2))



funcao_bolada(arg1, arg2)



print(arg1)

print(arg2)
def f(t, A=1, a=1, omega=2*np.pi):

    return A*np.exp(-a*t)*np.cos(omega*t)
result1 = f(1)

result2 = f(1, a=2)

result3 = f(1, omega=4*np.pi, A=4)



print(result1, result2, result3)
A = np.array([[3.5, 2.0, 0.0],[-1.5, 2.8, 1.9],[0, -2.5, 3.0]])

b = np.array([5,-1,2])



x = np.linalg.solve(A,b)



print(x)
import scipy.optimize
# definindo o sistema de equações como uma função do Python



def func (x):

    return [x[0]/(1.0+np.exp(-27.*x[1])*(x[0]/3.-1.))-5, 

            x[0]/(1.0+np.exp(-39.*x[1])*(x[0]/3.-1.))-6]

    

# estimativa inicial 

    

x0 = [10, 0.1]

    

# resolvendo!    

    

result = scipy.optimize.root(func, x0)    



# imprimindo resultado



print(result)
print(result.x)
import scipy.integrate
# definindo o sistema de equações

def dCdt (C,t,k1,k2):

    return [-k1*C[0], k1*C[0]-k2*C[1], k2*C[1]]

    

# parâmetros

k = (2,1)

    

# pontos no tempo

t = np.arange(0.0,5.1,0.1)



# condições iniciais

c0 = [5, 0, 0]



# resolvendo!

c = scipy.integrate.odeint(dCdt,c0,t,args=k)



#plotando

plt.plot(t,c)



# ajeitando o gráfico

plt.xlabel('$t$ (h)')

plt.ylabel('$C$ (mol/L)')

plt.legend(['$C_A$','$C_B$','$C_C$'])

plt.axis([0,5,0,5]);
from matplotlib import cm



x = np.linspace(-2,2,20)

y = np.linspace(-1,3,20)

X,Y = np.meshgrid(x,y)



Z = 100*(Y-X**2)**2 + (1-X)**2



fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_surface(X, Y, Z, cmap=cm.rainbow)



ax.set_xlabel('$x$')

ax.set_ylabel('$y$')

ax.set_zlabel('$f(x,y)$');
def rosenbrock (x):

    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
# estimativa inicial 

x0 = [0, 0]



# minimizando!

resultado = scipy.optimize.minimize(rosenbrock, x0)



print(resultado)
x = y = np.linspace(-512,512,1000)

X,Y = np.meshgrid(x,y)



Z = -(Y+47)*np.sin((abs(X/2+Y+47))**0.5) - X*np.sin((abs(X-(Y+47)))**0.5)



fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.plot_surface(X, Y, Z, cmap=cm.inferno);
def caixa_de_ovos(x):

    return (-(x[1]+47)*np.sin((abs(x[0]/2+x[1]+47))**0.5) 

           - x[0]*np.sin((abs(x[0]-(x[1]+47)))**0.5))
# limite de busca das variáveis

bounds = ((-512,512),(-512,512))



# minimizando!

resultado = scipy.optimize.differential_evolution(caixa_de_ovos, bounds)



print(resultado)
resultado = scipy.optimize.differential_evolution(caixa_de_ovos, bounds, 

                                                  popsize=30, strategy = 'best2bin')



print(resultado)
import pandas as pd
obj1 = pd.Series([5,6,8,9])

print(obj1)
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])

print(obj2)
print(obj1.values)

print(obj1.index)
print(obj2.values)

print(obj2.index)
obj2['b']
obj2['e'] = 9

obj2
sdata = {'Rio de Janeiro': 35000, 'São Paulo': 71000, 'Minas Gerais': 16000, 'Espírito Santo': 5000}

obj3 = pd.Series(sdata)

print(obj3)
states = ['São Paulo', 'Ceará', 'Rio de Janeiro', 'Rio Grande do Sul']

obj4 = pd.Series(sdata, index=states)

print(obj4)
obj3+obj4
nomes =['Tamefreu Chopin','Gumercindo Carrara','Acheropita Pacífico','Afrânia Salgueiro']



dados = {'P1': [8.0, 5.5, 4.3, 10.0],

         'P2': [2.0, 6.0, 7.4, 10.0]}



df1 = pd.DataFrame(dados, index = nomes)

df1
df1 = df1.sort_index()

df1
df1['P1']
df1['P2'].values
df1.loc['Acheropita Pacífico']
df1.loc['Gumercindo Carrara','P1']
df1['Média'] = 0.5*(df1['P1']+df1['P2'])

df1
df1['PF'] = [6.0,np.nan,1.5,4.9]



df1['Média Final'] = [0.5*(df1['Média'].loc[x]+df1['PF'].loc[x]) if pd.notnull(df1['PF'].loc[x]) \

                                                                 else df1['Média'].loc[x] \

                                                                 for x in df1.index]



df1['Situação'] = ['Aprovado(a)' if df1['Média Final'].loc[x]>=5 else 'Reprovado(a)' for x in df1.index]



df1
df1 = df1.sort_values(by='Média Final',ascending=False)

df1
df1.loc['Média da turma'] = df1.mean()

df1.loc['Desvio-padrão da turma'] = df1.std()

df1
df1.round(2)