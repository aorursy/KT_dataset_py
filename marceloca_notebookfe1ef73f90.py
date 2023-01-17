# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def avaliarFx(x):
    a = (x ** 3 - 4 * x + 2)
    return a

def avaliarGx(x):
    a = (x - 2) * (-1 / 2) + 1
    return a

def AvaliarFG(x):
    return avaliarFx(x)-avaliarGx(x)

k = 0 # contador de iterações

def bisseçao(a, b, tolerancia):
    global k
    if AvaliarFG(a)*AvaliarFG(b)<0: # verifica se os extremos da função se cruzão
        c=(a+b)/2
        k+=1
        if AvaliarFG(c)==0:
            print("interações:", k)
            return c
        elif AvaliarFG(c)*AvaliarFG(a)<0: # verifica de que lado esta a intersseção
            if abs(c - a) < tolerancia:  # verifica se já esta no intervalo de tolerancia
                print("interações:", k)
                return c
            else:
                return bisseçao(a, c, tolerancia)
        else:
            if abs(b - c) < tolerancia:  # verifica se já esta no intervalo de tolerancia
                print("interações:", k)
                return c
            else:
                return bisseçao(c, b, tolerancia)
    else:
            print("não é possivel garantir uma intersseção nesse intervalo")

def falsoponto(a, b, tolerancia):
    global k
    if AvaliarFG(a)*AvaliarFG(b)<0: # verifica se as funções se cruzam no intervalo dado
        c = (a*AvaliarFG(b) - b*AvaliarFG(a))/(AvaliarFG(b) - AvaliarFG(a)) # devolve um ponto atraves de uma media ponderada
        k+=1
        if AvaliarFG(c)==0:
            print("iterações:", k)
            return c
        elif AvaliarFG(c) * AvaliarFG(a) < 0: # verifica se a intersseção esta na direita ou esquerda
            if abs(AvaliarFG(c)) < tolerancia:
                print("iterações:", k)
                return c
            else:
                return falsoponto(a, c, tolerancia)
        else:
            if abs(AvaliarFG(c))<tolerancia:
                print("iterações:", k)
                return c
            else:
                return falsoponto(c, b, tolerancia)
    else:
        print("não é possivel garantir uma intersseção nesse intervalo")

tolerancia = float(input("escreva a tolerancia:"))

a = float(input("escreva o valor de A: "))

b = float(input("escreva o valor de B: "))

print("bisseção")
resultado= bisseçao(a,b,tolerancia)
if resultado != None:
            print("x=",resultado, " y=", avaliarGx(resultado))
k=0
print("falso ponto")
resultado= falsoponto(a, b, tolerancia)
if resultado != None:
            print("x=",resultado, " y=", avaliarGx(resultado))