# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
lAluno = []



class Aluno:

    Nome=""

    Nota=0.0

    Matrícula=""

    

def tudao():

        aurelio={}

        aurelio["lista"] = lAluno

        aurelio["nomeMN"]= nomeMN()

        aurelio["maiorN"]= maiorN()        

        aurelio["menorN"]=menorN()

        aurelio["media"]=media()        

        aurelio["total"]=len(lAluno)

        

        #print(aurelio)

        

        return aurelio

       

def maiorN():

     return max(x.nota for x in lAluno)



def nomeMN():

    for x in lAluno:

        if x.nota==maiorN():

            nomeM=x.nome

            return nomeM

        

def menorN():

    return min(float(x.nota) for x in lAluno)

        

def media():

    return (sum(float(x.nota) for x in lAluno)/len(lAluno))

    



def validaM(mat):

    for x in lAluno:

        if mat in x.matrícula:

            return 1

        return 0

    

def cadastra():

    print("Nome:")

    aluno.nome=input()

    print("Nota:")

    aluno.nota=input() 

    

    while True:

        print("Matrícula:")

        m=input()

        r=validaM(m)

        if r == 1:

            print("\nHey, essa matrícula já existe!\n")   

        else:

            aluno.matrícula=m

            break

    

    lAluno.append(aluno)

    

def dados():

    v=tudao()

    #Maior nota        

    print("Nome: %s" %v["nomeMN"])

    print("Maior nota: %.1f" %float(v["maiorN"]))

    

    #Menor nota

    print("Menor nota: %.1f" %v["menorN"])

    

    #Média das notas

    print("Média das notas: %.1f" %v["media"])

    

    #Total de alunos

    print("Total de alunos: %d" %v["total"])    

    

def listar():

    for x in lAluno:

        print("Nome: %s" %x.nome)

        print("Nota: %.1f" %float(x.nota))

        print("Matrícula: %s" %x.matrícula)

        

for x in range(0,2):

    aluno = Aluno()

    cadastra()



    

print("\n")

dados()

print("\n")

listar()

print("\n")