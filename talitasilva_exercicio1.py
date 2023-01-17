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
alunoL = []

class Aluno:

    nome = ""

    nota = 0.0

    matricula = ""  



def dicionario():

    dados = {}

    

    ret = maiorNota()

    

    dados["Maior"] = nomeAluno(ret)

    dados["Media"] = mediaNotas()

    dados["Menor"] = menorNota()

    dados["Total"] = len(alunoL)

    dados["Lista"] = listar()

    

    return dados

    



def validar_matricula(matricula):

    for x in alunoL:

        if matricula in x.matricula:  

            return 1

    return 0



def cadastrar():      

    retorno = 0

    print("Nome: ")

    aluno.nome = input("")

    print("Nota: ")

    aluno.nota = input("")

    

    while True:

        print("Matricula: ")

        mat = input("")

        retorno=validar_matricula(mat)

        if retorno == 1:

            print("Matricula já existe!")

        else:

            aluno.matricula = mat

            break

    

    alunoL.append(aluno)  



def maiorNota():

    maior=max(float(x.nota) for x in alunoL)

    return maior



def nomeAluno(ret):

    for x in alunoL:

        if ret == float(x.nota):

            #print("Aluno com maior nota: %s" % x.nome)

            return x.nome



def mediaNotas():

    media=sum(float(x.nota) for x in alunoL)/len(alunoL)

    return media



def menorNota():

    menor=min(float(x.nota) for x in alunoL)

    return menor



def listar():

    for x in alunoL:

        print("Nome: %s" % x.nome)

        print("Matricula: %s" % x.matricula)

        print("Nota: %.1f\n" % float(x.nota))

        

def dados():

    print("\n\nDADOS")

    

    dic = dicionario()

    

    #Maior

    print("Aluno com maior nota: %s" % dic["Maior"])

     

    #Media

    print("Media: %.1f" % dic["Media"])

    

    #Menor

    print("Menor nota: %s" % dic["Menor"])

    

    #Quantidade

    print("Número de alunos cadastrados: %d" % dic["Total"])



    #Listagem

    print("\nListagem de alunos")

    listar()



    

x=0

for x in range(0,2):

    aluno = Aluno()

    cadastrar()

    

dados()