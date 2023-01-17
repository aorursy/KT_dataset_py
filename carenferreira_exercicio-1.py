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
#A matricula teria o numero do aluno com auto incremento, gerado automticamente a cada aluno cadastrado

class Matricula:

    curso=""

    ano= int

    numAluno= 0

    def imprime(matricula):

        return "%s-%s%s" % (matricula.curso, matricula.ano, matricula.numAluno)

    

class Aluno:

    nome=""

    nota= int

    matricula=Matricula()



cont=0

alunos=[]



while cont<3:

    aluno = Aluno()

    aluno.nome = input("Nome:")

    aluno.nota = int(input("Nota:"))

    aluno.matricula.curso = input("Curso:")

    aluno.matricula.ano = input("Ano:")

    aluno.matricula.numAluno+=1

    alunos.append(aluno)

    cont+=1

    

def maiorNota(alunos,maior):

    cont=1

    for aluno in alunos:

        if cont == 1:

            maior = aluno.nota

            cont=0

        else:

            if aluno.nota > maior:

                maior = aluno.nota

                nome = aluno.nome

    return nome



def menorNota(alunos,menor):

    cont=1

    for aluno in alunos:

        if cont == 1:

            menor = aluno.nota

            cont=0

        else:

            if aluno.nota < menor:

                menor = aluno.nota

    return menor



def calcMedia(alunos):

    soma=0

    for aluno in alunos:

        soma += aluno.nota

        media = soma/len(alunos)

    return media



def listar(alunos):

    for aluno in alunos:

        print('Nome:{}'.format(aluno.nome))

        print('Nota:{}'.format(aluno.nota))

        print(aluno.matricula.imprime())

    

infoAlunos={}

def dAlunos(alunos, maior,menor,infoalunos):

    infoalunos["Nome maior nota"]=maiorNota(alunos,maior)

    infoalunos["Media nota"]=calcMedia(alunos)

    infoalunos["Menor nota"]=menorNota(alunos,menor)

    infoalunos["Qtd alunos"]=len(alunos)

    infoalunos["Lista alunos"]=listar(alunos)

    return infoAlunos



print("testando a função nome do aluno c/ maior nota: {}".format(maiorNota(alunos,maior)))

print("testando a função menor nota: {}".format(menorNota(alunos,menor)))

print("media:{}".format(calcMedia(alunos)))

print("dicionario:\n{}".format(dAlunos(alunos,maior,menor,infoAlunos)))



    

    
