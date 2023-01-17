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
# questao 1



vetor_or = [];

vetor = [];

Q = 100; # quantia de números que serão lidos

flag = 0;



for i in range(0,Q,1):

    vetor_or.append(0);

    

for i in range(Q):

    print("Digite um número inteiro positivo");

    n = int(input());

    if(n >= 1 and n <= Q): 

        indice = n-1;

        if(vetor_or[indice] != 0):

            flag = 1;

        else:

            vetor_or[indice] = 1;

    else:

        flag = 1;



if(flag == 1):

    print("Não possui todos os números entre 1 e ",Q);

else:

    print("Possui sequência inteira entre entre 1 e ",Q);





        
# questao 3



Q = 100; # quantia de números que serão lidos

flag = 0;

vetor = [];



for i in range(Q):

    print("Digite um número");

    x = int(input());

    vetor.append(x);

    

for i in range(0,Q,1):    

    for j in range(i+1,Q,1):

        if(vetor[i] > vetor[j]):

            aux = vetor[i];

            vetor[i] = vetor[j];

            vetor[j] = aux;



for i in range(Q):

    print("Vetor[",i,"] = ",vetor[i]);
## questao 4



vetor = [];

flag = 1;



while(flag):

    print("Digite um número( -1 como condição de parada)");

    n = int(input());

    if(n == -1):

        flag = 0;

    else:

        vetor.append(n);  



ax = 0;

c = 1;

I =0;        

flag = 1;

Q = len(vetor)      

for i in range(0,Q,1):     

    for j in range(i+1,Q,1): 

        if((vetor[i] == vetor[j]) and (vetor[i+1] == vetor[j+1])):

            I = i+1;

            while(flag): 

                if(vetor[i+c] == vetor[j+c]):

                    c+=1;

                    aux = j+c

                    if(aux >= len(vetor)):

                        flag = 0;

                else:

                    flag = 0;

                    break;

            

    

        

print("\n");  

print("i = ",I," e m = ",c);

                        

            

        

    

    

        

    