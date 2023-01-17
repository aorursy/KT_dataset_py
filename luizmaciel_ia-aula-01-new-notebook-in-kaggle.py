# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# PARA COMENTAR UM CÓDIGO DEVEMOS UTILIZAR O CARACTERE "#"
# VARIÁVEIS EM PYTHON NÃO TEM DEFINIÇÃO DE TIPO.
a = 5
# PARA EXIBIR UM TEXTO OU VARIÁVEL EM PYTHON:
print(a)
a = "alguma coisa"
print(a)
# A IDENTAÇÃO EM PYTHON INDICA ONDE COMEÇA E TERMINA UM BLOCO
# O CARACTERE ":" INDICA O FIM DE UM COMANDO.
if a == 5:
    print("O valor de a é igual 5")
else:
    print("O valor de a é diferente de 5")