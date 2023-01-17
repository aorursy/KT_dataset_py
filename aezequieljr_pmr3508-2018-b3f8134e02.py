# Importando o que utilizaremos no programa
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
# Realizando leitura da base de treinamento
adult = pd.read_csv("../input/myadultdb/train_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")
# Formato da base
adult.shape
# Estrutura da base
adult.head()
# Retirando linhas com dados faltantes
nadult = adult.dropna()
# Realizando leitura da base de teste
testAdult = pd.read_csv("../input/myadultdb/test_data.csv", sep=r'\s*,\s*', engine='python', na_values="?")