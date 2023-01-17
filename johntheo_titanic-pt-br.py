import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler        #Normalização dos dados
from sklearn.decomposition import PCA                   #Principal Component Analysis
from sklearn.model_selection import train_test_split    #Separação em treino e teste 
from sklearn.linear_model import LogisticRegression     #Modelo de classificação
data = pd.read_csv('../input/train.csv')
data.head()