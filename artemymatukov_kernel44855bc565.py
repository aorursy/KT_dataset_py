# Creación de gráficos.
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
# Manipulación de datos / álgebra lineal.
import numpy as np
import pandas as pd
# Utilidades.
from sklearn.preprocessing import scale, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
# Algoritmos
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier as xgb

from catboost import CatBoostClassifier as catb
# Otros
import warnings
telemetry = pd.read_xlsx("../input/artbad1/00.Fields  Master Data.xls", error_bad_lines=False)
