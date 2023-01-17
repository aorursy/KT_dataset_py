import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
 %matplotlib inline 
# Le 6 colonne sono le 6 braccia

# Gli indici nelle posizioni sono i tempi in nsecs

posizioni = pd.read_pickle("/kaggle/input/analisi-velocit-robot/Sezione_critica")

posizioni
posizioni.plot()
posizioni.iloc[57:60].plot()
# Nelle velocità gli indici sono il numero del campione perchè ho sbagliato a salvarli

velocita = pd.read_pickle("/kaggle/input/analisi-velocit-robot/velocita_critica")

velocita
velocita.index = posizioni.index

velocita
velocita.plot()
velocita.iloc[56:60].plot()
qua = pd.DataFrame()

# Dei 6 bracci testo sul primo, nessuno degli altri da risultati più significativi

qua["effettivo"] = posizioni[0].iloc[49:61]

qua["quadrato_precedente"] = posizioni[1].iloc[49:61].pow(2).shift(1)

qua["differenza"] = qua.effettivo - qua.quadrato_precedente

qua["errore %"] = qua.differenza / qua.effettivo

qua.iloc[1:]
qua = pd.DataFrame()

qua["effettivo"] = velocita[0].iloc[49:61]

qua["quadrato_precedente"] = velocita[1].iloc[49:61].pow(2).shift(1)

qua["differenza"] = qua.effettivo - qua.quadrato_precedente

qua["errore %"] = qua.differenza / qua.effettivo

qua.iloc[1:]
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(0,10,10),velocita[0].iloc[49:59])

r_value**2
p_value
slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(0,10,10)**2,velocita[0].iloc[49:59])

r_value**2
slope, intercept, r_value, p_value, std_err = stats.linregress(np.exp(np.linspace(0,10,10)),velocita[0].iloc[49:59])

r_value**2
p_value
posizioni_sess2 = pd.read_pickle("/kaggle/input/sessione-2-robot/sessione_2_posizioni")

velocita_sess2 = pd.read_pickle("/kaggle/input/sessione-2-robot/sessione_2_velocita")

tempo_sess2 = pd.read_pickle("/kaggle/input/tempi-sessione-2/sessione2_tempi")
max_val = 1744

posizioni_sess2.iloc[(max_val-12):max_val]
velocita_sess2.iloc[(max_val-12):max_val]
confronto = pd.DataFrame()

confronto["pos_primo"] = posizioni[0].iloc[49:61].reset_index(drop=True)

confronto["pos_secondi"] = posizioni_sess2.iloc[(max_val-12):max_val][0].reset_index(drop=True)

confronto["vel_primo"] = velocita[0].iloc[49:61].reset_index(drop=True)

confronto["vel_secondi"] = velocita_sess2.iloc[(max_val-12):max_val][0].reset_index(drop=True)

confronto