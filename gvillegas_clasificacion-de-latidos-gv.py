# Cargo los datos de train

import pickle
with open("../input/ecg-challenge-dataset/signals_train.pkl", 'rb') as f:
        lat=pickle.load(f)
with open("../input/ecg-challenge-dataset/annotations_train.pkl", 'rb') as f:
        ann=pickle.load(f)
with open("../input/ecg-challenge-dataset/patients_train.pkl", 'rb') as f:
        pat=pickle.load(f)
# Grafico el latido nro lat_idx
import numpy as np
from random import randint
import matplotlib.pyplot as plot
lat_idx=randint(0,len(lat))
f, axarr = plot.subplots(12, sharex=True)
f.set_figheight(15)
f.set_figwidth(15)
print("Latido nro: {}".format(lat_idx))
print("El latido es del tipo: {}".format(ann[lat_idx]))
print("El latido pertence al paciente {}".format(pat[lat_idx]))
for i in range(12):
    axarr[i].plot(lat[lat_idx].T[i])
plot.show()

# Cargo los datos de test

import pickle
with open("../input/ecg-challenge-dataset/signals_test.pkl", 'rb') as f:
        lat=pickle.load(f)
with open("../input/ecg-challenge-dataset/patients_test.pkl", 'rb') as f:
        pat=pickle.load(f)
# Grafico el latido nro lat_idx
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plot
lat_idx=randint(0,len(lat))
f, axarr = plot.subplots(12, sharex=True)
f.set_figheight(15)
f.set_figwidth(15)
print("Latido nro: {}".format(lat_idx))
print("El latido pertence al paciente {}".format(pat[lat_idx]))
for i in range(12):
    axarr[i].plot(lat[lat_idx].T[i])
plot.show()
