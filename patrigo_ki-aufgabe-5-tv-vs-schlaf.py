import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

data = np.array([ [0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8],

    [1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])

# Sortieren abhängig von x, damit es später einfacher "beim Denken" wird

# Achtung: np.sort macht nicht das, was sie erwarten würden... gar nicht ;)

data = data[data[:,0].argsort()]

x = data[:,0] # Get value in all tuples at position 0 (=Fernsehzeit)

y = data[:,1] # Get value in all tuples at position 1 (=Dauer Tiefschlaf)
def create_model():
    global x,y
    areas = []
    coefficients = []
    constants = []
    
    for i in range(0, len(x) - 1):
        areas.append([x[i],x[i+1]])
        coefficients.append((y[i+1]-y[i])/(x[i+1] - x[i]))
        constants.append(y[i] - coefficients[i] * x[i])
        
    for n in areas:
        n[0] = float("{0:.2f}".format(n[0])) # kürzt die float-Nummer auf eine Nachkommastellen
        n[1] = float("{0:.2f}".format(n[1]))
    return areas, coefficients, constants
def get_values(stellen):
    global areas, coefficients, constants
    results = []
    for stelle in stellen:
        result = None
        i = 0
        while i < len(areas) and result == None:
            if stelle >= areas[i][0] and stelle <= areas[i][1]:
                result = coefficients[i] * stelle + constants[i]
            i = i + 1
        if result == None:
            result = coefficients[i - 1] * stelle + constants[i - 1]
        results.append(result)
    k = 0
    while k < len(results):
        zahl = results[k]
        zahl = int(zahl * 100)
        results[k] = zahl / 100
        k = k + 1
    return results
areas, coefficients, constants = create_model()

times = []
i = 0.5
while(i <= 4):
    times.append(i)
    i = i + 0.5

plt.plot(x,y,'o') # Plotten der erhobenen Datenpunkten
values = get_values(x)
plt.plot(x,values,'--') # Plotten der lineare Verbindungen der erhobenen Datenpunkte. Dies nutzt die get_values Funktion um sicherzugehen, dass die Steigungen zumindest auf diese Punkte zutreffen
predictions = get_values(times)
plt.plot(times,predictions,'rx',markersize=8) # Plotten der 30 Minuten-Werte

d = {'TV Zeiten': times, 'Tiefschlaf in Stunden': predictions}
pd.DataFrame(data=d)