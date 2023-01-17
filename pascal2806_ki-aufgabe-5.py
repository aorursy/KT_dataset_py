import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go



data = np.array([ [0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8],

    [1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])

# Sortieren abhängig von x, damit es später einfacher "beim Denken" wird

# Achtung: np.sort macht nicht das, was sie erwarten würden... gar nicht ;)

data = data[data[:,0].argsort()]

#print(data)

x = data[:,0] # Get value in all tuples at position 0 (=Fernsehzeit)

y = data[:,1] # Get value in all tuples at position 1 (=Dauer Tiefschlaf)

plt.scatter(x,y) # 

plt.show()

def sage_wert_vorraus(wert):
    for i in range(len(data)):
        if(wert == data[i][0]):
            return data[i][1]
    m,b = berechne_funktion(wert)
    ergebnis = (m*wert) + b
    return ergebnis

def berechne_funktion(wert):
    if(wert <= data[len(data) - 1][0]):
        for i in range(len(data)):
            if(wert >= data[i][0]):
                kleiner = data[i]
                groesser = data[i + 1]
    else:
        kleiner = data[len(data) - 2]
        groesser = data[len(data) - 1]
    if(groesser[1] < kleiner[1]):
        m = (groesser[1] - kleiner[1]) / (groesser[0] - kleiner[0])
    else:
        m = (kleiner[1] - groesser[1]) / (groesser[0] - kleiner[0])
    b = kleiner[1] - (kleiner[0] * m)
    return m,b
    

tiefschlafzeiten = []
fernsehzeiten = []
fernsehzeit = 0.5
while(fernsehzeit <= 4):
    tiefschlafzeit = sage_wert_vorraus(fernsehzeit)
    fernsehzeiten.append(fernsehzeit)
    tiefschlafzeiten.append(tiefschlafzeit)
    print("Fernsehzeit: ",fernsehzeit, "Tiefschlafzeit: ",  round(tiefschlafzeit,2))
    fernsehzeit += 0.5
    
    


plt.scatter(fernsehzeiten,tiefschlafzeiten) # 

plt.show()


