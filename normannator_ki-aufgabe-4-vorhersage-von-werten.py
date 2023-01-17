import numpy as np
import matplotlib.pyplot as plt

data = np.array([ [0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8],
    [1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])

# Sortieren abhängig von x, damit es später einfacher "beim Denken" wird
# Achtung: np.sort macht nicht das, was sie erwarten würden... gar nicht ;)

data = data[data[:,0].argsort()]
print(data)
x = data[:,0] # Get value in all tuples at position 0 (=Fernsehzeit)
y = data[:,1] # Get value in all tuples at position 1 (=Dauer Tiefschlaf)

plt.scatter(x,y) # 
plt.show()
data = data.tolist()
def bestimmePunkt(aktx):
    global data
    nachbarpunkte = []
    index = -1
    for x,y in data:
        if(aktx < x):
            index=data.index([x,y])
            break
        elif(aktx == x):
            return y
        else:
            continue
    #Ende Schleife
    if(index == -1):
        return data[-1][1] #y Wert des letzten Elements der Liste
    elif(index == 0):
        return data[0][1]
    else:
        punkt1 = data[index-1]
        punkt2 = data[index]
        steigung = (punkt2[1]-punkt1[1])/(punkt2[0]-punkt1[0])
        Yaa = punkt1[1] - (punkt1[0]*steigung)
        return aktx * steigung + Yaa

#tabellarische Darstellung
resultData = []
print(" x " + "   |   " + " y ")
print("_____________")
for i in range(1,5,1):
    punkt1 = [i-0.5,bestimmePunkt(i-0.5)]
    punkt2 = [float(i),bestimmePunkt(i)]
    print(str(i-0.5) + "   |   " + str(punkt1[1]))
    print(str(float(i)) + "   |   " + str(punkt2[1]))
    resultData.append(punkt1)
    resultData.append(punkt2)
data = np.array([ [0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8],
    [1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])

# Sortieren abhängig von x, damit es später einfacher "beim Denken" wird
# Achtung: np.sort macht nicht das, was sie erwarten würden... gar nicht ;)

data = data[data[:,0].argsort()]
#print(data)
x = data[:,0] # Get value in all tuples at position 0 (=Fernsehzeit)
y = data[:,1] # Get value in all tuples at position 1 (=Dauer Tiefschlaf)
plt.scatter(x,y,edgecolors="b") # 

resultData = np.array(resultData)
resultData = resultData[resultData[:,0].argsort()]
#print(resultData)
resX = resultData[:,0]
resY = resultData[:,1]
plt.scatter(resX,resY,edgecolors="r")

plt.show()