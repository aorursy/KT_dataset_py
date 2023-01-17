import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
data = np.array([ [0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8], [1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])
data = data[data[:,0].argsort()]
x = data[:,0] # Get value in all tuples at position 0 (=Fernsehzeit)
y = data[:,1] # Get value in all tuples at position 1 (=Dauer Tiefschlaf)
plt.xlabel("TV Zeit")
plt.ylabel("Tiefschlafzeit")
plt.grid(True)
plt.scatter(x,y) # 
plt.plot(x,y)
plt.show()
data = np.array([ [0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8], [1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])
sumTV = 0
sumSchlaf = 0
for i in data :
    sumTV += i[0]
    sumSchlaf += i[1]
avgTV = sumTV / len(data)
avgSchlaf = sumSchlaf / len(data)
minTV = avgTV / (avgTV*10)
minSchlaf = avgSchlaf / (avgTV*10) 
print("Die durchschnittliche Tiefschlafzeit beträgt: ", avgSchlaf)
print("Die durchschnittliche TV-Zeit beträgt: ",avgTV)
print("Das macht pro 0.1 TV-Zeit durchschnittlich" , minSchlaf, "Stunden Tiefschlaf")
def schaetze(zeit):
    schaetzung =  minSchlaf * (zeit/minTV)
    return schaetzung
i = 0.5
schaetzungen = []
zeit = []
while i <= 4.0:
    zeit.append(i)
    schaetzungen.append(schaetze(i))
    i += 0.5
d = {'Zeit': zeit, 'Schätzung': schaetzungen}
df = pd.DataFrame(data=d)
df
x1 = zeit
y1 = schaetzungen
plt.xlabel("TV Zeit")
plt.ylabel("Tiefschlafzeit")
plt.grid(True)
plt.scatter(x1,y1)
plt.plot(x1,y1)
plt.show()
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
#Die Arrays müssen in 2D Arrays umgebaut werden, ansonsten funktioniert es nicht.
x2 = x.reshape(-1,1)
y2 = y.reshape(-1,1)
testintervall = np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
model = LinearRegression()
model.fit(x2, y2)
testintervall = testintervall.reshape(-1,1)
X_predict = testintervall
y_predict = model.predict(X_predict);
plt.xlabel("TV Zeit")
plt.ylabel("Tiefschlafzeit")
plt.grid(True)
plt.plot(testintervall,y_predict,"ro") # 
red_patch = mpatches.Patch(color='red', label='Schätzung')
blue_patch = mpatches.Patch(color='blue', label='TV-zeit')
plt.legend(handles=[red_patch,blue_patch])
plt.plot(x,y)
plt.show()
