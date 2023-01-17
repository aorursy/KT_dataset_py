import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = np.array([[0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8],[1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])
# Sortieren abhängig von x, damit es später einfacher "beim Denken" wird
# Achtung: np.sort macht nicht das, was sie erwarten würden... gar nicht ;)
data = data[data[:,0].argsort()]
tv = data[:,0] # Get value in all tuples at position 0 (=Fernsehzeit)
sleep = data[:,1] # Get value in all tuples at position 1 (=Dauer Tiefschlaf)
plt.plot(tv, sleep, 'ko')
plt.title("Data")
pd.DataFrame(data = {'TV Zeiten': tv, 'Tiefschlaf': sleep})
model = linear_model.LinearRegression().fit(tv.reshape(-1,1), sleep.reshape(-1,1))
lineX = np.array([i * 0.5 for i in range(0,9)])
lineY = model.predict(lineX.reshape(-1,1)).reshape(-1)
plt.plot(tv, sleep, 'ko', lineX, lineY, 'b-.')
plt.title("f(x) = " + str((lineY[1] - lineY[0]) / 4) + "x + " + str(lineY[0]))
plt.show()
resX = [lineX[i] for i in range(1, len(lineX))]
resY = [lineY[i] for i in range(1, len(lineY))]
plt.title("line with results")
plt.plot(lineX, lineY, 'b-.', resX, resY, 'gd')
pd.DataFrame(data = {'TV Zeiten': resX, 'Tiefschlaf': resY})