import pandas as pd


df = pd.read_csv("../input/heightweightcsv/Height_weight.csv")
df.keys()
# y = mx + c    y = dependent x = ind
x = df.Height

y = df.Weight

x.values
x = df.iloc[:, 0:1].values  

y = df.iloc[:, 1].values

x
# linear regression

# sklearn



from sklearn.linear_model import LinearRegression
MachineBrain = LinearRegression()  #class

 

MachineBrain.fit(x, y)


m = MachineBrain.coef_   #slope

c = MachineBrain.intercept_ #intercept
y_pred = m*1.47+c

y_pred
# prediction with real(training) data

y_predict = MachineBrain.predict(x)
y_predict
# prediction with new data

h1 = 1.71

h2 = 1.62

w = MachineBrain.predict([[h1], [h2]])

w
import matplotlib.pyplot as plt
plt.scatter(x,y)

plt.scatter([h1, h2],w, color = ["green", "yellow"])

plt.plot(x, y_predict, c = "red")

plt.xlabel("Height")

plt.ylabel("Weignt")

plt.show()