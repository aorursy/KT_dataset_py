import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error
data = pd.read_csv("../input/heights-and-weights/data.csv")
data.head(10)
data.info()
plt.figure(figsize = (16,9))
plt.scatter(data["Height"] , data["Weight"] , color = 'green')
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()
data["BMI"] = data["Weight"] / (data["Height"] * data["Height"])
data.head(10)
x = data.iloc[:,0:-1].values
y = data.iloc[:,-1].values
train_x , test_x , train_y , test_y = train_test_split(x,y,test_size = 0.2)
model = LinearRegression()
model.fit(train_x , train_y)
pred = model.predict(test_x)
print(r2_score(test_y , pred))