! conda install -y gdown

import gdown

gdown.download('https://drive.google.com/uc?id=1eDhwxZxEnMoKcaR-ATW21ReuKdwpMdrF', 'airline-safety.csv', True);

gdown.download('https://drive.google.com/uc?id=14FYQvB05JqfU4d1Wn1LSv_Jh0om7AI0M', 'tips.csv', True);
import numpy as np
a = np.array([12.50, 10.20, 25])
a
b = np.array([

    [19.50, 13, 12, 11 ],

    [12.50, 10, 25, 14]])
b
import numpy as np

x = np.array([21, 24, 22, 23, 27])

y = np.array([4, 5, 6, 7, 8])

x - y
b.shape
np.linspace(0,100,6)
np.arange(0,10,3)
np.random.rand(4,5)
np.random.rand(4,5)*100
c = np.arange(0,10,3)
c.dtype
import numpy as np

x.shape

np.linspace(15, 45, 5)

np.random.rand(3, 6)

x = np.array([3,4,5])
y = np.array([1,2,3])
print(x + y)

print(x - y)

print(x * y)

print(x / y)

print(x**2)
np.add(x,1)
np.subtract(x,2)
np.multiply(x,3)
np.divide(x,4)
np.power(x,5)
s = np.arange(12)*5 # note this is same as np.arange(0,12,1)*5

s
s[2]
s[2:5]
np.concatenate((s,a))
z = np.arange(9.0)

z
split_arr = np.split(z, 3)
split_arr
np.array(split_arr).shape
oreo = np.arange(0,50,10)
oreo[2]
milk = np.arange(0,75,15)
milk[0]
import pandas as pd
decades = pd.Series(np.array([10,20,30,40,50,60])) 
decades
import numpy as np

import pandas as pd

gdp = np.array([264.50, 250.07, 248.84, 242.83])

country = pd.Series(gdp, index =['Czech Republic', 'Iraq', 'Romania', 'Portugal'])

print(country)
animal_dict = {

     'Animal' : ["Hamster", "Alligator", "Hamster","Cat", "Snake", "Cat","Hamster", "Cat", "Cat", "Snake", "Hamster", "Hamster", "Cat", "Alligator"],

     'Age' : [1,9,4,13,14,10,2,4,14,7,14,2,1,7],

     'Weight': [7,13,8,12,11,8,10,14,9,11,10,10,9,14], 

     'Length' : [8,6,9,1,8,9,5,6,6,6,5,3,4,5] 

}
animal = pd.DataFrame(animal_dict)

animal
pd.unique(animal["Animal"])
animal.describe(include=[np.number])
animal[animal["Weight"] > 10]
animal[(animal["Length"] > 4) & (animal["Length"] < 8)] 
animal_groups = animal.groupby("Animal")

animal_groups['Weight'].mean()
import pandas as pd

un_dict = {

    'Area': ["Northern Africa", "Sub-Saharan Africa", "Eastern Asia", "Western Europe"],

    'Population Rate of Increase': [1.7, 2.7, 0.6, 0.3],

    'Fertility Rate': [3.2, 5.6, 1.5, 1.6],

    'Infant Mortality': [39.3, 87.3, 23.2, 4.3]

}
un = pd.DataFrame(un_dict)

un
un[(un["Infant Mortality"] > 29) & (un["Infant Mortality"] < 51)]
un_groups = un.groupby("Area")

un_groups['Infant Mortality'].max()
import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[1,4,9,16]) # plt.plot([x-coordinates], [y-coordinates])

plt.show()
plt.plot([1,2,3,4],[1,4,9,16]) # plt.plot([x-coordinates], [y-coordinates])

plt.title("First Plot")

plt.xlabel("X Label")

plt.ylabel("Y Label")

plt.show()
import numpy as np

height = np.array([167, 170, 149, 165, 155, 180, 166, 146, 159, 185, 145, 168, 172, 181, 169])

weight = np.array([86,74,66,78,68,79,90,73,70,88,66,84, 67, 84, 77])



#We can set the limit (lower, upper) for the x-axis and the y-axis using xlim and ylim, respectively.

plt.xlim(140, 200)

plt.ylim(60,100)

#A scatterplot can be generated through .scatter(x,y)

plt.scatter(height,weight)

plt.title("Comparing Height vs. Weight")

plt.xlabel("Height")

plt.ylabel("Weight")

plt.show()
import numpy as np

length = np.array([8,6,9,1,8,9,5,6,6,6,5,3,4,5])

weight = np.array([7,13,8,12,11,8,10,14,9,11,10,10,9,14])



plt.xlim(0,10)

plt.ylim(5,15)



plt.scatter(length,weight)

plt.title("Weight v Length of Animals")

plt.xlabel("Length")

plt.ylabel("Weight")

plt.show()
import seaborn as sns
# Run this code to import the data and read it in using 'pd.read_csv('file')'

tips = pd.read_csv('tips.csv')

ax = sns.scatterplot(x="total_bill", y="tip", data=tips) #plotting it
sns.catplot(x="day", y="total_bill", data=tips);
sns.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=tips);
ax = sns.scatterplot(x="size", y="tip", data=tips)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston

data = load_boston()
# Print the description of the dataset (first 1500 characters)

print(data.DESCR[:1400])
# data.data are all the houses with their attributes.

# data.target are the corresponding house prices

 

print("House 1, attributes: ", data.data[0])



# Show the actual price for these 2 houses

print("House 1, price: ", data.target[0])
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

clf = LinearRegression()

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

expected = y_test
plt.figure(figsize=(4, 3))

plt.scatter(expected, predicted)

plt.plot([0, 50], [0, 50], '--k')

plt.axis('tight')

plt.xlabel('True price ($1000s)')

plt.ylabel('Predicted price ($1000s)')

plt.tight_layout()