li = ["abc", 34, 4.34, 23]

print(li)
st = "Hello World"



print(st)



st = 'Hello World'



print(st)

  

st = """This is a multi-line string that uses triple quotes. """



print(st)

  

tu = (23, 'abc', 4.56, (2,3), 'def')

print(tu[1])

print(tu[-1])
for fruit in ['apple','banana','mango']:

    print("I like",fruit)
import numpy as np

np_2d = np.array([[1.73, 1.68, 1.71, 1.89, 1.79],

 [65.4, 59.2, 63.6, 88.4, 68.7]]) 
np_2d.shape 
import numpy as np

np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])

np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7]) 



bmi = np_weight / np_height ** 2



print(" BMI : ", bmi)
import numpy as np



np_city = np.array([[ 1.64, 71.78],

[ 1.37, 63.35],

[ 1.6 , 55.09],

[ 2.04, 74.85],

[ 2.04, 68.72],

[ 2.01, 73.57]])



print(np_city)

print(type(np_city))
print("Mean Height : ",np.mean(np_city[:,0]))
print("Median Height : ",np.median(np_city[:,0]))
np.corrcoef(np_city[:,0], np_city[:,1])
np.std(np_city[:,0])
fam = [1.73, 1.68, 1.71, 1.89]

tallest = max(fam)

print("Tallest : ", tallest)
height = np.round(np.random.normal(1.75,0.20,5000),2)

weight = np.round(np.random.normal(60.32,15,5000),2)

np_city = np.column_stack((height,weight))

print(np_city)