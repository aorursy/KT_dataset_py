# import numpy library
import numpy as np
# Creating a rank 1 Array
arr = np.array([1, 2, 3])
print("Array with Rank 1: \n",arr)
# Creating a rank 2 Array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print("Array with Rank 2: \n", arr)
# Creating an array from tuple
arr = np.array((1, 3, 2))
print("Array created using "
      "passed tuple:\n", arr)
# Initial Array
arr = np.array([[-1, 2, 0, 4],
                [4, -0.5, 6, 0],
                [2.6, 0, 7, 8],
                [3, -7, 4, 2.0]])
print("Initial Array: ")
print(arr)
# Printing a range of Array
# with the use of slicing method
sliced_arr = arr[:2, ::2]
print ("Array with first 2 rows and"
    " alternate columns(0 and 2):\n", sliced_arr)
# Printing elements at
# specific Indices
Index_arr = arr[[1, 1, 0, 3], 
                [3, 2, 1, 0]]
print ("Elements at indices (1, 3), "
    "(1, 2), (0, 1), (3, 0):\n", Index_arr)
# Defining Array 1
a = np.array([[1, 2],
              [3, 4]])
 
# Defining Array 2
b = np.array([[4, 3],
              [2, 1]])
# Adding 2 to every element
print ("Adding 2 to every element:\n", a + 2)
# Subtracting 1 from each element
print ("\nSubtracting 1 from each element:\n", b - 1)
# sum of array elements
# Performing Unary operations
print ("\nSum of all array "
       "elements: ", a.sum())
# Adding two arrays
# Performing Binary operations
print ("\nArray sum:\n", a + b)
# Integer datatype
# guessed by Numpy
x = np.array([1, 2])  
print("Integer Datatype: ")
print(x.dtype)  
# Float datatype
# guessed by Numpy
x = np.array([1.0, 2.0]) 
print("Float Datatype: ")
print(x.dtype)  
# Forced Datatype
x = np.array([1.0, 2.0], dtype = np.int64)   
print("Forcing a Datatype: ")
print(x.dtype)
# First Array
arr1 = np.array([[4, 7], [2, 6]], 
                 dtype = np.float64)
                  
# Second Array
arr2 = np.array([[3, 6], [2, 8]], 
                 dtype = np.float64)
# Addition of two Arrays
Sum = np.add(arr1, arr2)
print("Addition of Two Arrays: ")
print(Sum)
# Addition of all Array elements
# using predefined sum method
Sum1 = np.sum(arr1)
print("Addition of Array elements: ")
print(Sum1)
# Square root of Array
Sqrt = np.sqrt(arr1)
print("Square root of Array1 elements: ")
print(Sqrt)
# Transpose of Array
# using In-built function 'T'
Trans_arr = arr1.T
print("\nTranspose of Array: ")
print(Trans_arr)
from scipy import special
# 10^3
a = special.exp10(3)
print(a)
# 2^3
b = special.exp2(3)
print(b)
# sin(90)
c = special.sindg(90)
print(c)
# cos(45) 
d = special.cosdg(45)
print(d)
import scipy.integrate as integrate
import scipy.special as special
# integral of a=x^2 from 0-2
a = lambda x: x**2
b = integrate.quad(a,0,2)
print(b)
# create dictionary (a column for each fruit and a row for each customer purchase)
data = {
    'apples': [3, 2, 0, 1], 
    'oranges': [0, 3, 7, 2]
}
# pass it to the pandas DataFrame constructor
purchases = pd.DataFrame(data)
print(purchases)
# set customer names as our index
purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])
print(purchases)
# locate a customer's order by using their name
purchases.loc['Lily']
# Reading data from CSV
import pandas as pd
a = pd.read_csv('../input/test-data/house_prices.csv')
print(a)

# specific some columns
print(a[['price','bedrooms']])
# line chart
import plotly.express as px
fig = px.line(x=[1, 2, 3],y=[4,5,6])
fig.show()
# scatter plot
import plotly.express as px
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()
# bar chart
import plotly.express as px
data_canada = px.data.gapminder().query("country == 'Canada'")
fig = px.bar(data_canada, x='year', y='pop')
fig.show()
# x and y graph
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro-')
plt.ylabel('y')
plt.xlabel('x')
plt.suptitle('X-Y graph')
plt.show()
# load iris dataset
from sklearn import datasets
iris = datasets.load_iris()
# access to the features
print(iris.data)
# access to the target
print(iris.target)
# build a regression model from sample data
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit([[0,5],[6,8],[1,3]],[0,1,1])
reg.coef_
# Setup
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# Prepare the data
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
# Train the model
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# Evaluate the trained model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
# read image
image = cv2.imread('../input/opencv-samples-images/data/fruits.jpg')
# convert color to rgb
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(30, 30))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image)

# convert rgb to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(1, 3, 2)
plt.title("Gray")
plt.imshow(gray)

# thresholded image
ret,thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)


plt.subplot(1, 3, 3)
plt.title("Threshold")
plt.imshow(thresh)

plt.show()