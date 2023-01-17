# Bring in numpy(for numerical operations) and matplotlib.pyplot (for graphics)
from numpy import *
from matplotlib.pyplot import *
%matplotlib notebook
print("Hello, world!")
print("My name is Kerry Miller. I am excited to learn physical chemistry!")
# Addition
print(5 + 10) #integer
print(5.0 + 10.0) #floating point
#Subtraction
print(10 - 5)
#Multiplication
print(5 * 9)
#Division
print(5 / 2)
print(9.0 * 8.0 - 13.0)
print(9.0 * (8.0 - 13.0))
print(8.0 - 13.0 * 9)
print((8.0 - 13.0) * 9)
print((3.0/5.0)-((19.0/20.0)*4))

print(5**2) #Double asterisks (*) are used to represent an exponent
x = 5
X = 10
print(x, X) #I can print two separate things by separating them by a comma
apples = 10
bananas = 8
total_fruit = apples + bananas

print("There are", total_fruit, "pieces of fruit in the basket.")
me=70.5 # Specify your height, as a variable (e.g., me=71.0)

Miranda=67.0# Now specify somebody else's height, also as a variable (e.g., friend=66.0)

average=(me+Miranda)/2 # Now get the average as a third variable

print("The average height of Miranda and I is", average, "inches")# And print the third variable in a sentence, like the cell above.

# Assign the constants a-d
a = 3
b = 1
c = -3
d = -5

x=2# Assign a value to x

y=((a*x**3)+(b*x**2)+(c*x)+d)# Calculate y

print(y)# Print y
# Generate an array (sequence of points) between -5 and 5; the number of points is 50 by default
x = linspace(-5,5)
print(x)

# This generates 10 points between -5 and 5
x_10 = linspace(-5,5,10)
print(x_10)
print("an scalar is a variable that can only hace one value at a time, an array can have multiple values. Therefore, this is an array")
x_10=linspace(-3,4,10)

print(x_10)

# This prints the shape of variables x_10 and x
print(shape(x_10))
print(shape(x))
y = a*x**3 + b*x**2 + c*x + d
print(shape(y))
# This initializes the plot window
figure()

# This plots y as a function of x
plot(x,y,'m-.')
xlabel('this is the x-axis')
ylabel('y')
print("e^5 is", exp(5))
print("e^x is", exp(x))
# Initialize the plot window
figure()

x_100=linspace(-3,3,100)# Calculate a new x-array from -3 to 3, with 100 points

y=4*(e**(-x**2))# Calculate a new y-array 

plot(x,y)
# Plot y as a function of x

# Label the x and y axes

i=5
j = i**2
j=linspace(-3,3)
First_name = "Isiaah"
Last_name = "Crawford"
print("The President of the University of Puget Sound is", First_name, Last_name)
print("I am so glad I am learning to program using Python.")
a = 3
b = -4
x = linspace(0,1.5)
f_x = a*exp(b*x)
figure(3)
plot(x,f_x)
n = linspace(-3,3)
m = n**2
figure(4)
plot(n,m)
cars = 100.0
space_in_a_car = 4.0
drivers = 30
passengers = 90
cars_not_driven = cars - drivers
cars_driven = drivers
carpool_capacity = cars_driven * space_in_a_car
average_passengers_per_car = passengers / cars_driven

print("We have", passengers, "to carpool today.")
print("We need to put about", average_passengers_per_car, "in each car.")