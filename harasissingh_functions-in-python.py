def my_fun(x):

    return x**2
my_fun(3)
y = lambda x: x**2



#We are saying that lambda take x, Square it and return the answer in y
y(3)
# Using Simple Function



def my_sum(x,y,z):

    return x+y+z

    
result = my_sum(1,2,3)

result
# Using Lambda()



result = lambda x,y,z: x+y+z



result(1,2,3)
num_a = int(input("Enter 1st number:"))

num_b = int(input("Enter 2nd number:"))



mult_ab = lambda x, y : x * y



print("Multiplication = {}". format(mult_ab(num_a, num_b)))
# Defining 2 lists a and b



a = [1,4,5,6,9]

b = [1,7,9,11,4]
# Defining a function to be performed on the elements of lists



def my_sum(a,b):

    return a+b
# Now we can use map() to apply the function to the entire lists and generate a new list



c = list(map(my_sum,a,b))

c
input_list = [1,2,3,4]
output_list = list(map(lambda x: x**2, input_list))

output_list
def cube(x):

    return x*x*x
numbers = list(range (-10,11))

numbers
cubed = list(map(lambda x: cube(x), numbers))
cubed
prices = [20,33,56,11,78,65,34,12,79,88]

prices
# Return only even numbers



even_list = list(filter(lambda x: (x%2==0), prices))

even_list



# lambda will select only those numbers which return 0 
# Return only odd numbers



odd_list = list(filter(lambda x:(x%2==1), prices))

odd_list
# Return Prices that are greater than or equal to 56



out_list = list(filter(lambda x: (x>=56), prices))

out_list
# Return a value between 34 and 78



out_list = list(filter(lambda x: (x>=34 and x<=78), prices))

out_list
a = int(input("Enter lower bound:"))

b = int(input("Enter upper bound:"))



numbers = range(a,b)



c= list(filter(lambda x: (x>=0 and x%2==0), numbers))



print(c)