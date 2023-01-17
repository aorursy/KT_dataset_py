def hello():
    print("hello world")
hello()
def hello_name(name):
    print('hello', name)
hello_name("Charles")
name = "Charles"
hello_name(name)
def addAndMultiply(number1, number2):
    '''
    return the sum and the multiplication results of the arguments
    '''
    add_result = number1 + number2
    multiply_result = number1*number2
    return add_result, multiply_result

addAndMultiply(5,7)
resultAdd , resultMult = addAndMultiply(8,8)
print(resultAdd)
print(resultMult)
help(addAndMultiply)
def addAndMultiply(number1, number2):
    '''
    return the sum and the multiplication results of the arguments
    '''
    add_result = number1 + number2
    multiply_result = number1*number2
    
    temp= add_result, multiply_result
    
    return add_result, multiply_result
temp
addAndMultiply(1, 1)
temp
def squared(x):
    return x**2
squared(7)
squaredLambda = lambda x: x**2
squaredLambda(9)
type(squared)
type(squaredLambda)
x
list1 = [2, 18, 9, 22, 17, 24, 8, 12, 27]
print(squaredLambda(list1))
#create your function here
