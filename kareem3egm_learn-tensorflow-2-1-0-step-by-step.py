#import tensorflow

import tensorflow as tf



#Becuse tensorflow 2.1.0 enable eager_execution by defulte

#We must disable it to can use session

#We will know about eager_execution later



tf.compat.v1.disable_eager_execution()



#print tensorflow version

print("tensorflow version : ",tf.__version__)
#Define session 

session =  tf.compat.v1.Session()



#Define pi constant = 3.14

pi = tf.constant(3.14)



#Print value of it using session

print("pi = ",session.run(pi))



#Close session

session.close()
print(pi)
pi = tf.constant(3.14)

with tf.compat.v1.Session() as sess : 

    print(sess.run(pi))

a = tf.constant(3)

b = tf.constant(3,tf.float32)

c = tf.constant([2,5,6,9,6])

d = tf.constant([[2,5,6],

                 [8,5,8],

                 [8,8,1]])

e = tf.constant('I love TensorFlow')



with tf.compat.v1.Session() as sess : 

    print(sess.run(a))

    print(sess.run(b))

    print(sess.run(c))

    print(sess.run(d))

    print(sess.run(e))

a = tf.Variable(3)

b = tf.Variable(3,tf.float32)

c = tf.Variable([2,5,6,9,6])

d = tf.Variable([[2,5,6],[8,5,8], [8,8,1]])

e = tf.Variable('I love TensorFlow')



with tf.compat.v1.Session() as sess :

    sess.run(tf.compat.v1.global_variables_initializer())

    print(sess.run(a))

    print(sess.run(b))

    print(sess.run(c))

    print(sess.run(d))

    print(sess.run(e))

session = tf.compat.v1.Session()



pizza = tf.Variable(45.5)

juice = tf.Variable(7.25)

tomatos = tf.Variable(5)



session.run(tf.compat.v1.global_variables_initializer())



print('pizza price   = ',session.run(pizza))

print('juice price   = ',session.run(juice))

print('tomatos price = ',session.run(tomatos))



session.close()
session = tf.compat.v1.Session()



#Change variable value using = operator 

x = tf.Variable(5)

session.run(tf.compat.v1.global_variables_initializer())

print("x before change = ",session.run(x))



x = tf.Variable(10)

session.run(tf.compat.v1.global_variables_initializer())

print("x after change = ",session.run(x))



#Change variable value using assign keyord

#variableName = tf.compat.v1.assign(variableName,newValue)



x = tf.compat.v1.assign(x,20)

print("x after change using assign = ",session.run(x))
for i in range(5):

    x = tf.compat.v1.assign(x,x+10)

    print("x = ",session.run(x))

    

session.close()    
session = tf.compat.v1.Session()



#intialize constant = 2

two = tf.constant(2.0)



#intialize constant = 3.14

pi = tf.constant(3.14)



#intialize placeholder r

r  = tf.compat.v1.placeholder(tf.float32)



#intialize two_pi variable = 2.0*3.14

two_pi = two*pi



#equation = 2πr

c = two_pi*r



#Note : we use feed_dict argument to set palceholders variables 



with tf.compat.v1.Session() as sess : 

    result = session.run(c , feed_dict ={r:[1,2,3,4,5]})

    print(result)



session = tf.compat.v1.Session()



#intialize constant = 2

two = tf.constant(2.0)



#intialize constant = 3.14

pi = tf.constant(3.14)



#intialize placeholder r

r  = tf.compat.v1.placeholder(tf.float32)



#intialize two_pi variable = 2.0*3.14

two_pi = two*pi



#equation = 2πr

c = two_pi*r



#Note : we use feed_dict argument to set palceholders variables 

result = session.run(c, feed_dict = {r:[1,2,3,4,5]})

print(result)



session.close()
session = tf.compat.v1.Session()



x = tf.compat.v1.placeholder(tf.float32)

y = tf.compat.v1.placeholder(tf.float32)



eq = x**2 + y**0.5 



#Note : we can set palceholders variables in dictionry without use feed_dict keyword 

result = session.run(eq , {x:[5,2],y:[25,4]})

print(result)



session.close()
import matplotlib.pyplot as plt

%matplotlib inline

session = tf.compat.v1.Session()



w = tf.constant(5.0)

X = tf.compat.v1.placeholder(tf.float32)

b = tf.constant(2.0)



y = w*X + b

Xval = list(range(0,10))



print("X values : ",Xval)



result = session.run(y, feed_dict = {X:Xval})

print("y values : ",result)



plt.style.use('fivethirtyeight')

plt.plot(Xval,result,marker='o', markersize = 7,linewidth=3)

plt.show()



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(5.0)

num2 = tf.constant(10.5)

result = tf.add(num1,num2)



print(session.run(num1),' + ',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(5)

num2 = tf.constant([1,2,3,4,5])

result = tf.add(num1,num2)



print(session.run(num1),' + ',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



str1 = tf.constant('Hellow')

str2 = tf.constant(' World')

result = tf.add(str1,str2)



print(session.run(str1),' + ',session.run(str2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



str1 = tf.constant('Hellow')

str2 = tf.constant([' World',' Python',' Machine Learning'])

result = tf.add(str1,str2)



print(session.run(str1),' + ',session.run(str2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(5)

num2 = tf.constant(10)

result = tf.multiply(num1,num2)



print(session.run(num1),' x ',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(-5.25)

num2 = tf.constant(10.15)

result = tf.multiply(num1,num2)



print(session.run(num1),' x ',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(5)

num2 = tf.constant(10)

result = tf.subtract(num1,num2)



print(session.run(num1),' - ',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(-5.25)

num2 = tf.constant(10.15)

result = tf.multiply(num1,num2)



print(session.run(num1),' x ',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(25)

num2 = tf.constant(10)

result = tf.divide(num1,num2)



print(session.run(num1),' / ',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(-25.25)

num2 = tf.constant(10.0)

result = tf.divide(num1,num2)



print(session.run(num1),' / ',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(2)

num2 = tf.constant(10)

result = tf.pow(num1,num2)



print(session.run(num1),'^',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



num1 = tf.constant(-5.0)

num2 = tf.constant(3.0)

result = tf.pow(num1,num2)



print(session.run(num1),'^',session.run(num2),' = ',session.run(result))



session.close()
session = tf.compat.v1.Session()



# matrix size = 3x3

matrix1 = tf.constant([[1.0,2.0,3.0],

                       [4.0,5.0,6.0],

                       [7.0,8.0,9.0]])

# matrix size = 3x2

matrix2 = tf.constant([[1.0,2.0],

                       [3.0,4.0],

                       [5.0,6.0]])



# result will be matrix has size = 3x2

result = tf.matmul(matrix1,matrix2)



print(session.run(matrix1))

print('x')

print(session.run(matrix2))

print('=')

print(session.run(result))



session.close()
session = tf.compat.v1.Session()



# matrix size = 3x3

matrix1 = tf.constant([[1.04,2.20,3.40],

                       [4.10,5.01,6.20],

                       [7.051,8.0,9.10]])

# matrix size = 3x2

matrix2 = tf.constant([[1,2],

                       [3,4.0],

                       [5,6]])



# result will be matrix has size = 3x2

result = tf.matmul(matrix1,matrix2)



print(session.run(matrix1))

print('x')

print(session.run(matrix2))

print('=')

print(session.run(result))



session.close()
session = tf.compat.v1.Session()



# matrix size = 3x3

matrix = tf.constant([[1.0,2.0,3.0],

                       [4.0,5.0,6.0],

                       [7.0,8.0,9.0]])

# matrix size = 3x3

matrixTransport = tf.compat.v1.matrix_transpose(matrix)



# result will be matrix has size = 3x3

result = tf.matmul(matrix,matrixTransport)



print('original matrix')

print(session.run(matrix))

print('-------------------------------')



print('transport matrix')

print(session.run(matrixTransport))

print('-------------------------------')



print('original matrix x transport matrix : ')

print(session.run(result))



session.close()
graph = tf.compat.v1.get_default_graph()

for operation in graph.get_operations():

    print(operation.name)