# In this lesson we will explore the train_test_split module
# Therefore we need no more than the module itself and NumPy
import numpy as np
from sklearn.model_selection import train_test_split
a = np.arange(1,101)
a
b = np.arange(501,601)
b
train_test_split(a, shuffle = False)

# when we give "shuffle = False" then there will be no shuffling and data is split in the ratio,
# by default it takes 75:25 which gives us total 100
# Let's check out how this works, by default shuffle=True
train_test_split(a)

#Here we haven't specified the "random_state = integer" as a result if u run the code cell again and again 
#then the output will change and the integers will change it's position everytime(try it urself). so u need to speciy a 
#random_state which can be any integer(1,2,3, or anyyy..), so everytime it will be the same order.
#while writing and working with our code we will be running the code cell many times, so we need to specify random_state
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=20)
# Let's check the shapes
# Basically, we are checking how does the 'test_size' work
a_train.shape, a_test.shape
a_train
a_test
b_train.shape, b_test.shape
b_train
b_test