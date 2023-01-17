import numpy as np

#For this exercise, you are not required to code anything.  Instead, answer these three basic questions:

#A.) What is this code training against, and what is it working to accomplish?


X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1


#B.) What are the purposes of our syns?  More specifically, what are they and how do they impact our network?

for j in range(60000):
	l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
	l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
	l2_delta = (y - l2)*(l2*(1-l2))
	l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
	syn1 += l1.T.dot(l2_delta)
	syn0 += X.T.dot(l1_delta)
	
	if j%10000==0:
		print("Output: {}".format(l2))
        
        
#C.) Lastly, how many hidden layers do we have [if any], and what do they do [if they are there]?
#Answer "None" if there aren't any.


#If you feel stumped, the code's author, Andrew Trask, posted about it here:
#https://iamtrask.github.io/2015/07/12/basic-python-network/

#Welcome back to another exciting round of ANSWER MY QUESTIONS!
#Now that you have a few concepts under your belt, here are your last five questions (bonus round at the end)

#QUESTION A:  What is Keras?  What is it used for?

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

#QUESTION B:  Why are we splitting our dataset?  What purpose does using the same dataset use?

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

#QUESTION C:  What does the Sigmoid Function do?  What makes it so useful?


#Question D:  What is Relu and why are we using that instead?

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#FINAL QUESTION:  What is an Epoch?

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)

# calculate predictions
predictions = model.predict(X)

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


#Example from:
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/