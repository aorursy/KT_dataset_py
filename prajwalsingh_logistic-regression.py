import numpy as np

import matplotlib.style as style

from matplotlib import pyplot as plt

import pandas as pd



def cost_function(trainingData, thetai, m):

	

	cost = 0.0



	for data in trainingData:

		X = np.array([[1], [data[0]], [data[1]]],dtype='float32')

		H = thetai.dot(X)[0]

		Z = 1/(1+np.exp(-H))

		cost += ( - data[2] * np.log(Z) - (1-data[2]) * np.log(1-Z) )

	return float(cost)/float(m)





def logistic_reg(trainingData, m, testingData):

		

	learning_rate = 0.1

	thetai = np.random.random((1,3))

	# print(thetai)

	global_error = cost_function(trainingData, thetai, m)

	# print(global_error)

	prev_error = 0.0



	try:

		while prev_error != global_error and global_error >= 0.13:



			prev_error = global_error



			dthetai = np.zeros((1,3),dtype='float32')



			for data in trainingData:

				X = np.array([[1], [data[0]], [data[1]]],dtype='float32')

				H = thetai.dot(X)[0]

				Z = 1/(1+np.exp(-H))



				dthetai[0,0] += ( Z - data[2] ) * X[0]

				dthetai[0,1] += ( Z - data[2] ) * X[1]

				dthetai[0,2] += ( Z - data[2] ) * X[2]



			dthetai[0,0] /= float(m)

			dthetai[0,1] /= float(m)

			dthetai[0,2] /= float(m)



			thetai = thetai - (learning_rate * dthetai)



			global_error = cost_function(trainingData, thetai, m)



			print(global_error)

	except:

		pass



	pred = 0



	for data in testingData:

		X = np.array([[1], [data[0]], [data[1]]],dtype='float32')

		H = thetai.dot(X)[0]

		Z = 1/(1+np.exp(-H))



		if Z>=0.5:

			Z = 1

		else:

			Z = 0



		print('Target : {0} \t Predict : {1}'.format(data[2],Z))

		if data[2]==Z:

			pred+=1



	print('Accuracy : {0}%'.format((float(pred)/len(testingData))*100))



	# style.use('seaborn')

	# plt.scatter(trainingData[trainingData[:,2]==1,0],trainingData[trainingData[:,2]==1,1],color='r')

	# plt.scatter(trainingData[trainingData[:,2]==0,0],trainingData[trainingData[:,2]==0,1],color='b')

	# x = np.linspace(np.min(trainingData[:,1]),np.max(trainingData[:,1]),100)

	# y = thetai[0,0]+thetai[0,1]*x + thetai[0,2]*x

	# plt.plot(x,y, color='g')

	# # plt.scatter(data[data[:,2]==2,0],data[data[:,2]==2,1],color='g')

	# plt.xlabel('petal_length')

	# plt.ylabel('petal_width')

	# plt.show()





if __name__ == '__main__':



	# Data : sepal_length , sepal_width, petal_length, petal_width, species

	# setosa = 0

	# versicolor = 1

	# virginica = 2 -> 0



	df = pd.read_csv('/kaggle/input/iris.csv')

	# print(df['petal_length'])

	data = np.array(list(zip(df['petal_length'],df['petal_width'],df['species'])))



	temp = data[data[:,2]==1]

	trainingData = temp[:30,:]

	testingData = temp[30:,:]

	temp = data[data[:,2]==2]

	temp[:,2] = 0

	trainingData = np.append(trainingData, temp[:30,:], axis=0)

	testingData = np.append(testingData, temp[30:,:], axis=0)

	# print(trainingData)

	# print(len(trainingData))

	# print(len(testingData))

	# print(data[data[:,2]==0,0])

	# print(len(data[data[:,2]==1,1]))

	# print(len(data[data[:,2]==2,1]))

	# style.use('seaborn')

	# plt.scatter(trainingData[trainingData[:,2]==1,0],trainingData[trainingData[:,2]==1,1],color='r')

	# plt.scatter(trainingData[trainingData[:,2]==2,0],trainingData[trainingData[:,2]==2,1],color='b')

	# # plt.scatter(data[data[:,2]==2,0],data[data[:,2]==2,1],color='g')

	# plt.xlabel('petal_length')

	# plt.ylabel('petal_width')

	# plt.show()



	logistic_reg(trainingData, len(trainingData), testingData)