from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import pandas as pd





import seaborn as sns

#import pandas as pd

import numpy as np

#import matplotlib.pyplot as plt





dataset = pd.read_csv('../input/t1-100/T1_100.csv')

x = dataset.iloc[:,2:6].values

y= dataset.iloc[:,6].values



print(dataset.isnull().any().any())

print(dataset.isnull().sum())



#df = pd.DataFrame([[0.529, 5.0, 7.0, 4.0], [0, 3.0, 4.0, 2.0], [0.774, 10.0, 7.0, 6.0], [0.774, 10.0, 8.0, 5.0], [1, 3.0, 0.0, 2.0]])

#fig = plt.figure()

#ax = fig.add_subplot(111, projection='3d')

#ax.scatter(dataset.iloc[:,2].values, dataset.iloc[:,3].values, dataset.iloc[:,6].values, c=dataset.iloc[:,6].values, cmap='gray', s=50, vmin=0.,vmax=1)



#ax.set_xlabel('X Label')

#ax.set_ylabel('Y Label')

#ax.set_zlabel('Z Label')



#plt.show()





#plt.subplots(figsize=(17,14))

#sns.heatmap(dataset.corr(),annot=True,linewidths=0.5,linecolor="Black",fmt="1.1f")

#plt.title("Data Correlation",fontsize=50)

#plt.show()





#dataset.plot(kind="scatter",x="air_speed", y="energy", alpha=0.5, color="red")

#plt.xlabel("air_speed")

#plt.ylabel("energy")

#plt.title("air_speed impact")



#dataset.plot(kind="scatter",x="temp", y="energy", alpha=0.5, color="red")

#plt.xlabel("temp")

#plt.ylabel("energy")

#plt.title("temp impact")





df_norm = dataset[['air_pressure', 'air_speed', 'air_dir', 'temp', 'energy']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

df_norm.sample(n=5)



df_norm.describe()



#sns.pairplot( data=df_norm, vars=('air_pressure','air_speed','air_dir','temp'), hue='energy' )





train_test_per = 90/100.0

df_norm['train'] = np.random.rand(len(df_norm)) < train_test_per

df_norm.sample(n=5)









train = df_norm[df_norm.train == 1]

train = train.drop('train', axis=1).sample(frac=1)

train.sample(n=5)



test = df_norm[df_norm.train == 0]

test = test.drop('train', axis=1)

test.sample(n=5)



X = train.values[:,:4]

X[:5]



targets = [[1,0,0],[0,1,0],[0,0,1]]

y = np.array([targets[int(x)] for x in train.values[:,4:5]])

#y[:5]



num_inputs = len(X[0])

hidden_layer_neurons = 5

np.random.seed(4)

w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1

#w1



num_outputs = len(y[0])

w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1

#w2





                

                

                

#fig = plt.figure(figsize=(12, 12))

#ax = fig.gca()

#ax.axis('off')

#draw_neural_net(ax, .1, .9, .1, .9, [4, 5, 3])



# sigmoid function representation

_x = np.linspace( -5, 5, 50 )

_y = 1 / ( 1 + np.exp( -_x ) )

plt.plot( _x, _y )







learning_rate = 0.2 # slowly update the network

for epoch in range(100000):

    l1 = 1/(1 + np.exp(-(np.dot(X, w1)))) # sigmoid function

    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

    er = (abs(y - l2)).mean()

    l2_delta = (y - l2)*(l2 * (1-l2))

    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))

    w2 += l1.T.dot(l2_delta) * learning_rate

    w1 += X.T.dot(l1_delta) * learning_rate

print('Error:', er)









X = test.values[:,:4]

y = np.array([targets[int(x)] for x in test.values[:,4:5]])



l1 = 1/(1 + np.exp(-(np.dot(X, w1))))

l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))



np.round(l2,3)





yp = np.argmax(l2, axis=1) # prediction

res = yp == np.argmax(y, axis=1)

correct = np.sum(res)/len(res)



testres = test



testres['Prediction'] = yp



print(testres)

print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')






