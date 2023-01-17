import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score
import keras

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

import random

from sklearn.metrics import accuracy_score
class Modelcnn():



    def __init__(self,x_train,y_train,x_test,y_test,neuron1,neuron2,neuron3,neuron4,act1,act2,act3,act4,opt,bs,ep):

        self.neuron1=neuron1

        self.neuron2=neuron2

        self.neuron3=neuron3

        self.neuron4=neuron4

        self.act1=act1

        self.act2=act2

        self.act3=act3

        self.act4=act4

        self.opt=opt

        self.bs=bs

        self.ep=ep

        self.x_train=x_train

        self.y_train=y_train

        self.x_test=x_test

        self.y_test=y_test

        self.model=None

        self.preds=None

        self.predround=None



    def createModel(self,):

        self.model = Sequential()

        self.model.add(Conv2D(self.neuron1, kernel_size=(3,3), activation = self.act1, input_shape=(28, 28 ,1) ))

        self.model.add(MaxPooling2D(pool_size = (2, 2)))



        self.model.add(Conv2D(self.neuron2, kernel_size = (3, 3), activation = self.act2))

        self.model.add(MaxPooling2D(pool_size = (2, 2)))



        self.model.add(Conv2D(self.neuron3, kernel_size = (3, 3), activation = self.act3))

        self.model.add(MaxPooling2D(pool_size = (2, 2)))



        self.model.add(Flatten())

        self.model.add(Dense(self.neuron4, activation = self.act4))

        self.model.add(Dropout(0.20))

        self.model.add(Dense(24, activation = 'softmax'))



        self.model.compile(loss = keras.losses.categorical_crossentropy, optimizer=self.opt,metrics=['accuracy'])



        self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), epochs=self.ep, batch_size=self.bs)



    def predict(self,test_images,test_labels):

        y_pred = self.model.predict(test_images)

        self.preds = y_pred

        self.predround = y_pred.round()

        return(accuracy_score(test_labels, self.predround))
neurons=[16,32,64,128]

actfunc=['relu','sigmoid','tanh']

optfunc=[keras.optimizers.Adam(),keras.optimizers.SGD(),keras.optimizers.Adagrad(),keras.optimizers.Adamax()]

batchsize=[64,128,256]

epoch=[30,40,50]



class newGenClass():

    def __init__(self):

        self.neuron1=None

        self.neuron2=None

        self.neuron3=None

        self.neuron4=None

        self.act1=None

        self.act2=None

        self.act3=None

        self.act4=None

        self.opt=None

        self.bs=None

        self.ep=None

        self.vec1=None



    def selection(self):

        self.neuron1=random.choice(neurons)

        self.neuron2=random.choice(neurons)

        self.neuron3=random.choice(neurons)

        self.neuron4=random.choice(neurons)

        self.act1=random.choice(actfunc)

        self.act2=random.choice(actfunc)

        self.act3=random.choice(actfunc)

        self.act4=random.choice(actfunc)

        self.opt=random.choice(optfunc)

        self.bs=random.choice(batchsize)

        self.ep=random.choice(epoch)

        return ([self.neuron1,self.neuron2,self.neuron3,self.neuron4,self.act1,self.act2,self.act3,self.act4,self.opt,self.bs,self.ep])



    def crossover(self):

        vec1=self.selection()

        vec2=self.selection()

        n = random.randint(0,8)

        a = vec1[0:n]

        b = vec1[n:]

        c = vec2[0:n]

        d = vec2[n:]

        a.extend(d)

        c.extend(b)

        self.vec1=a

        

    def mutation(self):

        cr=self.vec1

        mut_percent=30

        x=random.randint(0,100)

        if(x>mut_percent):

            return(cr)

        else:

            n=random.randint(0,10)

            if(n<=3):

                newNeu=list(neurons)

                newNeu.remove(cr[n])

                cr[n]=random.choice(newNeu)

            elif(n<=7):

                newfuncs=list(actfunc)

                newfuncs.remove(cr[n])

                cr[n]=random.choice(newfuncs)

            elif(n==8):

                cr[n]=random.choice(optfunc)

            elif(n==9):

                newBatch=list(batchsize)

                newBatch.remove(cr[n])

                cr[n]=random.choice(newBatch)

            elif(n==10):

                newEpochs=list(epoch)

                newEpochs.remove(cr[n])

                cr[n]=random.choice(newEpochs)

            return(cr)
train = pd.read_csv('../input/sign_mnist_train.csv')

test = pd.read_csv('../input/sign_mnist_test.csv')
train.head()
train.shape
labels = train['label'].values
unique_val = np.array(labels)

np.unique(unique_val)
train.drop('label', axis = 1, inplace = True)
images = train.values

images = np.array([np.reshape(i, (28, 28)) for i in images])

images = np.array([i.flatten() for i in images])
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)
labels
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)
x_train = x_train / 255

x_test = x_test / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
plt.imshow(x_train[0].reshape(28,28))
test_labels = test['label']
test.drop('label', axis = 1, inplace = True)
test_images = test.values

test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])

test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
np.array(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images.shape
best_hyperparameters=[]

accuracy=0

total_tries=[]

count=0

acckeys=[]

accvals=[]
while(count<30):

    newGene = newGenClass()

    newGene.crossover()

    Mutated = newGene.mutation()

    print(Mutated)

    if(Mutated in total_tries):

        pass

    else:

        count+=1

        total_tries.append(Mutated)

        run = Modelcnn(x_train,y_train,x_test,y_test,Mutated[0],Mutated[1],Mutated[2],Mutated[3],Mutated[4],Mutated[5],Mutated[6],Mutated[7],Mutated[8],Mutated[9],Mutated[10])

        run.createModel()

        new_accuracy=run.predict(test_images,test_labels)

        acckeys.append(Mutated)

        accvals.append(new_accuracy)

        if(new_accuracy>accuracy):

            accuracy=new_accuracy

            best_hyperparameters=Mutated
for i in range(len(accvals)):

    for j in range(len(accvals)):

        if(accvals[i]>accvals[j]):

            accvals[i],accvals[j]=accvals[j],accvals[i]

            acckeys[i],acckeys[j]=acckeys[j],acckeys[i]
#Applying the Genetic Algorithm

nkey1 = acckeys[0]

nkey2 = acckeys[1]

gen_acc = accvals[0]

lest = [nkey1,nkey2]

genHyper= acckeys[0]

for i in range(5):

    newGen = []

    for j in range(11):

        q1 = random.choice(lest)

        newGen.append(q1[j])

    run = Modelcnn(x_train,y_train,x_test,y_test,newGen[0],newGen[1],newGen[2],newGen[3],newGen[4],newGen[5],newGen[6],newGen[7],newGen[8],newGen[9],newGen[10])

    run.createModel()

    new_accuracy=run.predict(test_images,test_labels)

    if(new_accuracy>gen_acc):

        gen_acc = new_accuracy

        genHyper = newGen

acckeys.append(genHyper)

accvals.append(gen_acc)
print("Hyperparameters found using the genetic algorithm are: ")

print(genHyper)

print("Best Accuracy found using the genetic algorithm for the above parameters are: ")

print(gen_acc)
'''

print("Hyperparameters vs Accuracy")

for i in range(len(accvals)):

    print("Hyperparameter",acckeys[i])

    print("Accuracy",accvals[i])

    print("")

'''