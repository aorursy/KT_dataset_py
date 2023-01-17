import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from scipy.io import arff
from io import StringIO
import tensorflow as tf
# Importing data
f = open('../input/bankurptcy-data/4year.arff','r') 
data, details = arff.loadarff(f)
f.close()
d=[]
for i in data:
    t=list(i)
    t=list(map(float,t))
    d.append(t)
df=pd.DataFrame(d)
df.fillna(method='ffill',inplace=True)
df.shape
df[64].value_counts()
# ratio of majority to minority:
9277/515
# defining weights based on imbalancing
weights={0:1., 1:18.}
y=df[64].values
del df[64]
x=df.values
x.shape,y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,stratify=y)
Y_test.shape[0]-Y_test[Y_test>0].sum()
Y_test.shape
def create_model(hidden_units,learning_rate):
    model=keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_units,activation='relu'))
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# definig call back
es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min')
new_dic=[(0.002, 24), (0.04, 8), (0.06, 64), (0.08, 80), (0.005, 48), (0.07, 40), (0.001, 20), (0.1, 52), (0.009, 36), (0.01, 76)]

accuracies={}
for i,params in enumerate(new_dic):
    accuracies[i]=[]
    for j in range(20):
        model=create_model(params[1],params[0])
        print('Fitting model for {}th iteration for {}th parameters'.format(j,i))
        X_train1,X_test1,Y_train1,Y_test1=train_test_split(X_train,Y_train,test_size=0.3,stratify=Y_train)
        model.fit(X_train1,Y_train1,validation_data=(X_test1,Y_test1),class_weight=weights,epochs=100,batch_size=32,verbose=0,callbacks=[es])
        accuracies[i].append(model.evaluate(X_train,Y_train)[1]*100)
        
        
import matplotlib.pyplot as plt
for i in range(10):
    plt.boxplot(accuracies[i])
    plt.title('Learning rate = {}, Hidden units = {}'.format(round(new_dic[i][0],2),new_dic[i][1]))
    plt.show()

# Applying algorithm
v_max=-1
i_opt=-1
for i in range(len(accuracies)):
    q1=np.percentile(accuracies[i],25,interpolation='midpoint')*100
    q2=np.percentile(accuracies[i],50,interpolation='midpoint')*100
    q3=np.percentile(accuracies[i],75,interpolation='midpoint')*100
    #print(q1,q2,q3)
    M=q2
    IQR=q3-q1
    v=(M*M)/IQR
    if(v>v_max):
        v_max=v
        i_opt=i
        #print('new v: {} and new i_opt: {}'.format(v_max,i_opt))
    print('For Lr: {}, Hu: {}, M: {} & IQR: {} => V: {}'.format(new_dic[i][0],new_dic[i][1],round(M,2),round(IQR,2),round(v,2)))
print('Best possible combiation: Learning rate: {}, Hidden units: {}'.format(new_dic[i_opt][0],new_dic[i_opt][1]))
# Checking test accuracy on optimal hyper parameters:
i_opt=0
model = create_model(new_dic[i_opt][1],new_dic[i_opt][0])
print('Fitting model')
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X_train,Y_train,test_size=0.3,stratify=Y_train)
model.fit(X_train1,Y_train1,validation_data=(X_test1,Y_test1),class_weight=weights,epochs=100,batch_size=32,verbose=0,callbacks=[es])
print("Testing accuracy: {} and loss: {}".format(model.evaluate(X_test,Y_test)[1],model.evaluate(X_test,Y_test)[0]))
prediction=model.predict(X_test)
predictions=[
]
for i in prediction:
    predictions.append(round(i[0]))
predictions1=[]
for label in predictions:
    if(label==0):
        predictions1.append([0,1])
    else:
        predictions1.append([1,0])
predictions=np.array(predictions1)
labels=list(Y_test)
ranks = np.zeros(len(labels))

for i in range(len(labels)) :
    if labels[i] in predictions[i] :
        firstOccurance = np.argmax(predictions[i]== labels[i])        
        for j in range(firstOccurance, len(labels)) :            
            ranks[j] +=1
ranks
plt.plot(ranks)
plt.title('CMC curve')
plt.show()
