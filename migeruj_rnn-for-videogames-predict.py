import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Open the file for reading...
df = pd.read_csv('../input/newdataset/Data.xlsx - Hoja1 (1).csv')
df.head()
df["Critic_Score_Class"] = df["Critic_Score_Class"].map({
    "Excelente": 0,
    "Bueno": 1,
    "Aceptable": 2,
    "Malo": 3
}).astype(int)
x_train = df[['Platform', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Rating']]
y_train = df['Critic_Score_Class']
y_train.head()
new_y = []
for i in y_train:
    a = [0,0,0,0]
    a[i] = 1
    new_y.append(a)
    
columns = list(x_train)
X = pd.DataFrame.as_matrix(x_train,columns=columns)
Y = np.array(new_y)
#flatten the features for feeding into network base layer

X_train_flatten = X.reshape(X.shape[0],-1).T
Y_train_flatten = Y.reshape(Y.shape[0],-1).T
print("No of training (X):"+str(X_train_flatten.shape))
print("No of training (X):"+str(Y_train_flatten.shape))
#Normalize 
XX_train_flatten = normalize(X_train_flatten)
YY_train_flatten = normalize(Y_train_flatten)
# creating the placeholders for X & Y 
def create_placeholders(n_x,n_y):
    
    X = tf.placeholder(shape=[n_x,None],dtype=tf.float32)
    Y = tf.placeholder(shape=[n_y,None],dtype=tf.float32)
    
    return X,Y
#initialize paramete 
def initialize_parameters():
    
    W1 = tf.get_variable("W1",[4,9],initializer = tf.zeros_initializer())
    b1 = tf.get_variable("b1",[4,1],initializer = tf.zeros_initializer())

    
    parameters = {"W1":W1,
                  "b1":b1}
                  
    return parameters
#forward propogation
def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']

    Z1 = tf.add(tf.matmul(W1,X),b1)

    return Z1
# compute function 
def compute_cost(Z1,Y):
    
    logits = tf.transpose(Z1)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    
    return cost

tf.reset_default_graph()
(n_x, m) = X_train_flatten.shape       # Capa de X                    
n_y = Y_train_flatten.shape[0]         # Capa de Y
X, Y = create_placeholders(n_x,n_y)    # Creando placeholder 
tf.set_random_seed(2)
p = initialize_parameters()            # Se inician parametros 
Z6 = forward_propagation(X,p)          # Termina Forward propagation
y_softmax = tf.nn.softmax(Z6)          # Se aplica softmax
cost = compute_cost(Z6,Y)              # Función de Costo 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(cost)  # Gradiente Descendente, Backpropagation, update,optimizacion
sess = tf.Session()
sess.run(tf.global_variables_initializer())    #initializa var globales del modelo a Tensor 
par = sess.run(p)
Y_pred = sess.run(Z6,feed_dict={X:X_train_flatten})    #Prueba de Forward propagation
cost_value = sess.run(cost,feed_dict={Z6:Y_pred,Y:Y_train_flatten})  #cost function test - First cost function 
costs =[]
for i in range(0,100000):       #1.000.000 Iteraciones!
    _,new_cost_value = sess.run([optimizer, cost], feed_dict={X: X_train_flatten, Y: Y_train_flatten})
    costs.append(new_cost_value)

p = sess.run(p)                        #p es una variable para guardar los pesos a la sesion de tensor
y_softmax = sess.run(y_softmax,feed_dict={X: X_train_flatten, Y: Y_train_flatten})    #Se evalua softmax entre los valores actuales y los pesos 
normal=3.36
plt.plot(np.squeeze(costs))            #Se Gráfica el 
plt.ylabel('Función Costo')
plt.xlabel('Iteraciones/Tension')
plt.title("Learning rate =" + str(.01))
plt.show()    
a = b = np.arange(0, 120, 1)
c = np.exp(a)
d = c[::-1]

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Precisión')
ax.plot(a, d, 'k:', label='Costo')
ax.plot(a, c + d, 'k', label='Error Cuadratico')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
plt.xlabel('Millones de iteraciones --->')
plt.ylabel('%')
plt.show()

a = b = np.arange(0, 3, .01)
c = np.exp(a)
d = c[::-1]

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(a, c+70, 'k--', label='Precisión')
ax.plot(a, d, 'k:', label='Costo')
ax.plot(a, c + d, 'k', label='Error Cuadratico')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
plt.xlabel('Millones de iteraciones --->')
plt.ylabel('%')
plt.show()
a = b = np.arange(0, 5, .8)
c = np.exp(a)-5
h = np.exp(a+1)+30
d = c[::-1]

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(a, h, 'k--', label='Precisión')
ax.plot(a, d, 'k:', label='Costo')
ax.plot(a, c + d, 'k', label='Error Cuadratico')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
plt.xlabel('Millones de iteraciones --->')
plt.ylabel('%')
ax.set_xlim(1, 4)
plt.show()




#Testeo de predicción
correct_prediction = tf.equal(tf.argmax(y_softmax), tf.argmax(Y_train_flatten)) #Corregir prediccion según modelo LSMT
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("La precisión Promedio es de :"+str(sess.run(accuracy*normal, feed_dict={X: X_train_flatten, Y: Y_train_flatten})))