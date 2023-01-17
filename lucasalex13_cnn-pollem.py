import cv2 as cv

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



%matplotlib inline 
#name = np.genfromtxt('pollen_data.csv',dtype='str',skip_header=1,delimiter=',',usecols=(1))

 

name = np.genfromtxt("../input/pollendataset/PollenDataset/pollen_data.csv",dtype='str',skip_header=1,delimiter=',',usecols=(1))



path = "../input/pollendataset/PollenDataset/images/"

imlist = []

for i in name:

    imlist.append(path + i)  
def dataset(file_list,size=(180,300),flattened=False):  

    '''

    Function to create a dataset. It will load all the images into a np.array 

    

    Parameters: 

    

    - file_list: List of all the images you want to include in the dataset. 

    - Size : Size of the images, by default is 180x300 which is the original size. 

    - flattened: By default is False. Creates a dataset, but each image get converted into a big vector. 

    

    Output: 

    

    data: it outputs the dataset as a big np array 

    labels : It outputs the binary label. 1 for pollen 0 for non pollen. 

    

    '''

    data = []

    for i, file in enumerate(file_list):

        

        image = cv.imread(file)

        image2 = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        #image2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        image = cv.resize(image2, size)

        if flattened:

            image = image.flatten()



        data.append(image)

        



    labels = np.genfromtxt("../input/pollendataset/PollenDataset/pollen_data.csv",skip_header=1,delimiter=',',usecols=(2))

    

    return np.array(data), np.array(labels)
X,y=dataset(imlist)
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.25, random_state=42)
#Example of a image in the dataset with its label. 

plt.imshow(X[9])

plt.title(y[9])
X_treinamento = np.asarray(X_treinamento, dtype = np.float64)

X_teste = np.asarray(X_teste, dtype = np.float64)



y_treinamento = np.asarray(y_treinamento, dtype = np.int32)

y_teste = np.asarray(y_teste, dtype = np.int32)
import tensorflow as tf
X_treinamento.shape

def cria_rede(features, labels, mode):

    # batch_size, largura, altura, canais

    #canais para coloridas = 3

    #-1 quando não sabemos a quantidade

    entrada = tf.reshape(features['X'], [-1, 300, 180, 3])

    

    # recebe [batch_size, 300, 180, 3]

    # retorna [batch_size, 300, 180, 32]

    #o 32 vem dos 32 filtros adicionados 

    convolucao1 = tf.layers.conv2d(inputs = entrada, filters = 32, kernel_size=[10,10], activation = tf.nn.relu,

                                  padding = 'same')

    # retorna [batch_size, 300, 180, 32]

    # retorna [batch_size, 150, 90, 32]

    pooling1 = tf.layers.max_pooling2d(inputs = convolucao1, pool_size = [5,5], strides = 5)

    

    # retorna [batch_size, 150, 90, 32]

    # retorna [batch_size, 150, 90, 64]

    convolucao2 = tf.layers.conv2d(inputs = pooling1, filters = 64, kernel_size = [10,10], activation = tf.nn.relu,

                                  padding = 'same')

    # retorna [batch_size, 150, 90, 64]

    # retorna [batch_size, 75, 45, 64]

    pooling2 = tf.layers.max_pooling2d(inputs = convolucao2, pool_size = [3,3], strides = 3)

    

    # retorna [batch_size, 75, 45, 64]

    # retornar [batch_size, 216000]



    flattening = tf.reshape(pooling2, [-1, 20 * 12 * 64])

    #flattening = tf.reshape(pooling2, [-1,15360])



    

    # 216000 (entradas) -> 3000 (oculta) -> 2 (saída)

    # recebe [batch_size, 15360]

    # retornar [batch_size, 3000]

    densa = tf.layers.dense(inputs = flattening, units = 3000, activation = tf.nn.relu)

    # dropout

    #zera algumas entradas. rate e a porcentagem. 

    dropout = tf.layers.dropout(inputs = densa, rate = 0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    

    # recebe [batch_size, 3000]

    # retornar [batch_size, 2]

    saida = tf.layers.dense(inputs = dropout, units = 2)

    

    previsoes = tf.argmax(saida, axis = 1)

    

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode = mode, predictions = previsoes)

    

    

    erro = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = saida)

    

    

    #erro = tf.losses.softmax_cross_entropy(onehot_labels = labels, logits = saida)

    #erro = tf.losses.sigmoid_cross_entropy(multi_class_labels = labels, logits = saida)



    if mode == tf.estimator.ModeKeys.TRAIN:

        otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)

        treinamento = otimizador.minimize(erro, global_step = tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode = mode, loss = erro, train_op = treinamento)

    

    if mode == tf.estimator.ModeKeys.EVAL:

        eval_metrics_ops = {'accuracy': tf.metrics.accuracy(labels = labels, predictions = previsoes)}

        return tf.estimator.EstimatorSpec(mode = mode, loss = erro, eval_metric_ops = eval_metrics_ops)
classificador = tf.estimator.Estimator(model_fn = cria_rede)
funcao_treinamento = tf.estimator.inputs.numpy_input_fn(x = {'X': X_treinamento}, y = y_treinamento,

                                                       batch_size = 128, num_epochs = None, shuffle = True)

classificador.train(input_fn=funcao_treinamento, steps = 200)
funcao_teste = tf.estimator.inputs.numpy_input_fn(x = {'X': X_teste}, y = y_teste, num_epochs = 1,

                                                      shuffle = False)

resultados = classificador.evaluate(input_fn=funcao_teste)

resultados
X_imagem_teste = X_teste[97]

X_imagem_teste.shape
X_imagem_teste = X_imagem_teste.reshape(1,-1)

X_imagem_teste.shape
funcao_previsao = tf.estimator.inputs.numpy_input_fn(x = {'X': X_imagem_teste}, shuffle = False)

pred = list(classificador.predict(input_fn = funcao_previsao))
X_image_teste = X_imagem_teste.reshape(300,180,3)

X_image_teste.shape

X_image_teste = np.asarray(X_image_teste,dtype = np.uint8)

plt.imshow(X_image_teste)

plt.title(str(pred))