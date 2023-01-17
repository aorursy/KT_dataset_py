import pandas as pd # Criação de data frames

from keras.layers import Input, Dense # Criação do autoencoder

from keras.models import Model, Sequential

from keras.datasets import mnist, fashion_mnist

import numpy as np # Manipulação numérica

import matplotlib.pyplot as plt # Criação de gráficos

import seaborn as sns # Criação de gráficos

from sklearn import decomposition # pca

def create_model(model_df, list_gng_models, list_modes = [] ):

    """

    Parameters

    ----------

        model_df : Modelo de data.frame a ser criado

        list_gng_models : Lista dos modelos GNG a serem adicionados no data.frane

        list_modes : Lista de modos a serem adicionados no data.frame

    

    Returns

    -------

        Data.frame com as informações dos modelos adicionados

    

    """

    for index in range(len(list_gng_models)):

            dict_info = pd.DataFrame.from_dict({'val_loss': list_gng_models[index].history['val_loss'],

                                                'loss': list_gng_models[index].history['loss'],

                                                'epoch': [i for i in range(1, len(list_gng_models[index].history['val_loss'])+1)],

                                                'mode': [list_modes[index] for i in range(1, len(list_gng_models[index].history['val_loss'])+1)],

                                                'error_min':[min(list_gng_models[index].history['val_loss']) for i in range(1, len(list_gng_models[index].history['val_loss'])+1)]

                                        })

            model_df = model_df.append(dict_info)

    return(model_df)
(x_train, _), (x_test, _) = mnist.load_data()



x_train = x_train.astype('float32') / 255.

x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print (x_train.shape)

print (x_test.shape)
# Criação da camada de entrada

input_img = Input(shape=(784,))



# Criação das representações de codificação

encoded_default = Dense(32, activation='relu')(input_img)

encoded_default = Dense(16, activation='relu')(encoded_default)



# Criação da camada de decodificação

decoded_default = Dense(32, activation='relu')(encoded_default)

decoded_default = Dense(784, activation='sigmoid')(decoded_default)



# Criação do modelo default

autoencoder_default = Model(input_img, decoded_default)
autoencoder_default.summary()
# Modelo para codificar os dados de entrada

encoder_default = Model(input_img, encoded_default)
# Definição da função de otimização e perda

autoencoder_default.compile(optimizer='adadelta', loss='binary_crossentropy')



# Treino do modelo

history_default = autoencoder_default.fit(x_train, x_train,

                epochs=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test, x_test))
# Criação das representações de codificação

encoded_changed_1 = Dense(256, activation='relu')(input_img)

encoded_changed_1 = Dense(128, activation='relu')(encoded_changed_1)

encoded_changed_1 = Dense(32, activation='relu')(encoded_changed_1) # code



# Criação da camada de decodificação 

decoded_changed_1 = Dense(128, activation='relu')(encoded_changed_1)

decoded_changed_1 = Dense(256, activation='relu')(decoded_changed_1)

decoded_changed_1 = Dense(784, activation='sigmoid')(decoded_changed_1)



# Criação do modelo default

autoencoder_changed_1 = Model(input_img, decoded_changed_1)
autoencoder_changed_1.summary()
# Modelo para codificar os dados de entrada

encoder_changed_1 = Model(input_img, encoded_changed_1)

# Definição da função de otimização e perda

autoencoder_changed_1.compile(optimizer='adadelta', loss='binary_crossentropy')



# Treino do modelo

history_changed_1 = autoencoder_changed_1.fit(x_train, x_train,

                epochs=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test, x_test))
# Criação das representações de codificação

encoded_changed_2 = Dense(1024, activation='relu')(input_img)

encoded_changed_2 = Dense(512, activation='relu')(encoded_changed_2)

encoded_changed_2 = Dense(256, activation='relu')(encoded_changed_2)

encoded_changed_2 = Dense(128, activation='relu')(encoded_changed_2) # code



# Criação da camada de decodificação 

decoded_changed_2 = Dense(256, activation='relu')(encoded_changed_2)

decoded_changed_2 = Dense(512, activation='relu')(decoded_changed_2)

decoded_changed_2 = Dense(1024, activation='relu')(decoded_changed_2)

decoded_changed_2 = Dense(784, activation='sigmoid')(decoded_changed_2)



# Criação do modelo default

autoencoder_changed_2 = Model(input_img, decoded_changed_2)
autoencoder_changed_2.summary()
# Modelo para codificar os dados de entrada

encoder_changed_2 = Model(input_img, encoded_changed_2)

# Definição da função de otimização e perda

autoencoder_changed_2.compile(optimizer='adadelta', loss='binary_crossentropy')



# Treino do modelo

history_changed_2 = autoencoder_changed_2.fit(x_train, x_train,

                epochs=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test, x_test))
# Primeira camada escondida do modelo default

firsth_default = Model(inputs=autoencoder_default.layers[0].input,

                                 outputs=autoencoder_default.layers[1].output)

predict_default_fh = firsth_default.predict(x_test)

# Decodificação do modelo criado

decoded_imgs_default = autoencoder_default.predict(x_test)





# Primeira camada escondida do modelo changed-1

firsth_changed_1 = Model(inputs=autoencoder_changed_1.layers[0].input,

                                 outputs=autoencoder_changed_1.layers[1].output)

predict_changed_1_fh = firsth_changed_1.predict(x_test)

# Decodificação do modelo criado

decoded_imgs_changed_1 = autoencoder_changed_1.predict(x_test)





# Primeira camada escondida do modelo changed-2

firsth_changed_2 = Model(inputs=autoencoder_changed_2.layers[0].input,

                                 outputs=autoencoder_changed_2.layers[1].output)

predict_changed_2_fh = firsth_changed_2.predict(x_test)

# Decodificação do modelo criado

decoded_imgs_changed_2 = autoencoder_changed_2.predict(x_test)
predict_changed_1_fh.shape
# Visualização dos primeiros da 10 digitos

n = 10 

plt.figure(figsize=(20, 10))

for i in range(n):

    # display original

    ax = plt.subplot(4, n, i + 1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    plt.title('Img original')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(4, n, i + 1 + n)

    plt.imshow(predict_default_fh[i].reshape(4, 4*2))

    plt.gray()

    plt.title('Default')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

      # display reconstruction

    ax = plt.subplot(4, n, i + 1 +  2*n)

    plt.imshow(predict_changed_1_fh[i].reshape(16, 16))

    plt.gray()

    plt.title('Changed-1')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

        # display reconstruction

    ax = plt.subplot(4, n, i + 2 + 29)

    plt.imshow(predict_changed_2_fh[i].reshape(32, 32))

    plt.gray()

    plt.title('Changed-2')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

plt.show()
# Visualização dos primeiros da 10 digitos

n = 10 

plt.figure(figsize=(20, 10))

for i in range(n):

    # display original

    ax = plt.subplot(4, n, i + 1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    plt.title('Img original')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(4, n, i + 1 + n)

    plt.imshow(decoded_imgs_default[i].reshape(28, 28))

    plt.gray()

    plt.title('Default')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

      # display reconstruction

    ax = plt.subplot(4, n, i + 1 +  2*n)

    plt.imshow(decoded_imgs_changed_1[i].reshape(28, 28))

    plt.gray()

    plt.title('Changed-1')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

        # display reconstruction

    ax = plt.subplot(4, n, i + 2 + 29)

    plt.imshow(decoded_imgs_changed_2[i].reshape(28, 28))

    plt.gray()

    plt.title('Changed-2')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

plt.show()
mnist_model = pd.DataFrame(columns = ['val_loss', 'loss', 'epoch', 'mode', 'error_min'])

mnist_model = create_model(model_df = mnist_model, list_gng_models = [history_default, history_changed_1,history_changed_2], 

                          list_modes = ["default", "changed-1","changed-2" ] )



mnist_model_melt=pd.melt(mnist_model, ['epoch', 'mode', 'error_min'])
# Reduzingdo em 64 componentes

pca_mnist = decomposition.PCA(n_components = 64)

pca_mnist_reduced = pca_mnist.fit_transform(x_test)
def reconstruction(X, trans):

    """

    Creates a reconstruction of an input record, X, using the topmost (n) vectors from the

    given transformation (trans)

    

    Note 1: In this dataset each record is the set of pixels in the image (flattened to 

    one row).

    """

    #vectors = [trans.components_[n] * X[n] for n in range(0, n)]

    

    # Invert the PCA transformation.

    ret = trans.inverse_transform(X)

    

    # This process results in non-normal noise on the margins of the data.

    # We clip the results to fit in the [0, 1] interval.

    ret[ret < 0] = 0

    ret[ret > 1] = 1

    return ret
# Camada de codificação

encoded_ae = Dense(64, activation='linear')(input_img)

# Camada de decoficação

decoded_ae = Dense(784, activation='linear')(encoded_ae)



# Criação do autoencoder

autoencoder_linear = Model(input_img, decoded_ae)
# Definição da função de otimização e perda

autoencoder_linear.compile(optimizer='adadelta', loss='mse')



# Treino do modelo

history_linear = autoencoder_linear.fit(x_train, x_train,

                epochs=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test, x_test))
predict_linear = autoencoder_linear.predict(x_test)

predict_pca = reconstruction(pca_mnist_reduced, pca_mnist)
n = 10 

plt.figure(figsize=(20, 10))

for i in range(n):

    # display original

    ax = plt.subplot(3, n, i + 1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    plt.title('Img original')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(3, n, i + 1 + n)

    plt.imshow(predict_pca[i].reshape(28, 28))

    plt.gray()

    plt.title('PCA')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

      # display reconstruction

    ax = plt.subplot(3, n, i + 1 +  2*n)

    plt.imshow(predict_linear[i].reshape(28, 28))

    plt.gray()

    plt.title('AE-Linear')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

       

plt.show()
(x_train_f, _), (x_test_f, _) = fashion_mnist.load_data()



x_train_f = x_train_f.astype('float32') / 255.

x_test_f = x_test_f.astype('float32') / 255.

x_train_f = x_train_f.reshape((len(x_train_f), np.prod(x_train_f.shape[1:])))

x_test_f = x_test_f.reshape((len(x_test_f), np.prod(x_test_f.shape[1:])))

print (x_train_f.shape)

print (x_test_f.shape)
# Criação da camada de entrada

input_img_f = Input(shape=(784,))



# Criação das representações de codificação

encoded_default_f = Dense(1024, activation='relu')(input_img_f)

encoded_default_f = Dense(32, activation='relu')(encoded_default_f)



# Criação da camada de decodificação

decoded_default_f = Dense(1024, activation='relu')(encoded_default_f)

decoded_default_f = Dense(784, activation='sigmoid')(decoded_default_f)



# Criação do modelo default

autoencoder_default_f = Model(input_img_f, decoded_default_f)
autoencoder_default_f.summary()
# Definição da função de otimização e perda

autoencoder_default_f.compile(optimizer='adadelta', loss='binary_crossentropy')



# Treino do modelo

history_default_f = autoencoder_default_f.fit(x_train_f, x_train_f,

                epochs=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test_f, x_test_f))
# Criação das representações de codificação

encoded_changed_1_f = Dense(400, activation='relu')(input_img_f)

encoded_changed_1_f = Dense(800, activation='relu')(encoded_changed_1_f)

encoded_changed_1_f = Dense(64, activation='relu')(encoded_changed_1_f)



# Criação da camada de decodificação

decoded_changed_1_f = Dense(800, activation='relu')(encoded_changed_1_f)

decoded_changed_1_f = Dense(400, activation='relu')(decoded_changed_1_f)

decoded_changed_1_f = Dense(784, activation='sigmoid')(decoded_changed_1_f)



# Criação do modelo default

autoencoder_changed_1_f = Model(input_img_f, decoded_changed_1_f)
autoencoder_changed_1_f.summary()
# Definição da função de otimização e perda

autoencoder_changed_1_f.compile(optimizer='adadelta', loss='binary_crossentropy')



# Treino do modelo

history_changed_1_f = autoencoder_changed_1_f.fit(x_train_f, x_train_f,

                epochs=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test_f, x_test_f))
# Criação das representações de codificação

encoded_changed_2_f = Dense(1024, activation='relu')(input_img_f)

encoded_changed_2_f = Dense(512, activation='relu')(encoded_changed_2_f)

encoded_changed_2_f = Dense(256, activation='relu')(encoded_changed_2_f)

encoded_changed_2_f = Dense(64, activation='relu')(encoded_changed_2_f)



# Criação da camada de decodificação

decoded_changed_2_f = Dense(256, activation='relu')(encoded_changed_2_f)

decoded_changed_2_f = Dense(512, activation='relu')(decoded_changed_2_f)

decoded_changed_2_f = Dense(1024, activation='relu')(decoded_changed_2_f)

decoded_changed_2_f = Dense(784, activation='sigmoid')(decoded_changed_2_f)



# Criação do modelo default

autoencoder_changed_2_f = Model(input_img_f, decoded_changed_2_f)
autoencoder_changed_2_f.summary()
# Definição da função de otimização e perda

autoencoder_changed_2_f.compile(optimizer='adadelta', loss='binary_crossentropy')



# Treino do modelo

history_changed_2_f = autoencoder_changed_2_f.fit(x_train_f, x_train_f,

                epochs=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test_f, x_test_f))
# Primeira camada escondida do modelo default

firsth_default = Model(inputs=autoencoder_default_f.layers[0].input,

                                 outputs=autoencoder_default_f.layers[1].output)

predict_default_fh = firsth_default.predict(x_test_f)

# Decodificação do modelo criado

decoded_imgs_default = autoencoder_default_f.predict(x_test_f)





# Primeira camada escondida do modelo changed-1

firsth_changed_1 = Model(inputs=autoencoder_changed_1_f.layers[0].input,

                                 outputs=autoencoder_changed_1_f.layers[1].output)

predict_changed_1_fh = firsth_changed_1.predict(x_test_f)

# Decodificação do modelo criado

decoded_imgs_changed_1 = autoencoder_changed_1_f.predict(x_test_f)





# Primeira camada escondida do modelo changed-2

firsth_changed_2_f = Model(inputs=autoencoder_changed_2_f.layers[0].input,

                                 outputs=autoencoder_changed_2_f.layers[1].output)

predict_changed_2_fh = firsth_changed_2_f.predict(x_test_f)

# Decodificação do modelo criado

decoded_imgs_changed_2 = autoencoder_changed_2_f.predict(x_test_f)
predict_changed_2_fh.shape
# Visualização dos primeiros da 10 digitos

n = 10 

plt.figure(figsize=(20, 10))

for i in range(n):

    # display original

    ax = plt.subplot(4, n, i + 1)

    plt.imshow(x_test_f[i].reshape(28, 28))

    plt.gray()

    plt.title('Img original')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(4, n, i + 1 + n)

    plt.imshow(predict_default_fh[i].reshape(32, 32))

    plt.gray()

    plt.title('Default')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

      # display reconstruction

    ax = plt.subplot(4, n, i + 1 +  2*n)

    plt.imshow(predict_changed_1_fh[i].reshape(20, 20))

    plt.gray()

    plt.title('Changed-1')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

        # display reconstruction

    ax = plt.subplot(4, n, i + 2 + 29)

    plt.imshow(predict_changed_2_fh[i].reshape(32, 32))

    plt.gray()

    plt.title('Changed-2')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

plt.show()
# Visualização dos primeiros da 10 digitos

n = 10 

plt.figure(figsize=(20, 10))

for i in range(n):

    # display original

    ax = plt.subplot(4, n, i + 1)

    plt.imshow(x_test_f[i].reshape(28, 28))

    plt.gray()

    plt.title('Img original')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(4, n, i + 1 + n)

    plt.imshow(decoded_imgs_default[i].reshape(28, 28))

    plt.gray()

    plt.title('Default')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

      # display reconstruction

    ax = plt.subplot(4, n, i + 1 +  2*n)

    plt.imshow(decoded_imgs_changed_1[i].reshape(28, 28))

    plt.gray()

    plt.title('Changed-1')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

        # display reconstruction

    ax = plt.subplot(4, n, i + 2 + 29)

    plt.imshow(decoded_imgs_changed_2[i].reshape(28, 28))

    plt.gray()

    plt.title('Changed-2')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

plt.show()
f_mnist_model = pd.DataFrame(columns = ['val_loss', 'loss', 'epoch', 'mode', 'error_min'])

f_mnist_model = create_model(model_df = f_mnist_model, list_gng_models = [history_default_f, history_changed_1_f,history_changed_2_f], 

                          list_modes = ["default", "changed-1","changed-2" ] )



f_mnist_model_melt=pd.melt(f_mnist_model, ['epoch', 'mode', 'error_min'])
f_mnist_model.to_csv('./agora_vai.csv')
# Reduzingdo em 64 componentes

pca_f_mnist = decomposition.PCA(n_components = 8)

pca_f_mnist_reduced = pca_f_mnist.fit_transform(x_test_f)
# Camada de codificação

encoded_ae_f = Dense(8, activation='linear')(input_img_f)



# Camada de decoficação

decoded_ae_f = Dense(784, activation='linear')(encoded_ae_f)



# Criação do autoencoder

autoencoder_linear_f = Model(input_img_f, decoded_ae_f)
autoencoder_linear_f.summary()
# Definição da função de otimização e perda

autoencoder_linear_f.compile(optimizer='adadelta', loss='mse')



# Treino do modelo

history_linear_f = autoencoder_linear_f.fit(x_train_f, x_train_f,

                epochs=50,

                batch_size=256,

                shuffle=True,

                validation_data=(x_test_f, x_test_f))
predict_linear_f = autoencoder_linear_f.predict(x_test_f)

predict_pca_f = reconstruction(pca_f_mnist_reduced, pca_f_mnist)
n = 10 

plt.figure(figsize=(20, 10))

for i in range(n):

    # display original

    ax = plt.subplot(3, n, i + 1)

    plt.imshow(x_test_f[i].reshape(28, 28))

    plt.gray()

    plt.title('Img original')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(3, n, i + 1 + n)

    plt.imshow(predict_pca_f[i].reshape(28, 28))

    plt.gray()

    plt.title('PCA')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

      # display reconstruction

    ax = plt.subplot(3, n, i + 1 +  2*n)

    plt.imshow(predict_linear_f[i].reshape(28, 28))

    plt.gray()

    plt.title('AE-Linear')

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

    

       

plt.show()