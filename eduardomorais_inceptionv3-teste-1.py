from keras.models import Model

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D,MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from keras.applications import InceptionV3

import numpy as np

import itertools

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import tensorflow as tf



# monta a matriz de confusão que mostra a relação da classe original e a classe prevista

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()



# monta os graficos da loss e da accuracy

def plot_training_curves(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    

    epochs = range(1, len(acc) + 1)





    plt.title('Training and validation loss')

    plt.plot(epochs, loss, 'bo', label='Training loss', color="r", linestyle = ":")

    plt.plot(epochs, val_loss, 'b', label='Validation loss', color="g" )

    

    plt.legend()

    plt.figure()

    

    plt.title('Training and validation accuracy')

    plt.plot(epochs, acc, 'bo', label='Training acc', color="b", linestyle = ":")

    plt.plot(epochs, val_acc, 'b', label='Validation acc', color="g")

    

    plt.legend()

    plt.figure()

    

    plt.show()



def print_evaluation(loss_value, accuracy_value):

    print ('Loss value: ', loss_value)

    print ('Accuracy value: ', accuracy_value)



    

def print_results(cm):

    tp = cm[0][0] 

    tn = cm[1][1]

    fn = cm[0][1]

    fp = cm[1][0]

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    sensitivity = tp / (tp + fn)

    specificity = tn / (tn + fp)

    print("Accuracy: %f \n Sensitivity : %f \n Specificity: %f" %(accuracy,sensitivity,specificity))

    



def fine_tune_inception():

    inception_model = InceptionV3(include_top=False, weights='imagenet')

    x = inception_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)

    x = Dropout(0.3)(x)



    predictions = Dense(2, activation='softmax')(x)



    model = Model(inputs=inception_model.input, outputs=predictions)

   

    #trava todas as camadas menos as ultimas 23, as primeiras camadas tem neuronios que procuram por bordas, contornos...

    # as ultimas 23 vao ser treinadas e os pesos ajustados por backpropagation quando poe a rede pra treinar

    for layer in model.layers[:-75]:

        layer.trainable = False



    return model



# salva o modelo em json pra nao ter que montar a estrutura da rede de novo

def save_model(model, file_path):

    model_json = model.to_json()

    with open('model.json', 'w') as json_file:

        json_file.write(model_json)

    

def train_network(model, class_weights, val_steps, callbacks, training_path, validation_path, val_batch_size, train_steps):

    



    # método que treina a rede, passando as imagens de treinamento e validação geradas acima

    return model.fit_generator(train_batches,

                              steps_per_epoch = train_steps,

                              class_weight = class_weights,

                              validation_data = valid_batches,

                              validation_steps = val_steps,

                              epochs = 1,

                              verbose = 1,

                              callbacks = callbacks)





def fine_tune_inceptionV3(train_batches, train_steps, class_weight, valid_batches, val_steps, file_path, callbacks):

    

    inception_model = InceptionV3(include_top=False, weights='imagenet')

    

    x = Conv2D(filters = 16, kernel_size = 3 , activation = 'relu', input_shape = (299, 299, 3))

    

    x = Conv2D(filters = 32, kernel_size = 3 , activation = 'relu')

    

    x = Conv2D(filters = 64, kernel_size = 3 , activation = 'relu')

    

    x = Conv2D(filters = 128, kernel_size = 3 , activation = 'relu')

    

    x = MaxPooling2D(pool_size = 3)

    

    x = inception_model.output

    

    x = Dropout(0.2)(x)

    

    x = Dense(128, activation='relu')(x)

    

    x = GlobalAveragePooling2D()(x)

    

    x = Dropout(0.2)(x)



    predictions = Dense(2, activation='softmax')(x)



    model = Model(inputs=inception_model.input, outputs=predictions)

   

    #for layer in inception_model.layers: layer.trainable = False

   

    #trava todas as camadas menos as ultimas 23, as primeiras camadas tem neuronios que procuram por bordas, contornos...

    # as ultimas 23 vao ser treinadas e os pesos ajustados por backpropagation quando poe a rede pra treinar

    

    model.compile(Adam(lr = 0.000095), loss = 'categorical_crossentropy', metrics = ['accuracy'])



    model.fit_generator(train_batches,

                              steps_per_epoch = train_steps,

                              class_weight = class_weights,

                              validation_data = valid_batches,

                              validation_steps = val_steps,

                              epochs = 50,

                              verbose = 1,

                              callbacks = callbacks)



    for layer in model.layers[:-75]: layer.trainable = False





    model.compile(Adam(lr = 0.000095), loss = 'categorical_crossentropy', metrics = ['accuracy'])





    history = model.fit_generator(train_batches,

                              steps_per_epoch = train_steps,

                              class_weight = class_weights,

                              validation_data = valid_batches,

                              validation_steps = val_steps,

                              epochs = 50,

                              verbose = 1,

                              callbacks = callbacks)



    return model, history

    

    

    


    

    # arquivo pra salvar os pesos(aprendizado do modelo)

file_path = 'weights.h5'





# caminhos

training_path = '../input/dermmel/DermMel/train_sep'

validation_path = '../input/dermmel/DermMel/valid'

test_path = '../input/dermmel/DermMel/test'



#numero de imagens de treinamento/validação e teste

num_train_samples = 10682

num_val_samples = 3562

num_test_samples = 3562



# tamanho do batch de treinamento/validacao/teste

# tem que ser baixo pq cpu nao aguenta numeros altos, em gpu usa numeros em potencia de 2

# 8, 16, 32, 64, 128

train_batch_size = 16

val_batch_size = 16

test_batch_size = 16



# numero de passos em cada iteracao(epoca) no treinamento

# por convenção nego usa o total de imagem do tipo: treinamento/validacao/teste dividido pelo tamanho do batch

train_steps = np.ceil(num_train_samples / train_batch_size)

val_steps = np.ceil(num_val_samples / val_batch_size)

test_steps = np.ceil(num_val_samples / val_batch_size)



# aqui fala pra dar uma importancia maior pra melanoma pq é a classe mais importante pra descobrir

# por padrao isso nao existe e todos os pesos de todas as classes sao iguais, isso faz com que a rede entenda que a classe 0, melanoma é mais importante

class_weights = {

        0: 5.1, # melanoma

        1: 1.0 # non-melanoma

}



# batches de treinamento, esse metodo gera imagens a partir do diretorio training_path e faz um rescale 1./255 pra deixar todos os pixels com valor entre 0 - 1 pra agilizar a computação

train_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(training_path,

                                    target_size = (299, 299),

                                    batch_size = val_batch_size,

                                    class_mode = 'categorical')

# batches de validacao, esse metodo gera imagens a partir do diretorio validation_path e faz um rescale 1./255 pra deixar todos os pixels com valor entre 0 - 1 pra agilizar a computação

valid_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(validation_path,

                                    target_size = (299, 299),

                                    batch_size = val_batch_size,

                                    class_mode = 'categorical')





# igual os que tao no metodo train_network()

test_batches = ImageDataGenerator(rescale = 1./255).flow_from_directory(test_path,

                                    target_size = (299, 299),

                                    batch_size = test_batch_size,

                                    class_mode = 'categorical',

                                    shuffle = False)

                                    
# funcoes usadas no treinamento

callbacks = [

        # salva a configuração dos pesos sempre que a accuracy é melhor

        ModelCheckpoint(file_path, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max'),

        # reduz a learning rate a cada 2 epocas que o valor da accuracy nao melhora

        ReduceLROnPlateau(monitor = 'val_acc', factor = 0.5, patience = 10, verbose = 1, mode = 'max', min_lr = 0.0000314159265359),

        # para o treinamento se por 5 épocas seguidas o valor da loss não abaixar

        EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 50, verbose = 1)

        ]



# treina a rede e recebe todos os dados que ela salva

model, history = fine_tune_inceptionV3(train_batches, train_steps, class_weights, valid_batches, val_steps, file_path, callbacks)

save_model(model, file_path)

# classes MELANOMA, NON-MELANOMA

test_labels = test_batches.classes



# essa função pega as imagens geradas pelo test_batches e passa elas pela rede e salvando o que a rede prevê pra cada imagem

predictions = model.predict_generator(test_batches, steps = val_steps, verbose = 1)

# método do scikitlearn que gera a matriz de confusao passando as classes e as previsoes

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))



print_results(cm)



plot_confusion_matrix(cm, ['melanoma', 'non-melanoma'])
plot_training_curves(history)