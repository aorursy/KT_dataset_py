import pandas as pd

trainData = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')       
testData = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')  
print('\t Filas,  Columnas', )
print('Train:\t', trainData.shape)
print('Test:\t', testData.shape)
def cehck_nulls(data):
    if data.isnull().any().any() == False:
        return print('los datos NO conetienen valores Null')
    else:
        return print('los datos SI conetienen valores Null')

cehck_nulls(trainData)
cehck_nulls(testData)
trainData.head(4).append(trainData.tail(3))
labels = {  0: "Camiseta / Top",
            1: "Pantalón",
            2: "Jersey",
            3: "Vestido",
            4: "Abrigo",
            5: "Sandalia",
            6: "Camisa",
            7: "Zapatilla de deporte",
            8: "Bolsa",
            9: "Botines"
         }

n_cat = len(labels)

def add_column_from_dict(data, col, new_col, dict_):
    data[new_col] = data[col].map(dict_)
    return data

add_column_from_dict(trainData, 'label', 'labelName', labels)
add_column_from_dict(testData, 'label', 'labelName', labels)
import matplotlib.pyplot as plt

def pie_plot(data, plotTitle):
    
    aux = data['labelName'].value_counts().to_frame('Freq')
    aux['labelName'] = aux.index 
    valores = aux['Freq']
    
    def pct_abs(values):
        def funct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%\n({v:d} it'.format(p = pct,v = val)
        return funct


    plt.figure(figsize = (16,8))

    ax1 = plt.subplot(121, aspect = 'equal')
    aux.plot(kind = 'pie', 
             y = 'Freq', 
             ax = ax1,
             autopct = pct_abs(valores), 
             labels = aux['labelName'], 
             legend = False,
             title = plotTitle,
             fontsize = 10)

    # plot table
    ax2 = plt.subplot(122)
    plt.axis('off')
    plt.show()
    
    
plot1 = pie_plot(trainData,'Distribución de la ropa para el conjunto de datos TRAIN')
plot2 = pie_plot(testData, 'Distribución de la ropa para el conjunto de datos TEST')
plt.show()
import numpy as np

def plot_image_sample(data, label_number, DataSetType, pf, pc):
    
    type_data = ('TRAIN' if DataSetType.lower().find("train") == label_number else 'TEST')
    
    # Obtenemos la etiqueta (diccionario)
    etiqueta = labels[label_number]
    # Eliminamos la primera columna (codigo etiqueta) y la última (nombre etiqueta)
    aux = data[data["label"] == label_number].sample(1)
    aux2 = aux.iloc[:, 1:-1]
    img = np.array(aux2).reshape(pf, pc)

    plt.imshow(img, cmap = 'gray')
    plt.grid(True)
    plot = plt.title('Ropa: ' + str(etiqueta) + '\nDatos: ' + str(type_data))
    

def matrix_image_sample(data, label_number, pf ,pc):
    
    pd.options.display.max_columns = None
    aux = data[data["label"] == label_number].sample(1)
    aux2 = aux.iloc[:, 1:-1]
    img = pd.DataFrame(np.array(aux2).reshape(pf, pc))

    return img 
pf = 28
pc = 28

plot_image_sample(trainData, 9, 'train', pf, pc)
matrix_image_sample(trainData, 9, pf, pc)
plot_image_sample(testData, 3, 'Test', pf, pc)
matrix_image_sample(testData, 3, pf, pc)
import keras

def preprocesamiento(data, pf, pc):
    
    out_Y = keras.utils.to_categorical(data.label, len(labels))
    x_vect = data.values[:,1:-1]  #transformamos el dataFrame en un ndarray, seleccionando solo los píxeles
    x_scaled = x_vect / 255 # Dividimos por 255 por literatura (convergencia del gradiente, evita le colapso)
    n_img = data.shape[0]
    out_X = x_scaled.reshape(n_img, pf, pc, 1) # redimensionamos el vector a (1,784) a (28, 28, 1)  
    
    out_X = out_X.astype(float)
    out_Y = out_Y.astype(float)
    
    return out_X, out_Y
x_train, y_train = preprocesamiento(trainData, pf, pc)
x_test, y_test = preprocesamiento(testData, pf, pc)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.3, random_state = 42)
def proc_data_to_plot(data):

    freq = []
    for i in range(len(data)):
        freq.append(np.argmax(data[i]))
        
    return pd.DataFrame(freq, columns = ['Label'])
    
    
    
Train_labels_to_plot = proc_data_to_plot(Y_train) 
Val_labels_to_plot = proc_data_to_plot(Y_val) 

Train_labels_to_plot = add_column_from_dict(Train_labels_to_plot, 'Label', 'labelName', labels)
Val_labels_to_plot = add_column_from_dict(Val_labels_to_plot, 'Label', 'labelName', labels)


plot1 = pie_plot(Train_labels_to_plot,'Distribución de la ropa para el conjunto de datos TRAIN')
plot2 = pie_plot(Val_labels_to_plot, 'Distribución de la ropa para el conjunto de datos de VALIDACION')
plt.show()
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D



#Parte 1 del modelo
model = Sequential()

LeakyReLU = lambda x: tf.keras.activations.relu(x, alpha=0.1)
model.add(Conv2D(32, 
                 kernel_size = (3, 3),
                 activation = LeakyReLU,
                 padding="same",
                 input_shape=(pf, pc, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))


#Parte 2 del modelo
model.add(Conv2D(64, 
                 kernel_size = (3, 3), 
                 activation = LeakyReLU,
                 padding="same"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))


#Parte 3 del modelo
model.add(Conv2D(128, (3, 3), activation = LeakyReLU))
model.add(Flatten())                               # Flatemos el tensor de pixeles:
model.add(Dense(128, activation = LeakyReLU))
model.add(Dropout(0.3))
model.add(Dense(n_cat, activation = 'softmax'))    # La ultima capa debe ser el nº de lables a predecir


model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics = ['accuracy'])
model.summary()
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG


SVG(model_to_dot(model).create(prog='dot', format='svg'))
batch = 70
epocas = 50
 
train_model = model.fit(X_train, Y_train,
                        batch_size = batch,
                        epochs = epocas,
                        verbose = 1,
                        validation_data = (X_val, Y_val))
score = model.evaluate(x_test, y_test, verbose = 0)
print('Perdida/Loss Test:', score[0])
print('Precision/Accuracy Test:', score[1])
import plotly.graph_objs as go

def interpolation_tracer(x, y, text, mode):
    fig.add_trace(go.Scatter(x = x, 
                             y = y, 
                             name = text,
                             mode = mode))
    fig.update_yaxes(range=[0,1])
    fig.update_xaxes(title_text = 'Épocas')
    fig.update_yaxes(title_text = 'Loss & Accuracy')
    
def layout_plot(Titulo):
    fig.update_layout(title = {'text': Titulo},
                      xaxis_title = "Accuracy",
                      yaxis_title = "Épocas",
                      legend_title = "Leyenda",
                      font = dict(family = "Courier New, monospace",
                                  size = 18,
                                  color = "RebeccaPurple"))
hist = train_model.history
acc = hist['accuracy']
val_acc = hist['val_accuracy']
loss = hist['loss']
val_loss = hist['val_loss']
epochs = list(range(1, len(acc) + 1))
    
fig = go.Figure()
interpolation_tracer(epochs, acc, 'Training accuracy', 'lines+markers')
interpolation_tracer(epochs, val_acc, 'Validation accuracy', 'lines+markers')
layout_plot('<b>Accuracy</b> entrenamiento y validación')
fig.show()

fig = go.Figure()
interpolation_tracer(epochs,loss,'Training loss', 'lines+markers')
interpolation_tracer(epochs,val_loss,'Validation loss', 'lines+markers')
layout_plot('<b>Loss</b> entrenamiento y validación')
fig.show()


pred = model.predict_classes(x_test)
y_true = testData.iloc[:,0].to_numpy()
n = len(pred[:10000])

GoodPred = np.where((pred[:10000] == y_true[:10000]) == True)[0]
BadPred  = np.where((pred[:10000] == y_true[:10000]) == False)[0]

print('Se han predicho correctamente ' + str(GoodPred.shape[0]) + 
      ' clases de ' + str(n) + '.\tAcc: ' + str(round((GoodPred.shape[0]/n)*100, 2)) + '%')

print('Se han predicho erróneamente ' + str(BadPred.shape[0]) +
      ' clases de ' + str(n) + '.\tAcc: ' + str(round((BadPred.shape[0]/n)*100, 2)) + '%')
from sklearn.metrics import confusion_matrix
import itertools

def Matriz_de_confusion(cm, clases,  normalize = False, title = 'Matriz de confusión', cmap = plt.cm.Oranges):
    
    plt.figure(figsize=(10 , 10) , dpi= 70)
    plt.imshow(cm , 
               interpolation = 'nearest' , 
               cmap = cmap ) 
    plt.suptitle(title, fontsize=20)
    tick_marks = np.arange(len(clases))
    plt.xticks(tick_marks, 
               clases,
               rotation = 45 )
    plt.yticks(tick_marks, 
               clases)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]) , range(cm.shape[1]) ):
        plt.text(j, i, format(cm[i, j] , fmt), 
        horizontalalignment = "center" ,
        color="white" if cm[ i, j] > thresh else "black" )
        
    plt.ylabel('Etiquetas reales')
    plt.xlabel('Etiquetas predichas') 
np.set_printoptions(precision = 2)
setLabels = [str(key) + str(': ') + labels[key] for key in labels]


Matriz_de_confusion(confusion_matrix(y_true, pred), 
                    clases = setLabels )