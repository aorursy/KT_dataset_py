import pandas as pd

# Leitura dos dataframes dos arquivos csv
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.head()
test_data.head()
import matplotlib.pyplot as plt

labels=train_data['benign_malignant'].value_counts().index
values=train_data['benign_malignant'].value_counts().values

plt.title('Número de exemplos de cada classe')
nclasses = plt.bar(labels, values)
nclasses[0].set_color('tab:orange')
plt.show()

print("Número de exemplos benignos: {}".format(values[0]))
print("Número de exemplos malignos: {}".format(values[1]))
import cv2
import numpy as np

benign_samples = train_data[train_data['benign_malignant']=='benign'].sample(20)

fig, ax = plt.subplots(5,4, figsize=(10,8))

for i in range(len(benign_samples)):
    img=cv2.imread(str("train/" + benign_samples['image_name'].iloc[i]+'.jpg'))
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    ax[i//4, i%4].imshow(img)
    ax[i//4, i%4].axis('off')

fig.suptitle('Imagens de amostras benignas', fontsize=22)       
plt.show()
malignant_samples = train_data[train_data['benign_malignant']=='benign'].sample(20)
fig, ax = plt.subplots(5,4, figsize=(10,8))

for i in range(len(malignant_samples)):
    img=cv2.imread(str("train/" + malignant_samples['image_name'].iloc[i]+'.jpg'))
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    ax[i//4, i%4].imshow(img)
    ax[i//4, i%4].axis('off')
        
fig.suptitle('Imagens de amostras malignas', fontsize=22)       
plt.show()
# Imports
import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


from tensorflow.keras.metrics import TruePositives
from tensorflow.keras.metrics import FalsePositives
from tensorflow.keras.metrics import TrueNegatives
from tensorflow.keras.metrics import FalseNegatives
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import AUC

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import roc_auc_score
# Leitura dos dataframes dos arquivos csv
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Adicionando a extensão aos nomes das imagens
train_data['image_name'] = train_data['image_name'].astype(str)+".jpg"
test_data['image_name'] = test_data['image_name'].astype(str)+".jpg"
image_height = 32
image_width = 32
batch_size = 32

# Carregando as imagens de treinamento com ImageDataGenerator
image_datagen = ImageDataGenerator(rescale = 1./255,          # normaliza valores dos pixels da imagem entre 0-1
                                   validation_split = 0.3)    # divide os dados do dataset em uma proporção de treinamento e validação


train_generator = image_datagen.flow_from_dataframe(dataframe=train_data,                   # dataframe com dados da imagem
                                                    directory="train",                      # diretório com as imagens
                                                    x_col="image_name",                     # nome da coluna do dataframe com os nomes das imagens
                                                    y_col="benign_malignant",               # nome da coluna do dataframe com a especificação das classes
                                                    class_mode="binary",                    # modo binário, irá selecionar as imagens em duas classes
                                                    target_size=(image_height,image_width), # tamanho das imagens de treinamento
                                                    batch_size=batch_size,                  # quantidade de imagens por pacote
                                                    subset="training",                      # subset de treinamento ou validação
                                                    color_mode="rgb")                       # modo de carregamento da imagem em 3 canais RGB

validation_generator = image_datagen.flow_from_dataframe(dataframe=train_data,
                                                         directory="train",
                                                         x_col="image_name",
                                                         y_col="benign_malignant",
                                                         class_mode="binary",
                                                         target_size=(image_height,image_width),
                                                         batch_size=batch_size,
                                                         subset="validation",
                                                         color_mode="rgb")


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(7,7), input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 100, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.summary()
METRICS = [
      TruePositives(name='tp'),
      FalsePositives(name='fp'),
      TrueNegatives(name='tn'),
      FalseNegatives(name='fn'), 
      BinaryAccuracy(name='accuracy'),
      Precision(name='precision'),
      Recall(name='recall'),
      AUC(name='auc'),
]
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = METRICS)
early_stopping = EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=3,
    mode='max',
    restore_best_weights=True)
# Fazer o Treinamento
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

model.fit(train_generator,
          steps_per_epoch=STEP_SIZE_TRAIN,
          validation_data=validation_generator,
          validation_steps=STEP_SIZE_VALID,
          epochs=50,
          callbacks=early_stopping,
          workers = 16)

model.save_weights('simple_model_weights.h5')
validation_predictions = model.predict(validation_generator,
                                       verbose=1,
                                       workers=16)
val_auc = roc_auc_score(y_true = validation_generator.classes, y_score=validation_predictions)
print(val_auc)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes, title):
    acc = accuracy_score(y_true, y_pred)
    title = title + " (Acurácia: " + str("{:10.4f}".format(acc)) + ")"

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_df = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize=(5.5,4))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
    plt.title(title)
    plt.ylabel('Label verdadeira')
    plt.xlabel('Label predita')
    plt.show()

# Plota a matriz de confusão dos dados de validação
plot_confusion_matrix(validation_generator.classes,
                      (validation_predictions>0.5).astype(int),
                      ['benign','malignant'], "Matriz de confusão")
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_data,
                                                  directory="test",
                                                  x_col="image_name",
                                                  y_col=None,
                                                  class_mode=None,
                                                  target_size=(image_height,image_width),
                                                  batch_size=batch_size)

# Realiza predição
predictions = model.predict(test_generator,
                            verbose=1,
                            workers=16)
submission = pd.DataFrame(test_data['image_name'].str.replace(r'.jpg', ''))
submission['target']=predictions

# Cria arquivo de submissão
submission.to_csv('simple_model_submission.csv',index=False)
# Carregamento das imagens

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['image_name'] = train_data['image_name'].astype(str)+".jpg"
test_data['image_name'] = test_data['image_name'].astype(str)+".jpg"

batch_size = 16       # Tamanho do pacote
image_height = 224    # Altura da imagem
image_width = 224     # Largura da imagem

# Realiza as augmentações nas imagens
image_datagen = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.3)

train_generator = image_datagen.flow_from_dataframe(dataframe=train_data,
                                                    directory="train",
                                                    x_col="image_name",
                                                    y_col="benign_malignant",
                                                    class_mode="binary",
                                                    target_size=(image_height,image_width),
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    color_mode="rgb")

validation_generator = image_datagen.flow_from_dataframe(dataframe=train_data,
                                                         directory="train",
                                                         x_col="image_name",
                                                         y_col="benign_malignant",
                                                         class_mode="binary",
                                                         target_size=(image_height,image_width),
                                                         batch_size=batch_size,
                                                         subset="validation",
                                                         color_mode="rgb")
# Criação do modelo

model = Sequential()
model.add(Conv2D(input_shape=(image_height,image_width,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=1, activation="softmax"))
model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = METRICS)
# Realiza treinamento

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

model.fit(train_generator,
          steps_per_epoch=STEP_SIZE_TRAIN,
          validation_data=validation_generator,
          validation_steps=STEP_SIZE_VALID,
          epochs=50,
          callbacks=early_stopping,
          workers = 16)

model.save_weights('vgg_model_weights.h5')
validation_predictions = model.predict(validation_generator,
                                       verbose=1,
                                       workers=16)
val_auc = roc_auc_score(y_true = validation_generator.classes, y_score=validation_predictions)
print(val_auc)
# Plota a matriz de confusão dos dados de validação
plot_confusion_matrix(validation_generator.classes,
                      (validation_predictions>0.5).astype(int),
                      ['benign','malignant'], "Matriz de confusão")
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_data,
                                                  directory="test",
                                                  x_col="image_name",
                                                  y_col=None,
                                                  class_mode=None,
                                                  target_size=(image_height,image_width),
                                                  batch_size=batch_size)

predictions = model.predict(test_generator,
                            verbose=1,
                            workers=16)


submission = pd.DataFrame(test_data['image_name'].str.replace(r'.jpg', ''))
submission['target'] = predictions

submission.to_csv('vgg_model_submission.csv',index=False)
# Carregamento das imagens
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['image_name'] = train_data['image_name'].astype(str)+".jpg"
test_data['image_name'] = test_data['image_name'].astype(str)+".jpg"


batch_size = 16       # Tamanho do pacote
image_height = 224    # Altura da imagem
image_width = 224     # Largura da imagem

# Realiza as augmentações nas imagens
image_datagen = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.3,
                                   rotation_range=30,      # Rotaciona aleatóriamente imagens em 30 graus
                                   zoom_range=0.15,        # Zoom aleatório
                                   width_shift_range=0.2,  # Shift de largura aleatório
                                   height_shift_range=0.2, # Shift de altura aleatório
                                   horizontal_flip=True)   # Flip horizontal aleatório

train_generator = image_datagen.flow_from_dataframe(dataframe=train_data,
                                                    directory="train",
                                                    x_col="image_name",
                                                    y_col="benign_malignant",
                                                    class_mode="binary",
                                                    target_size=(image_height,image_width),
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    color_mode="rgb")

validation_generator = image_datagen.flow_from_dataframe(dataframe=train_data,
                                                         directory="train",
                                                         x_col="image_name",
                                                         y_col="benign_malignant",
                                                         class_mode="binary",
                                                         target_size=(image_height,image_width),
                                                         batch_size=batch_size,
                                                         subset="validation",
                                                         color_mode="rgb")
# Criação do modelo

from tensorflow.keras.applications.resnet50 import ResNet50
# load model

model = Sequential()
model.add(ResNet50(input_shape=(image_height, image_width, 3), classes=2,include_top=False, weights=None))
model.add(Flatten())
model.add(Dense(units=1, activation="sigmoid"))


# summarize the model
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = METRICS)
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_generator.classes),
                                                 train_generator.classes)

class_weights = dict(zip(np.unique(train_generator.classes), class_weights))
print(class_weights)
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

model.fit(train_generator,
          steps_per_epoch=STEP_SIZE_TRAIN,
          validation_data=validation_generator,
          validation_steps=STEP_SIZE_VALID,
          epochs=50,
          callbacks=early_stopping,
          workers = 16,
          class_weight=class_weights)

model.save_weights('class_weights_balanced_weights.h5')
validation_predictions = model.predict(validation_generator,
                                       verbose=1,
                                       workers=16)
val_auc = roc_auc_score(y_true = validation_generator.classes, y_score=validation_predictions)
print(val_auc)
# Plota a matriz de confusão dos dados de validação
plot_confusion_matrix(validation_generator.classes,
                      (validation_predictions>0.5).astype(int),
                      ['benign','malignant'], "Matriz de confusão")
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_data,
                                                  directory="test",
                                                  x_col="image_name",
                                                  y_col=None,
                                                  class_mode=None,
                                                  target_size=(image_height,image_width),
                                                  batch_size=batch_size)

predictions = model.predict(test_generator,
                            verbose=1,
                            workers=16)


submission = pd.DataFrame(test_data['image_name'].str.replace(r'.jpg', ''))
submission['target'] = predictions

submission.to_csv('class_weight_balanced_submission.csv',index=False)
# Leitura dos dataframes dos arquivos csv
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Adicionando a extensão aos nomes das imagens
train_data['image_name'] = train_data['image_name'].astype(str)+".jpg"
test_data['image_name'] = test_data['image_name'].astype(str)+".jpg"

# Carregamento das imagens
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data['image_name'] = train_data['image_name'].astype(str)+".jpg"
test_data['image_name'] = test_data['image_name'].astype(str)+".jpg"


batch_size = 32       # Tamanho do pacote
image_height = 224    # Altura da imagem
image_width = 224     # Largura da imagem

# Realiza as augmentações nas imagens
image_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=30,      
                                   zoom_range=0.15,        
                                   width_shift_range=0.2,  
                                   height_shift_range=0.2, 
                                   horizontal_flip=True)   

# Nesse treinamento, todas as imagens serão usadas como dataset de treinamento
train_generator = image_datagen.flow_from_dataframe(dataframe=train_data,
                                                    directory="train",
                                                    x_col="image_name",
                                                    y_col="benign_malignant",
                                                    class_mode="binary",
                                                    target_size=(image_height,image_width),
                                                    batch_size=batch_size, subset="training",
                                                    color_mode="rgb")
import efficientnet.tfkeras as efn
# load model

model = Sequential([
        efn.EfficientNetB0(
            input_shape=(image_height, image_width, 3),
            weights='imagenet',
            include_top=False
        ),
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])



# summarize the model
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = METRICS)
early_stopping = EarlyStopping(
    monitor='auc', 
    verbose=1,
    patience=3,
    mode='max',
    restore_best_weights=True)
STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size

model.fit(train_generator,
          steps_per_epoch=STEP_SIZE_TRAIN,
          epochs=50,
          callbacks=early_stopping,
          workers = 16)
model.save_weights('tranfer_learning_weights.h5')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_data,
                                                  directory="test",
                                                  x_col="image_name",
                                                  y_col=None,
                                                  class_mode=None,
                                                  target_size=(image_height,image_width),
                                                  batch_size=batch_size)

predictions = model.predict(test_generator,
                            verbose=1,
                            workers=16)


submission = pd.DataFrame(test_data['image_name'].str.replace(r'.jpg', ''))
submission['target'] = predictions

submission.to_csv('effnet3_model_submission.csv',index=False)
# Imports
from pathlib import Path

from tensorflow.data import Dataset
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
class MixedDataGenerator(utils.Sequence):

    def __init__(self, values: pd.DataFrame,
                images: pd.Series, directory: str, labels: pd.Series=None,
                target_size: tuple=(32,32), batch_size: int=32, shuffle: bool=True):
        'Inicialização'

        self.values = values
        self.labels = labels
        self.images = images
        self.directory = Path(directory)
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_files = images.count()
        self.columns = self.values.columns
        self.indices = None
        self.values_gen = ImageDataGenerator()
        self.labels_indices = None
        if labels is not None:
            self.labels_indices = list(self.labels.unique())

        self.on_epoch_end()

    
    def __len__(self):
        'Indica a quantidades de pacotes por épocas'

        return (self.num_files + self.batch_size - 1) // self.batch_size

    
    def __getitem__(self, index):
        'Gera um pacote de dados'

        # Gera os índices do pacote
        if index >= len(self):
            raise ValueError('Asked to retrieve element {index}, '
                             'but the Sequence '
                             'has length {length}'.format(index=index,
                                                          length=len(self)))
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        x_images = [self.__process_image(image) for image in self.images.loc[indices]]
        images = np.array(x_images)
        x_values = self.values.loc[indices]
        values = np.array(x_values)
        if self.labels is not None:
            y_labels = [self.labels_indices.index(label) for label in self.labels.loc[indices]]
            labels = np.eye(len(self.labels_indices))[y_labels]
        # Gera os dados
        if self.labels is not None:
            return [images, values], labels
        else:
            return [images, values]


    def on_epoch_end(self):
        'Atualiza os índices após cada época'

        self.indices = np.arange(len(self.values))
        if self.shuffle:
            np.random.shuffle(self.indices)

    
    def __process_image(self, image_name):
        'Lê a imagem da memória'

        image = load_img(Path(self.directory, image_name), target_size=self.target_size)
        image = self.values_gen.apply_transform(image, dict(rescale=1./255))            # normaliza valores dos pixels da imagem entre 0-1
        array = img_to_array(image)
        image.close()

        return array
def treatNan(data):
    # remove nan na coluna 'sex'
    data = data[data.sex == data.sex]
    # remove nan na coluna 'age_approx'
    data = data[data.age_approx == data.age_approx]

    # substitui nan na coluna 'anatom_site_general_challenge' pelo atributo 'unknown'
    data.anatom_site_general_challenge.fillna(value='unknown', inplace=True)

    return data
# Leitura dos dataframes dos arquivos csv
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Adicionando a extensão aos nomes das imagens
train_data['image_name'] = train_data['image_name'].astype(str)+".jpg"
test_data['image_name'] = test_data['image_name'].astype(str)+".jpg"

train_data = treatNan(train_data)
test_data = treatNan(test_data)

train_data.head()
test_data.head()
# Separa os dados em treinamento e validação mantendo a proporção entre classes 'benign' e 'malignant'
train_data_ben = train_data[train_data.benign_malignant == 'benign']
train_data_mal = train_data[train_data.benign_malignant == 'malignant']

valid_data_ben = train_data_ben.sample(frac=0.3, random_state=1)
valid_data_mal = train_data_mal.sample(frac=0.3, random_state=1)

valid_data = valid_data_ben.append(valid_data_mal)

train_data.drop(index=valid_data.index)

valid_data = valid_data.reindex(np.random.permutation(valid_data.index))
valid_data.reset_index(drop=True, inplace=True)
train_data.reset_index(drop=True, inplace=True)
# Rótulos dos dados
y_train = train_data.benign_malignant
y_valid = valid_data.benign_malignant

# Dados com atributos para o treinamento
X_train = train_data[['sex', 'age_approx', 'anatom_site_general_challenge']]
X_valid = valid_data[['sex', 'age_approx', 'anatom_site_general_challenge']]
train_sex = X_train['sex'].str.get_dummies()
train_anatom = X_train['anatom_site_general_challenge'].str.get_dummies()

valid_sex = X_valid['sex'].str.get_dummies()
valid_anatom = X_valid['anatom_site_general_challenge'].str.get_dummies()

X_train = X_train.join([train_sex, train_anatom])
X_valid = X_valid.join([valid_sex, valid_anatom])

X_train.drop(columns=['sex', 'anatom_site_general_challenge'], inplace=True)
X_valid.drop(columns=['sex', 'anatom_site_general_challenge'], inplace=True)

train_columns = X_train.columns
valid_columns = X_valid.columns

for column in valid_columns:
    if column not in train_columns:
        X_train[column] = 0

for column in train_columns:
    if column not in valid_columns:
        X_valid[column] = 0
gen_train = MixedDataGenerator(values=X_train,
                            images=train_data['image_name'],
                            directory='train',
                            labels=y_train,
                            target_size=(32, 32),
                            batch_size=32)
gen_valid = MixedDataGenerator(values=X_valid,
                            images=valid_data['image_name'],
                            directory='train',
                            labels=y_valid,
                            target_size=(32, 32),
                            batch_size=32)
def createMLP(dimension):
    model = Sequential()
    model.add(Dense(units=8, input_dim=dimension, activation="relu"))
    model.add(Dense(units=4, activation="relu"))

    return model


def createCNN(shape):
    inputImage = Input(shape=shape)
    x = inputImage
    x = Conv2D(filters=32, kernel_size=(7,7), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Conv2D(filters=20, kernel_size=(3, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Flatten()(x)
    x = Dense(units = 10, activation = 'relu')(x)
    x = Dense(units = 4, activation = 'relu')(x)

    model = Model(inputImage, x)

    return model


#Criação do modelo multi-Input
mlp = createMLP(dimension=len(X_train.columns))
cnn = createCNN((32,32,3))
combinedInput = concatenate([cnn.output, mlp.output])

x = Dense(units=4, activation="relu")(combinedInput)
x = Dense(units=2, activation="sigmoid")(x)

model = Model(inputs=[cnn.input, mlp.input], outputs=x)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()
# Treinamento

model.fit(gen_train,
          epochs=4,
          validation_data=gen_valid,
          workers=16)
model.save_weights('cnn_melanoma.h5')
# Validação

validation_predictions = model.predict(gen_valid,
                                       verbose=1,
                                       workers=16)
validation_predictions = (validation_predictions > 0.5)
# Faz o classification report do modelo aplicado nos dados de validação
from sklearn.metrics import classification_report
report = classification_report(pd.get_dummies(y_valid),
                               validation_predictions)
print(report)
# Mostra a matriz de confusão dos dados de validação
plot_confusion_matrix(pd.get_dummies(y_valid).values.argmax(axis=1),
                      validation_predictions.argmax(axis=1),['benign', 'malignant'], "Matriz de confusão")
# Teste

X_test = test_data.drop(columns=['image_name', 'patient_id'])

test_sex = X_test['sex'].str.get_dummies()
test_anatom = X_test['anatom_site_general_challenge'].str.get_dummies()

X_test = X_test.join([test_sex, test_anatom])

X_test.drop(columns=['sex', 'anatom_site_general_challenge'], inplace=True)

test_columns = X_test.columns

for column in train_columns:
    if column not in test_columns:
        X_test[column] = 0


gen_test = MixedDataGenerator(values=X_test,
                            images=test_data['image_name'],
                            directory='test',
                            target_size=(32, 32),
                            batch_size=32)


predictions = model.predict(gen_test,
                            verbose=1,
                            workers=16)


# predictions é um vetor em que cada elemento é um vetor de dois valores, já que temos duas classes (benigno e maligno).
# Pega-se então a probabilidade do tumor ser maligno
submission = pd.DataFrame(test_data['image_name'].str.replace(r'.jpg', ''))
submission['target']=predictions[:,1]

# Cria arquivo de submissão
submission.to_csv('multi_input_submission.csv',index=False)
# Leitura dos dataframes dos arquivos csv
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Adicionando a extensão aos nomes das imagens
train_data['image_name'] = train_data['image_name'].astype(str)+".jpg"
test_data['image_name'] = test_data['image_name'].astype(str)+".jpg"

train_data = treatNan(train_data)
test_data = treatNan(test_data)


# Separa os dados em treinamento e validação mantendo a proporção entre classes 'benign' e 'malignant'
train_data_ben = train_data[train_data.benign_malignant == 'benign']
train_data_mal = train_data[train_data.benign_malignant == 'malignant']

valid_data_ben = train_data_ben.sample(frac=0.3, random_state=1)
valid_data_mal = train_data_mal.sample(frac=0.3, random_state=1)

valid_data = valid_data_ben.append(valid_data_mal)

train_data.drop(index=valid_data.index)

valid_data = valid_data.reindex(np.random.permutation(valid_data.index))
valid_data.reset_index(drop=True, inplace=True)
train_data.reset_index(drop=True, inplace=True)


# Rótulos dos dados
y_train = train_data.benign_malignant
y_valid = valid_data.benign_malignant

# Dados com atributos para o treinamento
X_train = train_data[['sex', 'age_approx', 'anatom_site_general_challenge']]
X_valid = valid_data[['sex', 'age_approx', 'anatom_site_general_challenge']]


train_sex = X_train['sex'].str.get_dummies()
train_anatom = X_train['anatom_site_general_challenge'].str.get_dummies()

valid_sex = X_valid['sex'].str.get_dummies()
valid_anatom = X_valid['anatom_site_general_challenge'].str.get_dummies()

X_train = X_train.join([train_sex, train_anatom])
X_valid = X_valid.join([valid_sex, valid_anatom])

X_train.drop(columns=['sex', 'anatom_site_general_challenge'], inplace=True)
X_valid.drop(columns=['sex', 'anatom_site_general_challenge'], inplace=True)

train_columns = X_train.columns
valid_columns = X_valid.columns

for column in valid_columns:
    if column not in train_columns:
        X_train[column] = 0

for column in train_columns:
    if column not in valid_columns:
        X_valid[column] = 0

gen_train = MixedDataGenerator(values=X_train,
                            images=train_data['image_name'],
                            directory='train',
                            labels=y_train,
                            target_size=(128,128),
                            batch_size=32)
gen_valid = MixedDataGenerator(values=X_valid,
                            images=valid_data['image_name'],
                            directory='train',
                            labels=y_valid,
                            target_size=(128,128),
                            batch_size=32)
#Criação do modelo multi-Input

mlp = createMLP(dimension=len(X_train.columns))
cnn = createCNN((128,128,3))

combinedInput = concatenate([cnn.output, mlp.output])

x = Dense(units=4, activation="relu")(combinedInput)
x = Dense(units=2, activation="sigmoid")(x)

model = Model(inputs=[cnn.input, mlp.input], outputs=x)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [AUC()])
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

class_weights = dict(zip([0,1], class_weights))
# Treinamento
model.fit(gen_train,
          epochs=4,
          validation_data=gen_valid,
          class_weight=class_weights,
          workers=16)
model.save_weights('cnn_melanoma.h5')
# Validação

validation_predictions = model.predict(gen_valid,
                                       verbose=1,
                                       workers=16)
validation_predictions = (validation_predictions > 0.5)


# class_labels = list(validation_generator.class_indices.keys())
# validation_true = validation_generator.classes

# Faz o classification report do modelo aplicado nos dados de validação
from sklearn.metrics import classification_report
report = classification_report(pd.get_dummies(y_valid),
                               validation_predictions)
print(report)


# Plota a matriz de confusão dos dados de validação
plot_confusion_matrix(pd.get_dummies(y_valid).values.argmax(axis=1),
                      validation_predictions.argmax(axis=1),['benign', 'malignant'], "Matriz de confusão")
# Teste

X_test = test_data.drop(columns=['image_name', 'patient_id'])

test_sex = X_test['sex'].str.get_dummies()
test_anatom = X_test['anatom_site_general_challenge'].str.get_dummies()

X_test = X_test.join([test_sex, test_anatom])

X_test.drop(columns=['sex', 'anatom_site_general_challenge'], inplace=True)

test_columns = X_test.columns

for column in train_columns:
    if column not in test_columns:
        X_test[column] = 0


gen_test = MixedDataGenerator(values=X_test,
                            images=test_data['image_name'],
                            directory='test',
                            target_size=(128, 128),
                            batch_size=32)


predictions = model.predict(gen_test,
                            verbose=1,
                            workers=16)


# predictions é um vetor em que cada elemento é um vetor de dois valores, já que temos duas classes (benigno e maligno).
# Pega-se então a probabilidade do tumor ser maligno
submission = pd.DataFrame(test_data['image_name'].str.replace(r'.jpg', ''))
submission['target']=predictions[:,1]

# Cria arquivo de submissão
submission.to_csv('multi_input_submission2.csv',index=False)