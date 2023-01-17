from keras import backend as K
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding
from keras.layers import Dropout, Conv2D, UpSampling2D, LeakyReLU, GaussianNoise, Multiply, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam, RMSprop
from keras.models import Model, Sequential
from keras.utils import to_categorical
import warnings  
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from numpy.random import randn
from numpy import asarray

%matplotlib inline
img_shape = (28, 28, 1)
dim = 100
num_classes = 10

# Keičiami parametrai:
# Po 10 realių vaizdų klasėje:
subset_size = 100 # Realių duomenų bendras kiekis. Kiekis kiekvienoje klasėje: subset_size/num_classes
# Po 30 realių vaizdų klasėje:
#subset_size = 300
# Po 50 realių vaizdų klasėje:
#subset_size = 500
# Po 70 realių vaizdų klasėje:
#subset_size = 700
# Po 100 realių vaizdų klasėje:
#subset_size = 1000

# CDCGAN mokymui
iterations = 5000 # Kiek iteracijų apmokyti CDCGAN
batch_size = 32 # Batch dydis CDCGAN tinklo mokymui
show_results = 500 # Kas kiek epochų spausdinti CDCGAN gaunamus rezultatus
synth_size = 30000 # Kiek kurti sintetinių paveikslėlių. Kiekis kiekvienoje klasėje: synth_size/num_classes

# Klasifikavimo tinklo treniravimui
epochs = 100 # Epochų skaičius klasifikavimo tinklo treniravimui su realiais duomenimis
epochs_s = 20 # Epochų skaičius klasifikavimo tinklo treniravimui su realiais + sintetiniais duomenimis
b_size = 32 # Batch dydis klasifikavimo tinklo mokymui
g_noise = Input(shape=(dim,))
g_labels = Input(shape=(1,))

label_embedding = Flatten()(Embedding(num_classes+1, num_classes*10)(g_labels))
model_input = Multiply()([g_noise, label_embedding])
 
gen = Dense(7*7*256)(model_input)
gen = BatchNormalization()(gen)
gen = LeakyReLU(0.2)(gen)
gen = Reshape((7,7,256))(gen)
gen = UpSampling2D()(gen)

gen = Conv2D(128, 5, padding='same')(gen)
gen = LeakyReLU(0.2)(gen)
gen = BatchNormalization()(gen)
gen = UpSampling2D()(gen)

gen = Conv2D(64, 3, padding='same')(gen)
gen = LeakyReLU(0.2)(gen)
gen = BatchNormalization()(gen)

gen_output = Conv2D(1, 3,  padding='same', activation='tanh')(gen)
gen = Model(inputs=[g_noise, g_labels], outputs=gen_output, name='Generatorius')
gen.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, decay=1e-6))
gen.summary()
discriminator_input = Input(shape=(28,28,1))

dis = GaussianNoise(0.2)(discriminator_input)
dis = Conv2D(64, 5, padding='same', strides=(2, 2))(dis)
dis = BatchNormalization(momentum=0.5)(dis)
dis = LeakyReLU(0.2)(dis)

dis = GaussianNoise(0.2)(dis)
dis = Conv2D(128, 5, padding='same', strides=(2, 2))(dis)
dis = BatchNormalization(momentum=0.5)(dis)
dis = LeakyReLU(0.2)(dis)

dis = Flatten()(dis)
dis = Dense(1000)(dis)
dis = LeakyReLU(0.2)(dis)
dis = Dropout(0.3)(dis)
dis_embedding = Flatten()(Embedding(num_classes+1, num_classes*100)(g_labels))
discriminator_merge = Multiply()([dis, dis_embedding])
dis = Dense(128)(discriminator_merge)
dis = LeakyReLU(0.2)(dis)
dis = Dropout(0.3)(dis)
dis_output = Dense(1, activation="sigmoid")(dis)

disc = Model(inputs=[discriminator_input, g_labels], outputs=dis_output, name='Diskriminatorius')
disc.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam())
disc.summary()
# Generatoriui pateikiamas vektorius ir žymė, pagal tai jis sugeneruoja atitinkamą skaičiaus paveikslėlį
z = Input(shape=(dim,))
label = Input(shape=(1,))
img = gen([z, label])

# Diskriminatoriaus svoriai išlaikomi pastovūs kol apmokomas generatorius
disc.trainable = False
prediction = disc([img, label])
# CDCGAN modelis
cdcgan = Model([z, label], prediction)
cdcgan.compile(loss='binary_crossentropy', optimizer=Adam())
def show_result_images(rows=2, columns=5):
    z = np.random.normal(0, 1, (rows * columns, dim))
    labels = np.arange(0, 10).reshape(-1, 1)
    gen_imgs = gen.predict([z, labels])
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(rows, columns, figsize=(10,4), sharey=True, sharex=True)
    itr = 0
    for i in range(rows):
        for j in range(columns):
            axs[i,j].imshow(gen_imgs[itr, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            axs[i,j].set_title("Skaičius: %d" % labels[itr])
            itr += 1
def load_mnist_data():
    with np.load('../input/mnist-numpy/mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)
def load_real():
    (X_train, y_train), (_, _) = load_mnist_data()
    X_train = (X_train - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    return [X_train, y_train]
def subset(dataset, n_samples=100, n_classes=10):
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = int(n_samples/n_classes)
    for i in range(n_classes):
        X_with_class = X[y == i]
        ix = randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(y_list)
def train_cdcgan(X_train, y_train, iterations, batch_size, show_results):

    real = np.ones(shape=(batch_size, 1))
    fake = np.zeros(shape=(batch_size, 1))
    
    for iteration in range(iterations):
        
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]
        
        z = np.random.normal(0, 1, size=(batch_size, dim))
        gen_imgs = gen.predict([z, labels])
        
        disc.train_on_batch([imgs, labels], real)
        disc.train_on_batch([gen_imgs, labels], fake)
        
        z = np.random.normal(0, 1, size=(batch_size, dim))
        labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
        cdcgan.train_on_batch([z, labels], real)
        
        if iteration % show_results == 0:
            print('Vykdoma iteracija:', iteration)
            show_result_images()
dataset = load_real()
X_train, y_train = subset(dataset, subset_size)

train_cdcgan(X_train, y_train, iterations, batch_size, show_results)
def big_data_CNN():
    cnn_input = Input(shape=(28,28,1))

    x = Conv2D(32, 3, padding='same', activation='relu')(cnn_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 5, padding='same', activation='relu', strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 5, padding='same', activation='relu', strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    cnn_output = Dense(10, activation='softmax')(x)
    cnn = Model(inputs=cnn_input, outputs=cnn_output)
    cnn.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer=Adam())
    return cnn
def small_data_CNN():
    cnn_input = Input(shape=(28,28,1))
    x = Conv2D(32, 5, padding='same', activation='relu')(cnn_input)
    x = Dropout(0.3)(x)
    x = Conv2D(64, 5, padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    cnn_output = Dense(10, activation='softmax')(x)
    cnn = Model(inputs=cnn_input, outputs=cnn_output)
    cnn.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer=RMSprop(lr=1e-4))
    return cnn
class LossAccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

history = LossAccHistory()
s_history = LossAccHistory()
(_, _), (X_test, y_test) = load_mnist_data()
X_test = np.expand_dims(X_test, axis=3)
X_test = (X_test - 127.5) / 127.5
y_test = to_categorical(y_test)
y_train_cnn = to_categorical(y_train)
cnn = small_data_CNN() #big_data_CNN() - pakeičiamas vietoj small_data_CNN(), kai apmokinama su 1000 realių duomenų
cnn.fit(x=X_train, y=y_train_cnn, batch_size=b_size, validation_data=(X_test, y_test), shuffle=True, verbose=2, epochs=epochs, callbacks=[history])
loss = history.losses
val_loss = history.val_losses
acc = history.acc
val_acc = history.val_acc

print('Aukščiausias pasiektas klasifikavimo tikslumas:', max(val_acc))

plt.xlabel('Epocha')
plt.ylabel('Tikslumas')
plt.title('Tikslumo raida su realiais duomenimis')
plt.plot(acc, 'mediumvioletred', label='Mokymosi')
plt.plot(val_acc, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs)[1::20])
plt.legend()
plt.ylim(0, 1.05)
plt.show()

plt.xlabel('Epocha')
plt.ylabel('Nuostolis')
plt.title('Nuostolių raida su realiais duomenimis')
plt.plot(loss, 'mediumvioletred', label='Mokymosi')
plt.plot(val_loss, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs)[1::20])
plt.legend()
plt.show()
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    latent_input = randn(latent_dim * n_samples)
    g_input = latent_input.reshape(n_samples, latent_dim)
    labels = randint(0, n_classes, n_samples)
    return [g_input, labels]

def save_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    plt.show()
latent_points, _ = generate_latent_points(dim, synth_size)
labels_gen = asarray([x for _ in range(int(synth_size/10)) for x in range(10)])
X_gen = gen.predict([latent_points, labels_gen])
X_plot = (X_gen + 1) / 2.0
save_plot(X_plot, 10)
y_gen = to_categorical(labels_gen)
y_both = np.concatenate([y_gen, y_train_cnn])

X_both = np.concatenate([X_gen, X_train])
cnn_both = big_data_CNN()
cnn_both.fit(x=X_both, y=y_both, batch_size=b_size, validation_data=(X_test, y_test), shuffle=True, verbose=2, epochs=epochs_s, callbacks=[s_history])
loss = s_history.losses
val_loss = s_history.val_losses
acc = s_history.acc
val_acc = s_history.val_acc

print('Aukščiausias pasiektas klasifikavimo tikslumas:', max(val_acc))

plt.xlabel('Epocha')
plt.ylabel('Tikslumas')
plt.title('Tikslumo raida su realiais + sintetiniais duomenimis')
plt.plot(acc, 'mediumvioletred', label='Mokymosi')
plt.plot(val_acc, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs_s)[1::2])
plt.legend()
plt.ylim(0, 1.05)
plt.show()

plt.xlabel('Epocha')
plt.ylabel('Nuostolis')
plt.title('Nuostolių raida su realiais + sintetiniais duomenimis')
plt.plot(loss, 'mediumvioletred', label='Mokymosi')
plt.plot(val_loss, 'lightseagreen', label='Validacijos')
plt.xticks(range(0,epochs_s)[1::2])
plt.legend()
plt.show()