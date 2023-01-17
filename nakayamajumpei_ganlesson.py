import os
import numpy as np
import cv2 as cv # pip install opencv-python
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Reshape, Dense, Flatten, Activation, Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

%matplotlib inline
# Image data directory path 検証用画像データファイルの格納パス
image_path = '../input'
# Print data category (Only "pokemon" will be printed) カテゴリ名を取得（ "pokemon" のみとなる想定）
classes = os.listdir(image_path)
#print(classes)

print(f'Total number of categories: {len(classes)}')

X_train = [] # List for images

# Image size 学習データの画像サイズ
image_size = 64
image_width = image_size
image_height = image_size

# Image plot limit 表示する画像サンプル数
plot_limit = 20
# 画像表示領域のサイズ、グリッド
dim=(4,5)
figsize=(16,8)

plt.figure(figsize=figsize)

# A dictionary which contains class and number of images in that class
counts = {}
for c in classes:
    if c == ".DS_Store": # Only on Mac environment
        continue

    image_class_path = os.path.join(image_path, c)

    # Count image files
    image_files = os.listdir(image_class_path)
    #print(image_files)
    counts[c] = len(image_files)
    
    count = 0
    for image_file in image_files:
        image = cv.imread(os.path.join(image_class_path, image_file))

        try:
            resized = cv.resize(image, (image_width, image_height)) # Resizing images to (image_width, image_height)
            X_train.append(resized) # Adds resized image data to list
            
            # Print image data sample
            if count < plot_limit:
                plt.subplot(dim[0],dim[1],count+1)
                plt.imshow(resized)
                plt.axis('off')

            image = None
            count = count + 1
            
        # If we can't read image - we skip it
        except:
            print(os.path.join(image_class_path, image_file), '[ERROR] can\'t read the file')
            image = None
            continue

plt.tight_layout()
plt.show()
# Number of images in each clsss plot
fig = plt.figure(figsize = (25, 5))
sns.barplot(x = list(counts.keys()), y = list(counts.values())).set_title('Number of images in each class')
plt.xticks(rotation = 90)
plt.margins(x=0)
plt.show()

print(f'Total number of images in dataset: {sum(list(counts.values()))}')
# Convert list with images to numpy array and reshape it 
X_train = np.array(X_train).reshape(-1, image_width, image_height, 3)
print(X_train.shape)

# Scaling data in array
X_train = X_train / 255.0
# @param width: int(image_width / 2)
# @param height: int(image_height) / 2)
def Generator(width, height):
    nch = 200
    model_input = Input(shape=[100])
    #x = Dense(nch*14*14, kernel_initializer='glorot_normal')(model_input) # 100 -> 200*14*14
    x = Dense(nch*width*height, kernel_initializer='glorot_normal')(model_input) # 100 -> 200*64*64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Reshape( [14, 14, nch] )(x) # 200*14*14 -> 14x14x200 (width)x(height)x(channel)
    x = Reshape( [width, height, nch] )(x) # 200*64*64 -> 64x64x200 (width)x(height)x(channel)
    x = UpSampling2D(size=(2, 2))(x) # 64x64x200 -> 128x128x200 // # 14x14x200 -> 28x28x200
    x = Conv2D(int(nch/2), (3, 3), padding='same', kernel_initializer='glorot_uniform')(x) # 128x128x200 -> 128x128x100 // # 28x28x200 -> 28x28x100
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nch/4), (3, 3), padding='same', kernel_initializer='glorot_uniform')(x) # 128x128x100 -> 128x128x50 // # 28x28x100 -> 28x28x50
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform')(x) # 28x28x50 -> 28x28x1
    x = Conv2D(3, (1, 1), padding='same', kernel_initializer='glorot_uniform')(x) # 128x128x50 -> 128x128x3
    model_output = Activation('sigmoid')(x)
    model = Model(model_input, model_output)
    
    print("#### Generator: ")
    model.summary()
    
    return model
def Discriminator(shape, dropout_rate=0.25, opt=Adam(lr=1e-4)):
    model_input = Input(shape=shape) # 128x128x3 // # 28x28x1
    x = Conv2D(256, (5, 5), padding = 'same', kernel_initializer='glorot_uniform', strides=(2, 2))(model_input) # 128x128x3 -> 64x64x256 // # 28x28x1 -> 14x14x256
    x = LeakyReLU(0.2)(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(512, (5, 5), padding = 'same', kernel_initializer='glorot_uniform', strides=(2, 2))(x) # 64x64x256 -> 32x32x512 // # 14x14x256 -> 7x7x512
    x = LeakyReLU(0.2)(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x) # 32x32x512 -> 32*32*512 // # 7x7x512 -> 7*7*512
    x = Dense(256)(x) # 32*32*512 -> 256 // # 7*7*512 -> 256
    x = LeakyReLU(0.2)(x)
    x = Dropout(dropout_rate)(x)
    model_output = Dense(2,activation='softmax')(x) # 256 -> 2
    model = Model(model_input, model_output)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    
    print("#### Discriminator: ")
    model.summary()
    
    return model
generator = Generator(int(image_width / 2), int(image_height / 2))
#print(X_train.shape[1:])
discriminator = Discriminator(X_train.shape[1:])
def combined_network(generator, discriminator, opt=Adam(lr=1e-3)):
    gan_input = Input(shape=[100])
    x = generator(gan_input)
    gan_output = discriminator(x)
    model = Model(gan_input, gan_output)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    # model.summary()
    
    return model
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
def train(step=3000, BATCH_SIZE=128):
    for e in tqdm(range(step)):
        # 1. バッチの学習で利用する画像の選択 
        # バッチサイズの分だけランダムに画像を選択
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]
        
        # バッチサイズの分だけランダムにノイズを生成し、generatorにより画像を生成
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)
        
        # 2. Discriminatorの学習をonに切り替える
        # Discriminatorが学習するように変更
        make_trainable(discriminator,True)
        
        # 3. Generatorによる生成画像を用いてDiscriminatorの学習
        # X = (バッチサイズ分のデータセットの画像, バッチサイズ分の生成画像)
        X = np.concatenate((image_batch, generated_images))
        
        # y = (バッチサイズ分のTrue(本物), バッチサイズ分のFalse(偽物))
        y = np.zeros([2*BATCH_SIZE,2])
        y[:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1      
        
        # Discriminatorのtrain
        discriminator.train_on_batch(X,y)
        
        # 4. Discriminatorの学習をoffに切り替える
        # Discriminatorが学習しないように変更
        make_trainable(discriminator,False)
    
        # 5. Generatorの学習
        # バッチサイズの分だけランダムにノイズを生成
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        
        # y = (バッチサイズ分のTrue(本物))
        # 実際には生成した画像なのでDiscriminatorとしては偽物と判断すべきだが、Genaratorの学習なので生成した画像を本物と判断するように学習させる
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        # Generatorのtrain
        GAN.train_on_batch(noise_gen, y2 )
make_trainable(discriminator, False)
GAN = combined_network(generator, discriminator)
train(step=2000)
n_ex=plot_limit

noise = np.random.uniform(0,1,size=[n_ex,100])
generated_images = generator.predict(noise)

plt.figure(figsize=figsize)
for i in range(generated_images.shape[0]):
    plt.subplot(dim[0],dim[1],i+1)
    img = generated_images[i,:,:, 0]
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.show()
