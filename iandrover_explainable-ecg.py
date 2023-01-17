import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

import os
print(os.listdir("../input"))

MI = pd.read_csv("../input/ptbdb_abnormal.csv") 
HC = pd.read_csv("../input/ptbdb_normal.csv") 

new_column_name = ['label']
for num in range(MI.shape[1]-1):
    tem = 'dim' + str(num)
    new_column_name.append(tem)
MI.columns = new_column_name    

column_name = ['label']
for num in range(HC.shape[1]-1):
    tem = 'dim' + str(num)
    column_name.append(tem)
HC.columns = column_name
import keras
train_MI=MI.iloc[0:7000]
test_MI=MI.iloc[7000:9000]
train_HC=HC.iloc[0:2500]
test_HC=HC.iloc[2500:3500]
train=[train_MI,train_HC]
train=pd.concat(train,sort=True)
test=[test_MI,test_HC]
test=pd.concat(test,sort=True)

ytrain=list(range(9500))
ytest=list(range(3000))
for i in range(9500): ytrain[i] = 1 if i<=7000 else 0
for j in range(3000): ytest[j] = 1 if j<=2000 else 0        
ytrain = keras.utils.np_utils.to_categorical(ytrain)
ytest = keras.utils.np_utils.to_categorical(ytest)

train=np.asarray(train)
train=train.reshape(9500, 188, 1)

test=np.asarray(test)
test=test.reshape(3000, 188, 1)
for i in range(3): plt.plot(MI.iloc[i])
plt.show()
for i in range(3): plt.plot(HC.iloc[i])
plt.show()
tf.keras.backend.clear_session()

input_ = tf.keras.Input(shape=(188,1))
x = tf.keras.layers.Conv1D(100, 5)(input_)
x = tf.keras.layers.Conv1D(100, 5, activation='relu')(x)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Conv1D(100, 5, activation='relu')(x)
x = tf.keras.layers.Conv1D(160, 5, activation='relu')(x)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Conv1D(100, 5, activation='relu')(x)
x = tf.keras.layers.Conv1D(100, 5, activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)
output_ = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=input_,outputs=output_,)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
ytrain.shape
history = model.fit(train, ytrain, validation_data=(test, ytest), epochs=3)
import tensorflow.keras.backend as K

def gradcam(model, x, index, layer_name):
    # 取得影像的分類類別
    preds = model.predict(x)
    pred_class = np.argmax(preds[index])
    
    # 取得影像分類名稱
    #pred_class_name = imagenet_utils.decode_predictions(preds)[0][0][1]
    
    # 預測分類的輸出向量
    pred_output = model.output[:, pred_class]
    
    # 最後一層 convolution layer 輸出的 feature map
    # ResNet 的最後一層 convolution layer
    last_conv_layer = model.get_layer(layer_name)
    
    # 求得分類的神經元對於最後一層 convolution layer 的梯度
    grads = K.gradients(pred_output, last_conv_layer.output)[0]
    
    # 求得針對每個 feature map 的梯度加總
    #pooled_grads = K.sum(grads, axis=(0, 1, 2))
    pooled_grads = K.sum(grads, axis=(0, 1))
    
    # K.function() 讓我們可以藉由輸入影像至 `model.input` 得到 `pooled_grads` 與
    # `last_conv_layer[0]` 的輸出值，像似在 Tensorflow 中定義計算圖後使用 feed_dict
    # 的方式。
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[index]])
    
    # 傳入影像矩陣 x，並得到分類對 feature map 的梯度與最後一層 convolution layer 的 
    # feature map
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    # 將 feature map 乘以權重，等於該 feature map 中的某些區域對於該分類的重要性
    for i in range(pooled_grads_value.shape[0]):
        conv_layer_output_value[:, i] *= (pooled_grads_value[i])
        
    # 計算 feature map 的 channel-wise 加總
    heatmap = np.sum(conv_layer_output_value, axis=-1)
    
    return heatmap, pred_class
def heatmap_():
    heatmap_2d = np.expand_dims(heatmap,1)
    heatmap_2d = cv2.resize(heatmap_2d, (100,188))

    figure = plt.figure(figsize=(80,4))
    ax = figure.add_subplot(111)
    x = np.arange(0, 188, 1)
    ax.plot(x, data*100)
    ax.imshow(np.transpose(heatmap_2d), cmap="Oranges", extent=[0, 188, -10, 120])
    ax.autoscale(False)
    ax.set_xlim(0, 187)
    ax.set_ylim(-10, 110)
    ax.set_title(title)
    plt.show()
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output_shape)
def heatmap_3x():
    def preprocess_heatmap(heatmap):
        heatmap_2d = np.expand_dims(heatmap,1)
        heatmap_2d = cv2.resize(heatmap_2d, (100,188))
        heatmap_2d = np.transpose(heatmap_2d)
        return heatmap_2d

    fig, ax = plt.subplots(1,len(name_list)+1,figsize=(20,4))
    x = np.arange(0, 188, 1)
    Collect_attention = list()
    for i in range(1,len(name_list)+1):
        ax[i].plot(x, data*100)
        heatmap_2d = preprocess_heatmap(Collect_heatmap[i-1])
        Collect_attention.append(heatmap_2d[0])
        ax[i].imshow(heatmap_2d, cmap="Oranges", extent=[0, 188, -10, 120])
        ax[i].set_xlim(0, 187)
        ax[i].set_ylim(-10, 110)
        ax[i].set_title(name_list[i-1])
    ax[0].plot(x, data, label="ECG")
    for i in range(len(name_list)):
        ax[0].set_title(title)
        ax[0].plot(x, Collect_attention[i]/np.max(Collect_attention[i]), label=name_list[i])
        ax[0].set_ylim(-0.2, 1)
    ax[0].legend()

    plt.show()
for i in range(20):
    Collect_heatmap = list()
    index = np.random.randint(0,len(test))
    name_list = ["conv1d_1", "conv1d_3", "conv1d_5"]
    for name in name_list:
        heatmap, pred_class = gradcam(model, test, index, name)
        Collect_heatmap.append(heatmap)
    data = test[index]
    if int(ytest[index][0]): title = "HC"
    else: title = "MI"
    heatmap_3x()
