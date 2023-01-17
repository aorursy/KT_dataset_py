import pandas as pd
import keras as keras
import numpy as np
import sklearn.model_selection as skm
import matplotlib.pyplot as plt
import matplotlib as mpl
import timeit
final_models = {"keras1":"", "keras2":""}
def plot_y_true_vs_y_pred(x, y_true, y_pred, idx):
    pred_data = {"SNo": list(idx), "Y_true": y_true, "Y_pred": y_pred}
    x = list(x)
    pred_data["X"] = x
    pred_data["Correct"] = np.equal(pred_data["Y_true"], pred_data["Y_pred"] )
    df_pred = pd.DataFrame(pred_data)
    df_pred_summary = df_pred.groupby(["Correct"]).count()
    print(df_pred_summary.head(10))
    num_sample_rows = 50
    #df_pred_fail = df_pred[df_pred["Y_true"]!=df_pred["Y_pred"]]
    #df_pred_fail = df_pred_fail.head(num_sample_rows)
    df_pred = df_pred[df_pred["Y_true"]!=df_pred["Y_pred"]]
    #df_pred = df_pred.sample(num_sample_rows)
    #df_pred = pd.concat([df_pred, df_pred_fail], axis=0)
    i = 1
    fig = plt.figure(figsize=(15,20))
    fig.suptitle("True Value vs Predicted Value", size=20)
    for idx,row in df_pred.head(40).iterrows():
        plt.subplot(7,7,i)
        img = row["X"].reshape(28,28)
        img_class_true = row["Y_true"]
        img_class_pred = row["Y_pred"]
        plt.imshow(img, cmap="Greys")
        plt.title(str(row["SNo"]) + ": " + str(img_class_true) + " Prediction:" + str(img_class_pred))
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        i = i+1
def plot_bad_data(x, y):
    data = {"Y_true": y}
    x = list(x)
    data["X"] = x
    df_data = pd.DataFrame(data)
    i = 1
    fig = plt.figure(figsize=(15,20))
    fig.suptitle("Confusing Data", size=20)
    for idx,row in df_data.iterrows():
        plt.subplot(7,7,i)
        img = row["X"].reshape(28,28)
        img_class_true = row["Y_true"]
        plt.imshow(img, cmap="Greys")
        plt.title(str(img_class_true))
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        i = i+1
from skimage.transform import resize

threshold_color = 100 / 255
size_img = 28
def find_left_edge(x):
    edge_left = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if (x[k, size_img*i+j] >= threshold_color):
                    edge_left.append(j)
                    break
            if (len(edge_left) > k):
                break
    return edge_left
def find_right_edge(x):
    edge_right = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for j in range(size_img):
            for i in range(size_img):
                if (x[k, size_img*i+(size_img-1-j)] >= threshold_color):
                    edge_right.append(size_img-1-j)
                    break
            if (len(edge_right) > k):
                break
    return edge_right
def find_top_edge(x):
    edge_top = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for i in range(size_img):
            for j in range(size_img):
                if (x[k, size_img*i+j] >= threshold_color):
                    edge_top.append(i)
                    break
            if (len(edge_top) > k):
                break
    return edge_top
def find_bottom_edge(x):
    edge_bottom = []
    n_samples = x.shape[0]
    for k in range(n_samples):
        for i in range(size_img):
            for j in range(size_img):
                if (x[k, size_img*(size_img-1-i)+j] >= threshold_color):
                    edge_bottom.append(size_img-1-i)
                    break
            if (len(edge_bottom) > k):
                break
    return edge_bottom
def stretch_image(x):
    #get edges
    edge_left = find_left_edge(x)
    edge_right = find_right_edge(x)
    edge_top = find_top_edge(x)
    edge_bottom = find_bottom_edge(x)
    
    #cropping and resize
    n_samples = x.shape[0]
    x = x.reshape(n_samples, size_img, size_img)
    for i in range(n_samples):      
        x[i] = resize(x[i][edge_top[i]:edge_bottom[i]+1, edge_left[i]:edge_right[i]+1], (size_img, size_img))
    x = x.reshape(n_samples, size_img ** 2)
    return x
train_data = pd.read_csv("../input/train.csv")
x_train_all = np.array(train_data.drop(["label"], axis=1))
print (x_train_all[0][x_train_all[0] > 0])
x_train_all = x_train_all/255
x_train_all = stretch_image(x_train_all)
#x_train_all [ x_train_all > 0] = 1

y_train_all = np.array(train_data["label"])
idx_all = range(x_train_all.shape[0])
if 1==2:
    bad_data_idx = [28290, 16301,14101,15065, 6389,7764,28611,20954,2316, 37056, 37887, 36569, 40257]
    plot_bad_data(x_train_all[bad_data_idx], y_train_all[bad_data_idx])
    x_train_all = np.delete(x_train_all, bad_data_idx, axis=0)
    y_train_all = np.delete(y_train_all, bad_data_idx)
    idx_all = np.delete(idx_all, bad_data_idx)
y_train_all = keras.utils.to_categorical(y_train_all, num_classes=10)
x_train, x_valid, y_train, y_valid, idx_train, idx_valid = skm.train_test_split(x_train_all, y_train_all,idx_all,  test_size=0.2)

test_data = pd.read_csv("../input/test.csv")
x_test = np.array(test_data)
x_test = x_test/255
x_test = stretch_image(x_test)
#x_test[x_test > 0]=1




datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,                                               
    horizontal_flip=False, shear_range=0.2)
datagen.fit(x_train.reshape(-1,28,28,1))


if 1==2:
    augmented_image = []
    augmented_image_labels = []
    x_train = x_train.reshape(-1,28,28,1)
    for num in range (0, x_train.shape[0]):
        augmented_image.append(x_train[num])
        augmented_image_labels.append(y_train[num])
        if num<500:
            augmented_image.append(keras.preprocessing.image.random_rotation(x_train[num], 15, row_axis=1, col_axis=1, channel_axis=2))
            augmented_image_labels.append(y_train[num])

        if num > 500 and num < 1000:
            augmented_image.append(keras.preprocessing.image.random_shear(x_train[num], 0.2, row_axis=1, col_axis=1, channel_axis=2))
            augmented_image_labels.append(y_train[num])

        if num > 1000 and num < 1500:
            augmented_image.append(keras.preprocessing.image.random_shift(x_train[num], 0, 0.2, row_axis=1, col_axis=1, channel_axis=2))
            augmented_image_labels.append(y_train[num])

    x_train = np.array(augmented_image).reshape(-1, 784)
    y_train = np.array(augmented_image_labels)
print (test_data.head(1))
s = pd.read_csv("../input/sample_submission.csv")
s.head(1)
print(x_valid.shape[0]*4)
print(x_train.shape[0])
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='acc', 
                                            patience=1, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000001)
def tensorflow_keras_model(x_train, y_train, x_valid, y_valid, num_classes,\
                                num_epochs, learning_rate):
    
    keras_model = keras.models.Sequential()
    keras_model.add(keras.layers.Conv2D(filters=32,kernel_size=(6,6),strides=(1,1), \
                                        padding="same",input_shape=(28,28,1)))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.MaxPool2D(strides=(2,2), padding="same"))
    keras_model.add(keras.layers.Dropout(rate=0.05))
    
    keras_model.add(keras.layers.Conv2D(filters=64,kernel_size=(6,6),strides=(1,1), \
                                            padding="same")) #,activation='relu'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.MaxPool2D(strides=(2,2), padding="same"))
    keras_model.add(keras.layers.Dropout(rate=0.05))
    
    keras_model.add(keras.layers.Flatten())
    keras_model.add(keras.layers.Dense(units=1024)) #, activation='relu'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.Dropout(rate=0.05))
    
    keras_model.add(keras.layers.Dense(units=num_classes))#, activation='softmax'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('softmax'))  
    
    #opt = keras.optimizers.Adam(lr=learning_rate)
    opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    keras_model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=["accuracy"])
    keras_model.fit(x_train.reshape(-1,28,28,1), y_train, epochs=num_epochs, callbacks=[learning_rate_reduction])
    final_models["keras1"] = keras_model
    keras_model.save("Model_MNIST_Keras_Lenet.h5")
    cur_y_pred = keras_model.predict(x_valid.reshape(-1,28,28,1))
    y_valid_argmax = np.argmax(y_valid, 1)
    y_pred_argmax = np.argmax(cur_y_pred, 1)
    y_correct = np.equal(y_valid_argmax, y_pred_argmax)
    acc = y_correct.sum()/y_pred_argmax.shape[0]
    return cur_y_pred, acc, y_valid_argmax,y_pred_argmax 
def exec_tensorflow_keras_model():
    num_epochs = 20 #50
    learning_rate=0.000001
    num_rows, num_features, num_classes = x_train.shape[0], x_train.shape[1], 10
    print(num_rows, num_features, num_classes)
    final_pred_base_model, acc,y_valid_argmax,y_pred_argmax  = \
        tensorflow_keras_model(x_train, y_train, x_valid, y_valid, num_classes, num_epochs, learning_rate)  

    print("Num Epoch:", num_epochs, " Accuracy:", acc) #, \
           #   " Weights:", " ".join(list(final_w.astype(str).flatten())), " Bias:", final_b)
    plot_y_true_vs_y_pred(x_valid, y_valid_argmax.reshape(len(y_valid)), y_pred_argmax.reshape(len(y_valid)))
#timeit.timeit(exec_tensorflow_keras_model, number=1)
#VGG -> 64->MAXPOOL->128->MAXPOOL->256->256->MAXPOOL->512->512->MAXPOOL-->512->512->MAXPOOL->FC4096->FC4096->FC1000
#.99154
def tensorflow_keras_model_2(x_train, y_train, x_valid, y_valid, num_classes,\
                                num_epochs, learning_rate):
    keras_model = keras.models.Sequential()
    keras_model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1), \
                                            padding="same",activation='relu', input_shape=(28,28,1)))    #From 64->128->256 changing to 32->64->128
    keras_model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1), \
                                            padding="same")) #,activation='relu'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu')) 
    keras_model.add(keras.layers.Conv2D(filters=32,kernel_size=(5,5),strides=(2,2), \
                                            padding="same")) #,activation='relu'))
    #keras_model.add(keras.layers.MaxPool2D(strides=(2,2), padding="same"))
    keras_model.add(keras.layers.Dropout(rate=0.05))
    
    keras_model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1), \
                                        padding="same",activation='relu',input_shape=(28,28,1)))
    keras_model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1), \
                                            padding="same")) #,activation='relu'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.Conv2D(filters=64,kernel_size=(5,5),strides=(2,2), \
                                            padding="same")) #,activation='relu'))
    #keras_model.add(keras.layers.MaxPool2D(strides=(2,2), padding="same"))
    keras_model.add(keras.layers.Dropout(rate=0.050))
                    
    
    
    keras_model.add(keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1), \
                                            padding="same",activation='relu'))
    keras_model.add(keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1), \
                                            padding="same")) #,activation='relu'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.Conv2D(filters=128,kernel_size=(5,5),strides=(2,2), \
                                            padding="same")) #,activation='relu'))
    #keras_model.add(keras.layers.MaxPool2D(strides=(2,2), padding="same"))
    keras_model.add(keras.layers.Dropout(rate=0.075))
    
    keras_model.add(keras.layers.Flatten())
    keras_model.add(keras.layers.Dense(units=1024)) # #  , activation='relu')) Chaging from 20148 to 1024
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.Dropout(rate=0.100))
    
    keras_model.add(keras.layers.Dense(units=128)) # #  , activation='relu'))  Changin from 256 tp 128
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.Dropout(rate=0.100))
    
    keras_model.add(keras.layers.Dense(units=num_classes))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('softmax'))  
    
    opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    #opt = keras.optimizers.Adam(lr=learning_rate)
    keras_model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=["accuracy"])

    keras_model.fit_generator(datagen.flow(x_train.reshape(-1,28,28,1), y_train, batch_size=32),steps_per_epoch=len(x_train) / 32,  \
                              epochs=num_epochs, callbacks=[learning_rate_reduction])
    final_models["keras2"] = keras_model
    keras_model.save("Model_MNIST_Keras_Resnet.h5")
   
    cur_y_pred = keras_model.predict(x_valid.reshape(-1,28,28,1))
    y_valid_argmax = np.argmax(y_valid, 1)
    y_pred_argmax = np.argmax(cur_y_pred, 1)
    y_correct = np.equal(y_valid_argmax, y_pred_argmax)
    acc = y_correct.sum()/y_pred_argmax.shape[0]
    return cur_y_pred, acc, y_valid_argmax,y_pred_argmax , keras_model
    
model2_list = []
num_model = 12
def exec_tensorflow_keras_model_2():
    num_epochs = 30 #50  With 30 epochs acc was 99.667
    learning_rate=0.00001
    
    num_rows, num_features, num_classes = x_train.shape[0], x_train.shape[1], 10
    for imodel in range(num_model):
        final_pred_base_model, acc,y_valid_argmax,y_pred_argmax, keras_model  = \
                    tensorflow_keras_model_2(x_train, y_train, x_valid, y_valid, num_classes, num_epochs, learning_rate)
        model2_list.append(keras_model)

        print("Num Epoch:", num_epochs, " Accuracy:", acc) #, \
           #   " Weights:", " ".join(list(final_w.astype(str).flatten())), " Bias:", final_b)
    plot_y_true_vs_y_pred(x_valid, y_valid_argmax.reshape(len(y_valid)), y_pred_argmax.reshape(len(y_valid)), idx_valid)
timeit.timeit(exec_tensorflow_keras_model_2, number=1)

def tensorflow_keras_model_3(x_train, y_train, x_valid, y_valid, num_classes,\
                                num_epochs, learning_rate):
    
    keras_model = keras.models.Sequential()
    keras_model.add(keras.layers.Conv2D(filters=6,kernel_size=(6,6),strides=(1,1), \
                                        padding="same",input_shape=(28,28,1)))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.MaxPool2D(strides=(2,2), padding="same"))
    keras_model.add(keras.layers.Dropout(rate=0.05))
    
    keras_model.add(keras.layers.Conv2D(filters=16,kernel_size=(6,6),strides=(1,1), \
                                            padding="same")) #,activation='relu'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.MaxPool2D(strides=(2,2), padding="same"))
    keras_model.add(keras.layers.Dropout(rate=0.05))
    
    keras_model.add(keras.layers.Conv2D(filters=120,kernel_size=(6,6),strides=(1,1), \
                                            padding="same")) #,activation='relu'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.MaxPool2D(strides=(2,2), padding="same"))
    keras_model.add(keras.layers.Dropout(rate=0.05))
    
    keras_model.add(keras.layers.Flatten())
    keras_model.add(keras.layers.Dense(units=120)) #, activation='relu'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.Dropout(rate=0.05))
    
    keras_model.add(keras.layers.Dense(units=120)) #, activation='relu'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('relu'))  
    keras_model.add(keras.layers.Dropout(rate=0.05))
    
    keras_model.add(keras.layers.Dense(units=num_classes))#, activation='softmax'))
    keras_model.add(keras.layers.BatchNormalization())
    keras_model.add(keras.layers.Activation('softmax'))  
    
    #opt = keras.optimizers.Adam(lr=learning_rate)
    opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    keras_model.compile(optimizer=opt,loss="categorical_crossentropy", metrics=["accuracy"])
    keras_model.fit(x_train.reshape(-1,28,28,1), y_train, epochs=num_epochs, callbacks=[learning_rate_reduction])
    final_models["keras1"] = keras_model
    keras_model.save("Model_MNIST_Keras_Lenet.h5")
    cur_y_pred = keras_model.predict(x_valid.reshape(-1,28,28,1))
    y_valid_argmax = np.argmax(y_valid, 1)
    y_pred_argmax = np.argmax(cur_y_pred, 1)
    y_correct = np.equal(y_valid_argmax, y_pred_argmax)
    acc = y_correct.sum()/y_pred_argmax.shape[0]
    return cur_y_pred, acc, y_valid_argmax,y_pred_argmax 
def exec_tensorflow_keras_model_3():
    num_epochs = 30 #50
    learning_rate=0.00001
    num_rows, num_features, num_classes = x_train.shape[0], x_train.shape[1], 10
    final_pred_base_model, acc,y_valid_argmax,y_pred_argmax  = \
        tensorflow_keras_model_3(x_train, y_train, x_valid, y_valid, num_classes, num_epochs, learning_rate)  

    print("Num Epoch:", num_epochs, " Accuracy:", acc) #, \
           #   " Weights:", " ".join(list(final_w.astype(str).flatten())), " Bias:", final_b)
    plot_y_true_vs_y_pred(x_valid, y_valid_argmax.reshape(len(y_valid)), y_pred_argmax.reshape(len(y_valid)), idx_valid)
#timeit.timeit(exec_tensorflow_keras_model_3, number=1)
#model1 = final_models["keras1"]
model2 = final_models["keras2"]
#y_pred_valid1 = model1.predict(x_valid.reshape(-1,28,28,1))
#y_pred_valid2 = model2.predict(x_valid.reshape(-1,28,28,1))
x_valid = x_valid.reshape(-1,28,28,1)
y_pred_valid2 = np.zeros( (x_valid.shape[0],10) ) 
for imodel in range(num_model): #len(model2_list)):
    y_pred_valid2 = y_pred_valid2 + model2_list[imodel].predict(x_valid)
    
y_pred_valid = y_pred_valid2 #0.4*y_pred_valid1 + 0.6*y_pred_valid2

y_pred_valid_final = np.argmax(y_pred_valid, axis=1)
y_correct = np.equal(y_pred_valid_final, np.argmax(y_valid, axis=1))
acc_final = y_correct.sum()/y_valid.shape[0]
print(acc_final)
#y_pred_valid1 = model1.predict(x_test.reshape(-1,28,28,1))
#y_pred_test2 = model2.predict(x_test.reshape(-1,28,28,1))
x_test = x_test.reshape(-1,28,28,1)
y_pred_test2 = np.zeros( (x_test.shape[0],10) ) 
for imodel in range(num_model):
    y_pred_test2 = y_pred_test2 + model2_list[imodel].predict(x_test)

y_pred_test = y_pred_test2 #0.4*y_pred_valid1 + 0.6*y_pred_valid2
y_pred_test_final = np.argmax(y_pred_test, axis=1)

dictionary_data = {"ImageId":np.arange(1, x_test.shape[0]+1), "Label":y_pred_test_final}
df_final = pd.DataFrame(dictionary_data)
df_final.to_csv("submission.csv", index=False)
#
plot_y_true_vs_y_pred(x_valid, np.argmax(y_valid, 1).reshape(len(y_valid)), y_pred_valid_final.reshape(len(y_valid)), idx_valid)



