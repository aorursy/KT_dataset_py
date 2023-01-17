import os

import numpy as np

import matplotlib.pyplot as plt



from imageio import imread

from skimage.transform import resize as imresize

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, roc_auc_score, roc_curve

import tensorflow as tf
plt.style.use("ggplot")

plt.set_cmap("binary")
# setup filepaths

uninfected_folder = "../input/cell_images/cell_images/Uninfected/"

parasit_folder = "../input/cell_images/cell_images/Parasitized/"

uninfected_list = os.listdir(uninfected_folder)

parasit_list = os.listdir(parasit_folder)
# have a look at few images

fig = plt.figure(figsize=(15,5))

axs = fig.subplots(2,10, sharex=True, sharey=True)



# few cosmetic cleanup

axs[0][0].set_ylabel("Uninfected cells", fontweight="bold", fontsize="x-large")

axs[1][0].set_ylabel("Parasitized cells", fontweight="bold", fontsize="x-large")

axs[0][0].set_yticks([])

axs[1][0].set_xticks([])

plt.tight_layout()



for ax in axs[0]:

    random_img = imread(uninfected_folder + uninfected_list[np.random.randint(0,len(uninfected_list))], as_gray=True)

    random_img = imresize(random_img, [64,64])

    ax.imshow(random_img)





for ax in axs[1]:

    random_img = imread(parasit_folder + parasit_list[np.random.randint(0,len(parasit_list))], as_gray=True)

    random_img = imresize(random_img, [64,64])

    ax.imshow(random_img)
uninf = imread(uninfected_folder + uninfected_list[np.random.randint(0,len(uninfected_list))], as_gray=True)

uninf = imresize(uninf, [64,64])

inf = imread(parasit_folder + parasit_list[np.random.randint(0,len(parasit_list))], as_gray=True)

inf = imresize(inf, [64,64])



# have a look at few images

fig = plt.figure(figsize=(20,5))

axs = fig.subplots(2,12, sharex=True, sharey=True)



# few cosmetic cleanup

axs[0][0].set_ylabel("Uninfected cells", fontweight="bold", fontsize="x-large")

axs[1][0].set_ylabel("Parasitized cells", fontweight="bold", fontsize="x-large")

axs[0][0].set_yticks([])

axs[1][0].set_xticks([])

plt.tight_layout()



for ax in axs[0]:

    random_img = imread(uninfected_folder + uninfected_list[np.random.randint(0,len(uninfected_list))], as_gray=True)

    random_img = imresize(random_img, [64,64])

    filter_img = np.ma.masked_where((random_img<50) | (random_img>150), random_img )

#     ax.hist(random_img.ravel())

    ax.imshow(filter_img)



for ax in axs[1]:

    random_img = imread(parasit_folder + parasit_list[np.random.randint(0,len(parasit_list))], as_gray=True)

    random_img = imresize(random_img, [64,64])

    filter_img = np.ma.masked_where((random_img<50) | (random_img>150), random_img )

#     ax.hist(random_img.ravel())

    ax.imshow(filter_img)
low_filter = 50

high_filter = 180



X = np.empty([len(uninfected_list)+len(parasit_list),25])

xpointer=0

for fname in uninfected_list:

    try:

        img = imread(uninfected_folder + fname, as_gray=True)

    except SyntaxError:

        continue

    imgdata = img[(img>low_filter)&(img<high_filter)]

    X[xpointer,:] = np.histogram(imgdata, bins=25, density=False)[0]

    xpointer += 1

for fname in parasit_list:

    try:

        img = imread(parasit_folder + fname, as_gray=True)

    except SyntaxError:

        continue

    imgdata = img[(img>low_filter)&(img<high_filter)]

    X[xpointer,:] = np.histogram(imgdata, bins=25, density=False)[0]

    xpointer += 1



y = np.empty(len(uninfected_list)+len(parasit_list), dtype=np.uint8)

y[:len(uninfected_list)]=0

y[len(uninfected_list):]=1
# random shuffling of indices which will be useful later

shuffle_indices = np.arange(0,len(X))

np.random.shuffle(shuffle_indices)
fig = plt.figure(figsize=(8,8))

axs = fig.subplots(4,4,sharex=True,sharey=True)

chosen_indices = np.random.choice(shuffle_indices, size=axs.size)

for ax,i in zip(axs.ravel(), chosen_indices):

    ax.plot(X[i,:])

    ax.set_title("class: {}".format(y[i]))

# shuffle the data

X_shuffled = X[shuffle_indices] / 5000

y_shuffled = y[shuffle_indices]

X_train, X_test, y_train, y_test = train_test_split(X_shuffled,y_shuffled,test_size=0.2)
fcols = tf.feature_column.numeric_column(key="x", shape=[25])

dnn_est = tf.estimator.DNNClassifier([50,50], feature_columns=[fcols],optimizer="Adam")
input_fn_train = tf.estimator.inputs.numpy_input_fn({"x":X_train}, y_train, batch_size=20, num_epochs=5, shuffle=False)

input_fn_test = tf.estimator.inputs.numpy_input_fn({"x":X_test}, y_test, shuffle=False)
%%time

dnn_est.train(input_fn_train,max_steps=1000)
dnn_est.evaluate(input_fn_test)
pred_gen = dnn_est.predict(input_fn_test)

y_pred = [p["logistic"] for p in pred_gen]

y_pred = np.array(y_pred).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred, drop_intermediate=True)

auc = roc_auc_score(y_test, y_pred)

plt.figure(figsize=(12,8))

plt.subplot(121)

plt.plot(fpr,tpr,label="AUC {0:.2f}".format(auc))

plt.plot([0,1],[0,1])

plt.xlabel("False positive rate", fontsize="xx-large")

plt.ylabel("True positive rate", fontsize="xx-large")

_ = plt.legend(loc="lower right", fontsize="xx-large")

plt.subplot(122)

plt.plot(thresholds, fpr, label="FPR")

plt.plot(thresholds, tpr, label="TPR")

plt.xlim(left=0, right=1)

plt.xlabel("Threshold \n(when proba>this value positive class is predicted)")

_ = plt.legend(loc="lower left", fontsize="xx-large")