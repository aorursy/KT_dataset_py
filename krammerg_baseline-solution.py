import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns #visualization

%matplotlib inline



np.random.seed(42)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler



from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



sns.set(style='white', context='notebook', palette='pastel')



IMSIZE = 28

NUM_CLASSES = 10
#print(os.listdir("../input"))

print(os.listdir("../input/dat18mnist-shallow/"))



#mnist_path = "../input/mnist-fhj/"

#mnist_train_path = "../input/mnist-fhj/train/"

#mnist_test_path = "../input/mnist-fhj/test/"

mnist_path = "../input/dat18mnist-shallow/"

mnist_train_path = "../input/dat18mnist-shallow/train/train/"

mnist_test_path = "../input/dat18mnist-shallow/test/test/"

mnist_extension = ".jpg"



train_ids_all = pd.read_csv(mnist_path+"train.csv")

test_ids_all = pd.read_csv(mnist_path+"test.csv")



def plot_diag_hist(dataframe, title='NoTitle'):

    f, ax = plt.subplots(figsize=(7, 4))

    ax = sns.countplot(x="label", data=dataframe, palette="GnBu_d")

    sns.despine()

    plt.title(title)

    plt.show()



plot_diag_hist(train_ids_all, title="Labels Training Data")



print("Shape of Training Data: {}".format(train_ids_all.shape))

print("Shape of Test Data: {}\n".format(test_ids_all.shape))



def get_full_path_train(idcode):

    return "{}{}{}".format(mnist_train_path,idcode,mnist_extension)



def get_full_path_test(idcode):

    return "{}{}{}".format(mnist_test_path,idcode,mnist_extension)





train_ids_all["path"] = train_ids_all["id_code"].apply(lambda x: get_full_path_train(x))

test_ids_all["path"] = test_ids_all["id_code"].apply(lambda x: get_full_path_test(x))
train_ids_all.head()
test_ids_all.head()
import cv2



def load_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    return img



def load_images_as_tensor(image_path, dtype=np.uint8):

    data = load_image(image_path).reshape((IMSIZE*IMSIZE,1))

    return data.flatten()



def show_image(image_path, figsize=None, title=None):

    image = load_image(image_path)

    if figsize is not None:

        fig = plt.figure(figsize=figsize)

    if image.ndim == 1:

        plt.imshow(np.reshape(image, (IMSIZE,-1)),cmap='gray')

    elif image.ndim == 2:

        plt.imshow(image,cmap='gray')

    elif image.ndim == 3:

        if image.shape[2] == 1:

            image = image[:,:,0]

            plt.imshow(image,cmap='gray')

        elif image.shape[2] == 3:

            plt.imshow(image)

        else:

            print("Invalid image dimension")

    if title is not None:

        plt.title(title)

        

def show_Nimages(image_filenames, classifications, scale=1):

    N=len(image_filenames)

    fig = plt.figure(figsize=(25/scale, 16/scale))

    for i in range(N):

        ax = fig.add_subplot(1, N, i + 1, xticks=[], yticks=[])

        show_image(image_filenames[i], title="C:{}".format(classifications[i]))

        

def show_Nrandomimages(N=10):

    indices = (np.random.rand(N)*train_ids_all.shape[0]).astype(int)

    show_Nimages(train_ids_all["path"][indices].values, train_ids_all["label"][indices].values)

    

def show_Nimages_of_class(classification=0, N=10):

    indices = train_ids_all[train_ids_all["label"] == classification].sample(N).index

    show_Nimages(train_ids_all["path"][indices].values, train_ids_all["label"][indices].values)
test_index = 2477

show_image(train_ids_all["path"][test_index], title="Class = {}".format(train_ids_all["label"][test_index]))
show_Nrandomimages(10)
show_Nimages_of_class(classification=2)
train_df, validation_df = train_test_split(train_ids_all, test_size=0.05)
from tqdm import tqdm



def load_training_data(image_filenames):

    N = image_filenames.shape[0]

    train_X = np.zeros((N,IMSIZE*IMSIZE), dtype=np.float32)

    for i in tqdm(range(image_filenames.shape[0])):

        img = load_images_as_tensor(image_filenames.iloc[i])

        train_X[i, :] = np.array(img, np.float32)/255

    return train_X

#load Training Data

train_X = load_training_data(train_df["path"])

train_y = train_df["label"].values



#load Validation data



#load Test data

test_X = load_training_data(test_ids_all["path"])
def plot_nice_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8,8))

    sns.heatmap(cm, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax, cmap=plt.cm.copper)

    plt.ylabel('true label')

    plt.xlabel('predicted label')

from sklearn.linear_model import LogisticRegression

logisticRegressor = LogisticRegression(multi_class="multinomial", solver="lbfgs")

logisticRegressor.fit(train_X, train_y)
#use score function of Logistig Regressor

logistigRegressionScore = logisticRegressor.score(train_X, train_y)

print("Score of Logistic Regressor on Training Set: {}".format(logistigRegressionScore))



#compute accuracy score

train_logReg_prediction = logisticRegressor.predict(train_X)

logisticRegressionAccuracyScore = accuracy_score(train_y, train_logReg_prediction)

print("Accuracy Score of Logistic Regressor on Training Set: {}".format(logisticRegressionAccuracyScore))

plot_nice_confusion_matrix(train_y, train_logReg_prediction)
coefs = logisticRegressor.coef_.copy()

plt.figure(figsize=(10, 5))

coef_scale = np.abs(coefs).max()

for i in range(10):

    l1_plot = plt.subplot(2, 5, i + 1)

    l1_plot.imshow(coefs[i].reshape(IMSIZE, IMSIZE), interpolation='nearest',

                   cmap=plt.cm.RdBu, vmin=-coef_scale, vmax=coef_scale)

    l1_plot.set_xticks(())

    l1_plot.set_yticks(())

    l1_plot.set_xlabel('Class %i' % i)

plt.suptitle('Coefficient vector for MNIST Classification')
#evaluate Algorithm on Validation set

#interpret results
# predict results for logistic regression

logReg_prediction = logisticRegressor.predict(test_X)

logreg_results = pd.Series(logReg_prediction,name="label")

submission = pd.concat([test_ids_all["id_code"],logreg_results], axis = 1)

submission.to_csv("logreg_submission.csv",index=False)
from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import KFold



folds = 5
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC