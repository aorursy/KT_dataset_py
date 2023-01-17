import os, cv2, itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

ROWS = 64
COLS = 64
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


# Sample from train and test data.
"""
sample_size = 5000
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
train_images = np.hstack([np.random.choice(train_dogs, sample_size, replace=False),
                          np.random.choice(train_cats, sample_size, replace=False)])

test_images =  test_images[:sample_size]
"""

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    """
    Returns:
        X(n_x, m)
        y(1, m) -- 1: dog, 0: cat
    """
    m = len(images)
    n_x = ROWS * COLS * CHANNELS
    
    X = np.ndarray((n_x, m), dtype=np.uint8)
    y = np.zeros((1, m))
    print ("X shape is {}".format(X.shape))
    
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        X[:, i] = np.squeeze(image.reshape((n_x, 1)))
        if 'dog' in image_file.lower():
            y[0, i] = 1
        elif 'cat' in image_file.lower():
            y[0, i] = 0
        else:# if neither dog nor cat exist, return the image index (this is the case for test data)
            y[0, i] = image_file.split('/')[-1].split('.')[0]
        if i%1000 == 0: print('Processed {} of {}'.format(i, m))
    
    return X, y

X_train, y_train = prep_data(train_images)
X_test, test_idx = prep_data(test_images)

print("Train shape: {}".format(X_train.shape))
print("Test shape: {}".format(X_test.shape))
classes = {0:'cat',
           1:'dog'}
def show_image(X, y, idx):
    image = X[idx]
    image = image.reshape((ROWS, COLS, CHANNELS))
    plt.figure(figsize=(4,2))
    plt.imshow(image)
    plt.title("This is a {}".format(classes[y[idx,0]]))
    plt.show()
    
def show_image_prediction(X, idx, model):
    image = X[idx].reshape(1, -1)
    image_class = classes[model.predict(image).item()]
    image = image.reshape((ROWS, COLS, CHANNELS))
    plt.figure(figsize=(4,2))
    plt.imshow(image)
    plt.title("Test {}: I think this is a {}".format(idx, image_class))
    plt.show()
show_image(X_train.T, y_train.T, 1)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
clf = LogisticRegressionCV()
X_train_lr, y_train_lr = X_train.T, y_train.T.ravel()
clf.fit(X_train_lr, y_train_lr)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print("Model accuracy: {:.2f}%".format(clf.score(X_train_lr, y_train_lr)*100))
plot_confusion_matrix(confusion_matrix(y_train_lr, clf.predict(X_train_lr)), ['cat', 'dog'])
X_test_lr, test_idx = X_test.T, test_idx.T
for i in np.random.randint(0, len(X_test_lr), 10):
    show_image_prediction(X_test_lr, i, clf)
submission = pd.DataFrame(np.hstack([test_idx, clf.predict_proba(X_test_lr)]), columns=['id', 'cat', 'dog'])
submission = submission.drop(['cat'], axis=1)
submission = submission.rename(index=str, columns={"dog": "label"})
submission['id'] = submission['id'].astype(int)
submission.sort_values('id', inplace=True)
submission.head()
submission.to_csv('STahamtan_Dog_vs_Cat_Submission.csv', index=False)