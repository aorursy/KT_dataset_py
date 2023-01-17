import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, InputLayer, Activation
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, Accuracy
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
metadata = pd.read_csv("../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv")
metadata = metadata.drop("Unnamed: 0", axis=1)
metadata.describe(include="all")
# Drop all information but "Label_1_Virus_category" and "Dataset_type"
metadata.drop(["Label_1_Virus_category", "Label_2_Virus_category"], axis=1, inplace=True)
# Check Dataset_type
print(metadata["Dataset_type"].unique())
# Check Label_1_Virus_category
print(metadata["Label"].unique())
# Replace name of "Label_2_Virus_category" to "Label"
if("Label_1_Virus_category" in metadata.columns):
    metadata["Label"] = metadata["Label_1_Virus_category"]
    metadata.drop("Label_1_Virus_category", axis=1, inplace=True)
metadata["image"] = np.asarray(np.nan, dtype=object)  # To make sure I can assign an array to the column, otherwise dtypes won't match
print(metadata.head())
print(f"\nDifferent values of labels distribution:\n{metadata['Label'].value_counts()}")
imgs_dir = "../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/**/*.*"

IMG_SZ = 127
for name in glob(imgs_dir, recursive=True):
    image_name = name.split("/")[-1]
    # Open image
    img = Image.open(name).convert("L")  # Some images have RGB components, not grayscaled
    # Resize all images to 127x127
    maxsize = (127, 127)
    img = ImageOps.fit(img,
                       maxsize,
                       Image.ANTIALIAS)
    # Add to DataFrame
    if(image_name.lower() in metadata["X_ray_image_name"].str.lower().values):
        index_df = metadata[(metadata["X_ray_image_name"] == image_name)].index[0]
        metadata.at[index_df, "image"] = np.asarray(img)
pneumonia = metadata.loc[metadata["Label"] == "Pnemonia", "image"]
plt.figure(figsize=(15,15))
ax = plt.subplot(3,2, 1)
plt.imshow(pneumonia[pneumonia.index[10]], cmap="gist_gray")
ax.add_patch(patches.Circle((110, 100), radius=15, linewidth=2, fill=False, color="r"))
ax.add_patch(patches.Circle((60, 40), radius=20, linewidth=2, fill=False, color="orange"))
ax.add_patch(patches.Circle((20, 100), radius=10, linewidth=2, fill=False, color="blue"))
plt.title("Pneumonia")
ax = plt.subplot(3,2, 3)
plt.imshow(pneumonia[pneumonia.index[20]], cmap="gist_gray")
ax.add_patch(patches.Circle((80, 70), radius=20, linewidth=2, fill=False, color="orange"))
ax.add_patch(patches.Circle((20, 100), radius=10, linewidth=2, fill=False, color="blue"))
plt.title("Pneumonia")
ax = plt.subplot(3,2, 5)
plt.imshow(pneumonia[pneumonia.index[35]], cmap="gist_gray")
ax.add_patch(patches.Circle((80, 60), radius=20, linewidth=2, fill=False, color="orange"))
plt.title("Pneumonia")
normal = metadata.loc[metadata["Label"] == "Normal", "image"]
ax = plt.subplot(3,2, 2)
plt.imshow(normal[normal.index[10]], cmap="gist_gray")
plt.title("Normal")
ax = plt.subplot(3,2, 4)
plt.imshow(normal[normal.index[20]], cmap="gist_gray")
plt.title("Normal")
ax = plt.subplot(3,2, 6)
plt.imshow(normal[normal.index[40]], cmap="gist_gray")
plt.title("Normal")
plt.show()
print(pd.get_dummies(metadata["Label"]))
print(pd.get_dummies(metadata["Label"]).values)
# Data reshaping for tensorflow
X = np.zeros((metadata.shape[0], 127, 127))
Y = np.zeros((metadata.shape[0], 2))
y_dummies = pd.get_dummies(metadata["Label"]).values

for i in range(0, metadata.shape[0]):
    X[i, :, :] = metadata["image"][i]
    Y[i, :] = y_dummies[i]
    
X = X.reshape(-1,127,127,1)
X = X / 255
# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=7)
lr = 1e-3  # Learning rate
dr = [0.3, 0.5]  # Dropout rate
bz = 128  # Batch size
epochs = 40  # Epochs

precision, recall, bin_acc = Precision(), Recall(), BinaryAccuracy(threshold=0.5)  # Metrics
adam = Adam(lr)  # Optimizer

def simpleConvoNet(learn_rt, dr, act_funct='relu', pool_size=(2,2)):
    k = np.array([64, 2])  # Number of hidden units
    lr = 1e-4  # Learning rate

    model = Sequential()
    model.epoch = 0

    model.add(InputLayer((127, 127, 1)))
    
    model.add(Conv2D(32, 3, activation=act_funct))
    model.add(MaxPooling2D(pool_size)) 
    model.add(Activation("relu"))
    
    model.add(Conv2D(64, 2, activation=act_funct))
    model.add(MaxPooling2D(pool_size)) 
    model.add(Activation("relu"))
    model.add(Dropout(dr[0]))
    
    model.add(Conv2D(64, 3, activation=act_funct))
    model.add(MaxPooling2D(pool_size)) 
    model.add(Activation("relu"))
    
    model.add(Conv2D(64, 3, activation=act_funct))
    model.add(MaxPooling2D(pool_size)) 
    model.add(Activation("relu"))
    
    model.add(Conv2D(64, 2, activation=act_funct))
    model.add(MaxPooling2D(pool_size)) 
    model.add(Activation("relu"))
    
    model.add(Flatten())
    model.add(Dense(k[0], activation='relu'))
    model.add(Dropout(dr[1]))
    model.add(Dense(k[1], activation='softmax'))

    model.compile(loss="mse",
                  optimizer=adam,
                  metrics=[precision, recall, bin_acc])

    return model

model = simpleConvoNet(learn_rt=lr, dr=dr)
model.summary()
history = model.fit(X_train, Y_train,
                    initial_epoch = model.epoch,
                    batch_size=bz, epochs=model.epoch + epochs,
                    verbose=1, validation_split=0.3)
accur = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(8,4))
plt.plot(epochs_range, accur, label='Train Set')
plt.plot(epochs_range, val_acc, label='Test Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.title('Model Accuracy')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(epochs_range, loss, label='Train Set')
plt.plot(epochs_range, val_loss, label='Test Set')
plt.legend(loc="best")
plt.xlabel('Epochs')
plt.title('Model Loss')
plt.show()
y_test_pred = model.predict(X_test)
THRESHOLD = 0.5
y_test_pred = np.where(y_test_pred > THRESHOLD, 1, y_test_pred)
y_test_pred = np.where(y_test_pred < THRESHOLD, 0, y_test_pred)

print(classification_report(Y_test, y_test_pred))
y_test_pred_rev = np.zeros(y_test_pred.shape[0])
for i in range(0, y_test_pred.shape[0]):
    y_test_pred_rev[i] = y_test_pred[i].argmax()    
    
Y_test_rev = np.zeros(Y_test.shape[0])
for i in range(0, Y_test.shape[0]):
    Y_test_rev[i] = Y_test[i].argmax()

plt.figure(figsize=(12,10))
sns.heatmap(confusion_matrix(Y_test_rev, y_test_pred_rev), xticklabels=pd.get_dummies(metadata["Label"]).columns, yticklabels=pd.get_dummies(metadata["Label"]).columns, annot=True, fmt="g")
plt.show()
def plot_roc_curve(fpr, tpr):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
fpr, tpr, _ = roc_curve(y_test_pred[:,0], Y_test[:, 0]) 
plot_roc_curve(fpr, tpr)
print(f"Area under the curve = {round(roc_auc_score(y_test_pred[:,0], Y_test[:, 0]), 2)}")