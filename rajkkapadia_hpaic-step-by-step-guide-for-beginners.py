# Some basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# PIL library to read images
# PIL library is the fastest as per my knowledge
from PIL import Image
# Read the train.csv fiel
train_csv = pd.read_csv("../input/train.csv")

# Training images path
TRAIN_PATH = "../input/train/"
# Testing images path
TEST_PATH = "../input/test/"

# Four colours for images
colours = ["red", "green", "blue", "yellow"]

# Training image ids
ids = train_csv["Id"]
# Training image labels
targets = train_csv["Target"]
train_csv.head()
ids[0]
# The whole set of images for one sample is as follows
print(TRAIN_PATH+ids[0]+"_"+colours[0]+".png")
print(TRAIN_PATH+ids[0]+"_"+colours[1]+".png")
print(TRAIN_PATH+ids[0]+"_"+colours[2]+".png")
print(TRAIN_PATH+ids[0]+"_"+colours[3]+".png")
green = np.asarray(Image.open(TRAIN_PATH+ids[0]+"_"+colours[1]+".png"))

plt.imshow(green)
plt.show()
target = targets[1].split(" ")
print(target)
# First create empty array and then fill 1 where needed
label = np.zeros((1, 28))
print(label)
for value in target:
    label[0, int(value)] = 1

print(label)
# Create your own batches here
batches = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
           1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
           1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
           1071]
numb_labels = 28
# Model fitting parameters
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

batch_id = 1
index = 0

for batch in batches:
    print("Processing batch number " + str(batch_id))
    # Create empty images and labels for batch
    images = np.zeros((batch, 512, 512, 1), dtype=np.float)
    labels = np.zeros((batch, numb_labels), dtype=np.float)
    
    for i in range(batch):
        
        # Get the image
        green = np.asarray(Image.open(TRAIN_PATH+ids[index]+"_"+colours[1]+".png"))
        index += 1
        # Add to images
        images[i] = green.reshape(512, 512, 1)/255
        
        # Same for labels
        target = targets[i].split(" ")
        
        for value in target:
            labels[i, int(value)] = 1
        
    print("Fitting the data to the model.")
    # Train the model
    # --> Youer model here
    batch_id += 1
    index += 1
test_csv = pd.read_csv("../input/sample_submission.csv")
test_csv.head()
ids_test = test_csv["Id"]
ids_test[0]
y_pred = np.zeros((len(ids_test), numb_labels), dtype=np.float)
images = np.zeros((1, 512, 512, 3), dtype=np.float)

for i in range(len(ids_test)):
    red = np.asarray(Image.open(TEST_PATH+ids_test[i]+"_"+colours[0]+".png"))
    green = np.asarray(Image.open(TEST_PATH+ids_test[i]+"_"+colours[1]+".png"))
    blue = np.asarray(Image.open(TEST_PATH+ids_test[i]+"_"+colours[2]+".png"))
    
    img_rgb = np.stack((red, green, blue), axis=-1)
    img_rgb = img_rgb/255
    
    images[0] = img_rgb
    
    # Your model
    # y_pred[i] = model.predict(images, verbose=1)
y_pred = (y_pred > 0.4).astype(int)
# Convert 1 and 0 into 0 to 27 digits for our labels
y_sub = []
for label_set in y_pred:
    index = 0
    l = ""
    for label in label_set:
        if label == 1:
            l += str(index)
            l += " "
            index += 1
        else:
            index += 1
    y_sub.append(l[0:-1])
        
# Prepare submission file
submission = pd.DataFrame({"Predicted":y_sub}, index=ids_test)
submission.to_csv("submission_one.csv", index=False)
