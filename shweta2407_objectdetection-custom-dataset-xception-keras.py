from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical, plot_model
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Dropout
from keras.layers import Flatten
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import cv2
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from keras.layers import GlobalAveragePooling2D, Dense, Input
from keras.applications.xception import Xception 
# READ IMAGES IN COLORED FORMAT
def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# CREATE IMAGE LIST
def create_image_list(image_path):
    image_list = []
    # ITERATE THROUGH IMAGES FOLDER
    for image in os.listdir(image_path):
        # APPEND THE NAME OF IMAGES TO THE LIST
        image_list.append(image)
    return image_list

# CREATE MASK FOR BOUNDING BOX
def create_mask(bb, image):
    # EXTRACT THE IMAGE SHAPE
    rows,cols,*_ = image.shape
    # CREATE A MATRIX OF ZERO OF THE IMAGE SHAPE
    mask = np.zeros((rows, cols))
    # FILL THE MATRIX CONTAINING THE BOUNDING BOX WITH VALUE 1
    mask[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return mask

# CONVERT RESIZED MASK TO BOUNDING BOX
def convert_to_bb(mask):
    # EXTRACT THE SHAPE OF THE MASK OF BOUNDING BOX CREATED
    cols, rows = np.nonzero(mask)
    # RETURN ZERO COORDINATES IF NO MASK
    if len(cols)==0: 
        return np.zeros(4, dtype=np.float32)
    # EXTRACT THE BOUNDING BOX COORDINATES
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

# RESIZE THE IMAGES AND SAVE IT IN ANOTHER FOLDER
def image_resize(image_path, new_path, bb, size):
    # READ THE IMAGE FILE
    image = read_image(image_path)
    # RESIZE THE IMAGE
    image_resized = cv2.resize(image, (int(1.49*size), size))
    # CREATE MASK FROM THE BOUNDING BOX
    mask = create_mask(bb, image)
    # RESIZE THE MASK 
    mask_resized = cv2.resize(mask, (int(1.49*size), size))
    # WRITE THE NEW IMAGE INTO ANOTHER FOLDER
    cv2.imwrite(new_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
    return new_path, convert_to_bb(mask_resized)

# PLOT THE BOUNDING BOX AROUND THE IMAGE
def plot_bb(path, bb):
    image = read_image(path)
    # CONVERT BOUNDING BOXES (BB) INTO FLOAT
    bb = np.array(bb, dtype=np.float32)
    # CREATE A RECTANGLE FROM THE BB
    rect_box = plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color='red',
                         fill=False, lw=3)
    # RENDER THE IMAGE
    plt.imshow(image)
    # APPLY THE BB TO THE CURRENT AXIS RENDERING IMAGE
    plt.gca().add_patch(rect_box)
# EXTRACT BOUNDING BOX FROM THE ANNOTATION FILE
def extract_bb(anno_path):
    # PARSE THE XML FILE TO EXTRACT BB COORDINATES AND CLASS_NAME
    root = ET.parse(anno_path).getroot()
    class_name = root.find("./object/name").text
    xmin = int(root.find("./object/bndbox/xmin").text)
    ymin = int(root.find("./object/bndbox/ymin").text)
    xmax = int(root.find("./object/bndbox/xmax").text)
    ymax = int(root.find("./object/bndbox/ymax").text)
    # RETURN BOUNDING BOX COORDINATES
    bb = [ymin, xmin, ymax, xmax]
    return bb, class_name

# GENERATE DATAFRAME
def generate_dataframe(image_list, anno_path, image_path, new_path, size):
    dataset = []
    for image in image_list:
        path = image_path + image
        a_path = anno_path + image.split('.')[0] + '.xml'
        # EXTRACT BB AND CLASS_NAME FROM ANNOTATION FILE
        bb, class_name = extract_bb(a_path)
        # FILENAME OF THE NEW RESIZED IMAGE
        n_path = new_path + image 
        # RESIZE THE IMAGE AND CORRESPONDING BOUNDING BOX 
        img_path, resized_bb = image_resize(path, n_path, bb, size)
        # APPEND EVERYTHING TO A DICTIONARY 
        data = dict()
        data['filename'] = img_path
        data['bb'] = resized_bb
        data['class_name'] = class_name
        # APPEND THE DICTIONARY TO THE LIST
        dataset.append(data)
    # APPEND THE LIST TO THE DATAFRAME 
    return pd.DataFrame(dataset) 
# prepare dataset
def generate_data_array(dataframe):
    
    train_img = []
    classes = []
    bounding_boxes = []
    
    for index, row in dataframe.iterrows():
        path = row['filename']
        x = read_image(path)
        
        # append image
        train_img.append(x)
        # append class labels
        classes.append(row['class_name'])
        # append bb
        bounding_boxes.append(row['bb'])
        
    return train_img, classes, bounding_boxes
!mkdir resized_image
image_path = '../input/currency-datasets/images/'
anno_path =  '../input/currency-datasets/annotations/'
new_path = './resized_image/'

# CREATE IMAGE LIST
image_list = create_image_list(image_path)
# SHUFFLE THE LIST
np.random.shuffle(image_list)

image_list[:5]
data = generate_dataframe(image_list, anno_path, image_path, new_path, 300)
len(data)
data.head()
train_img, classes, bounding_boxes = generate_data_array(data)
# ENCODE THE LABEL
le = LabelEncoder()
Y_class = le.fit_transform(classes)
# 0:'50Rs', 1:'500Rs', 2:'100Rs', 3:'10Rs', 4:'20Rs', 5:'200Rs', 6:'2000Rs'
le.inverse_transform([0,1,2,3,4,5,6])
X = np.array(train_img)
Y_class = to_categorical(Y_class)
Y_bb = np.array(bounding_boxes)

# CHECK THE DIMENSIONS
print("shape of X ", X.shape)
print("shape of Y_class ", Y_class.shape)
print("shape of Y_bb ", Y_bb.shape)
# model = VGG16(include_top=False, input_shape=(224, 333, 3))
# x = Flatten()(model.layers[-1].output)
 
image_input = Input(shape=(300, 447, 3))
base_model = Xception(include_top=False, input_tensor=image_input, weights='imagenet')
x = base_model.layers[-1].output
x = GlobalAveragePooling2D()(x)
# x = BatchNormalization()(x)
# x = Dropout(0.25)(x)

output_1 = Dense(4, activation='relu')(x)
output_2 = Dense(7, activation='softmax')(x)

model = Model(inputs = image_input, outputs =[output_1, output_2])
model.compile(loss=['mae', 'categorical_crossentropy'], optimizer='adam', metrics =['accuracy'])
# dense_loss = loss of bb
# dense_1_loss = loss of class
history = model.fit(X, [Y_bb, Y_class], epochs=200, batch_size=8, validation_split=0.2)
# PLOT BOUNDING BOX TRAINING LOSS AND VALIDATION LOSS

plt.plot(history.history['dense_loss'], color='deeppink')
plt.plot(history.history['val_dense_loss'], color='yellowgreen')
plt.title('bounding box loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
# PLOT CLASS LABEL TRAINING LOSS AND VALIDATION LOSS

plt.plot(history.history['dense_1_loss'], color='darkorange')
plt.plot(history.history['val_dense_1_loss'], color='cornflowerblue')
plt.title('class_label loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
test_image = './resized_image/2000_3.jpeg'

img = read_image(test_image)
img = np.array([img])
predict = model.predict(img)
predict
# INFERENCE CODE
currency_dict = {0:'50Rs', 1:'500Rs', 2:'100Rs', 3:'10Rs', 4:'20Rs', 5:'200Rs', 6:'2000Rs'}

def identify_currency(image_path):
    
    x = read_image(image_path)
    img = read_image(test_image)
    img = np.array([img])
    predict = model.predict(img)
    confidence = predict[1][0][np.argmax(predict[1][0])]
    curr = currency_dict[np.argmax(predict[1][0])]
    print(" The detected currency is {} and the model is {} % sure about it".format(curr, round(confidence*100, 2)))
    plot_bb(image_path, predict[0][0])
test_image = './resized_image/2000_3.jpeg'

identify_currency(test_image)
test_image = './resized_image/500_64.jpeg'

identify_currency(test_image)
test_image = './resized_image/200_46.jpeg'

identify_currency(test_image)
test_image = './resized_image/10_29 (copy).jpeg'

identify_currency(test_image)
test_image = './resized_image/50_43 (copy).jpeg'

identify_currency(test_image)
test_image = './resized_image/100_37.jpeg'

identify_currency(test_image)
test_image = './resized_image/20_19 (copy).jpeg'

identify_currency(test_image)
