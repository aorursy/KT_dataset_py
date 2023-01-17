!pip install tensorflow==2.0rc1 # Tensorflow 2.O is stable version currently

from IPython.display import clear_output
import cv2 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

clear_output()
!git clone https://github.com/abhinandanarya06/OCR.git
def imshow(image):
  plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
  plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
  plt.show()
keywords = [chr(c) for c in range(ord('a'), ord('z')+1)]
cap = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'Q', 'R', 'T']
keywords = keywords + cap
keywords = keywords + ['noise']
print('No of text classification classes taken : ', len(keywords))
data_images = list()
labels = list()
i = 0
for c in keywords:
    path = 'OCR/data/{}/'.format(c) # IF YOU WANT TO GET DATA FROM OTHER PATH,
    files = os.listdir(path)        # THEN PLEASE MODIFY "path" VARIABLE ACCORDINGLY
    for name in files:
        img = cv2.imread(path+name, 0)
        try:
            img = cv2.resize(img, (30, 30), interpolation = cv2.INTER_AREA)
        except:
            continue
        img = img / np.max(img.reshape(900)) # Pixel feature normalization
        data_images.append([img])
        labels.append(i)
    i += 1

data_images = np.array(data_images)
labels = np.array(labels)

DATASET_SIZE = labels.shape[0]
IMAGE_DATA_SHAPE = (DATASET_SIZE, 30, 30, 1)

data_images = data_images.reshape(IMAGE_DATA_SHAPE)
print('Image Data Shape : ', data_images.shape)
print('Labels Data Shape : ', labels.shape)
BATCH_SIZE = 1000

dataset = tf.data.Dataset.from_tensor_slices((data_images, labels))
dataset = dataset.shuffle(DATASET_SIZE)

VAL_SIZE = int(DATASET_SIZE * 0.2)

val_data = dataset.take(VAL_SIZE).batch(VAL_SIZE)
train_data = dataset.batch(BATCH_SIZE)
for img, label in train_data.take(10):
  imshow(img[0].numpy().reshape((30, 30))*255)
  print(' '*12, keywords[label[0]], end='\n\n')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (5, 5), activation='relu', input_shape = (30, 30, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation = 'relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(len(keywords))
])
model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_data, epochs=10, callbacks = [callback], validation_data=val_data)
clear_output()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
model.save('model.h5')
clear_output()
model = tf.keras.models.load_model('OCR/model.h5')
text_detector = tf.keras.Sequential(
    model,
    tf.keras.layers.Softmax()
)
X = 0   # X coordinate of the character in image space
Y = 1   # Y coordinate of the character in image space
POS = 1    # shows and return the (x,y) coordinate of image space
SHAPE = 2  # shows and return the (width, height) of character contour in image space
W = 0 # width of the character
H = 1 # height of the character
def check_in(c, region):
    x, y, w, h = region
    center_x = c[POS][X]+c[SHAPE][W]/2
    center_y = c[POS][Y]
    if (center_x > x-1 and center_x < x+w+1) and (center_y > y-1 and center_y < y+h+1):
        return True
    return False
def get_region(c, regions):
    for region in regions:
        if check_in(c, region):
            return region
    return False
def sort_chars(line):
    res = list()
    while len(line) > 0:
        mx = 100000
        m = 0
        for c in line:
            if c[POS][X] <= mx:
                mx = c[POS][X]
                m = c
        line.remove(m)
        res.append(m)
    return res
def sort_lines_by_yval(lines):
    res = list()
    while len(lines) > 0:
        mn = 100000
        m = 0
        for line in lines:
            if line[0][POS][Y] < mn:
              mn = line[0][POS][Y]
              m = line
        lines.remove(m)
        res.append(m)
    return res
def group_chars_by_line(characters):
  lines = list()
  linei = 0
  while len(characters) > 0:
      m = characters[0]
      my = m[1][1]
      my_plus_h = m[1][1]+m[2][1]
      lines.append([m])
      for c in characters[1:]:
          if my <= c[POS][Y]+c[SHAPE][H]/2 and c[POS][Y]+c[SHAPE][H]/2 <= my_plus_h:
              if my > c[POS][Y]:
                  my = c[POS][Y]
              if my_plus_h < c[POS][Y]+c[SHAPE][H]:
                  my_plus_h = c[POS][Y]+c[SHAPE][H]
              lines[linei].append(c)
              characters.remove(c)
      lines[linei]= sort_chars(lines[linei])
      linei += 1
      characters.remove(m)
  return lines
def apply_ocr(img, td):
    avg_text_height = 0 # will be assigned with Average Height of text character
    character_list = list() # will contain necessary text character to be extracted
    img = cv2.medianBlur(img,5)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,41,10) # will remove maximum noise and make image suitable for contour dt
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    black_map = img.copy()
    black_map[:, :] = 0
    i = 0
    for contour in contours:
        try:
            x,y,w,h = cv2.boundingRect(contour)
            try:
                cnt = img[y-2:y+h+2, x-2:x+w+2]
            except:
                cnt = img[y:y+h, x:x+w]
            cnt = cv2.resize(cnt, (30, 30), interpolation = cv2.INTER_AREA)
            cnt = cnt.reshape((1, 30, 30, 1))
            cnt = cnt / np.max(cnt.reshape(900))
            class_pred = np.argmax(td.predict(cnt))
            if class_pred < 40:
                avg_text_height += h
                cv2.rectangle(black_map, (x-w//7,y), (x+w+w//7, y+h), 255, -1)
                character_list.append([keywords[class_pred], (x,y), (w,h)])
                i += 1
        except:
            continue
    avg_text_height /= i # Average Height of text character
    text = group_chars_by_line(character_list)
    text = sort_lines_by_yval(text)
    contours, hierarchy = cv2.findContours(black_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    regions = list() # will contain regions of words for grouping chars by word
    for contour in contours:
        region = cv2.boundingRect(contour)
        if region[SHAPE + H] <= avg_text_height*7:
          regions.append(region)
    del contours

    TEXT = ''
    for l in text:
        char = l[0]
        region = get_region(char, regions)
        if not region:
            continue
        for char in l:
            if not check_in(char, region):
                TEXT = TEXT + ' '
                r = get_region(char, regions)
                if not r:
                    continue
                region = r
            TEXT = TEXT + char[0]
        TEXT = TEXT + '\n'
    return TEXT
test_images_path = 'OCR/sample_test_image/' # IF YOU PLACE TEST IMAGE INSIDE OCR DIRECTORY,
                                            # THEN PLEASE MODIFY "test_images_path" variable


imgs = [f for f in os.listdir(test_images_path) if f.endswith('.jpg')]
for img in imgs:
    print('*'*30, 'Text on {}'.format(img), '*'*30)
    img = cv2.imread(test_images_path + img, 0)
    TEXT = apply_ocr(img, text_detector)
    imshow(img)
    print(TEXT)
    print('-'*80, '\n\n')