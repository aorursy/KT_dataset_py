%load_ext autoreload
%autoreload 2
%pylab inline
%cd '/kaggle/input/face-detection-dataset/face-detection'
from keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform
from get_data import load_dataset, unpack
# First run will download 30 MB data from github

train_images, train_bboxes, train_shapes = load_dataset("data", "train")
val_images, val_bboxes, val_shapes = load_dataset("data", "val")
from graph import visualize_bboxes
visualize_bboxes(images=train_images,
                 true_bboxes=train_bboxes
                )
SAMPLE_SHAPE = (32, 32, 3)
from scores import iou_score # https://en.wikipedia.org/wiki/Jaccard_index

def is_negative_bbox(new_bbox, true_bboxes, eps=1e-1):
    """Check if new bbox not in true bbox list.
    
    There bbox is 4 ints [min_row, min_col, max_row, max_col] without image index."""
    for bbox in true_bboxes:
        if iou_score(new_bbox, bbox) >= eps:
            return False
    return True
# Write this function
def gen_negative_bbox(image_shape, bbox_size, true_bboxes):
    """Generate negative bbox for image."""
    tries = 1000
    for i in range(tries):
        corner_x = np.random.randint(max(image_shape[0] - bbox_size[0], 1))
        corner_y = np.random.randint(max(image_shape[1] - bbox_size[1], 1))
        new_bbox = [corner_x, corner_y, corner_x + bbox_size[0], corner_y + bbox_size[1]]
        if is_negative_bbox(new_bbox,true_bboxes):
            return new_bbox
    return None

def get_positive_negative(images, true_bboxes, image_shapes, negative_bbox_count=None):
    """Retrieve positive and negative samples from image."""
    positive = []
    negative = []
    image_count = image_shapes.shape[0]
    
    if negative_bbox_count is None:
        negative_bbox_count = len(true_bboxes)
    
    # Pay attention to the fact that most part of image may be black -
    # extract negative samples only from part [0:image_shape[0], 0:image_shape[1]]
    
    # Write code here
    # ...
    w, h, c = SAMPLE_SHAPE
    for true_bbox in true_bboxes:        
        image_index = true_bbox[0]
        pos_img = images[image_index][true_bbox[1]:true_bbox[1]+w, true_bbox[2]:true_bbox[2]+h, :]
        positive.append(pos_img)
    
    print(negative_bbox_count)
    for i in range(negative_bbox_count):
        image_index = np.random.choice(len(images))
        image_shape = image_shapes[image_index]
        image_true_bboxes = true_bboxes[true_bboxes[:, 0] == image_index, 1:]
        for j in range(100):
            bbox_size = (w, h)
            new_bbox = gen_negative_bbox(image_shape, bbox_size, image_true_bboxes)
            if new_bbox is None: continue
            neg_img = images[image_index][new_bbox[0]:new_bbox[2], new_bbox[1]:new_bbox[3]]
            negative.append(neg_img)
            break
        print(i)
    return positive, negative
def get_samples(images, true_bboxes, image_shapes):
    """Usefull samples for learning.
    
    X - positive and negative samples.
    Y - one hot encoded list of zeros and ones. One is positive marker.
    """
    positive, negative = get_positive_negative(images=images, true_bboxes=true_bboxes, 
                                               image_shapes=image_shapes)
    X = positive
    Y = [[0, 1]] * len(positive)
    
    X.extend(negative)
    Y.extend([[1, 0]] * len(negative))
    
    return np.array(X), np.array(Y)
X_train, Y_train = get_samples(train_images, train_bboxes, train_shapes)
X_val, Y_val = get_samples(val_images, val_bboxes, val_shapes)
out_file = '/kaggle/working/{}.npy'
np.save(out_file.format('X_train'), X_train)
np.save(out_file.format('Y_train'), Y_train)
np.save(out_file.format('X_val'), X_val)
np.save(out_file.format('Y_val'), Y_val)
out_file = '/kaggle/working/{}.npy'
X_train = np.load(out_file.format('X_train'))
Y_train = np.load(out_file.format('Y_train'))
X_val = np.load(out_file.format('X_val'))
Y_val = np.load(out_file.format('Y_val'))
# There we should see faces
from graph import visualize_samples
visualize_samples(X_train[Y_train[:, 1] == 1])
# There we shouldn't see faces
visualize_samples(X_train[Y_train[:, 1] == 0])
BATCH_SIZE = 64
K.clear_session()
from keras.preprocessing.image import ImageDataGenerator # Usefull thing. Read the doc.

datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.1,
                            )
datagen.fit(X_train)
import os.path
from keras.optimizers import Adam
# Very usefull, pay attention
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from graph import plot_history

model_path = '/kaggle/working/FDmodel.hdf5'
if os.path.isfile(model_path):
    os.remove(model_path)
    
def fit(model, datagen, X_train, Y_train, X_val, Y_val, class_weight=None, epochs=50, lr=0.001, verbose=False):
    """Fit model.
    
    You can edit this function anyhow.
    """
    
    if verbose:
        model.summary()

    model.compile(optimizer=Adam(lr=lr), # You can use another optimizer
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                  validation_data=(datagen.standardize(X_val), Y_val),
                                  epochs=epochs, steps_per_epoch=len(X_train) // BATCH_SIZE,
                                  callbacks=[ModelCheckpoint(model_path, save_best_only=True)],
                                  class_weight=class_weight,
            
                                 )  # starts training
    
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Activation, Input, Dropout, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

def generate_model(sample_shape):
    # Classification model
    # You can start from LeNet architecture
    x = inputs = Input(shape=sample_shape)

    # Write code here
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)

    # This creates a model
    predictions = Dense(2, activation='softmax')(x)
    return Model(inputs=inputs, outputs=predictions)

model = generate_model(SAMPLE_SHAPE)
model.summary()
# Attention: Windows implementation may cause an error here. In that case use model_name=None.
fit(model=model, datagen=datagen, X_train=X_train.astype('float32'), X_val=X_val.astype('float32'), Y_train=Y_train, Y_val=Y_val)
def get_checkpoint():
    return model_path

model.load_weights(get_checkpoint())
# FCNN

IMAGE_SHAPE = (176, 176, 3)

def generate_fcnn_model(image_shape):
    """After model compilation input size cannot be changed.
    
    So, we need create a function to have ability to change size later.
    """
    x = inputs = Input(image_shape)

    # Write code here
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (8, 8), activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Dropout(0.25)(x)

    # This creates a model
    predictions = Conv2D(2, (1, 1), activation='linear')(x)
    return Model(inputs=inputs, outputs=predictions)

fcnn_model = generate_fcnn_model(IMAGE_SHAPE)
fcnn_model.summary()
def copy_weights(base_model, fcnn_model):
    """Set FCNN weights from base model.
    """
    
    new_fcnn_weights = []
    prev_fcnn_weights = fcnn_model.get_weights()
    prev_base_weights = base_model.get_weights()
    
    # Write code here
    for prev_fcnn_weight, prev_base_weight in zip(prev_fcnn_weights, prev_base_weights):
        new_fcnn_weights.append(prev_base_weight.reshape(prev_fcnn_weight.shape))
        
    fcnn_model.set_weights(new_fcnn_weights)

copy_weights(base_model=model, fcnn_model=fcnn_model)
from graph import visualize_heatmap
predictions = fcnn_model.predict(np.array(val_images))
visualize_heatmap(val_images, predictions[:, :, :, 1])
# Detection
from skimage.feature import peak_local_max

def get_bboxes_and_decision_function(fcnn_model, images, image_shapes):      
    cropped_images = np.array([transform.resize(image, IMAGE_SHAPE, mode="reflect")  if image.shape != IMAGE_SHAPE else image for image in images])
    pred_bboxes, decision_function = [], []
   
    # Predict
    predictions = fcnn_model.predict(cropped_images)

    # Write code here
    for i in range(len(predictions)):
        img_shape = image_shapes[i]
        local_max_list = peak_local_max(predictions[i][:,:,1], num_peaks=5, min_distance=3, exclude_border=False)
        for local_max_orig in local_max_list:
            local_max = ((local_max_orig + 2)*176/37).astype(int)
            
            if local_max[0] < img_shape[0] and local_max[1] < img_shape[1]:
                bbox = [i] + [local_max[0]-16,local_max[1]-16,local_max[0]+16,local_max[1]+16]
                
                pred_bboxes.append(bbox)
                decision_function.append(predictions[i, local_max_orig[0], local_max_orig[1], 1])
        
    return pred_bboxes, decision_function
pred_bboxes, decision_function = get_bboxes_and_decision_function(fcnn_model=fcnn_model, images=val_images, image_shapes=val_shapes)

visualize_bboxes(images=val_images,
                 pred_bboxes=pred_bboxes,
                 true_bboxes=val_bboxes,
                 decision_function=decision_function
                )
from scores import best_match
from graph import plot_precision_recall

def precision_recall_curve(pred_bboxes, true_bboxes, decision_function):
    precision, recall, thresholds = [], [], []
    
    # Write code here
    threshold = min(decision_function) - 1
    max_th = max(decision_function) + 1
    num_steps = 100
    th_step = (max_th - threshold) / num_steps

    sorted_boxes = [[x] + y for y, x in sorted(zip(pred_bboxes, decision_function), key=lambda pair : pair[1])]
    
    for step in range(num_steps):        
        pred_bboxes_th = [x[1:] for x in sorted_boxes if x[0] > threshold]
        if len(pred_bboxes_th) > 0:
            matched, false_negative, false_positive = best_match(pred_bboxes_th, true_bboxes, decision_function)
        else:
            break
        
        prec = len(matched) / (len(matched) + len(false_positive))
        rec = len(matched) / (len(matched) + len(false_negative))
        
        thresholds.append(threshold)
        recall.append(rec)
        precision.append(prec)
        threshold += th_step
        
    return precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(pred_bboxes=pred_bboxes, true_bboxes=val_bboxes, decision_function=decision_function)
plot_precision_recall(precision=precision, recall=recall)
def get_threshold(thresholds, recall):
    return thresholds[np.argmax(np.asarray(recall) <= 0.85)] # Write this code

THRESHOLD = get_threshold(thresholds, recall)
def detect(fcnn_model, images, image_shapes, threshold, return_decision=True):
    """Get bboxes with decision_function not less then threshold."""
    pred_bboxes, decision_function = get_bboxes_and_decision_function(fcnn_model, images, image_shapes)   
    result, result_decision = [], []
    
    # Write code here
    for i in range(len(pred_bboxes)):
        if decision_function[i] >= threshold:
            result.append(pred_bboxes[i])
            result_decision.append(decision_function[i])
    
    if return_decision:
        return result, result_decision
    else:
        return result
pred_bboxes, decision_function = detect(fcnn_model=fcnn_model, images=val_images, image_shapes=val_shapes, threshold=THRESHOLD, return_decision=True)

visualize_bboxes(images=val_images,
                 pred_bboxes=pred_bboxes,
                 true_bboxes=val_bboxes,
                 decision_function=decision_function
                )

precision, recall, thresholds = precision_recall_curve(pred_bboxes=pred_bboxes, true_bboxes=val_bboxes, decision_function=decision_function)
plot_precision_recall(precision=precision, recall=recall)
test_images, test_bboxes, test_shapes = load_dataset("data", "test")

# We test get_bboxes_and_decision_function becouse we want pay attention to all recall values
pred_bboxes, decision_function = get_bboxes_and_decision_function(fcnn_model=fcnn_model, images=test_images, image_shapes=test_shapes)

visualize_bboxes(images=test_images,
                 pred_bboxes=pred_bboxes,
                 true_bboxes=test_bboxes,
                 decision_function=decision_function
                )

precision, recall, threshold = precision_recall_curve(pred_bboxes=pred_bboxes, true_bboxes=test_bboxes, decision_function=decision_function)
plot_precision_recall(precision=precision, recall=recall)
# First run will download 523 MB data from github

original_images, original_bboxes, original_shapes = load_dataset("data", "original")
# Write code here
# ...

# Write this function
def hard_negative(train_images, image_shapes, train_bboxes, X_val, Y_val, base_model, fcnn_model):
    pass
hard_negative(train_images=train_images, image_shapes=train_shapes, train_bboxes=train_bboxes, X_val=X_val, Y_val=Y_val, base_model=model, fcnn_model=fcnn_model)
model.load_weights("data/checkpoints/...")
copy_weights(base_model=model, fcnn_model=fcnn_model)

pred_bboxes, decision_function = get_bboxes_and_decision_function(fcnn_model=fcnn_model, images=val_images, image_shapes=val_shapes)

visualize_bboxes(images=val_images,
                 pred_bboxes=pred_bboxes,
                 true_bboxes=val_bboxes,
                 decision_function=decision_function
                )

precision, recall, thresholds = precision_recall_curve(pred_bboxes=pred_bboxes, true_bboxes=val_bboxes, decision_function=decision_function)
plot_precision_recall(precision=precision, recall=recall)
def multiscale_detector(fcnn_model, images, image_shapes):
    return []