import warnings

warnings.filterwarnings('ignore')
#importing libraries

from sklearn.ensemble import RandomForestClassifier

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,classification_report

from sklearn import tree

import graphviz

import random 

random.seed(3)
heart_data = pd.read_csv('../input/heart-disease-uci/heart.csv')

heart_data.head()
X_train = heart_data.drop('target',axis = 1)

y_train = heart_data['target']

X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,random_state = 3,test_size = 0.2)

clf_randomForest = RandomForestClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(X_train,y_train)
print(accuracy_score(y_test,clf_randomForest.predict(X_test)))

print(classification_report(y_test,clf_randomForest.predict(X_test)))
from IPython.display import Image

from subprocess import call

tree_graph = tree.export_graphviz(clf_randomForest.estimators_[0], out_file='tree.dot', feature_names=X_train.columns.tolist(),proportion = True,rounded = True,filled = True,precision = 2)

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

Image(filename = 'tree.png')
#permutation importance

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(clf_randomForest, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
#partial dependence plot



from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=clf_randomForest, dataset=X_test, model_features=X_test.columns.tolist(), feature='thalach')

# plot it

pdp.pdp_plot(pdp_goals, 'thalach')

plt.show()
#partial dependence plot



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=clf_randomForest, dataset=X_test, model_features=X_test.columns.tolist(), feature='ca')

# plot it

pdp.pdp_plot(pdp_goals, 'ca')

plt.show()
#partial dependence plot

features_to_plot = ['ca', 'thalach']

inter1  =  pdp.pdp_interact(model=clf_randomForest, dataset=X_test, model_features=X_test.columns.tolist(), features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')

plt.show()
#shap

row_to_show = 4

data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

clf_randomForest.predict_proba(data_for_prediction_array)
import shap  # package used to calculate Shap values

# Create object that can calculate shap values

explainer = shap.TreeExplainer(clf_randomForest)

# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
#Dependence Contribution Plots

shap_values = explainer.shap_values(X_train)

shap.dependence_plot('thalach',shap_values[1], X_train, interaction_index="ca")
#summary plot

shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values[1], X_train,auto_size_plot=False)


shap.force_plot(explainer.expected_value[0],shap_values[0] , X_train)
import lime

import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.astype(int).values,mode='classification',training_labels=y_train,feature_names=X_train.columns.tolist(),class_names=['true','false'])

#Let's take a look for the 100th row

i = 1

exp = explainer.explain_instance(X_train.loc[i,X_train.columns.tolist()].astype(int).values, clf_randomForest.predict_proba, num_features=13)
exp.show_in_notebook(show_table=True)
from lime import submodular_pick

# SP-LIME returns exaplanations on a sample set to provide a non redundant global decision boundary of original model

sp_obj = submodular_pick.SubmodularPick(explainer, X_train.values, clf_randomForest.predict_proba, num_features=13,num_exps_desired=3)

[exp.show_in_notebook() for exp in sp_obj.sp_explanations]
import os

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

from keras.models import load_model

import numpy as np

import matplotlib.pyplot as plt

from keras import models

from keras.applications import VGG16

from keras import backend as K

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input, decode_predictions

import cv2


model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#Visualize Intermediate activation 



model.load_weights('../input/catndog/model.h5')

img_path = '../input/cnn-image/dog.jpeg'



#Preprocesses the image into a 4D tensor

img = image.load_img(img_path, target_size=(128, 128))

img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.





plt.imshow(img_tensor[0])

plt.show()
#Visualize Intermediate activation 



layer_outputs = [layer.output for layer in model.layers[:8]]

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)



activations = activation_model.predict(img_tensor)



first_layer_activation = activations[0]

print(first_layer_activation.shape)
#Visualize Intermediate activation

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
#Visualize Intermediate activation

plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
#Visualize Intermediate activation



layer_names = []

for layer in model.layers[:8]:

    layer_names.append(layer.name)



images_per_row = 16



# Now let's display our feature maps

for layer_name, layer_activation in zip(layer_names, activations):

    # This is the number of features in the feature map

    n_features = layer_activation.shape[-1]



    # The feature map has shape (1, size, size, n_features)

    size = layer_activation.shape[1]



    # We will tile the activation channels in this matrix

    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols, images_per_row * size))



    # We'll tile each filter into this big horizontal grid

    for col in range(n_cols):

        for row in range(images_per_row):

            channel_image = layer_activation[0,:, :,col * images_per_row + row]

            # Post-process the feature to make it visually palatable

            channel_image -= channel_image.mean()

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            display_grid[col * size : (col + 1) * size,row * size : (row + 1) * size] = channel_image



    # Display the grid

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],

                        scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect='auto', cmap='viridis')

    

plt.show()

model = VGG16(weights='imagenet',include_top=False)

layer_name = 'block3_conv1'

filter_index = 0

layer_output = model.get_layer(layer_name).output

loss = K.mean(layer_output[:, :, :, filter_index])
# The call to `gradients` returns a list of tensors (of size 1 in this case)

# hence we only keep the first element -- which is a tensor.

grads = K.gradients(loss, model.input)[0]
# We add 1e-5 before dividing so as to avoid accidentally dividing by 0.

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
iterate = K.function([model.input], [loss, grads])

loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
# We start from a gray image with some noise

input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

# Run gradient ascent for 40 steps

step = 1.  # this is the magnitude of each gradient update

for i in range(40):

    # Compute the loss value and gradient value

    loss_value, grads_value = iterate([input_img_data])

    # Here we adjust the input image in the direction that maximizes the loss

    input_img_data += grads_value * step
def deprocess_image(x):

    # normalize tensor: center on 0., ensure std is 0.1

    x -= x.mean()

    x /= (x.std() + 1e-5)

    x *= 0.1



    # clip to [0, 1]

    x += 0.5

    x = np.clip(x, 0, 1)



    # convert to RGB array

    x *= 255

    x = np.clip(x, 0, 255).astype('uint8')

    return x
def generate_pattern(layer_name, filter_index, size=150):

    # Build a loss function that maximizes the activation

    # of the nth filter of the layer considered.

    layer_output = model.get_layer(layer_name).output

    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss

    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture

    iterate = K.function([model.input], [loss, grads])

    # We start from a gray image with some noise

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps

    step = 1.

    for i in range(40):

        loss_value, grads_value = iterate([input_img_data])

        input_img_data += grads_value * step

        

    img = input_img_data[0]

    return deprocess_image(img)
plt.imshow(generate_pattern('block3_conv1', 1))

plt.show()
for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:

    size = 64

    margin = 5



    # This a empty (black) image where we will store our results.

    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))



    for i in range(8):  # iterate over the rows of our results grid

        for j in range(8):  # iterate over the columns of our results grid

            # Generate the pattern for filter `i + (j * 8)` in `layer_name`

            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)



            # Put the result in the square `(i, j)` of the results grid

            horizontal_start = i * size + i * margin

            horizontal_end = horizontal_start + size

            vertical_start = j * size + j * margin

            vertical_end = vertical_start + size

            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img



    # Display the results grid

    plt.figure(figsize=(20, 20))

    plt.imshow(np.array(results,np.int32))

    plt.show()
#heatmaps of class activation

K.clear_session()

model = VGG16(weights='imagenet')
#heatmaps of class activation

img_path = '../input/cnn-image/elephant.jpeg'

# `img` is a PIL image of size 224x224

img = image.load_img(img_path, target_size=(224, 224))

# `x` is a float32 Numpy array of shape (224, 224, 3)

x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"

# of size (1, 224, 224, 3)

x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch

# (this does channel-wise color normalization)

x = preprocess_input(x)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])

np.argmax(preds[0])

# This is the "african elephant" entry in the prediction vector

african_elephant_output = model.output[:, 386]

# The is the output feature map of the `block5_conv3` layer,

# the last convolutional layer in VGG16

last_conv_layer = model.get_layer('block5_conv3')

# This is the gradient of the "african elephant" class with regard to

# the output feature map of `block5_conv3`

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry

# is the mean intensity of the gradient over a specific feature map channel

pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:

# `pooled_grads` and the output feature map of `block5_conv3`,

# given a sample image

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,

# given our sample image of two elephants

pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array

# by "how important this channel is" with regard to the elephant class

for i in range(512):

    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map

# is our heatmap of class activation

heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)

plt.matshow(heatmap)

plt.show()
# We use cv2 to load the original image

img = cv2.imread(img_path)

plt.imshow(img)

plt.show()

# We resize the heatmap to have the same size as the original image

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB

heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.6 here is a heatmap intensity factor

superimposed_img = heatmap * 0.6 + img

plt.imshow(np.array(superimposed_img,np.int32))

plt.show()






