# run only if you want to create class activation mappings for visualization
# installing old version of scipy
# warning will restart runtime on google colab
!pip install -I scipy==1.2.*
!pip install -q kaggle
# Put kaggle.json in the working directory
# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json
# downloading dataset
!kaggle datasets download jbeltranleon/xrays-chest-299-small
# downloading trained models
!kaggle datasets download SmolBoi96/CheXNet-Ensemble-Model
#shutil.rmtree('299_small')

# Unzip the data
!unzip -qq -n xrays-chest-299-small.zip
!unzip -qq -n CheXNet-Ensemble-Model.zip
# Switch directory and show its content
!cd 299_small && ls
import shutil

shutil.rmtree('input_299_small')
shutil.rmtree('sample_data')
shutil.rmtree('299_small/train')
#shutil.rmtree('299_small/train0')
#shutil.rmtree('299_small/train1')
#shutil.rmtree('299_small/train2')
#shutil.rmtree('299_small/train3')
#shutil.rmtree('299_small/train4')
#shutil.rmtree('299_small/train5')
#shutil.rmtree('299_small/train6')
#shutil.rmtree('299_small/train7')
#shutil.rmtree('299_small/train8')
import os

base_dir = '299_small'

# Directory to our training data
train_folder = os.path.join(base_dir, 'train')

# Directory to our validation data
val_folder = os.path.join(base_dir, 'val')

# Directory to our validation data
test_folder = os.path.join(base_dir, 'test')

# List folders and number of files
print("Directory, Number of files")
for root, subdirs, files in os.walk(base_dir):
    print(root, len(files))
#labels for the meta-learner
label = ["atelectasis","cardiomegaly","consolidation","effusion","infiltration","mass","no_finding","nodule","pneumothorax"]
#labels for each of the 9 sub-models
LABELS = [['effusion','infiltration','mass','no_finding'],
          ['effusion','infiltration','no_finding','nodule'],
          ['effusion','infiltration','no_finding','pneumothorax'],
          ['atelectasis','effusion','infiltration','no_finding'],
          ['atelectasis','infiltration','mass','no_finding'],
          ['atelectasis','infiltration','no_finding','nodule'],
          ['atelectasis','infiltration','no_finding','pneumothorax'],
          ['consolidation','infiltration','no_finding'],
          ['cardiomegaly','infiltration','no_finding']]
#test set sizes for each of the 9 sub-models 
SIZES = [[791,1910,428,2000],
         [791,1910,2000,541],
         [791,1910,2000,439],
         [843,791,1910,2000],
         [843,1910,428,2000],
         [843,1910,2000,541],
         [843,1910,2000,439],
         [262,1910,2000],
         [219,1910,2000]]
for i in range(9):
  os.mkdir('299_small/test' + str(i))
  for j in range(len(LABELS[i])):
    os.mkdir('299_small/test' + str(i) + "/" + LABELS[i][j])
import numpy as np

for i in range(9):
  for j in range(len(SIZES[i])):
    print(LABELS[i][j])
    path, dirs, files = next(os.walk('299_small/test/' + LABELS[i][j]))
    print(len(files))
    for k in range(len(files)):
      shutil.copy('299_small/test/' + LABELS[i][j]+'/'+files[k], '299_small/test' + str(i) + '/' + LABELS[i][j])
from tensorflow.keras.models import load_model
#load sub-models into a list
all_models = list()
for i in range(9):
	# define filename for this ensemble
	filename = 'Ensemble' + str(i) + '.hdf5'
	# load model from file
	model = load_model(filename)
	# add to list of members
	all_models.append(model)
	print('>loaded %s' % filename)
# update all layers in all models to not be trainable
for i in range(len(all_models)):
  submodel = all_models[i]
  submodel._name = 'ensemble'+str(i)
  for layer in submodel.layers:
    # make not trainable
    layer.trainable = False
		# rename to avoid 'unique layer name' issue
    layer._name = 'ensemble_' + str(i) + '_' + layer.name
#this model requires 9 seperate instances of the input
#to merge input layer later
def define_stacked_model(members):
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = concatenate(ensemble_outputs)
	hidden = Dense(18, activation='relu')(merge)
	output = Dense(9, activation='softmax')(hidden)
	model = Model(inputs=ensemble_visible, outputs=output)
	# plot graph of ensemble
	plot_model(model, show_shapes=True)#, to_file='model_graph.png')
	# compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# create an untrained ensemble model
model = define_stacked_model(all_models)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

#merging the input layers
input_shape = (299, 299, 3)
inputs = (Input(input_shape))
output = model([inputs,inputs,inputs,inputs,inputs,inputs,inputs,inputs,inputs])
new_model = Model(inputs,output)
from tensorflow.keras.utils import plot_model

plot_model(stacked_model3, to_file='model.png', show_shapes=True, show_layer_names=True,
    rankdir='LR', expand_nested=False, dpi=200)
from keras.preprocessing.image import ImageDataGenerator
# we train the meta-learner on the validation set and validate on the test set to prevent overfitting
train_folder = '299_small/val'
val_folder = '299_small/test'

# Batch size
bs = 16

# All images will be resized to this value
image_size = (299, 299)

# All images will be rescaled by 1./255. We apply data augmentation here.

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 16 using train_datagen generator
print("Preparing generator for train dataset")
train_generator = train_datagen.flow_from_directory(
    directory= train_folder, # This is the source directory for training images 
    target_size=image_size, # All images will be resized to value set in image_size
    batch_size=bs,
    class_mode='categorical')

# Flow validation images in batches of 16 using val_datagen generator
print("Preparing generator for validation dataset")
val_generator = val_datagen.flow_from_directory(
    directory= val_folder, 
    target_size=image_size,
    batch_size=bs,
    class_mode='categorical')
from tensorflow.keras import optimizers

new_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(amsgrad=True),
              metrics=['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint

Checkpointer = ModelCheckpoint('EnsemblePrime.hdf5', monitor='val_accuracy', save_best_only=True, verbose=1)
history = new_model.fit(
        train_generator, # train generator has 23777 train images
        steps_per_epoch=5948 // bs + 1,
        epochs=300,
        verbose = 1,
        validation_data=val_generator, # validation generator has 5948 validation images
        validation_steps=7433 // bs + 1,
        callbacks=[Checkpointer]
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_folder = '299_small/test'
test_datagen = ImageDataGenerator(rescale=1./255)

print("Preparing generator for test dataset")
test_generator = test_datagen.flow_from_directory(
    directory= test_folder, 
    target_size=(299,299),
    batch_size=16,
    shuffle=False,
    class_mode='categorical')
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

model0 = load_model( 'EnsemblePrime.hdf5' )
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

#merging the input layers
input_shape = (299, 299, 3)
inputs = (Input(input_shape))
output = model0([inputs,inputs,inputs,inputs,inputs,inputs,inputs,inputs,inputs])
new_model = Model(inputs,output)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_folder = '299_small/test'
test_datagen = ImageDataGenerator(rescale=1./255)

print("Preparing generator for test dataset")
test_generator = test_datagen.flow_from_directory(
    directory= test_folder, 
    target_size=(299,299),
    batch_size=16,
    shuffle=False,
    class_mode='categorical')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

input_shape = (299, 299, 3)
num_classes = 9
nb_samples = 7433
bs = 16

y_test = pd.get_dummies(pd.Series(test_generator.classes))
y_pred =  new_model.predict(test_generator,steps = nb_samples//bs + 1, verbose=1)
y_test = y_test.to_numpy()
y_preds = [np.argmax(i) for i in y_pred]
y_preds = np.asarray(y_preds)
y_tests = [np.argmax(i) for i in y_test]
y_tests = np.asarray(y_tests)

cnf_matrix1 = confusion_matrix(test_generator.classes, y_preds)

plt.figure(figsize=(12, 12))
sns.heatmap(cnf_matrix1, xticklabels=label, yticklabels=label, annot=True, fmt="d");
plt.title("Confusion matrix (Unnormalized)")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
cnf_matrix2 = confusion_matrix(y_tests, y_preds, normalize = 'true')

plt.figure(figsize=(12, 12))
sns.heatmap(cnf_matrix2, xticklabels=label, yticklabels=label, annot=True,fmt="0.3f",annot_kws={"size": 15});
plt.title("Confusion matrix for Ensemble Model (Row Normalized)")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

for i in range(num_classes):
    plt.plot(fpr[i], tpr[i],
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.title('ROC for Ensemble Model')
plt.legend(loc='best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
AUCsum = 0
for i in range(num_classes):
    print("AUC of class {0} = {1:0.4f}".format(label[i], roc_auc[i]))
    AUCsum +=roc_auc[i]
print("Average ROC AUC = {0:0.4f}".format(AUCsum/9))
from sklearn.metrics import precision_recall_curve

# Compute PR curve and area for each class
precision = dict()
recall = dict()
pr_auc = dict()

f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')

for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
    pr_auc[i] = auc(recall[i], precision[i])
for i in range(num_classes):
    plt.plot(recall[i], precision[i],
             label='PR curve of class {0} (area = {1:0.4f})'
             ''.format(i, pr_auc[i]))
plt.title('Precision Recall Curve for Ensemble Model')
plt.legend(loc=(0.05, -1.05), prop=dict(size=14))
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
AUCsum = 0
for i in range(num_classes):
    print("AUC of class {0} = {1:0.4f}".format(label[i], pr_auc[i]))
    AUCsum +=pr_auc[i]
print("Average PR AUC = {0:0.4f}".format(AUCsum/9))
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_folders = list()
test_datagens = list()
test_generators = list()

bs = 16

for i in range(9):
  test_folders.append('299_small/test' + str(i))
  test_datagens.append(ImageDataGenerator(rescale=1./255))

  print("Preparing generator for test dataset " + str(i))
  test_generators.append(test_datagens[i].flow_from_directory(
    directory= test_folders[i], 
    target_size=(299,299),
    batch_size=bs,
    shuffle=False,
    class_mode='categorical'))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

y_tests = list()
y_preds = list()

for j in range(9):
  y_test = pd.get_dummies(pd.Series(test_generators[j].classes))
  y_pred =  all_models[j].predict(test_generators[j],steps = test_generators[j].samples // bs + 1, verbose=1)
  y_test = y_test.to_numpy()
  preds = [np.argmax(i) for i in y_pred]
  y_preds.append(np.asarray(preds))
  tests = [np.argmax(i) for i in y_test]
  y_tests.append(np.asarray(tests))
SIZES = [[791,1910,428,2000],
         [791,1910,2000,541],
         [791,1910,2000,439],
         [843,791,1910,2000],
         [843,1910,428,2000],
         [843,1910,2000,541],
         [843,1910,2000,439],
         [262,1910,2000],
         [219,1910,2000]]
# For each of the 9 classes, find an instance of a correct classification and a misclassification
# each of the class has a class_idx variable that stores the following information: 
#[test set number, correctly classified image index, misclassified image index, correctly classified class, misclassified class]

print(y_tests[5][2])
print(y_preds[5][2])
print(y_tests[5][0 + 1])
print(y_preds[5][0 + 1])
print('atelectasis')
atelectasis_idx = [5,2,1,0,1]

print()
print(y_tests[8][0])
print(y_preds[8][0])
print(y_tests[8][0 + 6])
print(y_preds[8][0 + 6])
cardiomegaly_idx = [8,0,6,0,2]
print()

print(y_tests[7][0])
print(y_preds[7][0])
print(y_tests[7][0 + 2])
print(y_preds[7][0 + 2])
consolidation_idx = [7,0,2,0,1]
print()

print(y_tests[2][4])
print(y_preds[2][4])
print(y_tests[2][220])
print(y_preds[2][220])
effusion_idx = [2,4,220,0,3]
print('effusion')

print()
print(SIZES[3][0]+SIZES[3][1])
print(y_tests[3][2115])
print(y_preds[3][2115])
print(y_tests[3][2115 + 7])
print(y_preds[3][2115 + 7])
infiltration_idx = [3,2115-1634,2122-1634,2,1]
print('infiltration')

print()
print(SIZES[4][0]+SIZES[4][1])
print(y_tests[4][2753 + 3])
print(y_preds[4][2753 + 3])
print(y_tests[4][3000 + 1])
print(y_preds[4][3000 + 1])
mass_idx = [4,3001-2753,3,2,3]
print('mass')

print()
print(SIZES[6][0]+SIZES[6][1])
print(y_tests[6][3000])
print(y_preds[6][3000])
print(y_tests[6][3000 + 2])
print(y_preds[6][3000 + 2])
nofinding_idx = [6,3000-2753,3002-2753,2,1]
print()

print(SIZES[1][0]+SIZES[1][1]+SIZES[1][2])
print(y_tests[1][4701 + 4])
print(y_preds[1][4701 + 4])
print(y_tests[1][4701 + 1])
print(y_preds[1][4701 + 1])
nodule_idx = [1,4,1,3,2]
print('nodule')

print()
print(SIZES[6][0]+SIZES[6][1]+SIZES[6][2])
print(y_tests[6][4753 + 0])
print(y_preds[6][4753 + 0])
print(y_tests[6][5000 + 1])
print(y_preds[6][5000 + 1])
pneumothorax_idx = [6,0,5001-4753,3,2]
print('pneumothorax')

#print(np.where(y_preds[5] == 3))
#print(np.where(y_preds[5] == 2))
#print(np.where(y_preds[5] == 1))
#print(np.where(y_preds[5] == 0))
#print(y_tests[2][4701])
#print(y_tests[2][4700])
from PIL import Image
from vis.visualization import visualize_cam, overlay

def visualize_top_convolution(model,image_batch,classnb):
   
    # credit: https://github.com/raghakot/keras-vis/blob/master/applications/self_driving/visualize_attention.ipynb
    heatmap = visualize_cam(model, layer_idx=-1, filter_indices=classnb, seed_input=image_batch,penultimate_layer_idx=-3, grad_modifier='relu')
    img = image_batch.squeeze()

    # credit: (gaussian filter for a better UI) http://bradsliz.com/2017-12-21-saliency-maps/
    import scipy.ndimage as ndimage
    smooth_heatmap = ndimage.gaussian_filter(heatmap[:,:,2], sigma=5)

    nn = 5
    fig = plt.figure(figsize=(20,20))
    a = fig.add_subplot(1, nn, 1)
    plt.imshow(img)
    a.set_title("original",fontsize=10)
    plt.axis('off')
    a = fig.add_subplot(1, nn, 2)
    plt.imshow(overlay(img, heatmap, alpha=0.7))
    a.set_title("heatmap",fontsize=10)
    plt.axis('off')
    a = fig.add_subplot(1, nn, 3)
    plt.imshow(img)
    plt.imshow(smooth_heatmap, alpha=0.7)
    a.set_title("heatmap/gaussian",fontsize=10)
    plt.axis('off')
    plt.show()
import os

atelectasis_dir = '299_small/test'+str(atelectasis_idx[0])+'/atelectasis'
atelectasis_fnames = os.listdir(atelectasis_dir)
cardiomegaly_dir = '299_small/test'+str(cardiomegaly_idx[0])+'/cardiomegaly'
cardiomegaly_fnames = os.listdir(cardiomegaly_dir)
consolidation_dir = '299_small/test'+str(consolidation_idx[0])+'/consolidation'
consolidation_fnames = os.listdir(consolidation_dir)
effusion_dir = '299_small/test'+str(effusion_idx[0])+'/effusion'
effusion_fnames = os.listdir(effusion_dir)
infiltration_dir = '299_small/test'+str(infiltration_idx[0])+'/infiltration'
infiltration_fnames = os.listdir(infiltration_dir)
mass_dir = '299_small/test'+str(mass_idx[0])+'/mass'
mass_fnames = os.listdir(mass_dir)
nodule_dir = '299_small/test'+str(nodule_idx[0])+'/nodule'
nodule_fnames = os.listdir(nodule_dir)
nofinding_dir = '299_small/test'+str(nofinding_idx[0])+'/no_finding'
nofinding_fnames = os.listdir(nofinding_dir)
pneumothorax_dir = '299_small/test'+str(pneumothorax_idx[0])+'/pneumothorax'
pneumothorax_fnames = os.listdir(pneumothorax_dir)

filenames = os.listdir(atelectasis_dir)
filenames.sort()
rightfilename = filenames[atelectasis_idx[1]]
wrongfilename = filenames[atelectasis_idx[2]]

rightpath = os.path.join(atelectasis_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(atelectasis_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**Atelectasis Correct Prediction**")
visualize_top_convolution(all_models[atelectasis_idx[0]],rightimage,atelectasis_idx[3])
print("**Atelectasis mistaken for "+LABELS[atelectasis_idx[0]][atelectasis_idx[4]]+"**")
visualize_top_convolution(all_models[atelectasis_idx[0]],wrongimage,atelectasis_idx[4])
filenames = os.listdir(atelectasis_dir)
filenames.sort()
rightfilename = filenames[atelectasis_idx[1]]
wrongfilename = filenames[atelectasis_idx[2]]

rightpath = os.path.join(atelectasis_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(atelectasis_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**Atelectasis Correct Prediction**")
visualize_top_convolution(all_models[atelectasis_idx[0]],rightimage,atelectasis_idx[3])
print("**Atelectasis mistaken for "+LABELS[atelectasis_idx[0]][atelectasis_idx[4]]+"**")
visualize_top_convolution(all_models[atelectasis_idx[0]],wrongimage,atelectasis_idx[4])
filenames = os.listdir(cardiomegaly_dir)
filenames.sort()
rightfilename = filenames[cardiomegaly_idx[1]]
wrongfilename = filenames[cardiomegaly_idx[2]]

rightpath = os.path.join(cardiomegaly_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(cardiomegaly_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**Cardiomegaly Correct Prediction**")
visualize_top_convolution(all_models[cardiomegaly_idx[0]],rightimage,cardiomegaly_idx[3])
print("**Cardiomegaly mistaken for "+LABELS[cardiomegaly_idx[0]][cardiomegaly_idx[4]]+"**")
visualize_top_convolution(all_models[cardiomegaly_idx[0]],wrongimage,cardiomegaly_idx[4])
filenames = os.listdir(consolidation_dir)
filenames.sort()
rightfilename = filenames[consolidation_idx[1]]
wrongfilename = filenames[consolidation_idx[2]]

rightpath = os.path.join(consolidation_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(consolidation_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**consolidation correct prediction**")
visualize_top_convolution(all_models[consolidation_idx[0]],rightimage,consolidation_idx[3])
print("**consolidation mistaken for "+LABELS[consolidation_idx[0]][consolidation_idx[4]]+"**")
visualize_top_convolution(all_models[consolidation_idx[0]],wrongimage,consolidation_idx[4])
filenames = os.listdir(effusion_dir)
filenames.sort()
rightfilename = filenames[effusion_idx[1]]
wrongfilename = filenames[effusion_idx[2]]

rightpath = os.path.join(effusion_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(effusion_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**effusion correct prediction**")
visualize_top_convolution(all_models[effusion_idx[0]],rightimage,effusion_idx[3])
print("**effusion mistaken for "+LABELS[effusion_idx[0]][effusion_idx[4]]+"**")
visualize_top_convolution(all_models[effusion_idx[0]],wrongimage,effusion_idx[4])
filenames = os.listdir(infiltration_dir)
filenames.sort()
rightfilename = filenames[infiltration_idx[1]]
wrongfilename = filenames[infiltration_idx[2]]

rightpath = os.path.join(infiltration_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(infiltration_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**infiltration correct prediction**")
visualize_top_convolution(all_models[infiltration_idx[0]],rightimage,infiltration_idx[3])
print("**infiltration mistaken for "+LABELS[infiltration_idx[0]][infiltration_idx[4]]+"**")
visualize_top_convolution(all_models[infiltration_idx[0]],wrongimage,infiltration_idx[4])
filenames = os.listdir(mass_dir)
filenames.sort()
rightfilename = filenames[mass_idx[1]]
wrongfilename = filenames[mass_idx[2]]

rightpath = os.path.join(mass_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(mass_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**mass correct prediction**")
visualize_top_convolution(all_models[mass_idx[0]],rightimage,mass_idx[3])
print("**mass mistaken for "+LABELS[mass_idx[0]][mass_idx[4]]+"**")
visualize_top_convolution(all_models[mass_idx[0]],wrongimage,mass_idx[4])
filenames = os.listdir(nofinding_dir)
filenames.sort()
rightfilename = filenames[nofinding_idx[1]]
wrongfilename = filenames[nofinding_idx[2]]

rightpath = os.path.join(nofinding_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(nofinding_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**no finding correct prediction**")
visualize_top_convolution(all_models[nofinding_idx[0]],rightimage,nofinding_idx[3])
print("**no finding mistaken for "+LABELS[nofinding_idx[0]][nofinding_idx[4]]+"**")
visualize_top_convolution(all_models[nofinding_idx[0]],wrongimage,nofinding_idx[4])
filenames = os.listdir(nodule_dir)
filenames.sort()
rightfilename = filenames[nodule_idx[1]]
wrongfilename = filenames[nodule_idx[2]]

rightpath = os.path.join(nodule_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(nodule_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**nodule correct prediction**")
visualize_top_convolution(all_models[nodule_idx[0]],rightimage,nodule_idx[3])
print("**nodule mistaken for "+LABELS[nodule_idx[0]][nodule_idx[4]]+"**")
visualize_top_convolution(all_models[nodule_idx[0]],wrongimage,nodule_idx[4])
filenames = os.listdir(pneumothorax_dir)
filenames.sort()
rightfilename = filenames[pneumothorax_idx[1]]
wrongfilename = filenames[pneumothorax_idx[2]]

rightpath = os.path.join(pneumothorax_dir, rightfilename)
rightimage = mpimg.imread(rightpath)
rightimage = np.stack((rightimage,rightimage, rightimage), axis=2)

wrongpath = os.path.join(pneumothorax_dir, wrongfilename)
wrongimage = mpimg.imread(wrongpath)
wrongimage = np.stack((wrongimage,wrongimage, wrongimage), axis=2)

print("**pneumothorax correct prediction**")
visualize_top_convolution(all_models[pneumothorax_idx[0]],rightimage,pneumothorax_idx[3])
print("**pneumothorax mistaken for "+LABELS[pneumothorax_idx[0]][pneumothorax_idx[4]]+"**")
visualize_top_convolution(all_models[pneumothorax_idx[0]],wrongimage,pneumothorax_idx[4])