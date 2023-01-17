import numpy as np

import pandas as pd

import warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)
trees = pd.DataFrame(columns = ['Forest', 'Tree', 'Common names', 'Kingdom', 'Division', 'Class', 'Order', 'Family', 'Genus', 'Species'])



trees = trees.append({

  'Forest': 'Spruce/Fir',

  'Tree': 'Picea engelmannii',

  'Common names': ['Engelmann spruce', 'White spruce', 'Mountain spruce', 'Silver spruce'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Picea',

  'Species': 'P. engelmannii'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Spruce/Fir',

  'Tree': 'Abies lasiocarpa',

  'Common names': ['Subalpine fir', 'Rocky Mountain fir'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Abies',

  'Species': 'A. lasiocarpa'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Spruce/Fir',

  'Tree': 'Pinus resinosa',

  'Common names': ['Red pine', 'Norway pine'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Pinus',

  'Species': 'P. resinosa'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Spruce/Fir',

  'Tree': 'Pseudotsuga menziesii',

  'Common names': ['Douglas fir', 'Douglas-fir', 'Oregon pine' 'Columbian pine'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Pseudotsuga',

  'Species': 'P. menziesii'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Lodgepole Pine',

  'Tree': 'Pinus contorta',

  'Common names': ['Lodgepole pine', 'Shore pine', 'Twisted pine', 'Contorta pine'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Pseudotsuga',

  'Species': 'P. contorta'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Ponderosa Pine',

  'Tree': 'Pinus ponderosa',

  'Common names': ['Ponderosa pine', 'Bull pine', 'Blackjack pine', 'Western yellow-pine'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Pinus',

  'Species': 'P. ponderosa'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Cottonwood/Willow',

  'Tree': 'Populus deltoides',

  'Common names': ['Populus deltoides', 'Eastern cottonwood', 'Necklace poplar'],

  'Kingdom': 'Plantae',

  'Division': 'Magnoliophyta',

  'Class': 'Magnoliopsida',

  'Order': 'Malpighiales',

  'Family': 'Salicaceae',

  'Genus': 'Populus',

  'Species': 'P. deltoides'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Cottonwood/Willow',

  'Tree': 'Salix',

  'Common names': ['Willows', 'Sallows', 'Osiers'],

  'Kingdom': 'Plantae',

  'Division': 'Magnoliophyta',

  'Class': 'Magnoliopsida',

  'Order': 'Malpighiales',

  'Family': 'Salicaceae',

  'Genus': 'Salix',

  'Species': '~ 400 species'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Aspen',

  'Tree': 'Populus tremuloides',

  'Common names': ['Quaking aspen', 'Trembling aspen', 'American aspen', 'Quakies', 'Mountain aspen', 'Golden aspen', 'Trembling poplar', 'White poplar', 'Popple'],

  'Kingdom': 'Plantae',

  'Division': 'Magnoliophyta',

  'Class': 'Magnoliopsida',

  'Order': 'Malpighiales',

  'Family': 'Salicaceae',

  'Genus': 'Populus',

  'Species': 'P. tremuloides'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Douglas-fir',

  'Tree': 'Pseudotsuga menziesii',

  'Common names': ['Douglas fir', 'Douglas-fir', 'Oregon pine' 'Columbian pine'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Pseudotsuga',

  'Species': 'P. menziesii'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Krummholz',

  'Tree': 'Abies lasiocarpa',

  'Common names': ['Subalpine fir', 'Rocky Mountain fir'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Abies',

  'Species': 'A. lasiocarpa'

}, ignore_index = True)





trees = trees.append({

  'Forest': 'Krummholz',

  'Tree': 'Picea engelmannii',

  'Common names': ['Engelmann spruce', 'White spruce', 'Mountain spruce', 'Silver spruce'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Picea',

  'Species': 'P. engelmannii'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Krummholz',

  'Tree': 'Pinus contorta',

  'Common names': ['Lodgepole pine', 'Shore pine', 'Twisted pine', 'Contorta pine'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Pseudotsuga',

  'Species': 'P. contorta'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Krummholz',

  'Tree': 'Pinus flexilis',

  'Common names': ['Limber pine'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Pinus',

  'Species': 'P. flexilis'

}, ignore_index = True)



trees = trees.append({

  'Forest': 'Krummholz',

  'Tree': 'Pinus aristata',

  'Common names': ['Rocky Mountain bristlecone pine', 'the Colorado bristlecone pine'],

  'Kingdom': 'Plantae',

  'Division': 'Pinophyta',

  'Class': 'Pinopsida',

  'Order': 'Pinales',

  'Family': 'Pinaceae',

  'Genus': 'Pinus',

  'Species': 'P. aristata'

}, ignore_index = True)



# Export trees database

trees.to_csv('trees.csv', index = False)



trees
forest_types = np.array(['Spruce/Fir','Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'])

forests = pd.DataFrame(columns = ['Forest', 'Tree', 'Common names', 'Kingdom', 'Division', 'Class', 'Order', 'Family', 'Genus', 'Species'])



def add_to_list(f, x):

    l = []

    for i in trees.query('Forest == "' + f + '"')[x]:

        if i not in l:

            if type(i) == list:

                l.extend(i)

            else:

                l.append(i)

    if len(l) == 1:

        return l[0]

    else:

        return l



for forest in forest_types:

    forests = forests.append({

      'Forest': forest,

      'Tree': add_to_list(forest, 'Tree'),

      'Common names': add_to_list(forest, 'Common names'),

      'Kingdom': add_to_list(forest, 'Kingdom'),

      'Division': add_to_list(forest, 'Division'),

      'Class': add_to_list(forest, 'Class'),

      'Order': add_to_list(forest, 'Order'),

      'Family': add_to_list(forest, 'Family'),

      'Genus': add_to_list(forest, 'Genus'),

      'Species': add_to_list(forest, 'Species')

    }, ignore_index = True)



# Export forests database

forests.to_csv('forests.csv', index = False)



forests
# Read training file

X_train = pd.read_csv('../input/learn-together/train.csv', index_col = 'Id', engine = 'python')



# Define the dependent variable 

y_train = X_train['Cover_Type'].copy()



# Define a training set

X_train = X_train.drop(['Cover_Type'], axis = 'columns')
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict, cross_val_score



model = RandomForestClassifier(random_state = 1)

model.fit(X_train, y_train)

y_pred = cross_val_predict(model, X_train, y_train, n_jobs = -1)

scores = cross_val_score(model, X_train, y_train, scoring = 'accuracy', n_jobs = -1)

print('Accuracy: {:.4f}'.format(np.mean(scores)))
class_names = np.array([None, 'Spruce/Fir','Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz'])

class_names
import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



title = 'Confusion matrix'



# Compute confusion matrix

cm = confusion_matrix(y_train, y_pred)

# Only use the labels that appear in the data

classes = class_names[unique_labels(y_train, y_pred)]



fig, ax = plt.subplots()

im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

ax.figure.colorbar(im, ax=ax)

# We want to show all ticks...

ax.set(xticks=np.arange(cm.shape[1]),

       yticks=np.arange(cm.shape[0]),

       # ... and label them with the respective list entries

       xticklabels=classes, yticklabels=classes,

       title=title,

       ylabel='True label',

       xlabel='Predicted label')



# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

         rotation_mode="anchor")



# Loop over data dimensions and create text annotations.

fmt = 'd'

thresh = cm.max() / 2.

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(j, i, format(cm[i, j], fmt),

                ha="center", va="center",

                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()



np.set_printoptions(precision=2)



# plt.figure(figsize=(24, 24))

plt.show()
predicted_errors = dict()

for (x, y), element in np.ndenumerate(cm):

    if x != y:

        predicted_errors[class_names[x + 1] + ', ' + class_names[y + 1]] = element



        mean = np.array(list(predicted_errors.values())).mean()



print('Mean error: {:.0f}'.format(mean))



greater_than_mean_predicted_errors = {k:v for (k,v) in predicted_errors.items() if v > mean}



sorted_greater_than_mean_predicted_errors = sorted(greater_than_mean_predicted_errors.items(), key=lambda kv: kv[1])



sorted_greater_than_mean_predicted_errors
print('Spruce/Fir - Douglar-fir:   {:3d} / {:3d}'.format(cm[0, 5], cm[5, 0]))

print('Spruce/Fir - Krummholz:     {:3d} / {:3d}'.format(cm[0, 6], cm[6, 0]))

print('Krummholz - Lodgepole Pine: {:3d} / {:3d}'.format(cm[6, 1], cm[1, 6]))

print('Prediction error between Cottonwood/Willow and Aspen and the rest:')

print('Cottonwood/Willow - rest: {:3d} / {:3d} / {:3d} / {:3d} / {:3d}'.format(cm[3, 0], cm[3, 1], cm[3, 2], cm[3, 5], cm[3, 6]))

print('Rest - Cottonwood/Willow: {:3d} / {:3d} / {:3d} / {:3d} / {:3d}'.format(cm[0, 3], cm[1, 3], cm[2, 3], cm[5 ,3], cm[6, 3]))

print('Aspen - rest:             {:3d} / {:3d} / {:3d} / {:3d} / {:3d}'.format(cm[4, 0], cm[4, 1], cm[4, 2], cm[4, 5], cm[4, 6]))

print('Rest - Aspen:             {:3d} / {:3d} / {:3d} / {:3d} / {:3d}'.format(cm[0, 4], cm[1, 4], cm[2, 4], cm[5 ,4], cm[6, 4]))
print('Prediction error between:')

print('Cottonwood/Willow - Aspen:   {:3d} / {:3d}'.format(cm[3, 4], cm[4, 3]))