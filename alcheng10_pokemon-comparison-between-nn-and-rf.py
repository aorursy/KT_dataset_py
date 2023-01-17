import pandas as pd

import numpy as np

import pandas_profiling



#Suppress warnings

import warnings

warnings.filterwarnings("ignore")
# Import data

pokemon = pd.read_csv('../input/pokemon/pokemon.csv')



pokemon.head(10)
# Next let's do some EDA

pokemon.profile_report(style={'full_width':True})
pokemon.describe().transpose()
pokemon.dtypes
# We will drop abilities, for simplicity purposes

pokemon.drop(columns='abilities', inplace=True)



# Drop missing values

pokemon = pokemon.dropna(axis=1) #Rows with NaN



pokemon
# Next we need to split feature vs labels/targets

# We will arrayise the features



# Labels are the values we want to predict - in this case, whether a pokemon is legendary

labels = np.array(pokemon['is_legendary'])



# Remove the labels from the features

features = pokemon.drop('is_legendary', axis = 1) # axis 1 refers to the columns

feature_names = list(pokemon.drop('is_legendary', axis = 1).columns) # Get feature names



# While data is already in sparse matrix for many aspects, one-hot encoding required for remaining string values

pokemon_preprocessed = pd.get_dummies(features)



# Convert to numpy array

features = np.array(pokemon_preprocessed)



pokemon_preprocessed.head()
# Create training vs test data

# We'll use a 70/30 split

from sklearn.model_selection import train_test_split



train_features, test_features, train_labels, test_labels = train_test_split(

    features

    ,labels

   ,test_size=0.30 #30%

   ,random_state=42 #seed used by the random number generator

)



print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)
#NN are sensitive to feature scaling, so we'll scale our data - only train data

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaler.fit(train_features)



# Now apply the transformations to the data:

train_features = scaler.transform(train_features)

test_features = scaler.transform(test_features)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs' #‘lbfgs’ is an optimizer in the family of quasi-Newton methods.

                    ,alpha=1e-5 #L2 penalty (regularization term) parameter.

                    ,hidden_layer_sizes=(2,100) #We will pick 100 hidden layers, each 2 activation functions wide 

                    ,random_state=1 #seed used by the random number generator

                   )



# Fit training data to NN model

model_NN = clf.fit(train_features, train_labels)
# Predict using test data with NN model 

predict_NN = model_NN.predict(test_features)
# Evaluate results - show confusion matrix

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt



class_names = ['Is Legendary', 'Is not Legendary']

cm = confusion_matrix(predict_NN, test_labels)



# Reconvert back to DF

df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)



# Create seaborn heatmap plot

fig_NN = plt.figure() 



plt.ylabel('True label')

plt.xlabel('Predicted label')



heatmap = sns.heatmap(df_cm

                      , annot=True

                      , fmt="d",

                      cmap="Blues")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14, color='black')

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14, color='black')



fig_NN.show()
from sklearn.metrics import accuracy_score



print("Accuracy:")

print(accuracy_score(test_labels, predict_NN))
from sklearn.metrics import classification_report



print(classification_report(test_labels, predict_NN))
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100 # Number of trees in forest

                             ,max_depth=5 # Number of levels of tree

                             ,random_state=0 #seed used by the random number generator

                            )



# Fit training data to RF model

model_RF = clf.fit(train_features, train_labels)
predict_RF = model_RF.predict(test_features)
# Unlike NN, in RF we can peek under the hood

# Print out 1st tree in the forest

from sklearn.tree import export_graphviz

from graphviz import Source

from IPython.display import Image



export_graphviz(

    model_RF.estimators_[0]

    ,out_file='1_tree_limited.dot'

    ,feature_names=(pokemon_preprocessed.columns) # Grab feature names, minus the label

    ,class_names = ['Is Legendary', 'Is not Legendary']

   ,filled = True

    )



!dot -Tpng 1_tree_limited.dot -o 1_tree_limited.png -Gdpi=600

Image(filename = '1_tree_limited.png')
# Print out 10th tree in the forest

export_graphviz(

    model_RF.estimators_[9]

    ,out_file='10_tree_limited.dot'

    ,feature_names=(pokemon_preprocessed.columns) # Grab feature names, minus the label

    ,class_names = ['Is Legendary', 'Is not Legendary']

   ,filled = True

    )



!dot -Tpng 10_tree_limited.dot -o 10_tree_limited.png -Gdpi=600

Image(filename = '10_tree_limited.png')
# Evaluate results - show confusion matrix

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt



class_names = ['Is Legendary', 'Is not Legendary']

cm = confusion_matrix(predict_RF, test_labels)



# Reconvert back to DF

df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)



# Create seaborn heatmap plot

fig_RF = plt.figure() 



plt.ylabel('True label')

plt.xlabel('Predicted label')



heatmap = sns.heatmap(df_cm

                      , annot=True

                      , fmt="d",

                      cmap="Blues")

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14, color='black')

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14, color='black')



fig_RF.show()
from sklearn.metrics import accuracy_score



print("Accuracy:")

print(accuracy_score(test_labels, predict_RF))
from sklearn.metrics import classification_report



print(classification_report(test_labels, predict_RF))
print("NN accuracy: ")

print(round(accuracy_score(predict_NN, test_labels) * 100, 2), '%')
fig_NN
print("RF accuracy: ")

print(round(accuracy_score(predict_RF, test_labels) * 100, 2) ,'%')
fig_RF