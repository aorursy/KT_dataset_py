#lets use some machine learning packages
import pandas as pd  
from sklearn.tree import tree  

#import the data and make sure it has the shape we expect
dataset = pd.read_csv("../input/Rat_Sightings.csv")  
print (dataset.shape)  

size_of_dataset = 100

# Select a few columns that we think are important
# get_dummies allows us to use categorical data in a decsion tree
features = pd.get_dummies(dataset[['City','Location Type']][:size_of_dataset])
labels = dataset['Status'][:size_of_dataset]

#print(dataset[['City','Location Type']][:size_of_dataset])
#print(features)
#print(labels)

clf = tree.DecisionTreeClassifier(max_depth=6)
clf = clf.fit(features, labels)
# render an image of the decesion tree so we can see what is important
import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(clf, out_file="fourstar.dot", class_names = sorted(list(set(labels))), feature_names = sorted(list(set(features))))

#call a command line function to convert the image to something we can display online 
!dot -Tpng fourstar.dot -o fourstar.png -Gdpi=600

#then display the decsion tree
from IPython.display import Image
Image(filename = 'fourstar.png')