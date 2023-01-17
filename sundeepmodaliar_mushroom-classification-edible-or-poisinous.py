import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
mushroom=pd.read_csv("../input/mushrooms.csv")

mushroom.head()
mushroom.describe()

mushroom.info()
sns.countplot(x="cap-shape",data=mushroom,hue="class")
def cap_shape_to_class(shape):

    if shape=="s":

        return 6

    elif shape=="b":

        return 5

    elif shape=="x":

        return 4

    elif shape=="f":

        return 3

    elif shape=="k":

        return 2

    else:

        return 1

mushroom["cap-shape"]=mushroom["cap-shape"].map(cap_shape_to_class)
mushroom["class"]=mushroom["class"].map({"e":1,"p":0})
mushroom.corr()
sns.barplot(x="cap-surface",data=mushroom,y="class")
def cap_surface_to_class(surface):

    if surface=="f":

        return 4

    elif surface=="y":

        return 3

    elif surface=="s":

        return 2

    else:

        return 1

mushroom["cap-surface"]=mushroom["cap-surface"].map(cap_surface_to_class)
mushroom.head()
mushroom.corr()

sns.barplot(x="cap-color",data=mushroom,y="class")
def cap_color_to_class(color):

    if color=="r" or color=="u":

        return 4

    elif color=="c" or color=="w":

        return 3

    elif color=="n" or color=="g":

        return 2

    else:

        return 1

mushroom["cap-color"]=mushroom["cap-color"].map(cap_color_to_class)
mushroom.head()
sns.barplot(x="bruises",data=mushroom,y="class")
def bruises_to_class(bruises):

    if bruises=="t":

        return 2

    else:

        return 1

mushroom["bruises"]=mushroom["bruises"].map(bruises_to_class)
sns.barplot(x="odor",data=mushroom,y="class")
def odor_to_class(odor):

    if odor=="a" or odor=="l" or odor=="n":

        return 2

    else:

        return 1

mushroom["odor"]=mushroom["odor"].map(odor_to_class)
sns.barplot(x="gill-attachment",data=mushroom,y="class")
def gill_attachment_to_class(ga):

    if ga=='a':

        return 2

    else:

        return 1

mushroom["gill-attachment"]=mushroom["gill-attachment"].map(gill_attachment_to_class)
mushroom.columns

sns.barplot(x="gill-spacing",data=mushroom,y="class")
def gill_spacing_to_class(space):

    if space=="w":

        return 2

    else:

        return 1

mushroom["gill-spacing"]=mushroom["gill-spacing"].map(gill_spacing_to_class)
sns.barplot(x="gill-size",data=mushroom,y="class")
def gill_size_to_class(space):

    if space=="b":

        return 2

    else:

        return 1

mushroom["gill-size"]=mushroom["gill-size"].map(gill_size_to_class)
sns.barplot(x="gill-color",data=mushroom,y="class")
def gill_color_to_class(color):

    if color=="o" or color=="e" or color=="k" or color=="n" or color=="u":

        return 3

    elif color=="w" or color=="y" or color=="p":

        return 2

    else:

        return 1

mushroom["gill-color"]=mushroom["gill-color"].map(gill_color_to_class)
sns.barplot(x="stalk-shape",data=mushroom,y="class")
def stalk_shape_to_class(shape):

    if shape=="t":

        return 2

    else:

        return 1

mushroom["stalk-shape"]=mushroom["stalk-shape"].map(stalk_shape_to_class)
sns.barplot(x="stalk-root",data=mushroom,y="class")
def stalk_root_to_class(root):

    if root=="e" or root=="c" or root=="r":

        return 2

    else:

        return 1

mushroom["stalk-root"]=mushroom["stalk-root"].map(stalk_root_to_class)


sns.barplot(x="stalk-surface-above-ring",data=mushroom,y="class")
def stalk_surface_above_ring_to_class(surface):

    if surface=="s" or surface=="f" or surface=="y":

        return 2

    else:

        return 1

mushroom["stalk-surface-above-ring"]=mushroom["stalk-surface-above-ring"].map(stalk_surface_above_ring_to_class)


sns.barplot(x="stalk-surface-below-ring",data=mushroom,y="class")
def stalk_surface_below_ring_to_class(surface):

    if surface=="s" or surface=="f" or surface=="y":

        return 2

    else:

        return 1

mushroom["stalk-surface-below-ring"]=mushroom["stalk-surface-below-ring"].map(stalk_surface_below_ring_to_class)
sns.barplot(x="stalk-color-above-ring",data=mushroom,y="class")
def stalk_color_above_ring(color):

    if color=="g" or color=="e" or color=="o":

        return 4

    elif color=="w":

        return 3

    elif color=="p":

        return 2

    else:

        return 1

mushroom["stalk-color-above-ring"]=mushroom["stalk-color-above-ring"].map(stalk_color_above_ring)

    
sns.barplot(x="stalk-color-below-ring",data=mushroom,y="class")
def stalk_color_below_ring(color):

    if color=="g" or color=="e" or color=="o":

        return 5

    elif color=="w":

        return 4

    elif color=="p":

        return 3

    elif color=="n":

        return 2

    else:

        return 1

mushroom["stalk-color-below-ring"]=mushroom["stalk-color-below-ring"].map(stalk_color_below_ring)
sns.barplot(x="veil-color",data=mushroom,y="class")
def veil_color_to_class(color):

    if color=="n" or color=="o":

        return 3

    elif color=="w":

        return 2

    else:

        return 1

mushroom["veil-color"]=mushroom["veil-color"].map(veil_color_to_class)
sns.barplot(x="ring-number",data=mushroom,y="class")
def ring_number_to_class(ring):

    if ring=="o":

        return 2

    elif ring=="t":

        return 3

    else:

        return 1

mushroom["ring-number"]=mushroom["ring-number"].map(ring_number_to_class)
sns.barplot(x="ring-type",data=mushroom,y="class")
def ring_type_to_class(ring):

    if ring=="p" or ring=="f":

        return 3

    elif ring=="3":

        return 2

    else:

        return 1

mushroom["ring-type"]=mushroom["ring-type"].map(ring_type_to_class)
sns.barplot(x="spore-print-color",data=mushroom,y="class")
def spore_print_color_to_class(color):

    if color=="w":

        return 2

    elif color=="h" or color=="r" :

        return 1

    else:

        return 3

mushroom["spore-print-color"]=mushroom["spore-print-color"].map(spore_print_color_to_class)
sns.barplot(x="population",data=mushroom,y="class")
def population_to_class(pop):

    if pop=="v":

        return 1

    else:

        return 2

mushroom["population"]=mushroom["population"].map(population_to_class)
sns.barplot(x="habitat",data=mushroom,y="class")
def habitat_to_class(pop):

    if pop=="u" or pop=="p" or pop=="l":

        return 1

    else:

        return 2

mushroom["habitat"]=mushroom["habitat"].map(habitat_to_class)
mushroom.drop(columns=["veil-type"],inplace=True)
corr_matrix=mushroom.corr()

corr_matrix
X=mushroom.drop(columns=["class"])

y=mushroom["class"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_pred,y_test)

cm
print((1295+1378)/(1295+1378+8))