import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

# Libraries for displying the data. 
from IPython.core.display import HTML 
from ipywidgets import interact
from IPython.display import display

%matplotlib inline
training   = json.load(open("../input/train.json"))
test       = json.load(open("../input/test.json"))
validation = json.load(open("../input/validation.json"))
# A function to be mapped over the json dictionary. 
def joinFn(dat):
    return [dat[0]["url"][0], dat[1]["label_id"]]

trainingDF   = pd.DataFrame(list(map(joinFn, zip(training["images"],training["annotations"]))),columns=["url","label"])
validationDF = pd.DataFrame(list(map(joinFn, zip(validation["images"],validation["annotations"]))),columns=["url","label"])
testDF       = pd.DataFrame(list(map(lambda x: x["url"],test["images"])),columns=["url"])
trainingDF
validationDF.head()
testDF.head()
print("Number of classes: {0}".format(len( trainingDF["label"].unique())))
trainingDF["label"].value_counts().plot(kind='bar',figsize=(40,10),title="Number of Training Examples Versus Class").title.set_size(40)
def displayExamples(exampleIndex=0):
    outHTML = "<div>"
    for label in range(1,129):
        img_style = "width: 180px;height:180px; margin: 0px; float: left; border: 1px solid black;"
        captionDiv = "<div style='position:absolute;right:30px;color:red;font-size:30px;background-color:grey;padding:5px;opacity:0.5'>"+str(label)+"</div>"
        outHTML += "<div style='position:relative;display:inline-block'><img style='"+img_style+"' src='"+trainingDF[trainingDF.label == label].iloc[exampleIndex][0]+"'/>"+captionDiv+"</div>"
    outHTML += "</div>"
    display(HTML(outHTML))

displayExamples()
def displayCategoryExamples(category=0,nExamples=20):
    outHTML = "<div>"
    for idx in range(0,nExamples):
        img_style = "width: 180px;height:180px; margin: 0px; float: left; border: 1px solid black;"
        outHTML += "<div style='position:relative;display:inline-block'><img style='"+img_style+"' src='"+trainingDF[trainingDF.label == category].iloc[idx][0]+"'/></div>"
    outHTML += "</div>"
    display(HTML(outHTML))
    
displayCategoryExamples(7)
displayCategoryExamples(24)
def visCat(cat=1):
    displayCategoryExamples(cat,40)
    
interact(visCat, cat=(1,128))