import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/train.csv')
train_images = df.drop('label',axis=1)
labels = df['label']
train_images.head()
labels.head()
#Here is a very poorly way to see test_images and their respective labels as titles
fig,axes = plt.subplots(nrows=3,ncols=3)
image_index = 0
for x in range(0,3): 
    for y in range(0,3):
        axes[x,y].set_title(labels[image_index])
        axes[x,y].imshow(train_images.iloc[image_index].values.reshape(28,28))
        axes[x,y].axis('off')
        image_index+=1
plt.tight_layout()
plt.figure(figsize=(10,6))
sns.countplot(labels)
plt.grid()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, labels, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier
'''
Since we don't have a equal distribution of numbers
per class we are going to normalize the weights with class_weight
'''
dtree = DecisionTreeClassifier(class_weight='balanced')
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rdc = RandomForestClassifier(n_estimators=100)
rdc.fit(X_train,y_train)
predictions = rdc.predict(X_test)
print(classification_report(y_test,predictions))
