# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import tensorflow as tf

tf.random.set_seed(100)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import bs4
import os
import numpy as np
import pandas as pd 
path="/kaggle/input/road-sign-detection/annotations"
content=[]

for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    finalpath= os.path.join(path, filename)

    infile = open(finalpath,"r")
    contents = infile.read()
    soup = bs4.BeautifulSoup(contents,'xml')
    class_name=soup.find_all("name")
    name = soup.find_all('filename')
    width= soup.find_all("width")
    height=soup.find_all("height")
    depth=soup.find_all("depth")
    
    ls=[]
    for x in range(0,len(name)):
        for i in name:
            name=name[x].get_text()
            path_name="images/"+name
        class_name=class_name[x].get_text()
        height=int(height[x].get_text())
        depth=int(depth[x].get_text())
        width=int(width[x].get_text())
        f_name=filename
        ls.extend([f_name,path_name,width,height,depth,class_name])

    content.append(ls)

import pandas as pd
new_cols = ["f_name","path_name", "width","height","depth","class_name"]
data = pd.DataFrame(data = content, columns = new_cols)
data.class_name=data.class_name.map({'trafficlight':1, 'speedlimit':2, 'crosswalk':3, 'stop':4})
data.head()

print("wait it will take time")
data1=[]
from PIL import Image,ImageTk
import numpy
# try:
for a in data.path_name.values:
    image = Image.open("/kaggle/input/road-sign-detection/"+a).convert("RGB")
    image=image.resize((224,224),Image.ANTIALIAS)
    image=numpy.array(image.getdata()).reshape(224,224,3)
    data1.append(image)

# except:
print("done")
X=np.array(data1)

y=np.array(data.iloc[:,-1],dtype=int)



from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
c=to_categorical(y,dtype=int)
Y=c[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=787)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
model = Sequential()
model.add(Conv2D(128, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(128, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(4, activation='softmax'))

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test))
