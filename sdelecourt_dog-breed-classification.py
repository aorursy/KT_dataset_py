import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
path_to_data = '../input/dog-breed-identification'
df = pd.read_csv(os.path.join(path_to_data,'labels.csv'))
# Build a dataframe with the number of instances in each class
breed_distrib = df['breed'].value_counts()
breed_distrib.columns = ['breed', 'number']

# Horizontal bar plot
plt.figure(figsize=(30,100))
sns.set(style="whitegrid")
sns.set(font_scale=5)
ax = sns.barplot(breed_distrib,breed_distrib.index)
plt.show()
sns.set(font_scale=2)
n_breeds = len(breed_distrib.index)
print('Number of breeds : ', n_breeds)
for i in range(n_breeds):
    br = breed_distrib.index[i]
    path = df.loc[df['breed'] == br].iloc[0].id + '.jpg'
    path = os.path.join(path_to_data,'train',path)
    img = plt.imread(path)
    plt.axis('off')
    plt.imshow(img)
    plt.title(br)
    plt.show() 

from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
base_model = VGG16(weights=vgg16_weights)
base_model.summary()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


df['breed'] = LabelEncoder().fit_transform(df['breed'])
y = df['breed'] 
onehot = OneHotEncoder()
y = onehot.fit_transform(np.expand_dims(y, axis=1)).toarray()

#Generator
def generator(df):
    path_train = '../input/dog-breed-identification/train'
    while 1:
        for i in range(int(df.shape[0])):
            img_path = os.path.join(path_train, df.iloc[i]['id']+ '.jpg')
    
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            y = df.iloc[i]['breed']
            y = onehot.transform(y).toarray()
            #print(img.shape,np.array([y]).shape)
            yield (x,y)
                    
gen = generator(df)
X_pred = model.predict_generator(gen,steps=10221, verbose=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_pred, df.iloc[:10221]['breed'])
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=500)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)

print("Incredible accuracy of : ",acc)