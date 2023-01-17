# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns



from scipy.spatial.distance import squareform, pdist



from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix



from keras.models import Model

from keras.layers import Input, Dense, Concatenate, Reshape, Dropout

from keras.layers.embeddings import Embedding



from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot









cat1=5

cat2=10

#n=1000



def make_data(n):

    data = pd.DataFrame()

    data['cat1'] = np.random.randint(0,cat1,n)

    data['cat2'] = np.random.randint(0,cat2,n)

    data['label']=0

    data.loc[ (data.cat1==3) & (data.cat2==5), 'label' ]=1

    data.loc[ (data.cat2==7), 'label' ]=1

    data.head(20)

    return data



train = make_data(1000)

test = make_data(100000)



print(train.head(20))

    



num_categories1 = cat1 #size of the vocabulary

vector_size1 = 2 #dimension of dense embedding space



num_categories2 = cat2  #size of the vocabulary

vector_size2 = 3 #dimension of dense embedding space



perceptron_size = 5 #size of hidden layer above embeddings



input1 = Input(shape=(1,),name='cat1_input')

embedding1 = Embedding(num_categories1, vector_size1, input_length=1, name='embedding_cat1')(input1)

embedding1 = Reshape(target_shape=(vector_size1,))(embedding1)



input2 = Input(shape=(1,),name='cat2_input')

embedding2 = Embedding(num_categories2, vector_size2, input_length=1,name='embedding_cat2')(input2)

embedding2 = Reshape(target_shape=(vector_size2,))(embedding2)



x = Concatenate()([embedding1, embedding2])



x = Dense(perceptron_size, name='perceptron')(x)



output = Dense(1,activation='sigmoid')(x)



model = Model([input1,input2], output)

model.compile(loss='binary_crossentropy',optimizer='adam')



print(model.summary())



history = model.fit([train['cat1'],train['cat2']],

          train['label'],batch_size=10,epochs=10)



plt.plot(history.history['loss'])

plt.title('loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.show()



proba = model.predict([test['cat1'],test['cat2']])

predict = y_class = (proba>0.5)



matrix = confusion_matrix(test['label'], predict)

ax = sns.heatmap(matrix,annot=True,fmt='g',cmap='Blues')

ax.set_xlabel('Predicted')

ax.set_ylabel('True')

plt.show()



np.set_printoptions(precision=1)



print("cat1 vectors")

vectors1 = model.get_layer('embedding_cat1').get_weights()[0]

print(vectors1)



print("cat2 vectors")

vectors2 = model.get_layer('embedding_cat2').get_weights()[0]

print(vectors2)



#plot the distances between the trained embedded vectors, for both categorical variables

fig,axes = plt.subplots(1,2)



m=squareform(pdist(vectors1,metric='cosine'))

sns.heatmap(m,cmap='Blues',ax=axes[0])



m=squareform(pdist(vectors2,metric='cosine'))

sns.heatmap(m,cmap='Blues',ax=axes[1])

SVG(model_to_dot(model).create(prog='dot',format='svg'))