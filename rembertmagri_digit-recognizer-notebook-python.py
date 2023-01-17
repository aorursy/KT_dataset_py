import pandas as pd



df = pd.read_csv("../input/train.csv")



print(type(df)) # df type: pandas.core.frame.DataFrame



df.head()
print(df.columns)

#for column in df.columns:

#    print("* "+column+":", df[column].unique(), sep="\n")
images = df.drop('label',1)

labels = df[['label']]

pixels = images.unstack()

print("Total pixels:",pixels.size)
import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt



%matplotlib inline



mpl.rcParams['figure.figsize'] = [16.0, 8.0]



plt.subplot(2, 1, 1)

plt.hist(pixels.values, bins=range(0, 260, 5))

plt.xticks(np.arange(0, 260, 10))

plt.title("pixels")

plt.subplot(2, 1, 2)

plt.hist(labels.values)

plt.xticks(np.arange(0, 10))

plt.title("labels")
print('images (rows, columns): ',images.shape)

print('labels (rows, columns): ',labels.shape)
print("Original shape of image data for one number:",images.values[0].shape)

print("Reshaped image 28x28 pixels:",images.values[0].reshape(28,28).shape)
import matplotlib.cm as cm



%matplotlib inline



mpl.rcParams['figure.figsize'] = [15.0, 10.0]



for i in range(0, 15):

    plt.subplot(3, 5, (i+1))

    plt.imshow(images.values[i].reshape(28,28), cmap=cm.binary)

    plt.title(labels.iloc[i, 0])
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier



# mlp.score / Kaggle



# mlp = MLPClassifier(hidden_layer_sizes=(1000,500,100)) # 0.989500/0.96529

# mlp = MLPClassifier(hidden_layer_sizes=(2000,1000,200)) # 0.993048/0.96214

# mlp = MLPClassifier(hidden_layer_sizes=(1000,500)) # 0.988286/0.96371

# mlp = MLPClassifier(hidden_layer_sizes=(500,250,50)) # 0.987619/0.96943

mlp = MLPClassifier(hidden_layer_sizes=(88)) # 0.985214/0.95457

# mlp = MLPClassifier(hidden_layer_sizes=(264,89,30)) # 0.989500/0.96071

# mlp = MLPClassifier(hidden_layer_sizes=(392,196,98,48,22)) # score: 0.988643/0.96871



mlp.fit(images.values,labels.unstack())
print("Training set score: %f" % mlp.score(images.values,labels.unstack()))
df_test = pd.read_csv("../input/test.csv")

predictions = mlp.predict(df_test)
print(predictions)



submission=pd.DataFrame({

        "ImageId": list(range(1,len(predictions)+1)),

        "Label": predictions

    })

submission.to_csv("submission.csv", index=False, header=True)
import matplotlib.cm as cm



%matplotlib inline



mpl.rcParams['figure.figsize'] = [15.0, 20.0]



for i in range(0, 30):

    plt.subplot(6, 5, (i+1))

    plt.imshow(df_test.values[i].reshape(28,28), cmap=cm.binary)

    plt.title(predictions[i])
from sklearn.decomposition import PCA



pca = PCA(n_components=90)

pca.fit(images.values)

transform_images = pca.transform(images.values)

transform_test = pca.transform(df_test)



print(transform_images.shape)



# mlp.score / Kaggle



mlp = MLPClassifier(hidden_layer_sizes=(30)) # 0.970548/0.94571

mlp.fit(transform_images,labels.unstack())



print("Training set score: %f" % mlp.score(transform_images,labels.unstack()))



predictions = mlp.predict(transform_test)



submission=pd.DataFrame({

        "ImageId": list(range(1,len(predictions)+1)),

        "Label": predictions

    })

submission.to_csv("submission_PCA.csv", index=False, header=True)