import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import svm



import warnings



# Trick for plotting inline in Jupter Notebook

%matplotlib inline



# Ignoring warnings

warnings.filterwarnings("ignore")
def show_some_sample_images(dataset, k=5):

    '''

        Shows k random image samples from dataset.

        

        In the train dataset, there are 728 columns that represent the image.

        We need to reshape this 728 x 1 array to 28 x 28, in order to plot the image correctly.

        You can see it at line: "img.reshape((28, 28))"

        

        :param dataset: Pandas DataFrame

        :param k: Number of images to be shown

    '''

    sample = dataset.sample(n=k)

    for index in range(k):

        img = sample.iloc[index].as_matrix()

        img = img.reshape((28, 28))

        plt.figure(figsize = (20,2))

        plt.grid(False)

        plt.axis('off')

        plt.xticks([])

        plt.yticks([])

        plt.imshow(img)

        plt.show()
data = pd.read_csv('../input/train.csv')

data.head()
labels = data.iloc[0:10000, :1]

images = data.iloc[0:10000, 1:]
show_some_sample_images(images)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
clf = svm.SVC(kernel='linear')

clf = clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images, test_labels))
test_data=pd.read_csv('../input/test.csv')

results=clf.predict(test_data)



test_data['Label'] = pd.Series(results)

test_data['ImageId'] = test_data.index +1

sub = test_data[['ImageId','Label']]



sub.to_csv('submission.csv', index=False)