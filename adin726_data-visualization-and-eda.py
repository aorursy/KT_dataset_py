# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

import PIL



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/kuzushiji-recognition/train.csv"); len(train)
character_dict = pd.read_csv("/kaggle/input/kuzushiji-recognition/unicode_translation.csv"); len(character_dict)
ims = os.listdir("/kaggle/input/kuzushiji-recognition/train_images"); len(ims)
class JapChar():

    def __init__(self, char_data, im_id):

        self.char = char_data[0]

        self.x = int(char_data[1])

        self.y = int(char_data[2])

        self.width = int(char_data[3])

        self.height = int(char_data[4])

        self.im_id = im_id

        

    def get_area(self):

        return self.width * self.height



    def get_file(self):

        return "/kaggle/input/kuzushiji-recognition/train_images/" + self.im_id + ".jpg";

    

    def get_top_left(self):

        return [self.x, self.y]

    

    def get_bottom_right(self):

        return [self.x + self.width, self.y + self.height]

    

    def show(self):

        plt.figure(figsize = (6, 6))

        im = PIL.Image.open(self.get_file())

        im = im.crop(self.get_top_left()  + self.get_bottom_right())

        plt.imshow(im)
class ScripturePage:

    def __init__(self, im_data):

        self.id = im_data[0]

        if type(im_data[1]) is not float:

            split_labels = im_data[1].split()

            self.labels = [JapChar(split_labels[i: i+5], self.id) for i in range(0, len(split_labels), 5)]

        else:

            self.labels = []

        

    def get_file(self):

        return "/kaggle/input/kuzushiji-recognition/train_images/" + self.id + ".jpg";

    

    def show(self):

        plt.figure(figsize  = (10, 10))

        plt.imshow(plt.imread(self.get_file()))

        

    def get_im(self):

        return PIL.Image.open(self.get_file());

    

    def show_labeled(self):

        plt.figure(figsize  = (10, 10))

        ax = plt.gca()

        plt.imshow(self.get_im())

        

        for label in self.labels:

            box = Rectangle((label.x, label.y), label.width, label.height, fill = False, edgecolor = 'r')

            ax.add_patch(box)

            

        plt.show()
data = [ScripturePage(train.loc[i]) for i in range(len(train))]
page = data[0]
page.labels[25].show()
page.show()
page.show_labeled()
all_chars = []

for page in data:

    all_chars = all_chars + page.labels
len(all_chars)
all_char_areas = [char.get_area() for char in all_chars]

plt.figure(figsize=(10, 10))

_,_,_ = plt.hist(all_char_areas, 20)
char_freq = character_dict.copy()

codes = np.array(char_freq["Unicode"])

freqs = np.zeros(len(char_freq))



for char in all_chars:

    freqs[np.where(codes == char.char)[0]] += 1

    

char_freq["Frequency"] = freqs

char_freq.describe()