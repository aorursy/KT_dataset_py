#!/usr/bin/python

# -*- coding: utf-8 -*-



"""

=========================================================

Feature agglomeration

=========================================================



These images how similar features are merged together using

feature agglomeration.

"""

print(__doc__)



# Code source: Gaël Varoquaux

# Modified for documentation by Jaques Grobler

# License: BSD 3 clause



import numpy as np

import matplotlib.pyplot as plt



from sklearn import datasets, cluster

from sklearn.feature_extraction.image import grid_to_graph



digits = datasets.load_digits()

images = digits.images

X = np.reshape(images, (len(images), -1))

connectivity = grid_to_graph(*images[0].shape)



agglo = cluster.FeatureAgglomeration(connectivity=connectivity,

                                     n_clusters=32)



agglo.fit(X)

X_reduced = agglo.transform(X)



X_restored = agglo.inverse_transform(X_reduced)

images_restored = np.reshape(X_restored, images.shape)

plt.figure(1, figsize=(4, 3.5))

plt.clf()

plt.subplots_adjust(left=.01, right=.99, bottom=.01, top=.91)

for i in range(4):

    plt.subplot(3, 4, i + 1)

    plt.imshow(images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')

    plt.xticks(())

    plt.yticks(())

    if i == 1:

        plt.title('Original data')

    plt.subplot(3, 4, 4 + i + 1)

    plt.imshow(images_restored[i], cmap=plt.cm.gray, vmax=16,

               interpolation='nearest')

    if i == 1:

        plt.title('Agglomerated data')

    plt.xticks(())

    plt.yticks(())



plt.subplot(3, 4, 10)

plt.imshow(np.reshape(agglo.labels_, images[0].shape),

           interpolation='nearest', cmap=plt.cm.nipy_spectral)

plt.xticks(())

plt.yticks(())

plt.title('Labels')

plt.show()

'''

Basic Picture Viewer

====================



This simple image browser demonstrates the scatter widget. You should

see three framed photographs on a background. You can click and drag

the photos around, or multi-touch to drop a red dot to scale and rotate the

photos.



The photos are loaded from the local images directory, while the background

picture is from the data shipped with kivy in kivy/data/images/background.jpg.

The file pictures.kv describes the interface and the file shadow32.png is

the border to make the images look like framed photographs. Finally,

the file android.txt is used to package the application for use with the

Kivy Launcher Android application.



For Android devices, you can copy/paste this directory into

/sdcard/kivy/pictures on your Android device.



The images in the image directory are from the Internet Archive,

`https://archive.org/details/PublicDomainImages`, and are in the public

domain.



'''



import kivy

kivy.require('1.0.6')



from glob import glob

from random import randint

from os.path import join, dirname

from kivy.app import App

from kivy.logger import Logger

from kivy.uix.scatter import Scatter

from kivy.properties import StringProperty





class Picture(Scatter):

    '''Picture is the class that will show the image with a white border and a

    shadow. They are nothing here because almost everything is inside the

    picture.kv. Check the rule named <Picture> inside the file, and you'll see

    how the Picture() is really constructed and used.



    The source property will be the filename to show.

    '''



    source = StringProperty(None)





class PicturesApp(App):



    def build(self):



        # the root is created in pictures.kv

        root = self.root



        # get any files into images directory

        curdir = dirname(__file__)

        for filename in glob(join(curdir, 'images', '*')):

            try:

                # load the image

                picture = Picture(source=filename, rotation=randint(-30, 30))

                # add to the main field

                root.add_widget(picture)

            except Exception as e:

                Logger.exception('Pictures: Unable to load <%s>' % filename)



    def on_pause(self):

        return True





if __name__ == '__main__':

    PicturesApp().run()
