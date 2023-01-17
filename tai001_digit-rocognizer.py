# Import

from PIL import Image

from PIL import ImageFilter



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# SET THE RANDOM SEED SO RANDON SHUFFLE ALWAYS GIVES THE SAME RESULTS

data_root = '../input/' # Change me to store data elsewhere



# get training & test csv files as numpy arrays

full_test_set_df  = pd.read_csv(data_root+'test.csv')

full_train_set_df = pd.read_csv(data_root+'train.csv')



full_train_set_df.head(2)
full_train_set_df.groupby('label').size()
#full_test_set  = full_test_set_df.as_matrix()

full_train_set = full_train_set_df.as_matrix()



np.random.seed(42)

np.random.shuffle(full_train_set)



train_validation_split = 0.8



num_rows, num_columns = full_train_set.shape



training_sample_size = int(num_rows*train_validation_split)



train_set =  full_train_set[:training_sample_size]

X_train = train_set[:,1:]

Y_train = train_set[:,0]



validation_set = full_train_set[training_sample_size:]

X_validate = validation_set[:,1:]

Y_validate = validation_set[:,0]
train_set.shape
validation_set.shape
class ImageFactory:

    def __init__(self, w, h):

        self.w = w

        self.h = h

        

    def from_data(self, data):

        return Image.fromarray( np.uint8( data.reshape(self.w, self.h) ))







class LabeledImage:

    def __init__(self, image, label=''):

        self.label = label

        self.image = image

        

    def get_label(self):

        return self.label

    

    def get_image(self):

        return self.image



        

class LabeledImageGrid(object):

    def __init__(self, num_col, row_space= 0.35, column_space = 0.1):

        self.num_col = num_col

        self.row_space = row_space

        self.column_space = column_space

        self.labeled_images = []

        

    def add(self, labeled_image):    

        self.labeled_images.append(labeled_image)

        

    def addImage(self, image, label=''):

        self.labeled_images.append(LabeledImage(image, label) )



    def addRow(self, images):

        for image in images:

            self.labeled_images.append(LabeledImage(image) )

     

    def show(self):

        number_of_images = len(self.labeled_images)

        n_cols = self.num_col

        n_rows = 1 + int(number_of_images/n_cols)

        v_pad  = n_rows*self.row_space

        h_pad  = n_cols*self.column_space



        plt.figure( figsize=(n_cols+h_pad, n_rows+v_pad) )

        for i in range(number_of_images):

            plt.subplot(n_rows, n_cols, i + 1)

            plt.imshow( self.labeled_images[i].get_image(), cmap='gray')

            title_text = str(self.labeled_images[i].get_label())

            plt.title(title_text, size=10)

            plt.xticks(())

            plt.yticks(())

        plt.show()



def show_problem_images(problem_image_ndxs, image_data, predicted, expected):

    number_of_images = problem_image_ndxs.shape[0]

    image_grid = LabeledImageGrid(10)

    image_factory = ImageFactory(28,28)    

    

    for i in range(number_of_images):

        ndx = problem_image_ndxs[i]

        label = 'p:'+ str(predicted[ndx]) + '  a:' + str(expected[ndx])

        image = image_factory.from_data( image_data[ndx] )

        image_grid.add( LabeledImage(image, label) )

    image_grid.show()
image_grid = LabeledImageGrid(8)

image_factory = ImageFactory(28,28)



for ndx in range(10):

    image = image_factory.from_data(X_train[ndx])

    labeled_image = LabeledImage(image, str(ndx))

    image_grid.add(labeled_image)

    

image_grid.show()
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



#Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
#random_forest.fit(X_validate, Y_validate)

Y_pred = random_forest.predict(X_validate)

Y_pred_prob = random_forest.predict_proba(X_validate)

Y_error_ndx, = np.where(Y_validate!=Y_pred)



for i in range(5):

    print( Y_pred[i] )

    print( Y_pred_prob[i])



print ('---------------------------------------------------------')



for i in range(5):

    print( Y_pred[Y_error_ndx[i]] )

    print( Y_pred_prob[Y_error_ndx[i]] )



show_problem_images(Y_error_ndx[:20], X_validate, Y_pred, Y_validate)
Y_pred.shape
random_forest.score(X_validate, Y_validate)
X_train_normalized    = (X_train-128)/128

X_validate_normalized = (X_validate-128)/128





random_forest.fit(X_train_normalized, Y_train)



#Y_pred = random_forest.predict(X_test)



random_forest.score(X_train_normalized, Y_train)
random_forest.score(X_validate_normalized, Y_validate)
# Logistic Regression



#logisticRegression = LogisticRegression()



#logisticRegression.fit(X_train, Y_train)



#Y_pred = random_forest.predict(X_test)



#logisticRegression.score(X_train, Y_train)



#logisticRegression.score(X_validate, Y_validate)
# Image manipulation



def create_base_images(image):

    ret= []

    ret.append(image)

    ret.append(image.rotate( np.random.randint(-45,-25) ).filter(ImageFilter.SMOOTH) )

    ret.append(image.rotate( np.random.randint(25,45) ).filter(ImageFilter.SMOOTH) )

    return ret

    

def image_transfrom_chain(image):

    ret = []

    for image in create_base_images(image):

        rand1 = np.random.randint(0,8)

        rand2 = np.random.randint(0,8)

        box = (rand1, rand2, 20+rand1, 20+rand2)

        ret.append(image)

        new_image = Image.fromarray( np.zeros((784,), dtype=np.uint8 ).reshape(28, 28) )

        new_image.paste(image.crop(box),box=box)

        ret.append(new_image )

        ret.append( image.filter(ImageFilter.BLUR).filter(ImageFilter.SHARPEN ))

    return ret



image_factory = ImageFactory(28,28)



image = image_factory.from_data(X_train[0])   

image_cnt = len( image_transfrom_chain(image) )



image_grid = LabeledImageGrid(image_cnt)



for ndx in range(2):

    image = image_factory.from_data(X_train[ndx])

    images = image_transfrom_chain(image)

    image_grid.addRow(images)

    

image_grid.show()
X_train_expanded = []

Y_train_expanded = []



for ndx in range(X_train.shape[0]):

    image = image_factory.from_data(X_train[ndx])

    images = image_transfrom_chain(image)

    image_cnt = len(images)

    for image_ndx in range(image_cnt):

        image_data = np.array( images[image_ndx]).flatten()

        X_train_expanded.append(image_data)

        Y_train_expanded.append(Y_train[ndx])

        

X_train_expanded = np.array( X_train_expanded )

Y_train_expanded = np.array( Y_train_expanded )



print( X_train_expanded.shape )

print( Y_train_expanded.shape )
#random_forest = RandomForestClassifier(n_estimators=250)



#random_forest.fit(X_train_expanded, Y_train_expanded)



#random_forest.score(X_train_expanded, Y_train_expanded)
#random_forest.score(X_validate, Y_validate)
nearest_kneighbor = KNeighborsClassifier()

nearest_kneighbor.fit(X_train_expanded, Y_train_expanded)

print( nearest_kneighbor.score(X_train_expanded, Y_train_expanded))

print(  nearest_kneighbor.score(X_validate, Y_validate))