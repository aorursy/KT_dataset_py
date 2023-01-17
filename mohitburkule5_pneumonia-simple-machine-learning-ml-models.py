import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# refer https://www.youtube.com/watch?v=X_MaVHIBJHc&feature=youtu.be

try:



    import tensorflow as tf

    import cv2

    import os

    import pickle

    import numpy as np

    print("Library Loaded Successfully ..........")

except:

    print("Library not Found ! ")





class MasterImage(object):



    def __init__(self,PATH='', IMAGE_SIZE = 50):

        self.PATH = PATH

        self.IMAGE_SIZE = IMAGE_SIZE



        self.image_data = []

        self.x_data = []

        self.y_data = []

        self.CATEGORIES = []



        # This will get List of categories

        self.list_categories = []



    def get_categories(self):

        for path in os.listdir(self.PATH):

            if '.DS_Store' in path:

                pass

            else:

                self.list_categories.append(path)

        print("Found Categories ",self.list_categories,'\n')

        return self.list_categories



    def Process_Image(self):

        try:

            """

            Return Numpy array of image

            :return: X_Data, Y_Data

            """

            self.CATEGORIES = self.get_categories()

            for categories in self.CATEGORIES:                                                  # Iterate over categories



                train_folder_path = os.path.join(self.PATH, categories)                         # Folder Path

                class_index = self.CATEGORIES.index(categories)                                 # this will get index for classification



                for img in os.listdir(train_folder_path):                                       # This will iterate in the Folder

                    new_path = os.path.join(train_folder_path, img)                             # image Path



                    try:        # if any image is corrupted

                        image_data_temp = cv2.imread(new_path,cv2.IMREAD_GRAYSCALE)                 # Read Image as numbers

                        image_temp_resize = cv2.resize(image_data_temp,(self.IMAGE_SIZE,self.IMAGE_SIZE))

                        self.image_data.append([image_temp_resize,class_index])

                    except:

                        pass



            data = np.asanyarray(self.image_data)



            # Iterate over the Data

            for x in data:

                self.x_data.append(x[0])        # Get the X_Data

                self.y_data.append(x[1])        # get the label



            X_Data = np.asarray(self.x_data) / (255.0)      # Normalize Data

            Y_Data = np.asarray(self.y_data)



            # reshape x_Data



            X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)



            return X_Data,Y_Data

        except:

            print("Failed to run Function Process Image ")



    def pickle_image(self):



        """

        :return: None Creates a Pickle Object of DataSet

        """

        # Call the Function and Get the Data

        X_Data,Y_Data = self.Process_Image()



        # Write the Entire Data into a Pickle File

        #pickle_out = open('X_Data','wb')

        #pickle.dump(X_Data, pickle_out)

        #pickle_out.close()



        # Write the Y Label Data

        #pickle_out = open('Y_Data', 'wb')

        #pickle.dump(Y_Data, pickle_out)

        #pickle_out.close()



        #print("Pickled Image Successfully ")

        return X_Data,Y_Data



    def load_dataset(self):

            print('Could not Found Pickle File ')

            print('Loading File and Dataset  ..........')



            X_Data,Y_Data = self.pickle_image()

            return X_Data,Y_Data













a=MasterImage("../input/chest-xray-pneumonia/chest_xray/train",100)

X,Y=a.load_dataset()



a=MasterImage("../input/chest-xray-pneumonia/chest_xray/test",100)

X_test,Y_test=a.load_dataset()



a=MasterImage("../input/chest-xray-pneumonia/chest_xray/val",100)

X_val,Y_val=a.load_dataset()
#get features as single array (3d to 2d data )

X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))



X_val = X_val.reshape((X_val.shape[0],X_val.shape[1]*X_val.shape[2]))



X_test = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))

X.shape,Y.shape,X_val.shape,Y_val.shape,X_test.shape,Y_test.shape
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
models = [

    #KNeighborsClassifier(5),

     GaussianProcessClassifier(),

            GaussianNB(),

           DecisionTreeClassifier(max_depth=5,random_state=0),

           RandomForestClassifier( n_estimators=10, max_features=1,random_state=0),

          AdaBoostClassifier(),

          MLPClassifier(),

          SVC(),

    QuadraticDiscriminantAnalysis(),

   

    ]



trainedmodels=[]

for model in models:

  clf=model

  clf.fit(X,Y)

  ypred=clf.predict(X_test)

  yval=clf.predict(X_val)

  print(type(model).__name__," train ",accuracy_score(Y,clf.predict(X))," test ",accuracy_score(Y_test,ypred)," val ",accuracy_score(Y_val,yval))
