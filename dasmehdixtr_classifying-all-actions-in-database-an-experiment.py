# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

   """  for filename in filenames:

        print(os.path.join(dirname, filename))"""



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import scipy.io 
def combineMultipleDatas(data_names):

   datas = data_names[0]

   x = 0

   for data in data_names:

       if x == 0:

           result = datas.append(data,ignore_index=True)

       else:

           result = result.append(data,ignore_index=True)

       x = x+ 1

   return result
def dataTableOptimizerUpdated(mat_file):

    our_data = mat_file['d_skel']

    datas = []

    frame_size = len(our_data[0][0])-1

    for each in range(0,frame_size):

        data_flatten = our_data[:,:,each].flatten()

        data_flatten = data_flatten[np.newaxis]

        datas.append(data_flatten)

    return datas,frame_size
def Loader(path,chosen_class_number):

    full_list = []

    for entry in sorted(os.listdir(path)):

        if os.path.isfile(os.path.join(path, entry)):

            mat = scipy.io.loadmat(path+entry)

            all_data, frame = dataTableOptimizerUpdated(mat_file=mat)

            full_list.extend(all_data)

    #data_ready = dataTableForCluster2(data=full_list,joint_names=joint_names,column_names=col_names,frame=len(full_list))

    full_list = np.concatenate(full_list)

    data_re = pd.DataFrame(full_list)

    data_re['classs'] = np.full((1,len(data_re)),chosen_class_number).T

    return data_re
root_path = "//kaggle//input//human-action-recognition-dataset//"

path= root_path + "a1//"

a_1_files = Loader(path=path,chosen_class_number=1)

path = root_path + "a2//"

a_2_files = Loader(path,chosen_class_number=2)

path = root_path + "a3//"

a_3_files = Loader(path,chosen_class_number=3)

path = root_path + "a4//"

a_4_files = Loader(path,chosen_class_number=4)

path = root_path + "a5//"

a_5_files = Loader(path,chosen_class_number=5)

path = root_path + "a6//"

a_6_files = Loader(path,chosen_class_number=6)

path = root_path + "a7//"

a_7_files = Loader(path,chosen_class_number=7)

path = root_path + "a8//"

a_8_files = Loader(path,chosen_class_number=8)

path = root_path + "a9//"

a_9_files = Loader(path,chosen_class_number=9)

path = root_path + "a10//"

a_10_files = Loader(path,chosen_class_number=10)

path = root_path + "a11//"

a_11_files = Loader(path,chosen_class_number=11)

path = root_path + "a12//"

a_12_files = Loader(path,chosen_class_number=12)

path = root_path + "a13//"

a_13_files = Loader(path,chosen_class_number=13)

path = root_path + "a14//"

a_14_files = Loader(path,chosen_class_number=14)

path = root_path + "a15//"

a_15_files = Loader(path,chosen_class_number=15)

path = root_path + "a16//"

a_16_files = Loader(path,chosen_class_number=16)

path = root_path + "a17//"

a_17_files = Loader(path,chosen_class_number=17)

path = root_path + "a18//"

a_18_files = Loader(path,chosen_class_number=18)

path = root_path + "a19//"

a_19_files = Loader(path,chosen_class_number=19)

path = root_path + "a20//"

a_20_files = Loader(path,chosen_class_number=20)

path = root_path + "a21//"

a_21_files = Loader(path,chosen_class_number=21)

path = root_path + "a22//"

a_22_files = Loader(path,chosen_class_number=22)

path = root_path + "a23//"

a_23_files = Loader(path,chosen_class_number=23)

path = root_path + "a24//"

a_24_files = Loader(path,chosen_class_number=24)

path = root_path + "a25//"

a_25_files = Loader(path,chosen_class_number=25)

path = root_path + "a26//"

a_26_files = Loader(path,chosen_class_number=26)

path = root_path + "a27//"

a_27_files = Loader(path,chosen_class_number=27)
data_names = [a_1_files,a_2_files,a_3_files,a_4_files,a_5_files,a_6_files,a_7_files,a_8_files,a_9_files,a_10_files,a_11_files,a_12_files,a_13_files,a_14_files,a_15_files,a_16_files,a_17_files,a_18_files,a_19_files,a_20_files,a_21_files,a_22_files,a_23_files,a_24_files,a_25_files,a_26_files,a_27_files]

out = combineMultipleDatas(data_names=data_names)
x = out.drop(["classs"],axis=1)

y = out.classs.values
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train,y_train)





nb = GaussianNB()

nb.fit(x_train,y_train)



knn = KNeighborsClassifier(n_neighbors = 4) #n_neighbors = k

knn.fit(x_train,y_train)



rf = RandomForestClassifier(n_estimators=100,random_state=1)

rf.fit(x_train,y_train)



svm = SVC(random_state = 1)

svm.fit(x_train,y_train)





print("acc of svm is :",svm.score(x_test,y_test))

print('Random Forest accuracy on test data is : ',rf.score(x_test,y_test))

print("k={} nn score:{}".format(3,knn.score(x_test,y_test)))

print('accuracy of bayes in test data is :', nb.score(x_test,y_test))

print('acc_of_sgd is: ', sgd.score(x_test,y_test))
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix

class_names=['swipe left','swipe right','wave','clap','throw','arm cross','basketball shot','draw x','draw circle(clockwise)','draw circle(counter_cloclwise)','draw triangle','bowling','boxing','baseball swing','tennis swing','arm curl','tennis serve','push','knock','catch','pickup-throw','jog','walk','sit_to_stand','stand_to_sit','lunge','squat']

titles_options = [("Confusion matrix, without normalization", None),

                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:

    disp = plot_confusion_matrix(rf, x_test, y_test,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 normalize=normalize)

    disp.ax_.set_title(title)



    print(title)

    print(disp.confusion_matrix)



plt.show()
from sklearn.model_selection import cross_val_score



accuracy = cross_val_score(estimator = rf, X = x_train, y =y_train, cv = 10)

print("avg acc: ",np.mean(accuracy))

print("acg std: ",np.std(accuracy))