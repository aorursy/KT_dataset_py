# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_path = '../input/Creditcardprom.csv'

data = pd.read_csv(data_path)
data.head()
#data.columns  # <- returns an index 

data.columns.tolist()
data2 = data[['Income Range',

 #'Magazine Promo',

 'Watch Promo',

 'Life Ins Promo',

 'Credit Card Ins.',

 'Sex',

 'Age']]
data2 = data[ 

   [col for col in data.columns if col != 'Magazine Promo']

]

data2.head()
def create_label_encoder_dict(df):

    from sklearn.preprocessing import LabelEncoder

    

    label_encoder_dict = {}

    for column in df.columns:

        # Only create encoder for categorical data types

        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict
from sklearn.preprocessing import LabelEncoder
df = data2.copy()

label_encoder_dict = {}

for column in df.columns:

    print("About to create encoder for column %s" % column)

        # Only create encoder for categorical data types

    if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':

        print("This is a valid column %s" % column)

        label_encoder_dict[column]= LabelEncoder().fit(df[column])
label_encoders = create_label_encoder_dict(data2)

print("Encoded Values for each Label")

print("="*32)

for column in label_encoders:

    print("="*32)

    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))

    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)
#https://github.com/gggordon/data-mining-cmp4023-notebooks
# Apply each encoder to the data set to obtain transformed values

data3 = data2.copy() # create copy of initial data set

for column in data3.columns:

    if column in label_encoders:

        data3[column] = label_encoders[column].transform(data3[column])



print("Transformed data set")

print("="*32)

data3
data2




# separate our data into dependent (Y) and independent(X) variables

X_data = data3[['Income Range','Sex','Age', 'Watch Promo']]

Y_data = data3['Life Ins Promo']



from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
len(X_train)
X_test.shape
X_train
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
# Create the classifier with a maximum depth of 2 using entropy as the criterion for choosing most significant nodes

# to build the tree

clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=2) 

# Hint : Change the max_depth to 10 or another number to see how this affects the tree
# Build the classifier  by training it on the training data

clf.fit(X_train, y_train)
pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100) ], index = X_data.columns, columns = ['Feature Significance in Decision Tree'])
pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100) ],

            columns=['Feature Significance in Decision Tree'])
clf.classes_
import graphviz
dot_data = tree.export_graphviz(clf,out_file=None, 

                                feature_names=X_data.columns, 

                         class_names=label_encoders[Y_data.name].classes_,  

                         filled=True, rounded=True,  proportion=True,

                                node_ids=True, #impurity=False,

                         special_characters=True)
graph = graphviz.Source(dot_data) 

graph
data2['Watch Promo'].unique()
X_data.columns
gon = {

    'Income Range':'40-50,000',

    'Sex':'Male',

    'Watch Promo':'No',

    'Age':51

}
gon
label_encoders
gon_transformed = {

    'Income Range':label_encoders['Income Range'].transform( [gon['Income Range']]  )[0],

    'Sex':label_encoders['Sex'].transform( [gon['Sex']]  )[0],

    'Watch Promo':label_encoders['Watch Promo'].transform( [gon['Watch Promo']]  )[0],

    'Age':51

}

gon_transformed
# prediting which clas

clf.predict(

    pd.DataFrame(

        [gon_transformed]

    )

)
gon['Income Range']
k=(clf.predict(X_test) == y_test)
k.value_counts()




from sklearn.metrics import confusion_matrix



cm=confusion_matrix(y_test, clf.predict(X_test), labels=y_test.unique())

cm

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    import itertools

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
plot_confusion_matrix(cm,data2['Life Ins Promo'].unique())