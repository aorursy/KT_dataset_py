

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
import numpy as np
a=np.array([9,5,4,3,2,6])
b=np.array([5,8,6,9,2,1])
print("CHECK IF B HAS SAME VIEWS TO MEMORY IN A")
print(b.base is a)
print("CHECK IF A HAS SAME VIEWS TO MEMORY IN B")
print(a.base is b)
div_by_3=a%3==0
div1_by_3=b%3==0
print("Divisible By 3")
print(a[div_by_3])
print(b[div1_by_3])
b[::-1].sort()
print("SECOND ARRAY SORTED")
print(b)
print("SUM OF ELEMENTS OF FIRST ARRAY")
print(np.sum(a))
from matplotlib import pyplot as plt
SUBJECTS=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]
MARKS=[86,83,86,90,88] 
tick_label=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]
plt.bar(SUBJECTS,MARKS,tick_label=tick_label,width=0.8,color=['green','red','green','green','green'])
plt.xlabel('SUBJECTS') 
plt.ylabel('MARKS')
plt.title("STUDENT's MARKS DATASET")
plt.show()
from matplotlib import pyplot as plt
SUBJECTS=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]
MARKS=[86,83,86,90,88] 
tick_label=["ENGLISH","MATHS","SCIENCE","HISTORY","GEOGRAPHY"]
plt.bar(SUBJECTS,MARKS,tick_label=tick_label,width=0.8,color=['green','red','green','green','green'])
plt.xlabel('SUBJECTS') 
plt.ylabel('MARKS')
plt.title("STUDENT's MARKS DATASET")
plt.show()