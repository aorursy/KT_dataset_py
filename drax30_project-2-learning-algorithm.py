# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!ls '/usr/lib/jvm/'
import sys
sys.path
sys.path.append("/usr/lib/jvm/java-8-openjdk-amd64/bin/")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64/"
!apt update
!apt-get install build-essential python3-dev -y
!apt-get install python3-pil python3-pygraphviz -y
!apt install openjdk-8-jdk -y
!pip install javabridge --no-cache-dir
!pip install python-weka-wrapper3 --no-cache-dir

import weka.core.jvm as jvm
jvm.start()
# feature_selection=pd.read_csv('../input/cs3481-ass2/feature_selection_df.csv')
# feature_selection.diagnosis.value_counts()
# f_glass=glass
# from weka.filters import Filter
# remove = Filter(classname="weka.filters.unsupervised.attribute.Remove")
# remove.options=["-R", "5,9"]
# remove.inputformat(f_glass)
# f_glass=remove.filter(f_glass)
# f_glass
from weka.core.converters import Loader
from weka.classifiers import Classifier,Evaluation
from weka.core.classes import Random
loader=Loader("weka.core.converters.CSVLoader")
data=loader.load_file("../input/cs3481-ass2/feature_selection_df.csv")
data.class_is_last()
data,test=data.train_test_split(75,Random(1))
data.num_instances,test.num_instances
from weka.classifiers import Classifier,Evaluation
from weka.core.classes import Random
def return_classifier_and_estimate(cls_name,options,plot_roc=True):
    cvp=Classifier(classname='weka.classifiers.meta.CVParameterSelection')
    opt=['-W',cls_name,'-X','10']
    for par in options:
        opt.append('-P')
        opt.append(par)
    cvp.options=opt
    evl=Evaluation(data)
    evl.crossvalidate_model(cvp,data,10,Random(1))
    print(evl.percent_correct)
    print(evl.summary())
    print(evl.class_details()) # add detailed accuracy by class
    print(evl.matrix())
    if(plot_roc):
        import weka.plot.classifiers as plcls  # NB: matplotlib is required
        plcls.plot_roc(evl, class_index=[0, 1], wait=True)
    return cvp

def get_ytest_yprediction(test,classifier):
    evl=Evaluation(test)
    evl.test_model(classifier,test)
    ytest=[]
    ypred=[]
    for index, inst in enumerate(test):
        pred = classifier.classify_instance(inst)
        ypred.append(pred)
    return ytest,ypred
#EuclideanDistance
cvp=return_classifier_and_estimate('weka.classifiers.lazy.IBk',['K 1 10 10'])
cvp.build_classifier(data)
cvp
cvp=return_classifier_and_estimate("weka.classifiers.rules.JRip",["N 0.5 4 8","O 1 5 5"])
cvp.build_classifier(data)
cvp
cvp=return_classifier_and_estimate('weka.classifiers.rules.PART',['M 1 10 10','C 0.1 0.2 3'])
cvp.build_classifier(data)
cvp
cvp=return_classifier_and_estimate('weka.classifiers.trees.J48',['M 1 10 10','C 0.05 0.25 5'])
cvp.build_classifier(data)
cvp
# evl=Evaluation(data)
# evl.test_model(cvp,data)
# print(evl.percent_correct)

# J48=Classifier(classname='weka.classifiers.trees.J48')
# J48.options=['-M','5','-C','0.15']

# # J48.build_classifier(data)
# evl=Evaluation(data)
# evl.crossvalidate_model(J48,data,10,Random(1))
# # evl.test_model(J48,data)
# print(evl.percent_correct)
!pip install pydotplus
import pydotplus
from IPython.display import SVG

treeg = pydotplus.graph_from_dot_data(cvp.graph)
treeg.set_size('"10,10!"')

SVG(treeg.create_svg())
cvp=return_classifier_and_estimate('weka.classifiers.trees.REPTree',['M 1 10 10'])
cvp.build_classifier(data)
cvp
treeg = pydotplus.graph_from_dot_data(cvp.graph)
treeg.set_size('"10,10!"')

SVG(treeg.create_svg())
class_rf='weka.classifiers.trees.RandomForest'
cvp=Classifier(classname='weka.classifiers.meta.CVParameterSelection')
cvp.options=['-W',class_rf,'-X','10','-P','K 1 10 4','-P','M 1 10 4']

evl=Evaluation(data)
evl.crossvalidate_model(cvp,data,10,Random(1))
print(evl.percent_correct)
cvp.build_classifier(data)
cvp
RF=Classifier(classname=class_rf,options=['-K','1','-M','4','-print'])
evl=Evaluation(data)
evl.crossvalidate_model(RF,data,10,Random(1))
print(evl.percent_correct)
RF.is_drawable
data = pd.read_csv("../input/testdata/feature_selection_df.csv")
data.head()
type(data['diagnosis'])
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Load data
data = pd.read_csv("../input/testdata/feature_selection_df.csv")
y = data['diagnosis']
X = data.drop('diagnosis',axis=1)


# split data train 75 % and test 25 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Random Forest with Adaboosting
clf = RandomForestClassifier(n_estimators=50)
bclf = AdaBoostClassifier(base_estimator=clf, n_estimators=clf.n_estimators)
bclf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)
# ac = accuracy_score(y_test,y_pred)

# comment 10-fold cross-validation
scores = cross_val_score(bclf, X, y, cv=10)
print('Accuracy is: ',np.mean(scores))

cm = confusion_matrix(y_test,bclf.predict(X_test))
sns.heatmap(cm,annot=True,fmt="d")
plot_roc_curve(bclf, X_test, y_test)
plt.show()
# try GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
