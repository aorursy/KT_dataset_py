import warnings

warnings.filterwarnings('ignore')
!ls ../input/
import pandas as pd

ht_dt = pd.read_csv("../input/heart.csv", header = 'infer')
print("The heart dataset has {0} rows and {1} columns".format(ht_dt.shape[0], ht_dt.shape[1]))
ht_dt.head()  
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

                 'exang', 'oldpeak', 'slope', 'ca', 'thal']



target_name = 'target'



X = ht_dt[feature_names]

y = ht_dt[target_name]



print("Features of the dataset are {0}".format(X.columns.values))
from yellowbrick.features import Rank1D

# Instantiate the 1D visualizer with the Sharpiro ranking algorithm

visualizer = Rank1D(features=feature_names, algorithm='shapiro')



# Fit the data to the visualizer

visualizer.fit(X, y)  



# Transform the data

visualizer.transform(X) 



# visualise

visualizer.poof()                   
from yellowbrick.features import Rank2D

# covariance

visualizer = Rank2D(features=feature_names, algorithm='covariance') 

visualizer.fit(X, y)                

visualizer.transform(X)             

visualizer.poof()
#pearson

visualizer = Rank2D(features=feature_names, algorithm='pearson')

visualizer.fit(X, y)                

visualizer.transform(X)             

visualizer.poof()
#Feature set

feat_1 = ['age', 'trestbps', 'chol', 'thalach']    

feat_2 = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'oldpeak', 'slope', 'ca', 'thal']



from yellowbrick.features import RadViz

# Specify the features of interest and the classes of the target 

features = feat_1

classes = [0, 1]



# Instantiate the visualizer

visualizer = RadViz(classes=classes, features=features,size = (800,300))

visualizer.fit(X, y)      

visualizer.transform(X)  

visualizer.poof()
# Specify the features of interest and the classes of the target 

features = feat_2

classes = [0, 1]



# Instantiate the visualizer

visualizer = RadViz(classes=classes, features=features,size = (800,300))

visualizer.fit(X, y)      

visualizer.transform(X)  

visualizer.poof()
from yellowbrick.features import ParallelCoordinates

classes = [0, 1]

# Instantiate the visualizer for feat_1

visualizer = visualizer = ParallelCoordinates(

    classes=classes, features=feature_names,

    normalize='standard', size = (1200,500))



visualizer.fit(X, y)     

visualizer.transform(X)   

visualizer.poof()
# Classifier Evaluation Imports

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



#Yellowbrick

from yellowbrick.classifier import ClassificationReport,ConfusionMatrix



#Training & Test dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Instantiate the classification model and visualizer 

bayes = GaussianNB()

visualizer = ClassificationReport(bayes, classes=classes)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)  

g = visualizer.poof()
bayes = LogisticRegression()

visualizer = ClassificationReport(bayes, classes=classes)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)  

g = visualizer.poof()
logReg = LogisticRegression()

visualizer = ConfusionMatrix(logReg)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)

g = visualizer.poof()