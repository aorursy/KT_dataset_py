import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from sklearn import datasets
from sklearn.model_selection import train_test_split 

iris = datasets.load_iris()
iris_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
iris_data.head()
X = iris.data 
y = iris.target
print(y)
print(iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
# linear : modelin evaluation değerlendirilmesi linear olarak seçiyoruz.
# kernel'de c değeri : regularization değeridir. 

svm_predictions = svm_model_linear.predict(X_test)
svm_predictions
accuracy = svm_model_linear.score(X_test, y_test)
accuracy