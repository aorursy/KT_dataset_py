import numpy as np
import pandas as pd
myData = pd.read_csv('../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv')

print(myData.shape)

import numpy as np
import pandas as pd
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
print(myData.shape)
peek = myData.head(10)
print(peek)

import numpy as np
import pandas as pd
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
types = myData.dtypes
print(types)

import numpy as np
import pandas as pd
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
outcome_counts = myData.groupby('Outcome').size()
print(outcome_counts)

import numpy as np
import pandas as pd
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
pd.set_option ('max_colwidth', 100)
pd.set_option ('precision', 3)
correlations = myData.corr(method='pearson')
print(correlations)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
myData.plot.hist()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
myData.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
correlations = myData.corr()
myfig = plt.figure()
axis = myfig.add_subplot(111)      
cax = axis.matshow(correlations, vmin=-1, vmax=1)
myfig.colorbar(cax)
ticks = np.arange(0,9,1)
axis.set_xticks(ticks)
axis.set_yticks(ticks)
axis.set_xticklabels(names)
axis.set_yticklabels(names)
plt.show()

import pandas as pd
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
data = myData.values
X = data[:,0:8]
Y = data[:,8]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
set_printoptions(precision=3)
print(rescaledX[0:5,:])

from sklearn.preprocessing import Normalizer
import pandas as pd
from numpy import set_printoptions
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
data = myData.values
X = data[:,0:8]
Y = data[:,8]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
set_printoptions(precision=3)
print(normalizedX[0:5,:])

from sklearn.preprocessing import Binarizer
import pandas as pd
from numpy import set_printoptions
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
data = myData.values
X = data[:,0:8]
Y = data[:,8]
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
set_printoptions(precision=3)
print(binaryX[0:5,:])

import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
data = myData.values
X = data[:,0:8]
Y = data[:,8]
myFeature = SelectKBest(score_func=chi2, k=4)  
fit = myFeature.fit(X,Y)
set_printoptions(precision=3)
print(fit.scores_)

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
data = myData.values
X = data[:,0:8]
Y = data[:,8]
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Number of Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

import pandas as pd
from sklearn.decomposition import PCA
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
data = myData.values
X = data[:,0:8]
Y = data[:,8]
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
pca=PCA(0.98) 
X_new=pca.fit_transform(X) 
print (X_new.shape)

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
data = myData.values
X = data[:,0:8]
Y = data[:,8]
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

import pandas as pd
from sklearn.model_selection import train_test_split
myDataname='../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myDataname,names=names)
data = myData.values
X = data[:,0:8]
Y = data[:,8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)