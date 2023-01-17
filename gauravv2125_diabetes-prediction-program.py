# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import the dataset
import pandas as pd
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.hist(figsize=(8,8))
df.dtypes
# define the features and labels
features = df.drop(['Outcome'] , axis=1)
labels = df[['Outcome']]

features['BMI'] = features['BMI'].astype('int64')
features['DiabetesPedigreeFunction'] = features['DiabetesPedigreeFunction'].astype('int64')
# mapping the labels 
lc = labels['Outcome'].map({0:'b' , 1:'r'})
print(lc)
import matplotlib.pyplot as plt
plt.scatter( features['DiabetesPedigreeFunction'] , features['Pregnancies'] ,  c=lc )
plt.show()
# To find total no. of patients and no. of diagnostic measures used
print(features.shape)
print(labels.shape)
# Dimensionality Reduction of Features
# Standardization
from sklearn.preprocessing import StandardScaler
f1 = StandardScaler()
# fitting the data
f1.fit(features)
f1.mean_
f1.scale_
# Transforming the features
ff = pd.DataFrame(f1.transform(features))
ff
#PCA
from sklearn.decomposition import PCA
pca = PCA(8)
pca.fit(ff)
f_pca = pca.transform(ff)
evr = pca.explained_variance_ratio_
evr
import matplotlib.pyplot as plt
plt.xlabel('Dimensions')
plt.ylabel('Eigen_values')
plt.plot(range(1,9),evr)
plt.scatter(range(1,9),evr)
plt.show()
# Training and testing the data
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(f_pca , labels , test_size = 0.3)
# Selecting the algorithim
from sklearn.neighbors import KNeighborsClassifier
kmodel = KNeighborsClassifier(n_neighbors=8)
kmodel.fit(xtr,ytr)
print(kmodel.score(xtr,ytr))
print(kmodel.score(xts,yts))
# Trying with another algorithim
from sklearn.tree import DecisionTreeClassifier
dmodel = DecisionTreeClassifier(min_samples_leaf=6 , random_state=2)
dmodel.fit(xtr,ytr)
print(dmodel.score(xtr,ytr))
print(dmodel.score(xts,yts))
fi = features.iloc[765].values.reshape(1,8)
fi
#Transform the data using the values stored in components during standardization
fi2 = f1.transform(fi)
#Reduce the dimension using the information stored in pca
pca.transform(fi2)
print(dmodel.predict(pca.transform(fi2)))
print(kmodel.predict(pca.transform(fi2)))
# Enter info of patient to predict the onset of diabetes based on diagnostic measures
print('Enter the details of patients diagnostic measures like\n No. of times pregnant\n Glucose levels\n BloodPressure\n SkinThickness(mm)\n Insulin\n BMI(weight in kg/(height in m)^2)\n DiabetesPedigreeFunction\n Age resp. with comma sep')
Details = input()
print(Details.split(','))
fii = pd.DataFrame(Details.split(','))
fii = fii.transpose() 
#Transform the data using the values stored in components during standardization
fii2 = f1.transform(fii)
fii2
#Reduce the dimension using the information stored in pca
pca.transform(fii2)
print(dmodel.predict(pca.transform(fii2)))
print(kmodel.predict(pca.transform(fii2)))
if(dmodel.predict(pca.transform(fii2))==1):
    print('The Patient is Predicted to be diabetic')
    
else:
     print('The Patient is Predicted to be non-diabetic')