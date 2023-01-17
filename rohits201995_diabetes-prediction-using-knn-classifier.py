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
df=pd.read_csv('/kaggle/input/diabetes.csv')
df.head()

df['Outcome'].value_counts()
#Upsampling the data as the data set is imbalanced
from sklearn.utils import resample
df_majority=df[df.Outcome==0]
df_minority=df[df.Outcome==1]
df_minority_upsampled=resample(df_minority,replace=True,n_samples=500,random_state=100)
df_upsampled=pd.concat([df_majority,df_minority_upsampled])
df_upsampled['Outcome'].value_counts()
#Checking feature importance
import seaborn as sns
import matplotlib.pyplot as plt
X = df_upsampled.iloc[:,0:20]  #independent columns
y = df_upsampled.iloc[:,-1]    #target column i.e price range
#get correlations of each features in dataset
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(4).plot(kind='barh')
plt.show()
import seaborn as sns
sns.pairplot(df_upsampled,hue='Outcome')
sns.heatmap(df_upsampled.isnull(), cbar=False)
#Standardization
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X =  pd.DataFrame(SS.fit_transform(df_upsampled.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()
Y=df_upsampled.Outcome
Y.head(10)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=40)
#import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(10,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,Y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(11)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
from sklearn import metrics
y_pred=knn.predict(X_test)
print(metrics.confusion_matrix(y_test,y_pred))