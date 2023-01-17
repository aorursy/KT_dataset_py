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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

#from sklearn.cross_validation import StratifiedShuffleSplit

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split

#from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import PolynomialFeatures

import xgboost as xgb

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





# Load data

train_data=pd.read_csv('../input/gggggggg/train.csv')

submission_data=pd.read_csv('../input/gggggggg/sample_submission.csv')

test_data=pd.read_csv('../input/gggggggg/test.csv')



submission_data
train_data.drop(['id'],inplace=True,axis=1)

test_data.drop(['id'],inplace=True,axis=1)



all_features=pd.DataFrame(PolynomialFeatures(interaction_only=True).fit_transform(train_data.iloc[:,:-2]),

             columns=['Id','bone_length','rotting_flesh','hair_length','has_soul',

                      'bone_lengthXrotting_flesh','bone_lengthXhair_length','bone_lengthXhas_soul',

                      'rotting_fleshXhair_length','rotting_fleshXhas_soul','hair_lengthXhas_soul']).join(train_data['color'])
#Data preperation

all_features_id=all_features['Id']

all_features.drop(['Id'],axis=1, inplace=True)





#Deals with 'colour'

dummies=pd.get_dummies(all_features['color'],drop_first=False)

dummies=dummies.add_prefix('{}-'.format('color'))

all_features.drop(['color'],axis=1,inplace=True)

all_features=all_features.join(dummies)



all_features
all_features_test=pd.DataFrame(PolynomialFeatures(interaction_only=True).fit_transform(test_data.iloc[:,:-1]),

             columns=['Id','bone_length','rotting_flesh','hair_length','has_soul',

                      'bone_lengthXrotting_flesh','bone_lengthXhair_length','bone_lengthXhas_soul',

                      'rotting_fleshXhair_length','rotting_fleshXhas_soul','hair_lengthXhas_soul']).join(test_data['color'])



all_features_test



dummies=pd.get_dummies(all_features_test['color'],drop_first=False)

dummies=dummies.add_prefix('{}-'.format('color'))

all_features_test.drop(['color'],axis=1,inplace=True)

all_features_test=all_features_test.join(dummies)



all_features_test.drop(['Id'],axis=1,inplace=True)

all_features_test
le=LabelEncoder()

y_train=le.fit_transform(train_data.type.values)

clf=RandomForestClassifier(n_estimators=200)

clf=clf.fit(all_features,y_train)

indices=np.argsort(clf.feature_importances_)[::-1]



indices
print('Feature ranking')

for i in range(all_features.shape[1]):

    print('%d. feature %d %s (%f)' % (i+1,indices[i],all_features.columns[indices[i]],clf.feature_importances_[indices[i]]))



best_features=all_features.columns[indices[0:7]]



prediction=clf.predict(all_features_test)



predict_type=le.inverse_transform(prediction)



submission_data=pd.DataFrame({'Id':submission_data.id, 'type':predict_type})



submission_data



submission_data.to_csv('GoblinsGhoustGhouls',index=False)