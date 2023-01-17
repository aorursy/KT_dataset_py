#import dependencies



import numpy as np 

import matplotlib.pyplot as plt

import pandas as pd

import pickle



from keras.models import Sequential

from keras.layers import Dense

from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn.metrics import classification_report, confusion_matrix





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def reb(dfName):

  ufc = dfName

  

  # Separating the independent variables from dependent variables

  # selects all the rows and all the columns except for the last column

  x=ufc.drop(axis=1, labels=["Winner"])

  # selects all the rows and the last column

  y=ufc.loc[:, "Winner"]

  x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.20)





  rfc = RandomForestClassifier(random_state=101)

  rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')

  rfecv.fit(x_train, y_train)



  print('Optimal number of features: {}'.format(rfecv.n_features_))



  plt.figure(figsize=(16, 9))

  plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)

  plt.xlabel('Number of features selected', fontsize=14, labelpad=20)

  plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)

  plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)



  plt.show()



  print(np.where(rfecv.support_ == False)[0])



  x_train.drop(x_train.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)



  len(rfecv.estimator_.feature_importances_)



  len(x_train.columns)



  dset = pd.DataFrame()

  dset['attr'] = x_train.columns

  dset['importance'] = rfecv.estimator_.feature_importances_



  dset = dset.sort_values(by='importance', ascending=False)





  plt.figure(figsize=(16, 14))

  plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')

  plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)

  plt.xlabel('Importance', fontsize=14, labelpad=20)

  plt.show()



  ###@@@@@@@@@ THIS IS TO ***SAVE*** THE TRAINED MODEL FROM ABOVE @@@@@@@@@###

  pickle.dump(rfecv, open("normalised_features_never_drop_aggregate_df.pkl", "wb"))



  rfecv.estimator_.feature_importances_



  dset = pd.DataFrame()

  dset['attr'] = x_train.columns

  dset['importance'] = rfecv.estimator_.feature_importances_



  dset = dset.sort_values(by='importance', ascending=False)





  plt.figure(figsize=(16, 14))

  plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')

  plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)

  plt.xlabel('Importance', fontsize=14, labelpad=20)

  plt.show()



  new_x = x_train

  new_x["Winner"] = y_train



  new_x.shape

    

  return new_x
clean_data_df = pd.read_csv("../input/cleaned-data-no-draws/cleaned_data.csv")
clean_data_df.head()
clean_data_df.shape
cleaned_data_feature_eliminated = reb(clean_data_df)
cleaned_data_feature_eliminated.head()
cleaned_data_feature_eliminated.shape
cleaned_data_feature_eliminated.to_csv('cleaned_data_feature_eliminated.csv', index = False)
###@@@@@@@@@ THIS IS TO ***OPEN*** THE TRAINED MODEL FROM FROM THE STORED PICKLE FILE @@@@@@@@@###

### CHECKPOINT - can just load the pickle file and start running your analysis

#rfecv = pickle.load(open("normalised_features_never_drop_aggregate_df.pkl", "rb"))