import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing libraries.................

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score,f1_score,auc,confusion_matrix

from sklearn.utils import resample

from sklearn.ensemble import RandomForestClassifier
#Checking the dataset.....................

df=pd.read_csv('../input/diabetes2.csv')
df.head()
#Checking missing values..............

df.isnull().sum()
#Visualizing the Geography

df['Outcome'].value_counts().plot(kind='bar')
df['Outcome'].value_counts()
# Separate input features (X) and target variable (y)

y = df.Outcome

X = df.drop('Outcome', axis=1)

 

# Train model

clf_2 = LogisticRegression().fit(X, y)

 

# Predict on training set

pred_y_2 = clf_2.predict(X)

 

# Is our model still predicting just one class?

print( np.unique( pred_y_2 ) )

# [0 1]

 

# How's our accuracy?

print( accuracy_score(y, pred_y_2) )

# Separate majority and minority classes

df_majority = df[df.Outcome==0]

df_minority = df[df.Outcome==1]

 

# Upsample minority class

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # sample with replacement

                                 n_samples=500,    # to match majority class

                                 random_state=123) # reproducible results

 

# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

 



#Visualizing the Geography

df_upsampled['Outcome'].value_counts().plot(kind='bar')

# Separate input features (X) and target variable (y)

y = df_upsampled.Outcome

X = df_upsampled.drop('Outcome', axis=1)

 

# Train model

clf_2 = LogisticRegression().fit(X, y)

 

# Predict on training set

y_pred = clf_2.predict(X)

 

# Is our model still predicting just one class?

print( np.unique( y_pred ) )



# How's our accuracy?

print( accuracy_score(y, y_pred) )

# Separate majority and minority classes

df_majority = df[df.Outcome==0]

df_minority = df[df.Outcome==1]

 

# Downsample majority class

df_majority_downsampled = resample(df_majority, 

                                 replace=False,    # sample without replacement

                                 n_samples=268,     # to match minority class

                                 random_state=123) # reproducible results

 

# Combine minority class with downsampled majority class

df_downsampled = pd.concat([df_majority_downsampled, df_minority])

 

# Display new class counts

df_downsampled['Outcome'].value_counts().plot(kind='bar')





# Separate input features (X) and target variable (y)

y = df_downsampled.Outcome

X = df_downsampled.drop('Outcome', axis=1)

 

# Train model

clf_2 = LogisticRegression().fit(X, y)

 

# Predict on training set

pred_y_2 = clf_2.predict(X)

 

# Is our model still predicting just one class?

print( np.unique( pred_y_2 ) )

# [0 1]

 

# How's our accuracy?

print( accuracy_score(y, pred_y_2) )

from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score



# Separate input features (X) and target variable (y)

y = df.Outcome

X = df.drop('Outcome', axis=1)

 

# Train model

clf_3 = SVC(kernel='linear', 

            class_weight='balanced', # penalize

            probability=True)

 

clf_3.fit(X, y)

 

# Predict on training set

pred_y_3 = clf_3.predict(X)

 

# Is our model still predicting just one class?

print( np.unique( pred_y_3 ) )

# [0 1]

 

# How's our accuracy?

print("Accuracy score" ,accuracy_score(y, pred_y_3) )



 

# What about AUROC?

prob_y_3 = clf_3.predict_proba(X)

prob_y_3 = [p[1] for p in prob_y_3]

print("ROC acurracy", roc_auc_score(y, prob_y_3) )
# Separate input features (X) and target variable (y)

y = df.Outcome

X = df.drop('Outcome', axis=1)

 

# Train model

clf_4 = RandomForestClassifier()

clf_4.fit(X, y)

 

# Predict on training set

pred_y_4 = clf_4.predict(X)

 

# Is our model still predicting just one class?

print( np.unique( pred_y_4 ) )

# [0 1]

 

# How's our accuracy?

print("Accuracy_score", accuracy_score(y, pred_y_4) )

 

# What about AUROC?

prob_y_4 = clf_4.predict_proba(X)

prob_y_4 = [p[1] for p in prob_y_4]

print("ROC_score",roc_auc_score(y, prob_y_4) )
