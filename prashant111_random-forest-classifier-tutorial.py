# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # statistical data visualization

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import warnings



warnings.filterwarnings('ignore')
data = '/kaggle/input/car-evaluation-data-set/car_evaluation.csv'



df = pd.read_csv(data, header=None)
# view dimensions of dataset



df.shape
# preview the dataset



df.head()
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']





df.columns = col_names



col_names
# let's again preview the dataset



df.head()
df.info()
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']





for col in col_names:

    

    print(df[col].value_counts())   

df['class'].value_counts()
# check missing values in variables



df.isnull().sum()
X = df.drop(['class'], axis=1)



y = df['class']
# split data into training and testing sets



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# check the shape of X_train and X_test



X_train.shape, X_test.shape
# check data types in X_train



X_train.dtypes
X_train.head()
# import category encoders



import category_encoders as ce
# encode categorical variables with ordinal encoding



encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])





X_train = encoder.fit_transform(X_train)



X_test = encoder.transform(X_test)
X_train.head()
X_test.head()
# import Random Forest classifier



from sklearn.ensemble import RandomForestClassifier







# instantiate the classifier 



rfc = RandomForestClassifier(random_state=0)







# fit the model



rfc.fit(X_train, y_train)







# Predict the Test set results



y_pred = rfc.predict(X_test)







# Check accuracy score 



from sklearn.metrics import accuracy_score



print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# instantiate the classifier with n_estimators = 100



rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)







# fit the model to the training set



rfc_100.fit(X_train, y_train)







# Predict on the test set results



y_pred_100 = rfc_100.predict(X_test)







# Check accuracy score 



print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))
# create the classifier with n_estimators = 100



clf = RandomForestClassifier(n_estimators=100, random_state=0)







# fit the model to the training set



clf.fit(X_train, y_train)

# view the feature scores



feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)



feature_scores
# Creating a seaborn bar plot



sns.barplot(x=feature_scores, y=feature_scores.index)







# Add labels to the graph



plt.xlabel('Feature Importance Score')



plt.ylabel('Features')







# Add title to the graph



plt.title("Visualizing Important Features")







# Visualize the graph



plt.show()
# declare feature vector and target variable



X = df.drop(['class', 'doors'], axis=1)



y = df['class']
# split data into training and testing sets



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# encode categorical variables with ordinal encoding



encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])





X_train = encoder.fit_transform(X_train)



X_test = encoder.transform(X_test)
# instantiate the classifier with n_estimators = 100



clf = RandomForestClassifier(random_state=0)







# fit the model to the training set



clf.fit(X_train, y_train)





# Predict on the test set results



y_pred = clf.predict(X_test)







# Check accuracy score 



print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Print the Confusion Matrix and slice it into four pieces



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)



print('Confusion matrix\n\n', cm)



from sklearn.metrics import classification_report



print(classification_report(y_test, y_pred))