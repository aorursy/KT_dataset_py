# import libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
# data doesn't have headers, so let's create headers

_headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'car']
# read in cars dataset

df = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter06/Dataset/car.data', names=_headers, index_col=None)
df.head()
training, evaluation = train_test_split(df, test_size=0.3, random_state=0)
validation, test = train_test_split(evaluation, test_size=0.5, random_state=0)
# encode categorical variables

_df = pd.get_dummies(df, columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

_df.head()
# target column is 'car'



features = _df.drop(['car'], axis=1).values

labels = _df[['car']].values



# split 80% for training and 20% into an evaluation set

X_train, X_eval, y_train, y_eval = train_test_split(features, labels, test_size=0.3, random_state=0)



# further split the evaluation set into validation and test sets of 10% each

X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=0)
# train a Logistic Regression model

model = LogisticRegression()

model.fit(X_train, y_train)
# make predictions for the validation dataset

y_pred = model.predict(X_val)
#import libraries

from sklearn.metrics import confusion_matrix
confusion_matrix(y_val, y_pred)
#import libraries

from sklearn.metrics import precision_score
precision_score(y_val, y_pred, average='macro')
# import libraries

from sklearn.metrics import recall_score
recall_score = recall_score(y_val, y_pred, average='macro')

print(recall_score)
#import libraries

from sklearn.metrics import f1_score
f1_score = f1_score(y_val, y_pred, average='macro')

print(f1_score)
# import necessary library

from sklearn.metrics import accuracy_score
_accuracy = accuracy_score(y_val, y_pred)

print(_accuracy)
# import libraries

from sklearn.metrics import log_loss
_loss = log_loss(y_val, model.predict_proba(X_val))

print(_loss)