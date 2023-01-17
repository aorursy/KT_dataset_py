import pandas as pd
data = pd.read_csv("../input/bank_note_data.csv")
data.head()
import seaborn as sns

%matplotlib inline
sns.countplot(x='Class',data=data)
sns.pairplot(data,hue='Class')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data.drop('Class',axis=1))
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])

df_feat.head()
X = df_feat
y = data['Class']
X = X.as_matrix()

y = y.as_matrix()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
import tensorflow as tf

import tensorflow.contrib.learn.python



from tensorflow.contrib.learn.python import learn as learn
#classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]

classifier = learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=10)
tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")
classifier.fit(X_train, y_train, steps=200, batch_size=20)
note_predictions = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
lst = list(note_predictions)

print(confusion_matrix(y_test,lst))
print(classification_report(y_test,lst))
# !pip install -q tf-nightly-2.0-preview

# # Load the TensorBoard notebook extension

# %load_ext tensorboard
# import tensorflow as tf

# import datetime, os



# logs_base_dir = "./logs"

# os.makedirs(logs_base_dir, exist_ok=True)

# %tensorboard --logdir {logs_base_dir}
# %load_ext tensorboard.notebook

# %tensorboard --logdir logs

# %reload_ext tensorboard.notebook
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)
print(classification_report(y_test,rfc_preds))
rfc_score_train = rfc.score(X_train, y_train)

print("Training score: ",rfc_score_train)

rfc_score_test = rfc.score(X_test, y_test)

print("Testing score: ",rfc_score_test)
print(confusion_matrix(y_test,rfc_preds))
from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()

logis.fit(X_train, y_train)

logis_score_train = logis.score(X_train, y_train)

print("Training score: ",logis_score_train)

logis_score_test = logis.score(X_test, y_test)

print("Testing score: ",logis_score_test)
#decision tree

from sklearn.ensemble import RandomForestClassifier

dt = RandomForestClassifier()

dt.fit(X_train, y_train)

dt_score_train = dt.score(X_train, y_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(X_test, y_test)

print("Testing score: ",dt_score_test)
#Model comparison

models = pd.DataFrame({

        'Model'          : ['Logistic Regression',  'Decision Tree', 'Random Forest'],

        'Training_Score' : [logis_score_train,  dt_score_train, rfc_score_train],

        'Testing_Score'  : [logis_score_test, dt_score_test, rfc_score_test]

    })

models.sort_values(by='Testing_Score', ascending=False)