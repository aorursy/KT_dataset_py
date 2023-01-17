# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')

%matplotlib inline
# Importing the dataset

train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
train_data.head()
test_data.head()
train_rows = train_data.shape[0]
combined_data = pd.concat([train_data,test_data], sort=False)
combined_data
combined_data['HasCabin'] = combined_data['Cabin'].apply(lambda x: 1 if x is not None else o)

combined_data = combined_data.drop(columns=['Name', 'Ticket','Cabin'])

combined_data
combined_data.info()
#combined_data.to_csv("output_pre_impute.csv", index=False)
combined_data['Fare'] = combined_data.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.mean()))
combined_data['Age'] = combined_data.groupby(['Pclass'])['Age'].transform(lambda y: y.fillna(y.mean())).round(2)
combined_data['Embarked'] = combined_data.groupby(['Pclass'])['Embarked'].transform(lambda z: z.fillna(z.mode()[0]))
#combined_data.to_csv("output_post_impute.csv", index=False)
#combined_data[combined_data["PassengerId"] == 62]
combined_data.info()
catagorical_variables = [index for index, value in combined_data.dtypes.items() if value == "object"]
catagorical_variables
combined_data = pd.get_dummies(combined_data, columns=catagorical_variables, prefix=catagorical_variables, drop_first=True)
combined_data
new_train = combined_data.iloc[:train_rows]

new_test = combined_data.iloc[train_rows:]
new_train.head()
new_test.head()
X = new_train.drop(columns=['PassengerId', 'Survived'])

y = new_train["Survived"].values

columns = list(X.columns)
test_ids = new_test.iloc[:,0].values

test_set = new_test.iloc[:,1:]

test_set = test_set.drop(columns=['Survived'])
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

test_set = sc.transform(test_set)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100, stratify=y)
from sklearn.utils import class_weight

classes = ['Died (0)','Survived (1)']

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(y_train),

                                                 y_train)

class_weights
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=1000,

                               class_weight={0:class_weights[0],1:class_weights[1]},

#                                class_weight='balanced',

#                                class_weight={0:1,1:2},

                               random_state=0)

rf_model.fit(X_train,y_train)
y_pred_rf = rf_model.predict(X_test)
from sklearn.metrics import confusion_matrix

def confusion_matrix_plot(confusion_matrix,classes):

    accuracy = f"Accuracy: {(confusion_matrix[0][0] + confusion_matrix[1][1])/sum(sum(confusion_matrix))*100:.2f}%"

    df_cm = pd.DataFrame(confusion_matrix, index = classes, columns = classes)

    plt.figure(figsize = (10,7))

    sns.set(font_scale=1.4)

    sns.heatmap(df_cm, annot=True, fmt='g')

    plt.title(f"Confusion Matrix, {accuracy}")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()
from yellowbrick.classifier import ClassificationReport

def classification_report_plot(model,classes,X_train,X_test,y_train,y_test):

    visualizer = ClassificationReport(model,

                                  classes=classes, 

                                  support=True)

    visualizer.fit(X_train, y_train)  # Fit the visualizer and the model

    visualizer.score(X_test, y_test)  # Evaluate the model on the test data

    visualizer.show()
cm_rf = confusion_matrix(y_test, y_pred_rf)

confusion_matrix_plot(cm_rf,classes)
from sklearn.metrics import classification_report

rf_classification_report = classification_report(y_test, y_pred_rf,target_names=classes,output_dict=True)
sns.heatmap(pd.DataFrame(rf_classification_report).iloc[:-1, :].T, annot=True, fmt=".3f").set_title("Random Forest Classification Report")
#classification_report_plot(rf_model,classes,X_train,X_test,y_train,y_test)
# # Importing the Keras libraries and packages

# import keras

# from keras.models import Sequential

# from keras.layers import Dense
# # Initialising the ANN

# ann_model = Sequential()



# # Adding the input layer and the first hidden layer

# ann_model.add(Dense(activation="relu", input_dim=X_train.shape[1], units=16, kernel_initializer="uniform"))



# # Adding the second hidden layer

# ann_model.add(Dense(activation="relu", units=16, kernel_initializer="uniform"))



# # Adding the third hidden layer

# ann_model.add(Dense(activation="relu", units=16, kernel_initializer="uniform"))



# # Adding the output layer

# ann_model.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))



# # Compiling the ANN

# ann_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# # Fitting the ANN to the Training set

# ann_model.fit(X_train, y_train, batch_size = 32, epochs = 300, class_weight=class_weights, verbose=1)
# y_pred_ann = ann_model.predict(X_test)

# y_pred_ann = (y_pred_ann > 0.5) #Percentage estimates are converted to True and False
# cm_ann = confusion_matrix(y_test, y_pred_ann)

# confusion_matrix_plot(cm_ann,classes)
# ann_classification_report = classification_report(y_test, y_pred_ann,target_names=classes,output_dict=True)
# sns.heatmap(pd.DataFrame(ann_classification_report).iloc[:-1, :].T, annot=True, fmt=".3f").set_title("ANN Classification Report")
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', 

                gamma=0.05, 

                C=6,

                class_weight='balanced', 

                random_state=0)

# svm_model = SVC(kernel='linear', random_state=0)

svm_model.fit(X_train,y_train)
y_pred_svm = svm_model.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)

confusion_matrix_plot(cm_svm,classes)
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train,y_train)
y_pred_nb = nb_model.predict(X_test)
cm_nb = confusion_matrix(y_test, y_pred_nb)

confusion_matrix_plot(cm_nb,classes)
weight = new_train['Survived'].value_counts()[0]/new_train['Survived'].value_counts()[1]
from xgboost import XGBClassifier

xgb_model = XGBClassifier(n_estimators=1000, 

                          learning_rate= 0.01, 

                          scale_pos_weight=weight,

                          gamma=2)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

confusion_matrix_plot(cm_xgb,classes)
rf_model.fit(X,y)
# ann_model.fit(X,y)
svm_model.fit(X,y)
nb_model.fit(X,y)
xgb_model.fit(X,y)
test_pred_rf = rf_model.predict(test_set).astype(int)
# test_pred_ann = ann_model.predict(np.array(test_set))

# test_pred_ann = np.concatenate(test_pred_ann)

# test_pred_ann = (test_pred_ann > 0.5).astype(int)
test_pred_svm = svm_model.predict(test_set).astype(int)
test_pred_nb = nb_model.predict(test_set).astype(int)
test_pred_xgb = xgb_model.predict(test_set).astype(int)
results_rf = pd.DataFrame(

    {'PassengerId': test_ids,

     'Survived': test_pred_rf

    })
# results_ann = pd.DataFrame(

#     {'PassengerId': test_ids,

#      'Survived': test_pred_ann

#     })
results_svm = pd.DataFrame(

    {'PassengerId': test_ids,

     'Survived': test_pred_svm

    })
results_nb = pd.DataFrame(

    {'PassengerId': test_ids,

     'Survived': test_pred_nb

    })
results_xgb = pd.DataFrame(

    {'PassengerId': test_ids,

     'Survived': test_pred_xgb

    })
results_rf.to_csv("rf_prediction.csv", index=False)
# results_ann.to_csv("ann_prediction.csv", index=False)
results_svm.to_csv("svm_prediction.csv", index=False)
results_nb.to_csv("nb_prediction.csv", index=False)
results_xgb.to_csv("xgb_prediction.csv", index=False)