#Import Packages

import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Load data (Notebook set up in Kaggle) 



train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")

test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
# Extract Features that have high correlations with the target

# https://www.kaggle.com/wwu651/auto-feature-engineering-lgb



features=[]

cor=[]

for feature in train.iloc[:,2:].columns:

    if (train['target'].corr(train[feature])>0.05)|(train['target'].corr(train[feature])<-0.05):

        features.append(feature)

        cor.append(train['target'].corr(train[feature]))



df_corr=pd.DataFrame({'Features': features,'Correlations':cor}).sort_values(by='Correlations').set_index('Features')



df_corr.plot(kind='barh',figsize=(10,8))
# #Feature engineering with featuretools package

# #not used since it did not increase AUC



# import featuretools as ft



# features.append('ID_code')



# es = ft.EntitySet(id="santander")



# es = es.entity_from_dataframe(entity_id="santander",

#                               dataframe=train[features],

#                                index="ID_code")



# feature_matrix, feature_defs = ft.dfs(entityset=es,

#                                        target_entity="santander",

# #                                       agg_primitives=["skew", "std"],

#                                       trans_primitives=['multiply_numeric','add_numeric'],

#                                        max_depth=1)



# es_test = es.entity_from_dataframe(dataframe=test[features],

#                                              entity_id='test',

#                                              index='ID_code')



# feature_matrix_test, feature_defs_test = ft.dfs(entityset=es_test, 

#                                                  target_entity='test',

#                                                  trans_primitives=['multiply_numeric','add_numeric'],

#                                                  max_depth=1)
# #Merge new features on datasets 



# feature_matrix = feature_matrix[feature_matrix.columns[~feature_matrix.columns.isin(train.columns)]].reset_index()

# train = pd.merge(train, feature_matrix, on='ID_code', how='left')



# feature_matrix_test = feature_matrix_test[feature_matrix_test.columns[~feature_matrix_test.columns.isin(test.columns)]].reset_index()

# test = pd.merge(test, feature_matrix_test, on='ID_code', how='left')



# #Clear up RAM

# del feature_matrix, feature_defs, feature_matrix_test, feature_defs_test
import matplotlib.pyplot as plt



for col in train.columns[~train.columns.isin(['target', 'ID_code'])]:

    train[col].hist(histtype=u'step')



plt.show()
from sklearn.preprocessing import StandardScaler, MinMaxScaler



# Standard scale all variables except ID_code and the target

# Create Train and Test sets 



#Splitting train set in train and validation (validation = 0.2)



from sklearn.model_selection import train_test_split



X = train[train.columns[~train.columns.isin(['target', 'ID_code'])]]

y = train[['target']]



scale = StandardScaler().fit(X)

X_scaled = pd.DataFrame(scale.transform(X))



X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, stratify = y)
import matplotlib.pyplot as plt



for col in X_scaled.columns:

    X_scaled[col].hist(histtype=u'step')



plt.show()
# # Upsample the minority class

# # SMOTE generates random noise in the added columns of the minority class



# from imblearn.over_sampling import SMOTE

# from imblearn.under_sampling import RandomUnderSampler

# import plotly.graph_objects as go



# labels = ["Non Subscriber - 0", "Subscriber - 1"]

# values = y_train.target.value_counts()



# print(X_train.shape, y_train.shape)

# print(y_train.target.value_counts())

# donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=["rgb(153, 214, 255)", "rgb(0, 92, 153)"])])

# donut.show()



# #Apply oversampling with random noise in upsampled observations

# over = SMOTE("auto")

# X_train, y_train = over.fit_sample(X_train, y_train)



# labels = ["Non Subscriber - 0", "Subscriber - 1"]

# values = y_train.target.value_counts()



# donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=["rgb(153, 214, 255)", "rgb(0, 92, 153)"])])

# donut.show()

# print(X_train.shape, y_train.shape)

# print(y_train.target.value_counts())
# Set up Gaussian Naive Bayes Classifier



from sklearn.metrics         import accuracy_score, auc, f1_score, precision_recall_curve, roc_auc_score, confusion_matrix

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.naive_bayes     import GaussianNB, BernoulliNB

from sklearn.linear_model    import LogisticRegression

from sklearn.tree            import DecisionTreeClassifier

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt



bayes        = GaussianNB()

logistic     = LogisticRegression()

randomForest = RandomForestClassifier()

gboost       = GradientBoostingClassifier()

bernoulli    = BernoulliNB()

tree         = DecisionTreeClassifier()



models = {

#           "logistic"     :logistic,

#           "logistic_1"   :logistic,

#           "ada"          :ada,

#           "knn"          :knn,

#           "lda"          : lda,

#           "lda_svd"      : lda,

            "bayes"        :bayes

#             "bernoulli": bernoulli

#           "randomForest" :randomForest

#           "xgboost_tree" :xgboost,

#           "xgboost_lin" :xgboost,

#           "xgboost_dart" :xgboost,

#           #"svc"          : svc,

#           "gboost"       :gboost,

#           "tree"         :tree

         }



# Iniatiate Grid Values to loop through in grid search

grid_values = {

    "logistic":{"solver":["lbfgs", "newton-cg", "sag", "saga"],'penalty': ['l2'],'C':[0.1, 1, 5, 50], 

                "max_iter":[300, 90, 100, 150], "class_weight":[None]},

    "logistic_1":{"solver":["liblinear"],'penalty': ['l1'],'C':[0.2,0, 1, 0.5,3], "max_iter":[300, 90, 100, 150], 

                  "class_weight":[None]},

    "ada":{"n_estimators":[5,6]},

    "knn":{"n_neighbors":[150, 200,100, 250,300,500], "weights":["uniform"], "leaf_size":[50,100,300], "metric":["manhattan", "euclidean", "minkowski"]},



    "randomForest":{"max_depth":[5], "n_estimators":[300], "class_weight":[None], 

                    "max_features":["auto"]},

    "lda":{"solver":["lsqr","eigen"], "shrinkage":[None, "auto"]},

    "bayes":{},

    "bernoulli":{},

    "cat_nb":{},

    "lda_svd":{"solver":["svd"]},

    "gboost":{"loss":["deviance", "exponential"], "n_estimators":[100,200,300], "learning_rate":[0.003, 0.001]},

    "xgboost_tree":{"booster":["gbtree"], "eta":[0.1,0.2], "max_depth":[2,3,4]},

    "xgboost_lin":{"booster":["gblinear"], "lambda":[0, 0.8,0.2,1], "feature_selector":["cyclic","shuffle"], 

                   "top_k":[8,10,15,20,7]},

    "xgboost_dart":{"booster":["dart"], "skip_drop":[0.05,0.03,0.01], "sample_type":["uniform", "weighted"], 

                    "rate_drop":[0.3,0.4,0.5], "normalize_type":["tree", "forest"]},

    "svc":{'C':[0.001,0.01,1,5,2], "kernel":["linear", "poly", "rbf", "sigmoid"], "degree":[2,3,4,5]},

    "tree": {"max_depth": [3, 5, 4], "max_features":[None, "auto"], "class_weight":[None]}

              }

# Run GridSearch and print best parameters, Confusion Matrix and AUC



#Initiate empty dataframe to store model results

overview = pd.DataFrame()



#loop through models in models dictionary

for model in models:

    

    search_grid = grid_values[model]

    

    #grid search parameters in grid_values

    #scoring is based on roc_auc -> outcome of grid_clf_acc is best model from grid search

    grid_clf_acc = GridSearchCV(models[model], param_grid = search_grid, scoring = 'roc_auc')

    grid_clf_acc.fit(X_scaled, y)



    #Predict values for validation set

    y_pred = grid_clf_acc.predict(X_valid)

    

    proba_train   = pd.DataFrame(grid_clf_acc.predict_proba(X_train))[1]

    auc_train     = roc_auc_score(np.array(y_train),np.array(proba_train))

    

    #compute evaluation metrics

    probabilities = pd.DataFrame(grid_clf_acc.predict_proba(X_valid))[1]

    auc           = roc_auc_score(np.array(y_valid),np.array(probabilities))

    acc           = accuracy_score(y_valid,y_pred)

    f1            = f1_score(y_valid,y_pred)

    prec_recall   = precision_recall_curve(y_valid,y_pred)

    

    print("\n" , model, "\t", "AUC:", auc, grid_clf_acc.best_params_)

    

    #add model metrics to result dataframe

    overview[model] = [auc, auc_train, acc, f1, prec_recall, grid_clf_acc.best_params_]

    

    #print confusion matrix

    cmtx = pd.DataFrame(

    confusion_matrix(y_valid, y_pred), 

    index=['true:no', 'true:yes'], 

    columns=['pred:no', 'pred:yes'])



    print(cmtx)

    

    #print histogram of predicted probabilities

    probabilities.hist()

    plt.show()



#set index for result dataframe

overview.index = ["AUC", "AUC_Train", "Accuracy", "F1 Score", "Precision_Recall", "best params"]
# Build Neural Network 



from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import optimizers

import tensorflow as tf



checkpoint = ModelCheckpoint(filepath = "weights.hdf5", verbose=1, save_best_only=True)

callback = EarlyStopping(monitor="val_auc", patience=15, verbose=0, mode='max')



def get_model(activation = "relu"):

    

    model = Sequential()

    model.add(Dense(512, activation='relu', input_dim = len(X_train.columns)))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Dense(512, activation='sigmoid'))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))

    model.add(Dense(1, activation = "sigmoid"))

    

    loss = 'binary_crossentropy'

    optimizer = tf.keras.optimizers.Adam(lr=0.00002)

#     optimizer= 'adam'

    metrics = [tf.keras.metrics.AUC()]

    

    

    model.compile(loss=loss, optimizer= optimizer, metrics=metrics)

    return model
#Run Neural Network and predict on validation set (optimized towards AUC)



from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics         import roc_auc_score

from sklearn.metrics         import confusion_matrix



#get new model from function

nn           = KerasClassifier(build_fn=get_model)



#fit model on scaled data (full) and use 20% as validation data

history = nn.fit(X_scaled, y, batch_size = 128, validation_split=.2, epochs = 50, callbacks = [checkpoint, callback]) 



#calculate AUC and cufusion matrix

proba = pd.DataFrame(nn.predict_proba(X_valid))[1]

y_pred = nn.predict(X_valid)



auc           = roc_auc_score(np.array(y_valid),np.array(proba))



print(auc)



cmtx = pd.DataFrame(

confusion_matrix(np.array(y_valid), np.array(y_pred)), 

index=['true:yes', 'true:no'], 

columns=['pred:yes', 'pred:no'])



print(cmtx, "\n", auc)



#Print training performance

import matplotlib.pyplot as plt

try:

    # Plot training & validation accuracy values

    plt.plot(history.history[list(history.history.keys())[1]])

    plt.plot(history.history[list(history.history.keys())[3]])

    plt.title('Model AUC')

    plt.ylabel('AUC')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='best')

    plt.show()



    # Plot training & validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='best')

    plt.show()

except:

    next
model = get_model()



model.load_weights('/kaggle/working/weights.hdf5')



model.fit(X_scaled, y, batch_size = 128, validation_split=.2, epochs = 10, callbacks = [checkpoint, callback]) 
#Create new model and load weights (lowest loss in training)

model = get_model()



model.load_weights('/kaggle/working/weights.hdf5')

model.evaluate(X_valid, y_valid)



proba = pd.DataFrame(model.predict_proba(X_valid))[0]



y_pred = model.predict(X_valid)



#Check AUC for model with loaded weights and full model on validation set from bayes classifier 

#Note: this step only compares the two AUC scores, the data is likely to be included in the train data and therefore not used for model selection / validation

auc           = roc_auc_score(np.array(y_valid),np.array(proba))

print('Loaded Weights:', auc)



proba = pd.DataFrame(nn.predict_proba(X_valid))[1]

auc           = roc_auc_score(np.array(y_valid),np.array(proba))

print('Full Model:', auc)

#Scale test data and predict test data

X_test = test[test.columns[~test.columns.isin(['ID_code'])]]



scale = StandardScaler().fit(X_test)

test_scaled = pd.DataFrame(scale.transform(X_test))



test_pred = pd.DataFrame(model.predict_proba(test_scaled))[0] #Neural Net AUC 0.87

test_pred1 = pd.DataFrame(grid_clf_acc.predict_proba(test_scaled))[1] #Bayes AUC 0.89



submission = test[['ID_code']]

submission['target'] = np.array(test_pred)

submission.to_csv("submission.csv", index=False)



submission1 = test[['ID_code']]

submission1['target'] = np.array(test_pred1)

submission1.to_csv("submission1.csv", index=False)
submission.target.hist()
model.summary()