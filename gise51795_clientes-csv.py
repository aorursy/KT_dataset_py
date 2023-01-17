import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
import sklearn as sk
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import seaborn as sb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
!find /kaggle
#Import the dataset into a Pandas DataFrame
data = pd.read_csv('/kaggle/input/clientes/clientes.csv', sep=';')
data
#Print the first 5 rows in order to have a high overview of the data
data[:5]
data.info()
#High overview of the dataset
data.describe()
# Analize the variable objetive y
data['y'].value_counts()

#Frequency table of variables education x y
data.groupby(by=['education'])['y'].value_counts().unstack()
#Fill unknow with most frequent
data.loc[data['education']=='unknown','education'] ='secondary'
#Frequency table of variables education x y now complete
data.groupby(by=['education'])['y'].value_counts().unstack()
#Frequency table of variables poutcome x y
data.groupby(by=['poutcome'])['y'].value_counts().unstack()
#Frequency table of variables poutcome x y
data.groupby(by=['poutcome'])['y'].value_counts().unstack()
#Frequency table of variables month x y
data.groupby(by=['contact'])['y'].value_counts().unstack()
#Frequency table of variables marital x y
data.groupby(by=['marital'])['y'].value_counts().unstack()
#Frequency table of variables default x y
data.groupby(by=['default'])['y'].value_counts().unstack()
#Frequency table of variable job x y
data.groupby(by=['job'])['y'].value_counts().unstack()
#We have a look at correlations on the dataset
corr = data.corr()
f,ax = plt.subplots(figsize=(9,7))
sb.heatmap(corr, annot=True, square=True, fmt='.1g')
plt.show()
cat_vars = ['job','marital', 'education', 'contact', 'housing',
             'loan', 'poutcome','day'] 

cont_vars = ['age','balance','campaign', 'pdays', 'duration']

# Normalizing
from sklearn.preprocessing import StandardScaler
scaler = preprocessing.StandardScaler()
data[cont_vars] = StandardScaler().fit_transform(data[cont_vars])
data[cont_vars] = (data[cont_vars]).astype('float64')
#Encoding categorical variables
housing = preprocessing.LabelEncoder()
data['housing'] = housing.fit_transform(data['housing'])
data['housing']  = data['housing'].astype('category')

loan = preprocessing.LabelEncoder()
data['loan'] = loan.fit_transform(data['loan'])
data['loan']  = data['loan'].astype('category')

y = preprocessing.LabelEncoder()
data['y'] = y.fit_transform(data['y'])

# Other encodings
label_encoder = LabelEncoder()
mappings = []

# Desired label orders for categorical columns.

educ_order = [ 'primary','secondary','tertiary']
#month_order = ['jan','feb','mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
marital_order =[ 'married','single', 'divorced']
poutcome_order =['failure', 'other', 'unknown', 'success']

# using cat.codes for order, one hot for high cardinality and weak case of cardinality.

def ordered_labels(data, col, order):
    data[col] = data[col].astype('category')
    data[col] = data[col].cat.reorder_categories(order, ordered=True)
    data[col] = data[col].cat.codes.astype(int)

# Use dummy variables for occupation
data = pd.concat([data, pd.get_dummies(data['job'])],axis=1).drop('job',axis=1)

# Use dummy variables for contact
data = pd.concat([data, pd.get_dummies(data['contact'])],axis=1).drop('contact',axis=1)

# Use ordered cat.codes for months, and education
ordered_labels(data, 'education', educ_order)
ordered_labels(data, 'marital', marital_order)
ordered_labels(data, 'poutcome', poutcome_order)


#Drop variables of no use
data = data.drop('previous', axis=1)
data = data.drop('month', axis=1)
data = data.drop('default', axis=1)
data.head()
#Split between source and target variables
X, target = data.drop('y', axis=1), data['y']
X[:3]
# Split in train and test the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
#Convert from serie to DataFrame
target = pd.DataFrame(target)
len(X_train)
y_train[:3]
#Apply Smote for imbalanced datasets
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
k=10
seed= 100

y_train_arr= np.array(y_train).reshape(-1,)
X_train_arr= np.array(X_train)
sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
X_res, y_res = sm.fit_resample(X_train_arr, y_train_arr.ravel())

random_forest = RandomForestClassifier(max_depth=5, random_state=42)
random_forest.fit(X_res, y_res)
#Confusion Matrix
y_pred = random_forest.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
fi = pd.DataFrame({'feature': list(X_train.columns),
                   'importance': random_forest.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi.head()
predictions_train_rf = random_forest.predict(X_train)
roc_auc_score(y_train, predictions_train_rf)
predictions_test_rf = random_forest.predict(X_test)
roc_auc_score(y_test, predictions_test_rf)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt

y_true = y_test
y_probas = random_forest.predict_proba(X_test)
skplt.metrics.plot_roc(y_true, y_probas)
plt.show()
#Búsqueda de hiperparámetros

param_grid = {'criterion': ['gini'],
              'max_depth': [None, 30, ],
              'min_samples_split':[2,4],
              'min_samples_leaf':[1,2,],
              'min_weight_fraction_leaf':[0.,0.01],
              'max_leaf_nodes':[None,50],
              'min_impurity_decrease':[0.,0.05] 
            }
trees= RandomForestClassifier()
clf = GridSearchCV(trees, param_grid, cv=4,verbose=1,n_jobs=-1)
clf.fit(X_res,y_res)
import pandas as pd
results=pd.DataFrame(clf.cv_results_)
results.head()
results["mean_test_score"].hist(bins=20)
clf.best_params_
clf.best_score_
from sklearn.ensemble import RandomForestClassifier
best_rf = RandomForestClassifier(criterion='gini',max_depth=30,max_features=1, max_leaf_nodes=None, min_impurity_decrease=0,
                                       min_samples_leaf =1, min_samples_split=2, min_weight_fraction_leaf=0.00,
                                       )
best_rf.fit(X_res, y_res)
score = best_rf.score(X_res, y_res)
score2 = best_rf.score(X_test, y_test)
print("Training set accuracy: ", '%.3f'%(score))
print("Test set accuracy: ", '%.3f'%(score2))
predictions_train_best_rf = best_rf.predict(X_train)
roc_auc_score(y_train, predictions_train_best_rf)
predictions_test_best_rf = best_rf.predict(X_test)
roc_auc_score(y_test, predictions_test_best_rf)
#Confusion Matrix
y_pred = best_rf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
fi = pd.DataFrame({'feature': list(X_train.columns),
                   'importance': best_rf.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi.head()
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
y
y_true = y_test
y_probas = best_rf.predict_proba(X_test)
skplt.metrics.plot_roc(y_true, y_probas)
plt.show()
# Cross Validation
skf=5
X_train1 = K.cast_to_floatx(X_train)
X_train1 =np.array(X_train)
y_train1 = K.cast_to_floatx(y_train)
X_test1 =np.array(X_test)
X_test1 = K.cast_to_floatx(X_test)
y_test1 = K.cast_to_floatx(y_test)
X1 = K.cast_to_floatx(X)
y1 = K.cast_to_floatx(target)
reduce_lr = ReduceLROnPlateau(factor=0.01,
                              patience=10, min_lr=0.005)
earlystopper = EarlyStopping(patience=8, verbose=1)

check = ModelCheckpoint(filepath='basic_model_best.hdf5',
                                           monitor='val_loss', 
                                         # validation AUC
                                           save_best_only=True,
                                           mode='max')
def logistic_regression():
    model = Sequential()
    model.add(Dense(1, input_shape=(26,), activation='sigmoid'))
    model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy',[tf.keras.metrics.AUC()]])

    return model
X = pd.DataFrame(X)
X[:3]
logistic_model = logistic_regression()
logistic_model.fit(X_train1, 
          y_train1,
          epochs=150, batch_size= 128, 
          shuffle = 'True',
          verbose=0, 
          validation_data=(X_test1, y_test1),
          callbacks = [reduce_lr, earlystopper]
         )
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
ax1.plot(logistic_model.history.history['loss'])
ax1.plot(logistic_model.history.history['val_loss'])
ax2.plot(logistic_model.history.history['accuracy'])
ax2.plot(logistic_model.history.history['accuracy'])
y_pred[:3]
predictions = logistic_model.predict(X1)
predictions_test = logistic_model.predict(X_test1)
predicted_test = logistic_model.predict_proba(X_test1)
roc_auc_score(y_test, predictions_test)
logistic_model.evaluate(X_test)
coefs_, intercept_ = logistic_model.get_weights()
# Ordenamos el valor absoluto de los parámetros de mayor a menor
# Coeficientes ordenados en función de su de mayor importancia
abs_coefs_ = np.sort(np.abs(coefs_), axis=0)
for i in zip(X_train.columns,coefs_):
    print(i)
#With Cross-validation
classifier = KerasClassifier(logistic_regression, batch_size= 128,  
                             epochs=150, verbose=0)
cross_val_scores = cross_val_score(estimator=classifier, X=X1, y=y1, cv=skf, 
                                   scoring='accuracy')
cross_val_scores.mean(), cross_val_scores.std()
predictions[:3]
# Logistic Regression Confusion Matrix
val_predicts = logistic_model.predict(X_test)
y_pred = [1 * (x[0]>=0.5) for x in val_predicts]
#print(val_predicts)
#print(y_pred)
print(cm)
print(classification_report(y_test,y_pred))
predictions_train_lr = logistic_model.predict(X_train)
roc_auc_score(y_train, predictions_train_lr)
predictions_test_lr = logistic_model.predict(X_test)
roc_auc_score(y_test, predictions_test_lr)
predictions_test_lr = cross_val_scores.predict(X_test)
roc_auc_score(y_test, predictions_test_lr)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt

y_true = y_test
probs = logistic_model.predict_proba(X_test)
skplt.metrics.plot_roc(y_true, y_probas)
plt.show()
