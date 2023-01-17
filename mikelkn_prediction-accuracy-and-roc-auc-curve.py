import numpy as np
import pandas as pd
#Loading the data
diabetes = pd.read_csv("../input/diabetes.csv")
diabetes.head()
diabetes.shape
diabetes.describe()
#we check to see that there are no nans or missing data. 
diabetes.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
diabetes.Outcome.value_counts(normalize = True)
sns.set(style = 'darkgrid')
ax = sns.countplot(x = 'Outcome', data = diabetes)
diabetes.hist(figsize = (12, 12));
diabetic_patients = diabetes[diabetes['Outcome']==1]
diabetic_patients.hist(figsize = (12, 12));
#We find the number of zeros per column in the dataset
count_zeros = (diabetes == 0).astype(int).sum(axis =0)
count_zeros
diabetes_no_pregnancy = diabetes.drop(['Pregnancies'], axis = 1)
diabetes_no_pregnancy.head()
impute_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i in impute_columns:
    diabetes.loc[(diabetes[i]==0, i)] = diabetes[i].mean()

#we will use the sklearn function imputer
# from sklearn.preprocessing import Imputer

# imputer = Imputer(missing_values = 0, strategy = "mean")

# diabetes = imputer.fit_transform(diabetes_no_pregnancy)
    
any_zeros = (diabetes == 0).astype(int).sum(axis =0)
any_zeros
#we take a look at our original data set
diabetes.describe()
#We split the data into train, validation and test set.
#We use the StratifiedShuffleSplit from Sklearn to maintain the same ratio of predictors classes

from sklearn.model_selection import StratifiedShuffleSplit

feature_cols = diabetes.columns[:-1]
# Get the split indexes
strat_shuf_split = StratifiedShuffleSplit(n_splits=1, 
                                          test_size=0.3, 
                                          random_state=42)

train_idx, test_idx = next(strat_shuf_split.split(diabetes[feature_cols], diabetes.Outcome))

# Create the dataframes
X_train_1 = diabetes.loc[train_idx, feature_cols]
y_train = diabetes.loc[train_idx, 'Outcome']

X_test_1  = diabetes.loc[test_idx, feature_cols]
y_test  = diabetes.loc[test_idx, 'Outcome']
#First lets standardize the data except the Outcome column
from sklearn.preprocessing import StandardScaler

X_train = StandardScaler().fit_transform(X_train_1)
X_test = StandardScaler().fit_transform(X_test_1)
X_train.shape
X_test.shape
#comparing the ratios of classes in both splits. Very representative of the whole data set
y_train.value_counts(normalize = True)
y_test.value_counts(normalize = True)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, roc_curve

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#We define our plot roc_auc_curve

def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} on PIMA diabetes problem'.format(model_name),
           xlabel = 'False positive Rate', ylabel = 'True positive rate',
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
#we define a plot_multiple_roc to visualise all the model curves together

def plot_multiple_roc(y_preds, y_test, model_names):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    
    for i in range (0, len(y_preds)):
        false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_preds[i])
        label = ""
        if len(model_names) > i:
            label = model_names[i]
        ax.plot(false_positive_rate, true_positive_rate, label=label)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)
    ax.grid(True)
    
    ax.set(title='ROC Curves for PIMA diabetes problem',
           xlabel = 'False positive Rate', ylabel = 'True positive rate')
        
    if len(model_names) > 0:
        plt.legend(loc=4)
# We do not have a enough data to creat a validation set so we will use k fold cross validation to pick the best model
cv_results = []

labels = ['log_reg', 'linear_svc', 'svc_rbf', 'Random_forest']

models = [LogisticRegression(), LinearSVC(), SVC(kernel = 'rbf'), 
            RandomForestClassifier(n_estimators = 200)]

kf = KFold(n_splits=10, random_state = 42)

for i in models:
    result_cv = cross_val_score(i, X_train, y_train, cv= kf, scoring = 'accuracy')
   
    cv_results.append(result_cv.mean()* 100)
        
cross_val_df = pd.Series(cv_results, index = labels).sort_values(ascending=False).to_frame()
cross_val_df.rename(columns = {0: 'cross validation means b4 feat_sels(%)'}, inplace = True)
print('\nModels with their corresponding cross validation means')
cross_val_df
validation_probs = []
models_names = ['log_reg', 'linear_svc', 'svc_rbf', 'Random_forest']

for i in [LogisticRegression(),RandomForestClassifier(n_estimators = 200)]:
    i.fit(X_train, y_train)
    validation_probabilities = i.predict_proba(X_test)
    validation_probs.append(validation_probabilities[:, 1])
for j in [LinearSVC(), SVC(kernel = 'rbf')]:
    j.fit(X_train, y_train)
    svc_probabilities = j.decision_function(X_test)
    validation_probs.append(svc_probabilities)

#preliminaries
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD
#Building a single layered neural network
nn_model = Sequential()
nn_model.add(Dense(15, input_shape = (8,),activation = 'relu'))
#nn_model.add(Dropout(0.1))
nn_model.add(Dense(1, activation ='sigmoid'))
nn_model.summary()
#Creat a training and validation set from the training data
from sklearn.model_selection import train_test_split
x_train_nn, x_val, y_train_nn, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
#fitting and compiling the model with SGD optimizer
nn_model.compile(SGD(lr = 0.005), 'binary_crossentropy', metrics=['accuracy'])

history = nn_model.fit(x_train_nn, y_train_nn, validation_data = (x_val, y_val), epochs = 900, verbose = 1)

nn_predict = nn_model.predict_classes(X_test)
nn_proba_predict = nn_model.predict(X_test)
#lets take a look
nn_predict[:10]
nn_proba_predict[:10]
accuracy_nn = accuracy_score(y_test,nn_predict)

print('The accuracy for the Neural network is {:.3f}'.format(accuracy_nn))
print('The roc-auc for the Neural network is {:.3f}'.format(roc_auc_score(y_test,nn_proba_predict)))
#lets take a look at the run history
history.history.keys()
#Plotting the training and validation losses over the different epochs

fig, ax = plt.subplots()

ax.plot(history.history['loss'], 'r', marker = '.', label = 'Train loss')

ax.plot(history.history['val_loss'], 'g', marker = '.', label = 'Validation loss')

ax.legend()

Accuracies = pd.DataFrame(cross_val_df )
Accuracies.rename(columns = {"cross validation means b4 feat_sels(%)" : "Models_Accuracies"}, inplace = True)
Accuracies.loc['Neural_nets'] = accuracy_nn*100
Accuracies

validation_probs.append(nn_proba_predict)
model_names_all = ['log_reg', 'linear_svc', 'svc_rbf', 'Random_forest', 'Neural nets']
plot_multiple_roc(validation_probs, y_test, model_names_all)
rf_model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
rf_proba_predict = rf_model.predict_proba(X_test)[:,1]
rf_auc = roc_auc_score(y_test, rf_proba_predict)

rbf_svc = SVC(kernel ='rbf').fit(X_train, y_train)
rbf_proba_predict = rbf_svc.decision_function(X_test)
rbf_auc = roc_auc_score(y_test, rbf_proba_predict)

lin_svc = LinearSVC().fit(X_train, y_train)
lsvc_proba_predict = lin_svc.decision_function(X_test)
lsvc_auc = roc_auc_score(y_test, lsvc_proba_predict)

log_reg = LogisticRegression().fit (X_train, y_train)
log_reg_proba_predict = log_reg.predict_proba(X_test)[:,1]
log_reg_auc = roc_auc_score(y_test, log_reg_proba_predict)

nn_auc = roc_auc_score(y_test, nn_proba_predict)

AUCROC_scores = [log_reg_auc*100, lsvc_auc*100, rbf_auc*100, rf_auc*100, nn_auc*100]
labels = ['log_reg', 'linear_svc', 'svc_rbf', 'Random_forest', 'Neural_nets']


RocAuc_score = pd.Series(AUCROC_scores, index = labels).sort_values(ascending=False).to_frame()
RocAuc_score.rename(columns = {0: 'Roc_auc_scores'}, inplace = True)
print('\nModels with their corresponding Roc_AUC_score')
RocAuc_score

diabetic_patients = diabetes[diabetes['Outcome']==1]
non_diabetics = list(diabetes.columns[:-1])
correlations = diabetes[non_diabetics].corrwith(diabetic_patients)
correlations.sort_values(inplace= True)

correlations
ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='pearson correlation');
#####No idea why i am having this kind of correllation#####
#Using the random forest classifier to get the most important features

X = diabetes[diabetes.columns[:-1]]
Y = diabetes['Outcome']

rf = RandomForestClassifier(n_estimators = 200).fit(X, Y)
most_important = pd.Series(rf.feature_importances_ *100, index=X.columns).sort_values(ascending =False).to_frame()
most_important.rename(columns = {0: 'percentage importance'}, inplace = True)
most_important
diabetes_2 = diabetes.drop(['Pregnancies', 'Insulin', 'SkinThickness'], axis = 1)

diabetes_2.head()
feature_cols_2 = diabetes_2.columns[:-1]
# Get the split indexes
strat_shuf_split = StratifiedShuffleSplit(n_splits=1, 
                                          test_size=0.3, 
                                          random_state=42)

train_idx_2, test_idx_2 = next(strat_shuf_split.split(diabetes_2[feature_cols_2], diabetes_2.Outcome))

# Create the dataframes
X_train_2 = diabetes_2.loc[train_idx_2, feature_cols_2]
y_train_3 = diabetes_2.loc[train_idx_2, 'Outcome']

X_test_2  = diabetes_2.loc[test_idx_2, feature_cols_2]
y_test_3  = diabetes_2.loc[test_idx_2, 'Outcome']
X_train_3 = StandardScaler().fit_transform(X_train_2)
X_test_3 = StandardScaler().fit_transform(X_test_2)

# We do not have a enough data to creat a validation set so we will use k fold cross validation to pick the best model
cv_results_2 = []

labels = ['log_reg', 'linear_svc', 'svc_rbf', 'Random_forest']

models = [LogisticRegression(), LinearSVC(), SVC(kernel = 'rbf'), 
            RandomForestClassifier(n_estimators = 200)]

kf_2 = KFold(n_splits=10, random_state = 42)

for i in models:
    result_cv_2 = cross_val_score(i, X_train_3, y_train_3, cv= kf_2, scoring = 'accuracy')
   
    cv_results_2.append(result_cv_2.mean()* 100)
        
feature_sel_cross_val = pd.Series(cv_results_2, index = labels).sort_values(ascending=False).to_frame()
feature_sel_cross_val.rename(columns = {0: 'cross_val with feature selection'}, inplace = True)
print('\nModels with their corresponding cross validation means after feature selection:')
feature_sel_cross_val
validation_probs_fs = []
models_names = ['log_reg_fs', 'linear_svc_fs', 'svc_rbf_fs', 'Random_forest_fs']

for i in [LogisticRegression(),RandomForestClassifier(n_estimators = 200)]:
    i.fit(X_train_3, y_train_3)
    validation_probabilities = i.predict_proba(X_test_3)
    validation_probs_fs.append(validation_probabilities[:, 1])
for j in [LinearSVC(), SVC(kernel = 'rbf')]:
    j.fit(X_train_3, y_train_3)
    svc_probabilities = j.decision_function(X_test_3)
    validation_probs_fs.append(svc_probabilities)

#Our input_shape has changed, was 8, now it is 5
#Building a single layered neural network
nn_model_2 = Sequential()
nn_model_2.add(Dense(15, input_shape = (5,),activation = 'relu'))
#nn_model.add(Dropout(0.1))
nn_model_2.add(Dense(1, activation ='sigmoid'))
#Creat a training and validation set from the training data
from sklearn.model_selection import train_test_split
x_train_nn3, x_val_3, y_train_nn3, y_val_3 = train_test_split(X_train_3, y_train_3, test_size = 0.2, random_state = 42)
#fitting and compiling the model with SGD optimizer
nn_model_2.compile(SGD(lr = 0.005), 'binary_crossentropy', metrics=['accuracy'])

history_2 = nn_model_2.fit(x_train_nn3, y_train_nn3, validation_data = (x_val_3, y_val_3), epochs = 900)
nn_2_predict_fs = nn_model_2.predict_classes(X_test_3)
nn_2_proba_predict_fs = nn_model_2.predict(X_test_3)
accuracy_nn_2 = accuracy_score(y_test_3,nn_2_predict_fs)

print('The accuracy for the Neural nets after feature selection is {:.3f}'.format(accuracy_nn_2))
print('The roc-auc for the Neural nets after feature selection is {:.3f}'.format(roc_auc_score(y_test_3,nn_2_proba_predict_fs)))
history_2.history.keys()
#Plotting the training and validation losses over the different epochs

fig, ax = plt.subplots()

ax.plot(history_2.history['loss'], 'r', marker = '.', label = 'Train_loss_2')

ax.plot(history_2.history['val_loss'], 'g', marker = '.', label = 'Validation_loss_2')

ax.legend()

Accuracies_fs = pd.DataFrame(feature_sel_cross_val )
Accuracies_fs.rename(columns = {"cross_val with feature selection" : "Models_Accuracies with feature_sel"}, inplace = True)
Accuracies_fs.loc['Neural_nets'] = accuracy_nn_2*100
Accuracies_fs
validation_probs_fs.append(nn_2_proba_predict_fs)
all_models_names = ['log_reg', 'linear_svc', 'svc_rbf', 'Random_forest', 'Neural_nets']
plot_multiple_roc(validation_probs_fs, y_test, all_models_names)
rf_model_fs = RandomForestClassifier(n_estimators=200).fit(X_train_3, y_train_3)
rf_proba_predict_fs = rf_model_fs.predict_proba(X_test_3)[:,1]
rf_auc_fs = roc_auc_score(y_test_3, rf_proba_predict_fs)

rbf_svc_fs = SVC(kernel ='rbf').fit(X_train_3, y_train_3)
rbf_proba_predict_fs = rbf_svc_fs.decision_function(X_test_3)
rbf_auc_fs = roc_auc_score(y_test_3, rbf_proba_predict_fs)

lin_svc_fs = LinearSVC().fit(X_train_3, y_train_3)
lsvc_proba_predict_fs = lin_svc_fs.decision_function(X_test_3)
lsvc_auc_fs = roc_auc_score(y_test_3, lsvc_proba_predict_fs)

log_reg_fs = LogisticRegression().fit (X_train_3, y_train_3)
log_reg_proba_predict_fs = log_reg_fs.predict_proba(X_test_3)[:,1]
log_reg_auc_fs = roc_auc_score(y_test_3, log_reg_proba_predict_fs)

nn_auc_fs = roc_auc_score(y_test_3, nn_2_proba_predict_fs)

AUCROC_scores_fs = [log_reg_auc_fs*100, lsvc_auc_fs*100, rbf_auc_fs*100, rf_auc_fs*100, nn_auc_fs*100]
labels = ['log_reg', 'linear_svc', 'svc_rbf', 'Random_forest', 'Neural_nets']


RocAuc_score_fs = pd.Series(AUCROC_scores_fs, index = labels).sort_values(ascending=False).to_frame()
RocAuc_score_fs.rename(columns = {0: 'Roc_auc_scores with feat selection'}, inplace = True)
print('\nModels with their corresponding Roc_AUC_score')
RocAuc_score_fs
#Creating a submission file

Pima_diabetes_submission = pd.DataFrame( {'Outcome': rbf_proba_predict_fs})
Pima_diabetes_submission.to_csv('Pima_Diabetes_Prediction.csv', index = False)
