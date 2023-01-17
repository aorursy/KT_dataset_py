# This is importatnt to reproduce the same results

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)



import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve



import seaborn as sns

import tensorflow as tf



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, BatchNormalization, Activation

from tensorflow.keras.optimizers import Adam
def get_features_to_keep(data, correlation_threshold = 0):

  # This function returns the list of features to keep.

  # This list contains any features whose correlation with Class field is strictly greater than correlation_threshold

  # If correlation_threshold is None, we will be keeping all features.

  if correlation_threshold is not None:

    list_of_features = corr.iloc[:, data.columns == 'Class']

    list_of_features = list_of_features[list_of_features['Class'] > correlation_threshold].index.tolist()

  else:

    list_of_features = []

  

  return list_of_features



def get_intersections(c1, c2):

    # This function checks the intersection between 2 curves c1 and c2

    # First it checks if there is an exact match of values.

    # If not, it will check where there was a change of sign for c1 - c2

    intersections = np.argwhere(c1 == c2).flatten()

    if len(intersections) == 0:

        intersections = np.argwhere(np.diff(np.sign(c1 - c2))).flatten()

    

    return intersections
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data_file = os.path.join(dirname, filename)



# Read data and print the dataframe head

data = pd.read_csv(data_file)

print("Data dimension: " + str(data.shape))

data.head()
# Finding any missing value treatment in the dataset.

data_na = data.isna().any()



num_na = (data_na == True).sum()

if num_na > 0:

    data_na_positions = np.where(data_na == True)

    print("Data unavailable for the following columns:")

    print(data_na_positions)

else:

    print("No missing data")
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data = data.drop(['Amount'],axis=1)

data = data.drop(['Time'],axis=1)



print("Amount column has been normalised under the name of NormalizedAmount")

print("Time column has been dropped")
# Correlation

data.corrwith(data.Class).plot.bar(

        figsize = (20, 10), title = "Correlation with class", fontsize = 15,

        rot = 45, grid = True)
# HeatMap



# Include all the columns that you do not want them to be included in the heatmap

columns_not_to_include = [] #["Amount", "Time"]



if len(columns_not_to_include) > 0:

    corr = data.iloc[:, np.all([data.columns != c for c in columns_not_to_include], axis=0)].corr()

else:

    corr = data.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Change the second arguments if you want to minimise the input data by keeping only features that have a correlation with Class that is greater than 0.2 for example

list_of_features = get_features_to_keep(data, correlation_threshold=None)



if len(list_of_features) > 0:

  reduced_data = data.iloc[:, np.any([data.columns == c for c in list_of_features], axis=0)]

  print("Input size reduced from %d to %d" % (data.shape[1] - 1, reduced_data.shape[1] - 1))

else:

  reduced_data = data

  print("Input size remains the same: %d" % (reduced_data.shape[1] - 1))





X = reduced_data.iloc[:, reduced_data.columns != 'Class']

y = reduced_data.iloc[:, reduced_data.columns == 'Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)



print("Train dataset has %d samples, %d input features and %d output features" % (X_train.shape[0], X_train.shape[1], y_train.shape[1]))

print("Test dataset has %d samples, %d input features and %d output features" % (X_test.shape[0], X_test.shape[1], y_test.shape[1]))
%%time

decision_tree = DecisionTreeClassifier(random_state=0,

                                       criterion='gini',

                                       max_depth=10,

                                       max_leaf_nodes=10)

decision_tree.fit(X_train, y_train)
# Predicting Test Set

y_pred      = decision_tree.predict(X_test)

y_pred_prob = decision_tree.predict_proba(X_test)



accuracy  = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall    = recall_score(y_test, y_pred)

f1        = f1_score(y_test, y_pred)



print("Decision Tree Predicion on Test Set:")

print("------------------------------------")

print("Accuracy:\t%.4f" % (accuracy))

print("Precision:\t%.4f" % (precision))

print("Recall:\t\t%.4f" % (recall))

print("F1 Score:\t%.4f" % (f1))



result_dict = {

    'Model': 'Decision Tree',

    'Accuracy': accuracy,

    'Precision': precision,

    'Recall': recall,

    'F1 Score': f1

}



try:

    results = results.append(result_dict, ignore_index=True)

except NameError:

    results = pd.DataFrame([['Decision Tree', accuracy, precision, recall, f1, 0]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC AUC'])
precisions, recalls, ths = precision_recall_curve(y_test, y_pred_prob[:,1])



# Get the middle threshold, the closest to 0.5 which is used in the classification

th_idx = np.abs(ths - 0.5).argmin()



#plt.figure(figsize=(8,5))



# Plotting the precision, recall curves

plt.plot(ths, precisions[:-1], "b--", label="Precision")

plt.plot(ths, recalls[:-1], "g--", label="Recall")



# Plotting a vertical line at threshold = 0.5 which is the one used to predict

plt.axvline(x=ths[th_idx], color='r', linestyle='--')



# Plotting the intersection of the precision and recall curves

#idx = np.argwhere(np.diff(np.sign(precisions - recalls))).flatten()

idx = get_intersections(precisions, recalls)

plt.plot(ths[idx], recalls[idx], 'ro')



for i in range(len(idx)):

    if i > 0:

        if ((idx[i] - idx[i-1]) / idx[i]) > 0.01:

            plt.annotate('thr = %.1f, prec, rec = %.4f' % (ths[idx[i]], recalls[idx[i]]),

                         xy=(ths[idx[i]], precisions[idx[i]]),

                         xytext=(-40, -50),

                         textcoords='offset points',

                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    else:

        plt.annotate('thr = %.1f, prec, rec = %.4f' % (ths[idx[i]], recalls[idx[i]]),

                     xy=(ths[idx[i]], precisions[idx[i]]),

                     xytext=(-40, -50),

                     textcoords='offset points',

                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))



plt.plot(ths[th_idx], precisions[th_idx], 'ro')

plt.annotate('precision = %.4f' % (precisions[th_idx]),

             xy=(ths[th_idx], precisions[th_idx]),

             xytext=(10, 20),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

plt.plot(ths[th_idx], recalls[th_idx], 'ro')

plt.annotate('recall = %.4f' % (recalls[th_idx]),

             xy=(ths[th_idx], recalls[th_idx]),

             xytext=(0, -30),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))



plt.xlabel('Threshold')

plt.title('Precision / Recall Curves')

plt.legend()



plt.show()
cm    = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



sns.heatmap(df_cm, annot=True, fmt='g')
dt_fpr, dt_tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

roc_auc = auc(dt_fpr, dt_tpr)



results.iloc[-1, results.columns.get_loc('ROC AUC')] = roc_auc



plt.title('Desicion Tree ROC Curve')



plt.plot(dt_fpr, dt_tpr, label='AUC = %0.4f'% roc_auc)

plt.legend(loc='lower right')



plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



plt.show()
%%time

random_forest = RandomForestClassifier(random_state=0,

                                       max_depth=12,

                                       n_estimators=250,

                                       max_leaf_nodes=100,

                                       n_jobs=-1)

random_forest.fit(X_train, np.ravel(y_train))
# Predicting Test Set

y_pred      = random_forest.predict(X_test)

y_pred_prob = random_forest.predict_proba(X_test)



accuracy  = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall    = recall_score(y_test, y_pred)

f1        = f1_score(y_test, y_pred)



print("Random Forest Predicion on Test Set:")

print("------------------------------------")

print("Accuracy:\t%.4f" % (accuracy))

print("Precision:\t%.4f" % (precision))

print("Recall:\t\t%.4f" % (recall))

print("F1 Score:\t%.4f" % (f1))



result_dict = {

    'Model': 'Random Forest',

    'Accuracy': accuracy,

    'Precision': precision,

    'Recall': recall,

    'F1 Score': f1

}



try:

    results = results.append(result_dict, ignore_index=True)

except NameError:

    results = pd.DataFrame([['Random Forest', accuracy, precision, recall, f1, 0]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC AUC'])
precisions, recalls, ths = precision_recall_curve(y_test, y_pred_prob[:,1])



# Get the middle threshold, the closest to 0.5 which is used in the classification

th_idx = np.abs(ths - 0.5).argmin()



# Plotting the precision, recall curves

plt.plot(ths, precisions[:-1], "b--", label="Precision")

plt.plot(ths, recalls[:-1], "g--", label="Recall")



# Plotting a vertical line at threshold = 0.5 which is the one used to predict

plt.axvline(x=ths[th_idx], color='r', linestyle='--')



# Plotting the intersection of the precision and recall curves

idx = get_intersections(precisions, recalls)

plt.plot(ths[idx], recalls[idx], 'ro')



for i in range(len(idx)):

    if i > 0:

        if ((idx[i] - idx[i-1]) / idx[i]) > 0.01:

            plt.annotate('thr = %.2f, prec, rec = %.4f' % (ths[idx[i]], recalls[idx[i]]),

                         xy=(ths[idx[i]], precisions[idx[i]]),

                         xytext=(0, -50),

                         textcoords='offset points',

                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    else:

        plt.annotate('thr = %.2f, prec, rec = %.4f' % (ths[idx[i]], recalls[idx[i]]),

                     xy=(ths[idx[i]], precisions[idx[i]]),

                     xytext=(-40, -50),

                     textcoords='offset points',

                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        





plt.plot(ths[th_idx], precisions[th_idx], 'ro')

plt.annotate('precision = %.4f' % (precisions[th_idx]),

             xy=(ths[th_idx], precisions[th_idx]),

             xytext=(30, -10),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

plt.plot(ths[th_idx], recalls[th_idx], 'ro')

plt.annotate('recall = %.4f' % (recalls[th_idx]),

             xy=(ths[th_idx], recalls[th_idx]),

             xytext=(30, 10),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))



# Plotting an even better threshold

th_idx = np.abs(ths - 0.39).argmin()

plt.axvline(x=ths[th_idx], color='orange', linestyle='--')



plt.plot(ths[th_idx], precisions[th_idx], 'ro')

plt.annotate('%.4f' % (precisions[th_idx]),

             xy=(ths[th_idx], precisions[th_idx]),

             xytext=(-70, 10),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

plt.plot(ths[th_idx], recalls[th_idx], 'ro')

plt.annotate('%.4f' % (recalls[th_idx]),

             xy=(ths[th_idx], recalls[th_idx]),

             xytext=(-50, -10),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))



plt.xlabel('Threshold')

plt.title('Precision / Recall Curves')

plt.legend()



plt.show()
cm    = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



sns.heatmap(df_cm, annot=True, fmt='g')
rf_fpr, rf_tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

roc_auc = auc(rf_fpr, rf_tpr)



results.iloc[-1, results.columns.get_loc('ROC AUC')] = roc_auc



plt.title('Random Forest ROC Curve')



plt.plot(rf_fpr, rf_tpr, label='AUC = %0.4f'% roc_auc)

plt.legend(loc='lower right')



plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



plt.show()
# Create model

nn_model = Sequential()



nn_model.add(Dense(16, input_dim = X_train.shape[1]))

nn_model.add(BatchNormalization())

nn_model.add(Activation('relu'))

nn_model.add(Dense(12))

nn_model.add(BatchNormalization())

nn_model.add(Activation('relu'))

nn_model.add(Dense(1, activation = 'sigmoid'))



nn_model.compile(optimizer=Adam(lr=1e-3) , loss='binary_crossentropy', metrics = [tf.keras.metrics.AUC()])



# Model Summary

nn_model.summary()
%%time



epochs=20

batch_size=512



h = nn_model.fit(X_train, y_train,

                 batch_size=batch_size,

                 epochs=epochs)
y_pred_prob = nn_model.predict(X_test)

y_pred      = (y_pred_prob > 0.5)



accuracy  = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall    = recall_score(y_test, y_pred)

f1        = f1_score(y_test, y_pred)



print("Neural Network Predicion on Test Set:")

print("-------------------------------------")

print("Accuracy:\t%.4f" % (accuracy))

print("Precision:\t%.4f" % (precision))

print("Recall:\t\t%.4f" % (recall))

print("F1 Score:\t%.4f" % (f1))



result_dict = {

    'Model': 'Keras NN',

    'Accuracy': accuracy,

    'Precision': precision,

    'Recall': recall,

    'F1 Score': f1

}



try:

    results = results.append(result_dict, ignore_index=True)

except NameError:

    results = pd.DataFrame([[' Neural Networks', accuracy, precision, recall, f1, 0]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
precisions, recalls, ths = precision_recall_curve(y_test, y_pred_prob)



# Get the middle threshold, the closest to 0.5 which is used in the classification

th_idx = np.abs(ths - 0.5).argmin()



# Plotting the precision, recall curves

plt.plot(ths, precisions[:-1], "b--", label="Precision")

plt.plot(ths, recalls[:-1], "g--", label="Recall")



# Plotting a vertical line at threshold = 0.5 which is the one used to predict

plt.axvline(x=ths[th_idx], color='r', linestyle='--')



# Plotting the intersection of the precision and recall curves

idx = get_intersections(precisions, recalls)

plt.plot(ths[idx], recalls[idx], 'ro')



for i in range(len(idx)):

    if i > 0:

        if ((idx[i] - idx[i-1]) / idx[i]) > 0.01:

            plt.annotate('thr = %1.f, prec, rec = %.4f' % (ths[idx[i]], recalls[idx[i]]),

                         xy=(ths[idx[i]], precisions[idx[i]]),

                         xytext=(0, -50),

                         textcoords='offset points',

                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    else:

        plt.annotate('thr = %.1f, prec, rec = %.4f' % (ths[idx[i]], recalls[idx[i]]),

                     xy=(ths[idx[i]], precisions[idx[i]]),

                     xytext=(-20, -50),

                     textcoords='offset points',

                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        





plt.plot(ths[th_idx], precisions[th_idx], 'ro')

plt.annotate('precision = %.4f' % (precisions[th_idx]),

             xy=(ths[th_idx], precisions[th_idx]),

             xytext=(15, 20),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

plt.plot(ths[th_idx], recalls[th_idx], 'ro')

plt.annotate('recall = %.4f' % (recalls[th_idx]),

             xy=(ths[th_idx], recalls[th_idx]),

             xytext=(30, 0),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))



# Plotting an even better threshold

th_idx = np.abs(ths - 0.42).argmin()

plt.axvline(x=ths[th_idx], color='orange', linestyle='--')



plt.plot(ths[th_idx], precisions[th_idx], 'ro')

plt.annotate('%.4f' % (precisions[th_idx]),

             xy=(ths[th_idx], precisions[th_idx]),

             xytext=(-70, 10),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

plt.plot(ths[th_idx], recalls[th_idx], 'ro')

plt.annotate('%.4f' % (recalls[th_idx]),

             xy=(ths[th_idx], recalls[th_idx]),

             xytext=(-50, -10),

             textcoords='offset points',

             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))





plt.xlabel('Threshold')

plt.title('Precision / Recall Curves')

plt.legend()



plt.show()
cm    = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



sns.heatmap(df_cm, annot=True, fmt='g')
nn_fpr, nn_tpr, thresholds = roc_curve(y_test, y_pred_prob)

roc_auc = auc(nn_fpr, nn_tpr)



results.iloc[-1, results.columns.get_loc('ROC AUC')] = roc_auc



plt.title('Keras Model ROC Curve')



plt.plot(nn_fpr, nn_tpr, label='AUC = %0.4f'% roc_auc)

plt.legend(loc='lower right')



plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



plt.show()
print(results)
plt.title('ROC Curve - Model Comparison')



plt.plot(dt_fpr, dt_tpr, label='Decision Tree')

plt.plot(rf_fpr, rf_tpr, label='Random Forest')

plt.plot(nn_fpr, nn_tpr, label='Keras')

plt.legend(loc='lower right')



plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



plt.show()