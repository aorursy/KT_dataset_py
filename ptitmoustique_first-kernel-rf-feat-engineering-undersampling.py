# All necessary import in order to process this kernel 



import numpy as np

import pandas as pd

from sklearn.feature_extraction import DictVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support

import matplotlib.pyplot as plt

import itertools
#all additional functions in order to threat with graphs and feature preprocessing





#####################################################################

# Function preprocessing_vectorized_cat_features

#  Input : Dataframe, a list of header column with categorical features

#  Output : a DataFrame with vectorized column

#####################################################################

def preprocessing_vectorized_cat_features(cat_features, df):

    categories = list([])

    for cat_feature in cat_features:

        l = sorted(list(df[cat_feature].unique()))

        i = 0

        while i < len(l):

            l[i] = cat_feature + '_' + l[i]

            i = i + 1

        categories.append(l)

    labels = list([])

    for categorie in categories:

        labels += categorie



    dv = DictVectorizer(sparse=False)



    vectorized_df = pd.DataFrame(df[cat_features]).convert_objects(convert_numeric=True)

    vectorized_df = dv.fit_transform(vectorized_df.to_dict(orient='records'))

    vectorized_df = pd.DataFrame(vectorized_df, columns=labels)

    return pd.DataFrame(vectorized_df)





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

 

def plot_pr_curve(model, X, y):

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 

              'green', 'blue','black']



    plt.figure(figsize=(5,5))



    j = 1

    for i,color in zip(thresholds,colors):

        y_pred_test = rf.predict_proba(X_test)[:, 1] > i



        precision, recall, thresholds = precision_recall_curve(y_test.as_matrix(),y_pred_test)

        print(precision_recall_fscore_support(y_test.as_matrix(),y_pred_test, average='macro'))

        # Plot Precision-Recall curve

        plt.plot(recall, precision, color=color,

                 label='Threshold: %s'%i)

        plt.xlabel('Recall')

        plt.ylabel('Precision')

        plt.ylim([0.0, 1.05])

        plt.xlim([0.0, 1.0])

        plt.title('Precision-Recall example')

        plt.legend(loc="lower left")

    

    plt.show()

    

def plot_roc_curve(model, X, y):

    y_pred_test = rf.predict_proba(X_test)[:, 1]



    false_positive_rateRF, true_positive_rateRF, thresholdsRF = roc_curve(y_test, y_pred_test)

    roc_aucRF = auc(false_positive_rateRF, true_positive_rateRF)

    print(roc_aucRF)



    plt.figure()

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rateRF, true_positive_rateRF, 'b', label='AUC = %0.2f' % roc_aucRF)

    plt.legend(loc='lower right')

    plt.plot([0, 1], [0, 1], 'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

    
df_data = pd.read_csv("../input/HR_comma_sep.csv")



# Separate categorical features and continious features in order to make different preprocessing

cat_features = list([])

cont_features = list([])

for column in df_data.columns:

        if isinstance(df_data[column][0], str):

            cat_features.append(column)

        else :

            cont_features.append(column)



df_data = pd.concat([df_data[cont_features], 

                     preprocessing_vectorized_cat_features(cat_features, df_data)], axis=1)

print(df_data.head())

#remove the output

df_data_outputs = df_data['left']

df_data = df_data.drop(['left'], axis=1)





X_train, X_test, y_train, y_test = train_test_split(df_data, df_data_outputs, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, max_features=None, random_state=24)



rf.fit(X_train, list(y_train.values))



#plot roc curve :

plot_roc_curve(rf, X_test, y_test)



#plot pr curve :

plot_pr_curve(rf, X_test, y_test)

df_data = pd.read_csv("../input/HR_comma_sep.csv")



# Separate categorical features and continious features in order to make different preprocessing

cat_features = list([])

cont_features = list([])

for column in df_data.columns:

        if isinstance(df_data[column][0], str):

            cat_features.append(column)

        else :

            cont_features.append(column)



df_data = pd.concat([df_data[cont_features], 

                     preprocessing_vectorized_cat_features(cat_features, df_data)], axis=1)



df_data_train, df_data_test = train_test_split(df_data, train_size=0.8, random_state=42)

# df_data_train, df_data_val = train_test_split(df_data_train, train_size=0.9)





# df_data_train = pd.DataFrame(data_matrix, columns=df_data_train.columns)

df_resample_one = df_data_train[df_data_train['left'] == 1]

df_resample_zero = df_data_train[df_data_train['left'] == 0]

df_resample_zero = df_resample_zero.sample(n=len(df_resample_one.index), random_state=64)

# print(df_resample_one)

# df_resample_not =df_data_train[df_data_train['SoldFlag'] == 0].sample(n=25000, replace=True).as_matrix()

df_resample = np.concatenate((df_resample_one.as_matrix(), df_resample_zero.as_matrix()), axis=0)

# print(df_resample_not)

df_resample = pd.DataFrame(df_resample, columns=df_data.columns)

df_data_train = df_resample



y_train = df_data_train['left']



X_train = df_data_train.drop(['left'], axis=1)



y_test = df_data_test['left']

X_test = df_data_test.drop(['left'], axis=1)



rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, max_features=None, random_state=42)



rf.fit(X_train, list(y_train.values))



#plot roc curve :

plot_roc_curve(rf, X_test, y_test)



#plot pr curve :

plot_pr_curve(rf, X_test, y_test)



df_data = pd.read_csv("../input/HR_comma_sep.csv")



df_data.loc[df_data['Work_accident'] == 0, 'Work_accident'] = 'no'

df_data.loc[df_data['Work_accident'] == 1, 'Work_accident'] = 'yes'



#allows to create some linear combination from the salary

df_data.loc[df_data['salary'] == 'low', 'salary'] = 1.0

df_data.loc[df_data['salary'] == 'medium', 'salary'] = 2.0

df_data.loc[df_data['salary'] == 'high', 'salary'] = 3.0



df_data['time_per_proj'] = df_data['average_montly_hours']/df_data['number_project']

df_data['value_for_industry'] = df_data['number_project'] * df_data['last_evaluation']

df_data['employee_value'] = df_data['value_for_industry'] / df_data['average_montly_hours']

df_data['global_value'] = df_data['employee_value'] * df_data['time_spend_company']

df_data['cost_per_year'] = df_data['salary'] * df_data['average_montly_hours']

df_data['global_cost'] = df_data['cost_per_year'] * df_data['time_spend_company']

df_data['global_ratio'] = df_data['global_value'] / df_data['global_cost']

df_data['cost_per_work_unit'] = df_data['value_for_industry'] / df_data['cost_per_year']

df_data['global_satis'] = df_data['satisfaction_level'] * df_data['time_spend_company']

df_data['satis_per_project'] = df_data['satisfaction_level'] / df_data['number_project']



# Separate categorical features and continious features in order to make different preprocessing

cat_features = list([])

cont_features = list([])

for column in df_data.columns:

        if isinstance(df_data[column][0], str):

            cat_features.append(column)

        else :

            cont_features.append(column)



df_data = pd.concat([df_data[cont_features], 

                     preprocessing_vectorized_cat_features(cat_features, df_data)], axis=1)



df_data_train, df_data_test = train_test_split(df_data, train_size=0.8, random_state=42)

# df_data_train, df_data_val = train_test_split(df_data_train, train_size=0.9)





# df_data_train = pd.DataFrame(data_matrix, columns=df_data_train.columns)

df_resample_one = df_data_train[df_data_train['left'] == 1]

df_resample_zero = df_data_train[df_data_train['left'] == 0]

df_resample_zero = df_resample_zero.sample(n=len(df_resample_one.index), random_state=64)

# print(df_resample_one)

# df_resample_not =df_data_train[df_data_train['SoldFlag'] == 0].sample(n=25000, replace=True).as_matrix()

df_resample = np.concatenate((df_resample_one.as_matrix(), df_resample_zero.as_matrix()), axis=0)

# print(df_resample_not)

df_resample = pd.DataFrame(df_resample, columns=df_data.columns)

df_data_train = df_resample



y_train = df_data_train['left']



X_train = df_data_train.drop(['left'], axis=1)



y_test = df_data_test['left']

X_test = df_data_test.drop(['left'], axis=1)



rf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, max_features=None, random_state=24)



rf.fit(X_train, list(y_train.values))



#plot roc curve :

plot_roc_curve(rf, X_test, y_test)



#plot pr curve :

plot_pr_curve(rf, X_test, y_test)
importances = rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_test.shape[1]):

    print("%d. feature %s (%f)" % (f + 1, df_data.columns[indices[f]], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X_test.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X_test.shape[1]), indices)

plt.xlim([-1, X_test.shape[1]])

plt.show()
