# Imports

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.impute import KNNImputer

from sklearn import preprocessing

import traceback

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from imblearn.over_sampling import SMOTE 

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.preprocessing import PolynomialFeatures





# Data Loading

dataset = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

dataset.columns = [x.strip() for x in dataset.columns]
plot_df = dataset

plot_df['Color'] = plot_df['SARS-Cov-2 exam result'].apply(lambda x: 2 if x == 'positive' else 1)

feat_to_plot = [

    'Hematocrit',

    'Hemoglobin',

    'Platelets',

    'Mean platelet volume', 

    'Red blood Cells',

    'Lymphocytes',

    'Mean corpuscular hemoglobin concentration (MCHC)',

    'Leukocytes',

    'Basophils',

    'Mean corpuscular hemoglobin (MCH)',

    'Eosinophils',

    'Mean corpuscular volume (MCV)',

    'Monocytes',

    'Red blood cell distribution width (RDW)']

#for i in range(len(feat_to_plot)):

#    for j in range(i+1, len(feat_to_plot)):

#        fig, ax = plt.subplots()

#        plot_df.plot.scatter(x=feat_to_plot[j], y=feat_to_plot[i], c='Color', colormap='coolwarm', ax=ax)

#        plt.axhline(y=np.mean(plot_df[plot_df['Color']==2][feat_to_plot[i]]), color='r', linewidth=1)

#        plt.axhline(y=np.mean(plot_df[plot_df['Color']==1][feat_to_plot[i]]), color='b', linewidth=1)

#        plt.axvline(x=np.mean(plot_df[plot_df['Color']==2][feat_to_plot[j]]), color='r', linewidth=1)

#        plt.axvline(x=np.mean(plot_df[plot_df['Color']==1][feat_to_plot[j]]), color='b', linewidth=1)



# Reducing the number of plots

combs = [

    ['Eosinophils','Leukocytes'],

    ['Platelets','Leukocytes'],

    ['Eosinophils','Platelets'],

]

for c in combs:

    x, y = c

    fig, ax = plt.subplots()

    plot_df.plot.scatter(x=x, y=y, c='Color', colormap='coolwarm', ax=ax)

    plt.axhline(y=np.mean(plot_df[plot_df['Color']==2][x]), color='r', linewidth=1)

    plt.axhline(y=np.mean(plot_df[plot_df['Color']==1][x]), color='b', linewidth=1)

    plt.axvline(x=np.mean(plot_df[plot_df['Color']==2][y]), color='r', linewidth=1)

    plt.axvline(x=np.mean(plot_df[plot_df['Color']==1][y]), color='b', linewidth=1)
# Fixing pH column

dataset['Urine - pH'] = dataset['Urine - pH'].apply(lambda x: x if type(x) == float else np.nan)



# Data Frame with results

results = pd.DataFrame({

    'Group':[],

    'TotalAcc':[],

    'Acc.Positive':[],

    'Acc.Negative':[],

    'Acc.Weighted':[],

    'Train.Rows':[],

    'Test.Rows':[],

    'Method':[],

    'Features':[]

})



coefficients = pd.DataFrame({

    'Experiment': [],

    'Group': [],

    'Attribute': [],

    'Coefficient': [],

    'Coefficient (abs)': []

})



experiment_number = 1



# Tests seem to be done in batch (several tests are done together), so here I'm analyzing with respect to a given group

groups = ['Parainfluenza 1','Platelets','Influenza B, rapid test','Influenza B']

for g in groups:

    # Filtering per group

    pre_prep = dataset[~dataset[g].isnull()]

    

    # Dropping missing data

    pre_prep = pre_prep.dropna(axis=1, thresh=len(pre_prep)*0.7) # Require at least 50% data in each column

    

    # Generating dummies for categorical variables

    obj_only = pre_prep.select_dtypes(include=['object']).set_index('Patient ID')

    num_only = pre_prep.select_dtypes(exclude=['object'])

    num_only.loc[:,'Patient ID'] = dataset['Patient ID']

    num_only = num_only.set_index('Patient ID')

    prepared = pd.get_dummies(obj_only, columns=set(obj_only)-{'SARS-Cov-2 exam result','Patient ID'}, dummy_na=False).join(num_only)



    prepared = prepared.drop(columns=['Color']).reset_index(drop=True)

    

    # Encoding Labels

    le = preprocessing.LabelEncoder()

    prepared.loc[:,'Label'] = le.fit_transform(prepared['SARS-Cov-2 exam result'])



    # Dropping label column for ease of work afterwards

    prepared = prepared.drop(columns=['SARS-Cov-2 exam result']).reset_index(drop=True)

    labels = prepared['Label'].copy().reset_index(drop=True)

    prepared = prepared.drop(columns=['Label']).reset_index(drop=True)



    # Imputing missing data

    my_imputer = KNNImputer(n_neighbors=2, weights="uniform")

    prepared = pd.DataFrame(my_imputer.fit_transform(prepared), columns=prepared.columns)

    prepared



    features = prepared.columns



    # Model Search

    best_model = None



    try:



        X = prepared

        y = labels

        

        # Running each experiment 3 times

        for k in range(3):

            # Train-test split

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)



            # Oversampling minority class

            sm = SMOTE(sampling_strategy='all')

            X_train, y_train = sm.fit_resample(X_train, y_train)



            # Scaling data

            scaler = preprocessing.StandardScaler().fit(X_train)

            X_train = scaler.transform(X_train)

            X_test = scaler.transform(X_test)



            # Getting indices of each class

            idx_positive = y_test == (le.transform(['positive'])[0])

            idx_negative = y_test == (le.transform(['negative'])[0])



            # Decision tree should be the simplest method to evaluate

            for method in [tree.DecisionTreeClassifier(class_weight='balanced'), 

                           LogisticRegression(class_weight='balanced', max_iter=500), 

                           RandomForestClassifier(class_weight='balanced'),

                           svm.SVC(class_weight='balanced', kernel='linear',C=0.5),

                           KNeighborsClassifier(n_neighbors=3)]:

                model = method.fit(X_train, y_train)

                

                # Calculating accuracy in each class

                acc_pos = accuracy_score(y_test[idx_positive], model.predict(X_test[idx_positive]))

                acc_neg = accuracy_score(y_test[idx_negative], model.predict(X_test[idx_negative]))

                total_acc = accuracy_score(y_test, model.predict(X_test))

                w_acc = (acc_pos+acc_neg)/2



                if method.__class__.__name__ == 'LogisticRegression':

                    best_model = model

                    coefficients = coefficients.append(pd.DataFrame({

                        'Experiment': experiment_number,

                        'Group': g,

                        'Attribute': X.columns,

                        'Coefficient': model.coef_[0],

                        'Coefficient (abs)': abs(model.coef_[0])

                    }), ignore_index=True)

                    experiment_number += 1



                # Calculating accuracy on each class

                results = results.append({

                    'Group': g,

                    'TotalAcc': total_acc,

                    'Acc.Positive': acc_pos,

                    'Acc.Negative': acc_neg,

                    'Acc.Weighted': w_acc,

                    'Train.Rows': len(y_train),

                    'Test.Rows': len(y_test),

                    'Method': method.__class__.__name__,

                    'Features': " | ".join(X.columns)

                }, ignore_index=True)

                

                



    except Exception as e:

        traceback.print_exc()



results.groupby(['Group','Method','Features']).mean().reset_index().sort_values(by='Acc.Weighted', ascending=False)
coefficients.sort_values(by=['Group', 'Experiment', 'Coefficient (abs)'], ascending=False)[0:50]