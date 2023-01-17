%matplotlib inline

# File loading

import pandas as pd

import numpy as np

from sklearn import preprocessing



data = pd.read_csv('../input/mushrooms.csv');



columns = data.columns





#encoder = preprocessing.LabelEncoder()

#encoder.fit(np.unique(data))

encoder_labels = []

histo = [];

for ind in np.arange(0, len(columns)):

    encoder = preprocessing.LabelEncoder()

    data[columns[ind]] = encoder.fit_transform(data[columns[ind]])

    #data[columns[ind]] = encoder.transform(data[columns[ind]])

    

    # We save the histograms, just in case.

    dummy_histo, dummy = np.histogram(data[columns[ind]], bins=len(np.unique(data[columns[ind]])))

   

    

    # If we find a feature with a single label we just remove it.

    if len(dummy_histo) == 1:

        del data[columns[ind]]

        print('Removing feature:'+repr(columns[ind]))

    else:

        histo.append(dummy_histo);

        encoder_labels.append(encoder)

            

columns = data.columns
# We can check that both classes (edible and poissonous) have a similar number of samples.

print('N=0: ', histo[0][0])

print('N=1: ', histo[0][1])
from sklearn.cross_validation import train_test_split

labels = columns[columns != 'class']



X_train, X_test, y_train, y_test = train_test_split(data[labels], data['class'], test_size=0.20, random_state=42);



from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression



parameters = { 

    'penalty':['l1','l2'],

    'C':[.001,.01,.1, 1, 10, 100, 1000],              

    }

machine = LogisticRegression(random_state = 44, n_jobs = -1, class_weight = 'balanced')



clf = GridSearchCV(machine, parameters, n_jobs = -1, scoring = 'f1', cv = 5)  # scoring='roc_auc'

clf.fit(X_train, y_train)

clf.grid_scores_
model_log_reg = clf.best_estimator_ 
from sklearn.ensemble import RandomForestClassifier



parameters = { 

    'n_estimators':[10,20],

    'max_features':[.3, 1],

    'max_depth':[1, 3, 5, 10, None],

    'min_samples_leaf':[100, 10, 1]

    }

machine = RandomForestClassifier(random_state = 44, n_jobs = -1, class_weight = 'balanced')



clf = GridSearchCV(machine, parameters, n_jobs = -1, scoring = 'f1', cv = 5)  # scoring='roc_auc'

clf.fit(X_train, y_train);
clf.grid_scores_
model_RF = clf.best_estimator_;
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def compute_scores(y_test, y_pred):

    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_pred)

    

    print('Accuracy = '+repr(acc))

    print('F-score =  '+repr(f1))

    print('AUC = '+repr(auc))

    

    return acc, f1, auc



print('\n********\nLOGISTIC REGRESSION')

acc_log, f1_log, auc_log = compute_scores(y_test, model_log_reg.predict(X_test));

print('\n********\nRANDOM FOREST')

acc_RF, f1_RF, auc_RF = compute_scores(y_test, model_RF.predict(X_test));
import matplotlib.pyplot as plt

plt.stem(model_RF.feature_importances_)

plt.ylabel('Feature weight')

plt.xlabel('Feature index')
top_idx = [ 4,7,18,19,11];
dummy_top = [0, 5,8,19,20,12];

data[columns[dummy_top]].corr()
def plot_histogram(data, feature_index):

    feature = labels[feature_index]



    import matplotlib.pyplot as plt

    import numpy as np

    from matplotlib.ticker import FormatStrFormatter

    

    # EDIBLE MUSHROOMS

    samples = data[data['class'] == 0];  # Edible mushrooms

    samples = samples[labels[top_idx]];

    #plt.hist(samples['odor'],label='aaa')





    #fig, ax = plt.subplots()

    fig = plt.figure()

    ax = plt.subplot(111)

    counts, bins, patches = ax.hist(samples[feature], facecolor='blue', edgecolor='gray',

                                    bins = np.arange(0,np.max(data[feature])+2),

                                    range=(0,np.max(data[feature])))

    print('Counts E: ',counts);

    

    ax.set_xticks(bins)

    # Label the raw counts and the percentages below the x-axis...

    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for count, x in zip(encoder_labels[feature_index+1].classes_, bin_centers):

        # Label the raw counts

        ax.annotate(count, xy=(x, 0), xycoords=('data', 'axes fraction'),

            xytext=(0, -18), textcoords='offset points', va='top', ha='center')



    plt.title('Histogram comparison. Feature: "'+feature+'".')

    

    # POISSONOUS MUSHROOMS

    samples = data[data['class'] == 1];  # Edible mushrooms

    samples = samples[labels[top_idx]];



    

    counts, bins, patches = ax.hist(samples[feature], facecolor='red', edgecolor='gray',

                                    bins = np.arange(0,np.max(data[feature])+2),

                                    range=(0,np.max(data[feature])),

                                    alpha=0.75

                                   )

    ax.set_xticks(bins)



    # Label the raw counts and the percentages below the x-axis...

    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for count, x in zip(encoder_labels[feature_index+1].classes_, bin_centers):

        # Label the raw counts

        ax.annotate(count, xy=(x, 0), xycoords=('data', 'axes fraction'),

            xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    

    plt.ylabel('# Counts')

        

    plt.legend(['Edible','Poissonous'])

    

    print('Counts P: ',counts);
plot_histogram(data, top_idx[0])
plot_histogram(data, top_idx[1])
plot_histogram(data, top_idx[2])
plot_histogram(data, top_idx[3])
plot_histogram(data, top_idx[4])