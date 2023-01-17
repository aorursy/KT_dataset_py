import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import scipy.stats as stats

import seaborn as sns

from matplotlib import rcParams

import csv

import collections

from time import time



from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint as sp_randint



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



%matplotlib notebook

%pylab notebook
test = '/kaggle/input/starcraft-2-player-prediction-challenge-2019/TEST.CSV'

train_long = '/kaggle/input/starcraft-2-player-prediction-challenge-2019/TRAIN_LONG.CSV'

train = '/kaggle/input/starcraft-2-player-prediction-challenge-2019/TRAIN.CSV'

test_long = '/kaggle/input/starcraft-2-player-prediction-challenge-2019/TEST_LONG.CSV'

sample_sub = '/kaggle/input/starcraft-2-player-prediction-challenge-2019/SAMPLE_SUBMISSION.CSV'
def formatTrain(file):

    profile = []

    avatar = []

    moves = []

    with open(file) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',', quotechar="'")

        for row in csv_reader:

            profile.append(row[0])

            avatar.append(row[1])

            moves.append(row[2:])

        d = { 'Profile':profile, 'Avatar':avatar, 'Moves':moves}  

        df = pd.DataFrame(data=d)

    return df



def formatTest(file):

    avatar = []

    moves = []

    with open(file) as csvfile:

        csvreader = csv.reader(csvfile, delimiter=',', quotechar="'")

        for row in csvreader:

            avatar.append(row[0])

            moves.append(row[1:])

    d = {'Avatar':avatar, 'Moves':moves}  

    df = pd.DataFrame(data=d)

    return df
df_train = formatTrain(train)

df_train.head()
def get_unique_moves(df):

    moves = df['Moves']

    unique_moves = set()

    for move in moves:

        for a in move:

            a = a.strip()

            if a[0]!='t':

                unique_moves.add(a)

    return unique_moves       
unique_moves = sorted(list(get_unique_moves(df_train)))

print(unique_moves)
def get_unique_times(df):

    moves = df['Moves']

    timeframe = set()

    for time in moves:

        for a in time:

            a = a.strip()

            if a[0]=='t':

                timeframe.add(int(a[1:]))

    return timeframe       



def get_unique_temporal_moves(df, times):

    times_count = {time:0 for time in times}

    times_count [0] = 0

    moves = df['Moves']

    for move in moves:

        t, inf, sup = 0, 0, 0

        for a in move:

            a = a.strip()

            if a[0]=='t':

                sup = move.index(a)

                count = sup - inf - 1

                times_count [t] += count

                inf = sup

                t+= 5

    return times_count
unique_times = sorted(list(get_unique_times(df_train)))

unique_temporal_moves = get_unique_temporal_moves(df_train, unique_times)
plt.bar(range(len(unique_temporal_moves)), list(unique_temporal_moves.values()), align='center')

plt.xticks(range(len(unique_temporal_moves)), list(unique_temporal_moves.keys()))

plt.axvline(x=60, color = "red")

plt.axvline(x=120, color = "red")

plt.show()
agg = df_train.groupby('Profile').size().reset_index(name='counts')

print(agg)
def get_time(df):

    game = []

    for _, row in df.iterrows():

        duration = 0

        for ele in row['Moves']:

            if str(ele).startswith("t"):

                time = int(str(ele)[1:])

                if time > duration:

                    duration = time

        game.append(duration)

    return game
times = get_time(df_train)

plt.hist(times,50, facecolor='blue', alpha=0.5)

plt.show()
def get_average_time(df):

    profiles = list(set(df_train['Profile']))

    players = {profile:0 for profile in profiles}

    for _, row in df.iterrows():

        duration = 0

        for ele in row['Moves']:

            if str(ele).startswith("t"):

                time = int(str(ele)[1:])

                if time > duration:

                    duration = time

        player = row['Profile']

        players[player] += duration

    df_players = pd.Series(players).to_frame('Average time')

    return df_players
df_average_time = get_average_time(df_train)

df_average_time.head()
def define_features(df):  

    actions = unique_moves

    t60_actions = ["t60_" + moves for moves in unique_moves ]

    features = []

    for index, row in df.iterrows():

        #avatar = row["Avatar"]

        player_moves = row["Moves"]

        moves_count = {move:0 for move in actions}

        moves_60_count = {move:0 for move in t60_actions}

        t60_moves = 0

        t120_moves = 0

        duration = 1

        for action in player_moves:

            if str(action).startswith("t"):

                time = int(str(action)[1:])

                if time > duration:

                    duration = time

            if action in unique_moves:

                moves_count[action]+=1

        for key, value in moves_count.items():

            moves_count[key]= value/duration

        if "t60" in player_moves:

            t60 = player_moves[0:player_moves.index("t60")]

            for action in t60:

                if action in unique_moves:

                    t60_moves += 1

                    moves_60_count["t60_"+action]+=1

        if "t120" in player_moves:

            t120 = player_moves[player_moves.index("t60"):player_moves.index("t120")]

            for action in t120:

                if action in unique_moves:

                    t120_moves += 1

        #current = [avatar, *[moves_count[move] for move in actions], t60_moves, duration, t120_moves,*[moves_60_count[move] for move in t60_actions]]

        current = [*[moves_count[move] for move in actions], t60_moves, duration, t120_moves,*[moves_60_count[move] for move in t60_actions]]

        features.append(current)

        #new_df = pd.DataFrame(features, columns=["Avatar", *actions, "t60_moves", "duration", "t120_moves", *t60_actions])

        new_df = pd.DataFrame(features, columns=[*actions, "t60_moves", "duration", "t120_moves", *t60_actions])

    return new_df
train_features = define_features(df_train)

#train_features = pd.get_dummies(train_features, columns = ["Avatar"])



train_features.head()
# From Base to hotkey32

fig, axs = plt.subplots(9, 2, figsize=(15, 15), facecolor='w', edgecolor='k')

fig.subplots_adjust(hspace = .2, wspace=.1)

axs = axs.ravel()

i = 0

right = False

for label in train_features :

    if label == "hotkey41":

        right = True

    if right:

        axs[i].hist(train_features[label], bins=20)

        axs[i].set_title("Histogram of {}".format(label))

        axs[i].set_xlabel('Frequency {}'.format(label))

        i = i + 1

    if label == "hotkey92":

        break
# From Base to hotkey32

fig, axs = plt.subplots(7, 2, figsize=(15, 15), facecolor='w', edgecolor='k')

fig.subplots_adjust(hspace = .2, wspace=.1)

axs = axs.ravel()

i = 0

for label in train_features :

    axs[i].hist(train_features[label], bins=20)

    axs[i].set_title("Histogram of {}".format(label))

    axs[i].set_xlabel('Frequency {}'.format(label))

    i = i + 1

    if label == "hotkey32":

        break
def convert(output):

    output_df = pd.DataFrame(output, columns=['prediction'])

    output_df.index = range(1,len(output_df)+1)

    output_df.index.name = 'RowId'

    return output_df



def save(output_df, name):

    output_df.to_csv('./out_'+name+'.csv')
save(train_features,"matrix")
X = train_features

y = df_train.Profile
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
accuracy = []

for i in range(1,100):

    # Create Decision Tree classifer object

    clf = DecisionTreeClassifier(max_depth=i)



    # Train Decision Tree Classifer

    clf = clf.fit(X_train,y_train)



    #Predict the response for test dataset

    y_pred = clf.predict(X_test)

    

    accuracy.append(int(metrics.accuracy_score(y_test, y_pred)*100))

    
# Bar Plot

fig = plt.figure(figsize = (6, 4))

title = fig.suptitle("Decision Tree Accuracy", fontsize=14)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax = fig.add_subplot(1,1, 1)

ax.set_xlabel("Max depth")

ax.set_ylabel("Accuracy") 

w_q = ([i for i in range(1,100)], accuracy)

ax.tick_params(axis='both', which='major', labelsize=8.5)

bar = ax.bar(w_q[0], w_q[1], color='steelblue', 

        edgecolor='black', linewidth=1)
# Create Decision Tree classifer object

clf = DecisionTreeClassifier(max_depth=45)



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



print("Accuracy of the prediction for the binary decision tree: ", int(metrics.f1_score(y_test, y_pred,average='micro')*100), "%")
clf = RandomForestClassifier(n_estimators=100)

# Utility function to report best scores

def report(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")





# specify parameters and distributions to sample from

param_dist = {"max_depth": sp_randint(10, 30),

              "max_features": sp_randint(5, 20),

              "min_samples_split": sp_randint(2, 10),

              "bootstrap": [False],

              "criterion": ["gini", "entropy"],

             "random_state":sp_randint(0, 15)}



# run randomized search

n_iter_search = 20

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,

                                   n_iter=n_iter_search, cv=5)



start = time.time()

random_search.fit(X, y)

print("RandomizedSearchCV took %.2f seconds for %d candidates"

      " parameter settings." % ((time.time() - start), n_iter_search))

report(random_search.cv_results_)
rd_forest_clf = RandomForestClassifier(n_estimators=100, bootstrap= False, criterion='entropy', max_depth= 23, max_features= 8,min_samples_split= 3, random_state=0).fit(X_train,y_train)

predicted_rd_forest = rd_forest_clf.predict(train_features)

print("Accuracy of the prediction for the binary decision tree: ", int(metrics.f1_score(y, predicted_rd_forest,average='micro')*100), "%")
importances = rd_forest_clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rd_forest_clf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

plt.figure()

plt.title("Feature importances")

plt.bar(range(train_features.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(train_features.shape[1]), indices)

plt.xlim([-1, train_features.shape[1]])

plt.show()
# Correlation Matrix Heatmap

f, ax = plt.subplots(figsize=(20, 15))

corr = X.corr()

hm = sns.heatmap(round(corr,2), annot=False, ax=ax, cmap="coolwarm",fmt='.1f',

                 linewidths=.05)

f.subplots_adjust(top=0.93)

t= f.suptitle('Feature Correlation Heatmap', fontsize=10)



plt.savefig('Feature Correlation Heatmap.png')
confusion_matrix(y, predicted_rd_forest)
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    fig.tight_layout()

    return ax
class_names = df_train.groupby('Profile').size().reset_index(name='counts')

classes_names = list(class_names['Profile'])
np.set_printoptions(precision=2)



# Plot normalized confusion matrix

plot_confusion_matrix(y, predicted_rd_forest, classes=classes_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()
def compare_misclassified_labels(y_true, y_pred):

    misclass_gt = y_true[y_true != y_pred]

    misclass_pred = y_pred[y_true != y_pred]

    misclass_df = pd.DataFrame({'ground truth' : misclass_gt,

                                'predicted' : misclass_pred})

    return misclass_df



misclass_df = compare_misclassified_labels(y, predicted_rd_forest)

misclass_df.to_csv("./out_misclassified.csv", index = False)

misclass_df.head()
df_test = formatTest(test)

df_test.head()
test_features = define_features(df_test)

#test_features = pd.get_dummies(test_features, columns = ["Avatar"])



test_features.head()
predicted_rd_forest = rd_forest_clf.predict(test_features)
predicted_rd_fr_converted = convert(predicted_rd_forest)

save(predicted_rd_fr_converted, "rndForest_prediction")

predicted_rd_fr_converted.head()