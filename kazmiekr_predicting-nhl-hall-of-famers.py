import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # viz

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score, precision_score, recall_score



%matplotlib inline
master_df = pd.read_csv('../input/Master.csv')

master_df.head()
awards_df = pd.read_csv('../input/AwardsPlayers.csv')

awards_df.head()
master_df = master_df[['playerID', 'birthCountry', 'hofID', 'firstName', 'lastName', 'height', 'weight', 'shootCatch', 'firstNHL', 'lastNHL', 'pos']]

master_df.head()
master_df.info()
master_df = master_df.dropna(subset=['playerID', 'birthCountry', 'height', 'weight', 'shootCatch', 'firstNHL', 'lastNHL'])

master_df.info()
master_df["nhlLength"] = master_df["lastNHL"] - master_df["firstNHL"] + 1

master_df["isInHf"] = np.where(master_df['hofID'].notnull(), 1, 0)

master_df["fullName"] = master_df["firstName"] + ' ' + master_df["lastName"]
awards_df.head()
awards_count_df = awards_df[['playerID', 'award']].groupby(['playerID'], as_index=False).count().sort_values(by='award', ascending=False)

awards_count_df = awards_count_df.rename(columns={'award':'awardCount'})

awards_count_df.head(25)
master_df = pd.merge(master_df, awards_count_df, how='left', on='playerID')

master_df['awardCount'] = master_df['awardCount'].fillna(0)

master_df.tail(25)
master_df.describe()
master_df.describe(include=['O'])
master_df[['birthCountry', 'isInHf']].groupby(['birthCountry'], 

          as_index=False).sum().sort_values(by='isInHf', ascending=False).head(10)
master_df[['birthCountry', 'isInHf']].groupby(['birthCountry'], 

          as_index=False).mean().sort_values(by='isInHf', ascending=False).head(10)
pd.crosstab(master_df['birthCountry'], master_df['isInHf'])
master_df[['shootCatch', 'isInHf']].groupby(['shootCatch'], 

          as_index=False).count().sort_values(by='isInHf', ascending=False)
master_df.loc[master_df['shootCatch'] == 'B', 'shootCatch'] = 'L'

master_df[['shootCatch', 'isInHf']].groupby(['shootCatch'], 

          as_index=False).count().sort_values(by='isInHf', ascending=False)
master_df[['pos', 'isInHf']].groupby(['pos'], 

          as_index=False).count().sort_values(by='isInHf', ascending=False)
master_df.loc[master_df['pos'] == 'L/D', 'pos'] = 'L'

master_df.loc[master_df['pos'] == 'D/L', 'pos'] = 'D'

master_df.loc[master_df['pos'] == 'L/C', 'pos'] = 'L'

#master_df.loc[master_df['pos'] == 'L', 'pos'] = 'F'

#master_df.loc[master_df['pos'] == 'R', 'pos'] = 'F'

master_df.loc[master_df['pos'] == 'F', 'pos'] = 'L'

master_df[['pos', 'isInHf']].groupby(['pos'], 

          as_index=False).mean().sort_values(by='isInHf', ascending=False)
hof_df = master_df[master_df["isInHf"] == 1].sort_values(by='nhlLength', ascending=True)

print(hof_df.describe())

hof_df.head(25)
def plot_distribution( df, var, target, **kwargs):

    row = kwargs.get('row', None)

    col = kwargs.get('col', None)

    facet = sns.FacetGrid(df , hue=target, aspect=3, row=row, col=col)

    facet.map(sns.kdeplot, var, shade=True)

    facet.set(xlim=(df[var].min() , df[var].max()))

    facet.add_legend()



def plot_categories(df, cat, target, **kwargs):

    row = kwargs.get('row', None)

    col = kwargs.get('col', None)

    facet = sns.FacetGrid(df, row=row, col=col)

    facet.map(sns.barplot, cat, target)

    facet.add_legend()



categories_to_plot = ['shootCatch', 'pos']

for category in categories_to_plot:

	plot_categories(master_df, category, 'isInHf')



features_to_plot = ['height', 'weight', 'awardCount', 'nhlLength']

for feature in features_to_plot:

	plot_distribution(master_df, feature, 'isInHf')
countryIx = 0

countryMap = {}



def convertCountryMap(country):

    global countryIx, countryMap

    if country in countryMap.keys():

        return countryMap[country]

    else:

        countryMap[country] = countryIx

        countryIx += 1

        return countryMap[country]

    

master_df['birthCountry'] = master_df.birthCountry.apply(convertCountryMap)

print(countryMap)

master_df.head(25)
master_df["shootCatch"] = master_df['shootCatch'].map( {'L': 1, 'R': 0} )

master_df["pos"] = master_df['pos'].map( {'C': 0, 'F': 1, 'D': 2, 'G': 3, 'L': 4, 'R': 5} )

master_df.head(25)
master_df[master_df["isInHf"] == 1].sort_values(by='nhlLength', ascending=True).head(15)
new_hof = ['burepa01',

'oatesad01',

'sakicjo01',

'sundima01',

'chelich01',

'niedesc01',

'shanabr01',

'hasekdo01',

'forsbpe01',

'modanmi01',

'blakero01',

'lidstni01',

'fedorse01',

'houslph01',

'prongch01',

'lindrer01',

'vachoro01',

'makarse01'

]



new_hof_players_df = master_df[master_df['playerID'].isin(new_hof)]

print(new_hof_players_df.shape)



#for playerId in new_hof:

    #master_df.loc[master_df['playerID'] == playerId, 'isInHf'] = 1
season_threshold = 2000



# We're saying that a player is eligible for the HoF if they haven't been selected

# and they retired at some point since our threshold (2000) and that they aren't currently

# playing, ( lastNHL == 2011 )

# NOTE: Since this was 6 years ago, there is an obvious gap in player retirements that we'll miss in predictions

eligible_players = master_df[(master_df['isInHf'] == 0) & (master_df['lastNHL'] >= season_threshold) & (master_df['lastNHL'] < 2011)]

print(eligible_players.shape)



# Players that have a low chance of being inducted if they haven't already

retired_players = master_df[master_df['lastNHL'] < season_threshold]

print(retired_players.shape)
retired_players.head()
cols = ['playerID', 'fullName', 'birthCountry', 'height', 'weight', 'shootCatch', 'nhlLength', 'pos', 'isInHf', 'awardCount']

retired_players = retired_players[cols]

eligible_players = eligible_players[cols]

new_hof_players_df = new_hof_players_df[cols]

eligible_players.head()
def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 240 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )

    

plot_correlation_map( retired_players )
cols_to_drop = ['playerID', 'fullName', 'isInHf']



# Break the dataset into a training and cross validation set

train=retired_players.sample(frac=0.75,random_state=200)

test=retired_players.drop(train.index)



X_train = train.drop(cols_to_drop, axis=1)

Y_train = train['isInHf']

X_test = test.drop(cols_to_drop, axis=1).copy()

Y_test = test['isInHf']



# Dataset with unknown HoF predictions

X_player_test = eligible_players.drop(cols_to_drop, axis=1).copy()



# Create a test set of players that have made the HF since the last year of available data in this data set

# We'll just fill the Y with 1's

X_new_hof_test = new_hof_players_df.drop(cols_to_drop, axis=1).copy()

Y_new_hof_test = np.ones(X_new_hof_test.shape[0], dtype=np.int)



X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
classifiers = {}

# Run a classifier instance against the training set

def run_classifier(classifier):

    # Get the name of the classifier instance

    class_name = classifier.__class__.__name__

    print("Testing {0}".format(class_name))

    print("----------------------------------------\n")

    

    # Fit the classifier to the training set

    classifier.fit(X_train, Y_train)

    # Find out the training accuracy

    training_accuracy = round(classifier.score(X_train, Y_train) * 100, 2)

    print("Training Score: {0}".format(training_accuracy))

    

    # Use the trained data to predict the next HoF's

    Y_pred = classifier.predict(X_test)

    

    # How accurate was it against the known set of recent HoF's

    hof_accuracy = test_current_hf(classifier)

    

    # Cross validation scores

    f1_score = f1(Y_test, Y_pred)

    

    

    print("\n----- Predictions -----\n")

    # For all the classifiers except LinearSVC, we'll determine the probability in addition

    # to the prediction

    if class_name != 'LinearSVC':

        (positive_predictions, _) = show_prediction_with_probabilities(classifier, X_player_test)

    else:

        (positive_predictions, _) = show_prediction(classifier, X_player_test)

        

    # Store our results per classifier for later analysis    

    classifiers[class_name] = {

        'Training Accuracy': training_accuracy,

        'F1 Score': f1_score,

        'Current HoF Accuracy': hof_accuracy

    }

    return positive_predictions



# Given our set of known recent HoF additions, does the classifier think they should have made it?

def test_current_hf(classifier):

    accuracy = round(classifier.score(X_new_hof_test, Y_new_hof_test) * 100, 2)

    print("Accuracy with Current HF: {0}".format(accuracy))

    return accuracy



# Display from cross validation scores, F1, precision, recall

def f1(Y_test, Y_pred):

    macro = f1_score(Y_test.values, Y_pred, average='macro')

    print("F1 Score: {0}".format(formatPercent(macro)))

    precision = precision_score(Y_test.values, Y_pred, average='macro')

    print("Precision Score: {0}".format(formatPercent(precision)))

    recall = recall_score(Y_test.values, Y_pred, average='macro')

    print("Recall Score: {0}".format(formatPercent(recall)))

    return macro



def formatPercent(num):

    return round(num, 2)

    

def show_prediction_with_probabilities(classifier, X_player_test, display=True):

    prediction = classifier.predict(X_player_test)

    prob = classifier.predict_proba(X_player_test)

    pred_df = pd.DataFrame({

        "playerID": eligible_players["playerID"],

        "fullName": eligible_players["fullName"],

        "isInHfPrediction": prediction,

        #"prob1": prob[:,0],

        "probability": prob[:,1]

    })

    positive_predictions = pred_df[pred_df['isInHfPrediction'] == 1]

    if display == True:

        print(positive_predictions.sort_values(by='probability', ascending=False))    

    return positive_predictions, pred_df



def show_prediction(classifier, X_player_test):

    prediction = classifier.predict(X_player_test)

    pred_df = pd.DataFrame({

        "playerID": eligible_players["playerID"],

        "fullName": eligible_players["fullName"],

        "isInHfPrediction": prediction

    })

    positive_predictions = pred_df[pred_df['isInHfPrediction'] == 1]

    print(positive_predictions)

    return positive_predictions, pred_df
nn = MLPClassifier(alpha=1)

nn_pred = run_classifier(nn)
logreg = LogisticRegression()

logreg_pred = run_classifier(logreg)
svc = SVC(probability=True)

svc_pred = run_classifier(svc)
coeff_df = pd.DataFrame(retired_players.columns.drop(['playerID', 'fullName', 'isInHf']))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
knn = KNeighborsClassifier(n_neighbors = 3)

knn_pred = run_classifier(knn)
gaussian = GaussianNB()

gaussian_pred = run_classifier(gaussian)
linear_svc = LinearSVC()

linear_svc_pred = run_classifier(linear_svc)
decision_tree = DecisionTreeClassifier()

decision_tree_pred = run_classifier(decision_tree)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest_pred = run_classifier(random_forest)
model_performance = pd.DataFrame.from_dict(classifiers, orient='index')

model_performance.sort_values(by='F1 Score', ascending=False)
gaussian_pred.sort_values(by='probability', ascending=False).head(10)
def filter_out_new_hof(df):

    return df[~df['playerID'].isin(new_hof)].sort_values(by='probability', ascending=False)



filter_out_new_hof(gaussian_pred).head(5)
filter_out_new_hof(logreg_pred).head(5)
# Grab his record

teemu = master_df[master_df['playerID'] == 'selante01'][cols]

# Remove cols not needed

teemu.drop(cols_to_drop, axis=1)

# Add his as an eligible player

eligible_players = eligible_players.append(teemu)

# Add him to the test set

X_player_test = eligible_players.drop(cols_to_drop, axis=1).copy()
# Rerun the prediction using our existing NN classifier

(_, all_nn_pred) = show_prediction_with_probabilities(nn, X_player_test, display=False)

filter_out_new_hof(all_nn_pred).head(10)
# Rerun the prediction using our existing Gaussian classifier

(_, all_gaussian_pred) = show_prediction_with_probabilities(gaussian, X_player_test, display=False)

filter_out_new_hof(all_gaussian_pred).head(10)
(_, all_logreg_pred) = show_prediction_with_probabilities(logreg, X_player_test, display=False)

filter_out_new_hof(all_logreg_pred).head(10)
missing_hof = ['andreda01', 'recchma01']

master_df[master_df['playerID'].isin(missing_hof)]
all_gaussian_pred[all_gaussian_pred['playerID'].isin(missing_hof)]
all_logreg_pred[all_logreg_pred['playerID'].isin(missing_hof)]