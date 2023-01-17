import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



pd.set_option('display.max_rows', 80)

pd.set_option('display.max_columns', 80)

pd.set_option('display.width', 1000)
games = pd.read_csv('../input/csgo-ratings/csgo_games_ratings.csv')



# I left the resulting rating changes for each game in the data, but we won't need this.

games.drop(['t1_rating_change', 't2_rating_change'], axis=1, inplace=True)

games.shape
games.head(10)
t1_wins = games[games['t1_win'] == 1]

t2_wins = games[games['t1_win'] == 0]



print('Number of t1 wins: ', len(t1_wins))

print('Number of t2 wins: ', len(t2_wins))
def equalize_wins(games):

    t1_wins = games[games['t1_win'] == 1]

    t2_wins = games[games['t1_win'] == 0]



    min_wins = min(len(t1_wins), len(t2_wins))

    max_wins = max(len(t1_wins), len(t2_wins))



    if len(t2_wins) < len(t1_wins):

        reduced_t1_wins = t1_wins.sample(frac=(min_wins / max_wins), random_state=1)

        games = pd.concat([reduced_t1_wins, t2_wins], axis=0)

    else:

        reduced_t2_wins = t2_wins.sample(frac=(min_wins / max_wins), random_state=1)

        games = pd.concat([reduced_t2_wins, t1_wins], axis=0)



    return games.sample(frac=1, random_state=1)

    

games = equalize_wins(games)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))



ax1.hist(games['rating_difference'], bins=300)

ax1.set_ylabel('Number of Games', {'fontsize': 12})

ax1.set_xlabel('Rating Difference', {'fontsize': 12})

ax1.set_title('All Games', {'fontsize': 16})



non_zero = games[games['rating_difference'] != 0]



ax2.hist(non_zero['rating_difference'], bins=300)

ax2.set_ylabel('Number of Games', {'fontsize': 12})

ax2.set_xlabel('Rating Difference', {'fontsize': 12})

ax2.set_title('Games With Non-Zero Rating Difference', {'fontsize': 16})

plt.show()
zero_games = games[games['rating_difference'] == 0]

print('Win rate for zero difference games:', 

      round((zero_games['t1_win'].sum() / len(zero_games)), 4))
def win_rates(history):

    t1_better_t2 = history[history['rating_difference'] >= 0]

    t1_better_percentage = t1_better_t2['t1_win'].sum() / len(t1_better_t2)



    return round(t1_better_percentage, 4)



non_zero_games = games[games['rating_difference'] != 0].copy()

print('Win rate for all games:', win_rates(games))

print('Win rate for non-zero rating_difference games:', win_rates(non_zero_games))
# Returns a dataframe containing the win rates for games with a greater absolute rating_difference than i, along with the number of games in the sample

def win_rate(games):

    rating_win_rate = []

    games['rating_difference'] = games['rating_difference'].astype(int)

    for i in range(0, games['rating_difference'].abs().max()):

        t1_better_t2 = games[games['rating_difference'] > i]

        t1_worse_t2 = games[games['rating_difference'] < -i]



        # We are interested in the t1_worse_t2 games where t1_win is 0, this is captured below.

        wins = t1_better_t2['t1_win'].sum() + (len(t1_worse_t2) - t1_worse_t2['t1_win'].sum())

        num_games = len(t1_better_t2) + len(t1_worse_t2)

        win_rate = wins / num_games



        rating_win_rate.append({'rating_difference': i, 

                                'win_rate': win_rate, 

                                'num_games': num_games})



    return pd.DataFrame(rating_win_rate)
def plot_win_rating(games):

    fig, ax1 = plt.subplots(figsize=(7, 7))

    ax1.set_title('Win Rate vs Rating Difference', {'fontsize': 16})



    ax1.bar(x=games['rating_difference'], 

            height=games['num_games'], 

            width=2,

            alpha=0.4)

    ax1.set_xlabel('Rating difference', {'fontsize': 12})

    ax1.set_ylabel('Number of Games', {'fontsize': 12})



    ax2 = fig.add_subplot(sharex=ax1, frameon=False)

    ax2.plot(games['win_rate'], color='r')

    ax2.yaxis.tick_right()

    ax2.yaxis.set_label_position('right')

    ax2.set_ylabel('Win Rate', {'fontsize': 12})



    plt.show()
rating_win_rate = win_rate(games)

plot_win_rating(rating_win_rate)
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import scale



from scipy import stats



import tensorflow as tf

from tensorflow import keras



results = []
def decision_tree_classifier(games):

    x = np.array(games['rating_difference']).reshape(-1, 1)

    y = games['t1_win']



    x = scale(x)



    hyperparameters = {

        'max_depth': [2, 3, 4, 5],

        'min_samples_split': [2, 3, 4, 5],

        'min_samples_leaf': [1, 2, 3, 4, 5],

        'max_leaf_nodes': [5, 6, 7, 8, 9]



    }

    dtc = DecisionTreeClassifier()

    grid = GridSearchCV(dtc, param_grid=hyperparameters, cv=5)

    grid.fit(x, y)



    return grid.best_params_, grid.best_score_

  

include_zero = decision_tree_classifier(games)

non_zero = decision_tree_classifier(non_zero_games)

print('With zero rating difference \nBest accuracy: ', round(include_zero[1], 4), 

      '\nBest parameters: ', include_zero[0])

print('\nWithout zero rating difference \nBest accuracy: ', round(non_zero[1], 4), 

      '\nBest parameters: ', non_zero[0])



results.append({'model': 'Decision Tree', 

                'non_zero': round(non_zero[1], 4), 

                'include_zero': round(include_zero[1], 4)}

              )
def k_nearest_neighbors(games):    

    x = np.array(games['rating_difference']).reshape(-1, 1)

    y = games['t1_win']



    x = scale(x)



    hyperparameters = {

        'n_neighbors': [10, 20, 30, 40, 50, 60, 70],

        'weights': ['uniform', 'distance'],

        'algorithm': ['brute', 'auto'],

        'p': [1, 2]



    }

    knc = KNeighborsClassifier()

    grid = GridSearchCV(knc, param_grid=hyperparameters, cv=5)

    grid.fit(x, y)



    return grid.best_params_, grid.best_score_

    

include_zero = k_nearest_neighbors(games)

non_zero = k_nearest_neighbors(non_zero_games)

print('With zero rating difference \nBest accuracy: ', round(include_zero[1], 4), 

      '\nBest parameters: ', include_zero[0])

print('\nWithout zero rating difference \nBest accuracy: ', round(non_zero[1], 4), 

      '\nBest parameters: ', non_zero[0])



results.append({'model': 'K Neighbors', 

                'non_zero': round(non_zero[1], 4), 

                'include_zero': round(include_zero[1], 4)}

              )
def random_forest_classifier(games):    

    x = np.array(games['rating_difference']).reshape(-1, 1)

    y = games['t1_win']



    x = scale(x)



    hyperparameters = {

        'n_estimators': [10, 15, 20, 25],

        'max_depth': [2, 3, 4],

        'min_samples_split': [5, 10, 15, 20],

        'max_leaf_nodes': [20, 25, 30, 35],

        'random_state': [1]



    }

    rfc = RandomForestClassifier()

    grid = GridSearchCV(rfc, param_grid=hyperparameters, cv=5)

    grid.fit(x, y)



    return grid.best_params_, grid.best_score_

    

include_zero = random_forest_classifier(games)

non_zero = random_forest_classifier(non_zero_games)

print('With zero rating difference \nBest accuracy: ', round(include_zero[1], 4), 

      '\nBest parameters: ', include_zero[0])

print('\nWithout zero rating difference \nBest accuracy: ', round(non_zero[1], 4), 

      '\nBest parameters: ', non_zero[0])



results.append({'model': 'Random Forest', 

                'non_zero': round(non_zero[1], 4), 

                'include_zero': round(include_zero[1], 4)}

              )
def cvc(games):

    x = np.array(games['rating_difference']).reshape(-1, 1)

    y = games['t1_win']



    x = scale(x)



    hyperparameters = {}



    svc = SVC()

    grid = GridSearchCV(svc, param_grid=hyperparameters, cv=10)

    grid.fit(x, y)



    return grid.best_params_, grid.best_score_

    

include_zero = cvc(games)

non_zero = cvc(non_zero_games)

print('With zero rating difference \nBest accuracy: ', round(include_zero[1], 4))

print('\nWithout zero rating difference \nBest accuracy: ', round(non_zero[1], 4))



results.append({'model': 'CVC', 

                'non_zero': round(non_zero[1], 4), 

                'include_zero': round(include_zero[1], 4)}

              )
def sgd(games):

    x = np.array(games['rating_difference']).reshape(-1, 1)

    y = games['t1_win']



    x = scale(x)



    hyperparameters = {

        'loss': ['log', 'hinge', 'squared_loss'],

        'max_iter': [10000],

        'shuffle': [False]

    }

    sgdc = SGDClassifier()

    grid = GridSearchCV(sgdc, param_grid=hyperparameters, cv=10)

    grid.fit(x, y)



    return grid.best_params_, grid.best_score_

    

include_zero = sgd(games)

non_zero = sgd(non_zero_games)



print('With zero rating difference \nBest accuracy: ', round(include_zero[1], 4), 

      '\nBest parameters: ', include_zero[0])

print('\nWithout zero rating difference \nBest accuracy: ', round(non_zero[1], 4), 

      '\nBest parameters: ', non_zero[0])



results.append({'model': 'SGD', 

                'non_zero': round(non_zero[1], 4), 

                'include_zero': round(include_zero[1], 4)}

              )
results_df = pd.DataFrame(results)



fig, ax = plt.subplots(figsize=(7, 7))



ax.bar(x=results_df['model'], 

       height=results_df['non_zero'], 

       label='Excluding Zero Difference')

ax.bar(x=results_df['model'], 

       height=results_df['include_zero'], 

       label='Including Zero Difference')



ax.set_ylim(bottom=0.5)

ax.set_title('Accuracy of Models', {'fontsize':16})

ax.set_ylabel('Accuracy', {'fontsize':12})

ax.set_xlabel('Model', {'fontsize':12})

ax.legend(loc='upper right')



plt.show()
def create_model():



    model = keras.Sequential()

    model.add(keras.layers.Dense(2, activation="softmax"))



    opt = tf.keras.optimizers.Adam()



    model.compile(loss='sparse_categorical_crossentropy',

                  optimizer=opt,

                  metrics=['accuracy'])



    return model



# Don't turn x into an nparray yet, 

# so we can track the non scaled values during the shuffle to keep comparisons intuitive.

x = non_zero_games['rating_difference']

non_zero_games['scaled_x'] = scale(x)

y = np.array(non_zero_games['t1_win'])



train_x, test_x, train_y, test_y = train_test_split(non_zero_games[['rating_difference', 

                                                                    'scaled_x']], 

                                                    y, 

                                                    test_size=0.2, 

                                                    random_state=1, 

                                                    shuffle=True

                                                   )



model = create_model()

model.fit(np.array(train_x['scaled_x']), 

          train_y, 

          validation_data=(np.array(test_x['scaled_x']), 

          test_y), 

          epochs=100, 

          verbose=2

         )
# This function matches the nparrays we fed into our model with their dataframe counterparts 

def prob_details(x, y):    

    probabilities = model.predict(np.array(x['scaled_x']))

    with_ratings = pd.concat([pd.DataFrame(probabilities, 

                                           columns=['t2_confidence', 't1_confidence']),

                              x['rating_difference'].reset_index(drop=True),

                              pd.DataFrame(y, columns=['t1_win'])], 

                             axis=1

                            )

    return with_ratings
test_details = prob_details(test_x, test_y)



fig, ax1 = plt.subplots(figsize=(7, 7))

ax1.set_title('Confidence vs Rating Difference', {'fontsize': 16})

ax1.bar(test_details['rating_difference'], test_details['t1_confidence'])

ax1.set_xlabel('Rating Difference', {'fontsize': 12})

ax1.set_ylabel('Confidence', {'fontsize': 12})



plt.plot()
def plot_confidence_win_rate(games):

    plt.figure(figsize=(7, 7))

    

    bin_means, bin_edges, binnumber = stats.binned_statistic(games['rating_difference'], 

                                                             games['t1_confidence'], 

                                                             'mean', 

                                                             bins=10)

    plt.hlines(bin_means, 

               bin_edges[:-1], 

               bin_edges[1:], 

               colors='r', 

               alpha=0.8, 

               label='Average Confidence')

    

    bin_means, bin_edges, binnumber = stats.binned_statistic(games['rating_difference'], 

                                                             games['t1_win'], 

                                                             'mean', 

                                                             bins=10)

    plt.hlines(bin_means, 

               bin_edges[:-1], 

               bin_edges[1:], 

               colors='b', 

               alpha=0.8, 

               label='Average Win Rate')

    

    plt.legend(loc='upper left')

    plt.title('Win Rates and Model Confidence', {'fontsize':16})

    plt.ylabel('Confidence / Win Rate', {'fontsize':12})

    plt.xlabel('Rating Difference', {'fontsize':12})

    plt.show()
plot_confidence_win_rate(test_details)
train_details = prob_details(train_x, train_y)

plot_confidence_win_rate(train_details)