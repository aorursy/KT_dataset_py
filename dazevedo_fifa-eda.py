import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import random

from matplotlib.colors import ListedColormap



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix

#load data



def load_data(name):

    df = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/' + name + '.csv')

    df.dob=pd.to_datetime(df.dob)

    

    return df



fifa_15 = load_data('players_15')

fifa_16 = load_data('players_16')

fifa_17 = load_data('players_17')

fifa_18 = load_data('players_18')

fifa_19 = load_data('players_19')

fifa_20 = load_data('players_20')



fifa_20.head()
nan_columns = fifa_20.isnull().sum() > 0

nan_columns
#Plot number of Nans in each column



nan_columns = fifa_20.isnull().sum() > 0

fifa_20.isnull().sum()[nan_columns].plot.bar(figsize = (30, 10))



plt.xlabel('Data Columns', fontsize=18), plt.xticks(fontsize=18)

plt.ylabel('Nº of Nans', fontsize=18), plt.yticks(fontsize=18)

plt.show()

print('Players Position:', fifa_20[fifa_20['rb'].isnull()]['player_positions'].unique())


def create_main_position(data):

        data.loc[:, 'main_position'] = data['player_positions'].apply(lambda x: x.split(',')[0])





create_main_position(fifa_15)

create_main_position(fifa_16)

create_main_position(fifa_17)

create_main_position(fifa_18)

create_main_position(fifa_19)

create_main_position(fifa_20)
tactic_433 = {

        'name': "433",

        'positions': ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

        }

    

tactic_442 = {

        'name': "442",

        'positions': ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'CDM', 'RM', 'ST', 'ST']

        }

    

tactic_352 = {

        'name': "352",

        'positions': ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']

        }





def get_best_team(data, nationality, chosen_tactic):



    best_team = []



    print("\nBest Team using tactic {} of {}:\n{}".format(chosen_tactic['name'], nationality, '='*35))



    for i in chosen_tactic['positions']:

        potential_players = data[(data['nationality'] == nationality) 

                                    & (data['player_positions'].str.contains(i))

                                   ].sort_values(['overall'], ascending=False)



        ind = 0

        while potential_players.iloc[ind].short_name in best_team:

            ind +=1



        best_team.append(potential_players.iloc[ind].short_name)

        print(i + ':', potential_players.iloc[ind].short_name + ' (' + str(potential_players.iloc[ind].overall) + ')')





#Examples



get_best_team(fifa_20, 'Portugal', tactic_433)



get_best_team(fifa_18, 'Spain', tactic_442)



get_best_team(fifa_20, 'Germany', tactic_352)







def get_best_talents(data, initial_overall=80):

    

    data['overall_diff'] = data.potential - data.overall

    

    data.sort_values(['overall_diff'], ascending = False, inplace = True)

    

    display(data.loc[data.overall >= initial_overall, ['short_name', 'age', 'nationality', 'club', 'player_positions', 'overall', 'potential']].head(10))





print("\n{} Wonderkids {}\n".format('='*30, '='*30))

#wonderkids

get_best_talents(fifa_17.copy(), 50)



print("\n\n{} Promising Players {}\n".format('='*25, '='*25))

#promising players

get_best_talents(fifa_15.copy(), 80)



print("\n\n{} Special Players {}\n".format('='*25, '='*25))

#future world star players

get_best_talents(fifa_20.copy(), 85)
def player_evolution(all_data, names):

    

    fig, ax = plt.subplots(figsize=(15, 10))

    

    for n in names:

        overalls = []

        

        for d in all_data:

            player_overall = d.loc[d['short_name'] == n, 'overall']

            overalls.append(player_overall.iloc[0])



        plt.plot(overalls, 'o-', label=n)



    plt.ylim([70, 100])

    plt.ylabel('Overall')

    plt.xlabel('Time')

    plt.legend()

    

    plt.title('Players Evolution')

    

    ax.set_xticklabels(['_', 'Fifa 15', 'Fifa 16', 'Fifa 17', 'Fifa 18', 'Fifa 19', 'Fifa 20'])



    

    plt.show()

        

player_evolution([fifa_15, fifa_16, fifa_17, fifa_18, fifa_19, fifa_20], ['A. Griezmann', 'R. Varane', 'M. ter Stegen', 'J. Rodríguez'])
def potential_team(data):

    #data = data[~(data['team_position'].isin(['RES', 'SUB']))]

    

    potential_data = data.groupby('club')['potential'].mean().sort_values(ascending = False)[:15]

    

    plt.figure(figsize = (15, 7))

    potential_data.plot(kind='bar')

    plt.ylim(70, 90)

    plt.ylabel('Potential Overall Mean')

    plt.show()

    

potential_team(fifa_20.copy())


def overall_vs_value(data):

    

    best_overall = data.groupby('club')['overall'].mean().sort_values(ascending = False)[:15]

    best_value = data.groupby('club')['value_eur'].mean().sort_values(ascending = False)[:15]

    

    

    fig, ax1 = plt.subplots(figsize = (23, 10))



    color = 'red'

    ax1.set_xlabel('Teams')

    ax1.set_ylabel('Overall', color=color)

    ax1.plot(best_overall, 'o-', color=color)

    ax1.tick_params(axis='y', labelcolor=color)



    ax2 = ax1.twinx()



    color = 'blue'

    ax2.set_ylabel('Club Value', color=color)  # we already handled the x-label with ax1

    ax2.plot(best_value, 'o-', color=color)

    ax2.tick_params(axis='y', labelcolor=color)



    fig.tight_layout()

    plt.show()





overall_vs_value(fifa_20.copy())
def stats_players(data):

    

    pos_dict = {

        'GK': 'GK',

        'RW': 'ATA',

        'ST': 'ATA',

        'LW': 'ATA',

        'CAM': 'MED',

        'CB': 'DEF',

        'CM': 'MED',

        'CDM': 'MED',

        'CF': 'ATA',

        'LB': 'DEF',

        'RB': 'DEF',

        'RM': 'MED',

        'LM': 'MED',

        'LWB': 'DEF',

        'RWB': 'DEF'

    }

    

    data['general_position'] = data['main_position'].map(pos_dict)

    

    #use only the stats/attributes columns

    data = data.iloc[:, np.append([2, 14, -2, -1], np.arange(31, len(data.columns) - 28))]

    

    data.drop(['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 'player_traits'], axis = 1, inplace = True)

    

    data.fillna(0, inplace = True)

    

    return data

    

    

    

stats_data = stats_players(fifa_20.copy())
stats_data.head()
def plot_correlation(df):

    plt.figure(figsize=(40,20))



    sns.set(font_scale=1.3)



    ax = sns.heatmap(

        df, 

        vmin=-1, vmax=1, center=0,

        cmap=sns.color_palette("Blues"),

        square=True,

        annot=True, annot_kws={"size": 11}

    )

    ax.set_xticklabels(

        ax.get_xticklabels(),

        rotation=70,

        horizontalalignment='right'

    );



    plt.show()



plot_correlation(stats_data.iloc[:, 4:].corr())



def main_attributes_position(data, target_position = 'DEF'):

    

    data = data.iloc[0:5000, :]

    data['target'] = [1 if x == target_position else 0 for x in data['general_position']]

    

    data = data.drop(['short_name', 'player_positions', 'main_position', 'general_position'], axis = 1)

    

    X_train, X_test, y_train, y_test = train_test_split(data.drop(['target'], axis = 1), data['target'], test_size=0.2, random_state=20)

    

    feature_names = np.array(X_train.columns.values)

    

    log_model = LogisticRegression()

    log_model.fit(X_train, y_train)

    

    n_feat = 5

    

    pos_inds = log_model.coef_[0].argsort()[-n_feat:][::-1]

    neg_inds = log_model.coef_[0].argsort()[:n_feat]

    

    print('Attributes with most Positive Impact:\n', feature_names[pos_inds])

    print('\nAttributes with most Negative Impact:\n', feature_names[neg_inds])    

    

    

    #Test

    y_pred = log_model.predict(X_test)

    print("\n\nTest Accuracy: ", round(accuracy_score(y_test, y_pred), 2))

    print("Test Recall: ", round(recall_score(y_test, y_pred), 2))

    





main_attributes_position(stats_data.copy(), 'DEF')


def positions_distribution(data, target_positions):

    

    data = data.iloc[0:5000, :]

    

    data = data[data['general_position'].isin(target_positions)]

    

    

    pos_dict = {

        'GK': 0,

        'DEF': 1,

        'MED': 2,

        'ATA': 3

    }

    

    data = data.sample(frac=1)

    target = data['general_position'].map(pos_dict).values

    

    data_attributes = data.drop(['short_name', 'player_positions', 'main_position', 'general_position'], axis = 1)

    

    pca_model = PCA(n_components=2)

    reduced_data = pca_model.fit_transform(data_attributes)

    

    colors = ListedColormap(['grey', 'tab:red', 'tab:olive', 'tab:blue'])

    

    fig, ax = plt.subplots(figsize=(12, 10))

    #ax = fig.add_subplot(111, projection='3d')

    

    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=target, cmap=colors)

    

    plt.legend(handles=scatter.legend_elements()[0], labels=target_positions)

    plt.xlabel('Component 1')

    plt.ylabel('Component 2')

    plt.title('Position Distribution: ' + ', '.join(target_positions))

    

    plt.show()

    

    

positions_distribution(stats_data.copy(), ['GK', 'DEF', 'MED', 'ATA'])

positions_distribution(stats_data.copy(), ['DEF', 'MED', 'ATA'])

    
def all_positions_distribution(data):

    

    data = data.iloc[0:5000, :]

    data = data[data['main_position'] != 'GK']

    

    pos_dict = {}

    

    l = ['CB', 'LB', 'RB', 'LWB', 'RWB', 'CMD', 'CM', 'CAM', 'RM', 'LM', 'RW', 'LW', 'CF', 'ST']



    for i, p in enumerate(l):

        pos_dict[p] = i

    

    data = data.sample(frac=1)

    target = data['main_position'].map(pos_dict).values

    

    data_attributes = data.drop(['short_name', 'player_positions', 'main_position', 'general_position'], axis = 1)

    

    pca_model = PCA(n_components=2)

    reduced_data = pca_model.fit_transform(data_attributes)

    

    colors = plt.get_cmap('viridis_r')

    

    fig = plt.figure(figsize=(18, 12))

    

    

    for i, p in enumerate(l):

        c = colors(i/len(l))

        inds = [j for j, x in enumerate(data['main_position']) if x == p]

        plt.scatter(reduced_data[inds, 0], reduced_data[inds, 1], label=p, color=c)

    

    

    plt.legend(loc='upper right')

    plt.xlabel('Component 1')

    plt.ylabel('Component 2')

    plt.title('Position Distribution: ' + ', '.join(l))

    

    plt.show()

    

all_positions_distribution(stats_data.copy())


def best_position(data):

    

    data = data.iloc[0:1000, :]



    X_train, X_test, y_train, y_test = train_test_split(data.drop(['main_position'], axis = 1), data['main_position'], test_size=0.2, random_state=8)

    

    n_test = 5



    X_train = X_train.drop(['short_name', 'player_positions', 'general_position'], axis = 1)



    test_names = X_test['short_name'].values

    X_test = X_test.drop(['short_name', 'player_positions', 'general_position'], axis = 1)

    

    

    k = 5

    knn_model = KNeighborsClassifier(n_neighbors=k)

    knn_model.fit(X_train, y_train)



    

    #Test

    y_pred = knn_model.predict(X_test)

    

    conf_matrix = confusion_matrix(y_test, y_pred, labels=y_test.unique())

    

    print("Confusion Matrix:")

    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(10, 10))

    x = np.arange(0, 14)

    my_xticks = y_test.unique()

    plt.xticks(x, my_xticks)

    plt.yticks(x, my_xticks)

    plt.show()

    

    print("\n{} Examples: ".format(n_test))

    for i in range(n_test):

        pred_pos = knn_model.predict(X_test.iloc[[i]])[0]

        

        print('Name: {}\nPrediction: {}\n True Pos: {}\n'.format(test_names[i], pred_pos, y_test.iloc[i]))

    

    

    

best_position(stats_data.copy())







def init_population(data, n, positions_target):



    population = []

    for _ in range(n):

        indv = []

        for j in positions_target:

            p = data[data['main_position'] == j].sample(1)

            indv.append(p.iloc[0].short_name)

        population.append(indv)



    return population

    







#One point Crossover

def crossover(indv_1, indv_2):



    if random.random() < prob_crossover:



        pos = random.randint(0,len(indv_1))

        f1 = indv_1[:pos] + indv_2[pos:]

        f2 = indv_2[:pos] + indv_1[pos:]

        return [f1, f2]



    else:

        return [indv_1,indv_2]





def random_mutation(indv, positions, data):

    new_indv = indv.copy()



    for i in range(len(indv)):

        if  random.random() < prob_mutation:

            p = data[data['main_position'] == positions[i]].sample(1)



            new_indv[i] = p.iloc[0]['short_name']



    return new_indv





def evaluate_population(original_data, population, budget):

    alpha = 0.8

    beta = 0.2

    scores = []

    for indv in population:



        accumulated_value = 0

        ages = 0

        for i in indv:

            p = original_data[original_data['short_name'] == i]

            

            accumulated_value += p.iloc[0].value_eur + 0.2 * p.iloc[0].value_eur

            ages += p.iloc[0].age

        

        

        scores.append(round(alpha * np.abs(budget - accumulated_value) + beta * np.mean(ages), 2))



    return scores



def indv_value(indv, data):



    accumulated_value = 0

    for i in indv:

        p = data[data['short_name'] == i]

        accumulated_value += p.iloc[0].value_eur

        

    return accumulated_value



def argsort_list(seq):

    return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]





def elitism(original_data, population, offspring, budget):

    

    scores_pop = evaluate_population(original_data, population, budget)

    scores_offs = evaluate_population(original_data, offspring, budget)

    

    comp_elite = int(len(population) * elitism_size)

    

    population = [population[ind] for ind in argsort_list(scores_pop)]

    offspring = [offspring[ind] for ind in argsort_list(scores_offs)]

    

    new_population = population[:comp_elite] + offspring[:len(population) - comp_elite]



    return new_population





def evolve(data, target_positions, budget):



    num_iters = 80



    scores = np.zeros(num_iters)

    best_indv = None

    best_score = float('+inf')



    population = init_population(data, n_pop, target_positions)



    #print("Initial Population: ", population)

    

    for it in range(num_iters):



        pool = population

        

        #Crossover

        parents = []

        for i in  range(0, n_pop - 1, 2):

            indiv_1= pool[i]

            indiv_2 = pool[i+1]

            parents.extend(crossover(indiv_1, indiv_2)) 

        

        #Mutation

        offspring = []

        for indv in parents:

            offspring.append(random_mutation(indv, target_positions, data))

        

        #Select the best inviduals: Elitism

        population = elitism(fifa_20, population, offspring, budget)

        pop_scores = evaluate_population(data, population, budget)

                

        for i in range(len(pop_scores)):



            if pop_scores[i] < best_score:

                best_indv = pop_scores[i]

                best_players = population[i]

                best_score = pop_scores[i]



            scores[it] = np.mean([pop_scores[s] for s in range(len(pop_scores))])



        if it % 5 == 0:

            print("Iteration ", it, ": Average Score:", round(scores[it], 3), "Best so far:", round(best_score, 3))



        

    plt.figure(figsize=(12, 8))

    plt.plot(scores)

    plt.xlabel('Iterations')

    plt.ylabel('Score')

    plt.title('Score Evolution')

    plt.show()

    

    print('Best players for {}:'.format(target_positions))



    for i in best_players:

        p = data[data['short_name'] == i]

        print('{}:\n\tAge: {}, Overall: {}'.format(i, p.iloc[0].age, p.iloc[0].overall))

            

    print('\nMax Budget: ', budget)

    print('Budget Used: ', indv_value(best_players, data))





target_pos = ['ST', 'CM', 'GK']

b = 100000000

n_pop = 30

prob_mutation = 0.2

prob_crossover = 0.8

elitism_size = 0.6

evolve(fifa_20, target_pos, b)
