import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_20.csv")

df = df.drop(columns=df.columns[df.isna().any()].tolist())

df = df.drop(columns=['sofifa_id','short_name','player_url','player_positions','real_face'])
df.head()
from sklearn.preprocessing import LabelEncoder



le_club = LabelEncoder()

le_nation = LabelEncoder()

le_body = LabelEncoder()

le_foot = LabelEncoder()

le_work_rate = LabelEncoder()





df["club"] = le_club.fit_transform(df["club"])

df["nationality"] = le_nation.fit_transform(df["nationality"])

df["body_type"] = le_body.fit_transform(df["body_type"])

df["preferred_foot"] = le_foot.fit_transform(df["preferred_foot"])

df["work_rate"] = le_work_rate.fit_transform(df["work_rate"])



df['dob'] = pd.to_datetime(df['dob']).dt.strftime("%Y").astype(int)



labels = df['long_name']

df = df.drop(columns=['long_name'])
NUMBER_OF_EXAMPLES = df.shape[0]

NUMBER_OF_ATTRIBUTES = df.shape[1]



LATTICE_SIZE = (10,10)

MAX_NUMBER_OF_ITERATIONS = (LATTICE_SIZE[0]) * (LATTICE_SIZE[1])

LEARNING_RATE_INIT = 0.01

SIGMA_INIT = 0.01



LEARNING_CONSTANT = 1000

NEIGHBOURHOOD_CONSTANT = 1000
# Initializtion and normalization



from sklearn import preprocessing



X = df.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

X = min_max_scaler.fit_transform(X)



weights = np.random.random((*LATTICE_SIZE,NUMBER_OF_ATTRIBUTES))

# Helper Function

def find_neighbours(i,j):

    neighbour_list = []

    

    try:

        _ =  weights[i][j+1]

        neighbour_list.append((i,j+1))

    except IndexError:

        pass

    

    try:

        _ =  weights[i][j-1]

        neighbour_list.append((i,j-1))

    except IndexError:

        pass

    

    try:

        _ =  weights[i+1][j]

        neighbour_list.append((i+1,j))

    except IndexError:

        pass

    

    try:

        _ =  weights[i-1][j]

        neighbour_list.append((i-1,j))

    except IndexError:

        pass

    

    return neighbour_list


def find_winner(x):

    distance = []

    for p in range(LATTICE_SIZE[0]):

        tmp = []

        for q in range(LATTICE_SIZE[1]):

            dist = np.sqrt(np.sum(np.square(weights[p][q] - x)))

            tmp.append(dist)

        distance.append(tmp)

        

    winner_neuron =  np.where(distance == np.min(distance))

    winner_neuron = winner_neuron[0][0],winner_neuron[1][0]

    

    neighbours =  find_neighbours(*winner_neuron)

    

    return winner_neuron, neighbours,distance[winner_neuron[0]][winner_neuron[1]]

    



def update_weights(x,winner_neuron, neighbours,n):

    learning_rate =LEARNING_RATE_INIT  * np.exp(-n/LEARNING_CONSTANT)

    weights[winner_neuron[0]][winner_neuron[1]] += learning_rate*(x - weights[winner_neuron[0]][winner_neuron[1]])

    

    for n_x,n_y in neighbours:

        lateral_dist = weights[winner_neuron[0]][winner_neuron[1]] -weights[n_x][n_y]

        sigma = SIGMA_INIT * np.exp(-n/NEIGHBOURHOOD_CONSTANT)

        tropo_dist =np.exp( -(lateral_dist**2) /(2* (sigma**2)))

        

        

        

        weights[n_x][n_y] += learning_rate*tropo_dist*(x - weights[n_x][n_y])    
# Driver Code





for n in range(MAX_NUMBER_OF_ITERATIONS):

    total_distance = 0

    for i in range(NUMBER_OF_EXAMPLES):

        winner, neighbours, distance = find_winner(X[i])

        update_weights(X[i],winner, neighbours,n)

        

        total_distance += distance

    

    print(f"{n+1} : {total_distance}")



    
cluster_density = np.zeros(LATTICE_SIZE)
for i in range(NUMBER_OF_EXAMPLES):

    winner, _, _ = find_winner(X[i])

    cluster_density[winner[0]][winner[1]] +=1
cluster_density
[val for x in cluster_density for val in x  if val >=8]