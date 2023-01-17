import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

import numpy as np



data_frame = pd.read_csv('../input/iris/Iris.csv')
data_frame.head()
labels = list(set(data_frame['Species'].values))

labels_dict = dict(zip([i for i in labels], [i for i in range(len(labels))]))



y = np.array([labels_dict[data_frame.iloc[i]['Species']] for i in range(len(data_frame))], dtype=np.int)



X = data_frame.iloc[:,1:-1].values



train_X, test_X, train_y, test_y = train_test_split(X, y, shuffle=True)
scaler = StandardScaler()



scaler.fit(train_X)



test_X = scaler.transform(test_X)

train_X = scaler.transform(train_X)
N_LAYERS = 3

N_NEURONS_PER_LAYER = 10

MAX_ITER = 600



layers = tuple([N_NEURONS_PER_LAYER for i in range(N_LAYERS)])



model = MLPClassifier(hidden_layer_sizes=layers, max_iter=MAX_ITER)



model.fit(train_X, train_y)
predictions = model.predict(test_X)

n_correct = sum(predictions == test_y)



print(f'Accuracy: {round(100*n_correct/len(test_y))}%')