import numpy as np

import pandas as pd
def get_answer(visits):

    probs = np.zeros(7, dtype=np.float64)

    visits = (np.array([int(s) for s in visits.split(' ') if s != '']) - 1) % 7

    values, counts = np.unique(visits, return_counts=True)

    for i in range(values.size):

        probs[values[i]] = counts[i]

    probs /= visits.size

    

    weights = np.empty(7, dtype=np.float64)

    for i in range(7):

        weights[i] = np.prod((1 - probs)[:i]) * probs[i]

    return weights.argmax() + 1
data = pd.read_csv('train.csv')
pd.DataFrame(data={

    'id': data['id'],

    'nextvisit': data['visits'].apply(get_answer).apply(lambda x: ' ' + str(x)),

}).to_csv('result.csv', index=False)