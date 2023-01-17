import numpy as np

import pandas as pd

from collections import Counter

from operator import itemgetter



class KNN:

    

    def __init__(self, X, y, K = None, inverse_weighted=False, l2_norm = True):

        self.X = np.array(X)

        self.y = y

        self.K = K

        if (self.K is None):

            self.K = int(np.floor(np.sqrt(self.X.shape[0])))

        self._inverse_weighted = inverse_weighted

        self.l2_norm = l2_norm

        

    def _euclidean(self, x1, x2):

        return np.sqrt(np.dot(x1 - x2, x1 - x2))

    

    def _manhattan(self, x1, x2):

        return np.sum(np.absolute(x1 - x2))

    

    def _distances(self, x):

        distances = []

        for index, row in enumerate(self.X):

            distances.append((index, self._euclidean(x, row) if self.l2_norm else self._manhattan(x, row)))

        return distances



    def _predict_instance_non_weighted(self, x):

        distances = self._distances(x)

        votes = [x[0] for x in sorted(distances, key=lambda y: y[1])][:self.K]

        cntr = Counter(votes)

        return y[cntr.most_common(1)[0][0]]

                          

    def _predict_instance_weighted(self, new_data):

        distances = self._distances(new_data)

        epsilon = 1e-5

        cw = [(self.y[t[0]], 1/(t[1] + epsilon)) for t in sorted(distances, key=lambda x: x[1])[:self.K]]

        vote_dict = {}

        for c, w in cw:

            if c in vote_dict:

                vote_dict[c] += w

            else:

                vote_dict[c] = w

        return max(vote_dict.items(), key=itemgetter(1))[0]

    

    def predict(self, new_data):

        new_data = np.array(new_data)

        if new_data.ndim == 1:

            if self._inverse_weighted:

                return self._predict_instance_weighted(new_data)

            else:

                return self._predict_instance_non_weighted(new_data)

        else:

            predictions = []

            for x in new_data:

                if self._inverse_weighted:

                    predictions.append(self._predict_instance_weighted(x))

                else:

                    predictions.append(self._predict_instance_non_weighted(x))

            return predictions

  

# if __name__ == "__main__":

    

#     iris = pd.read_csv("iris.csv")



#     X = iris.iloc[:, :4]

#     y = iris["class"]

    

#     knn = KNN(X=X, y=y, inverse_weighted=False, l2_norm=True)

#     print("Automatically calculated K: ", knn.K)

#     accuracy = sum(knn.predict(X) == y)/len(y)

#     print("Accuracy on entire dataset: {:.0%}".format(accuracy))