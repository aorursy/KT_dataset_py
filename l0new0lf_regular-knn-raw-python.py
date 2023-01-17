Xs = [

        [0], 

        [1], 

        [2], 

        [3],

        [4]

]

ys = [0, 0, 1, 1, 2]



from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(Xs, ys)



print(neigh.predict([[4.1]])) #close to target `2`
neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')

neigh.fit(Xs, ys)



print(neigh.predict([[4.1]])) #close to target `2`
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.model_selection import train_test_split as split
iris = datasets.load_iris()

X = iris.data

y = iris.target



X_train, X_test, y_train, y_test = split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
class Distances:

    @staticmethod

    def norm(X, p):

        """

        X : 1-D Vector (np.ndarray)

        P : Int (>= 1)

        

        Returns L2 Norm of 1-D Vector (np.ndarray)

        """

        scalar = np.sum(np.power(X, p))

        return np.power(scalar, (1/p))

    

    @staticmethod

    def euclidean(X1, X2):

        """

        X1 - 1-D vector of len `m dims` (np.ndarray)

        X2 - 1-D vector of len `m dims` (np.ndarray)

        """

        if (X1.shape != X2.shape): 

            raise Exception('X1 and X2 must be 1-D vecs of same dims')

        

        # calculate diffs

        X2_minus_X1 = X2-X1

        # calculate L2 norm and return

        return Distances.norm(X2_minus_X1, p=2)

    

    @staticmethod

    def mse(X1, X2):

        """

        X1 - 1-D vector of len `m dims` (list / np.ndarray)

        X2 - 1-D vector of len `m dims` (list / np.ndarray)

        """

        # validate

        if (len(X1) != len(X2)): 

            raise Exception('X1 and X2 must be 1-D vecs of same dims')

            

        # claculate diffs

        X2_minus_X1 = X2-X1

        # return loss

        dists = Distances.norm(X2_minus_X1, p=2)

        # return mse (l2 square)

        return (1/len(X1)) * np.power(dists, 2)
class KNN:

    """

    TIME  : ~ O(mn)

    SPACE : ~ O(mn) whole training set needed!

    """

    def __init__(self, k=5, distance_metric=Distances.euclidean):

        self.k = k

        self.dist = distance_metric

    

    def fit(self, X_train, y_train):

        """ Simply store 

        

        SPACE: O(mn)

        """

        self.X_train = X_train

        self.y_train = y_train

    

    def predict(self, X_test):

        y_pred = [self._predict(x_q) for x_q in X_test]

        return y_pred

    

    def _predict(self, x_q):

        # 1. get sorted distances and their idxs

        asc_sorted_dist_idxs, asc_sorted_dists = self.__get_sorted_distance_idxs(x_q)

        # 2. get top k classes w/ least distances

        top_k_ys, top_k_wts = self.__get_top_k_nearest_classes_w_wts(asc_sorted_dist_idxs, asc_sorted_dists)

        # 3. majority vote from top_k_ys

        pred_class = self.__majority_vote(top_k_ys, top_k_wts, n_top=1)

        return pred_class[0]

    

    # ==================================================

    # Helpers start

    # ==================================================    

    def __get_sorted_distance_idxs(self, x_q):

        """ Returns indices of `n` sorted distances 

        

        TIME  : O(nm)

        """

        # validate

        if (x_q.shape != self.X_train[0].shape): raise Exception('x_q and samples of X must be vecs of same dims')    

        

        # calculate distances

        dists = np.array([self.dist(x_q, x_i) for x_i in self.X_train])

        asc_idxs = np.argsort(dists) # O(nlogn). Could be made 0(1) by tracking the smallest in loop above

        return asc_idxs, dists[asc_idxs]

            

    def __get_top_k_nearest_classes_w_wts(self, asc_dist_idxs, asc_sorted_dists):

        """ 

        + Returns `k` class labels from which majority vote is taken 

        + `y` must have numeric class labels starting from `0`

        """

        top_k_ys     = self.y_train[asc_dist_idxs][:self.k]

        top_k_dists  = asc_sorted_dists[:self.k]

        top_k_wts    = np.array([(1/dist) for dist in top_k_dists])

        return top_k_ys, top_k_wts

        

    @staticmethod

    def __majority_vote(top_k_ys, top_k_wts, n_top=1):

        """ `top_k_ys` must be a subset of numeric class labels starting from `0` """

        labels, cnts = np.unique(top_k_ys, return_counts=True)

        # descending (need highest)

        desc_cnts_idxs = np.argsort(cnts)[::-1]

        # return mostly occuring label(s)

        return labels[desc_cnts_idxs][:n_top]

        

    # ==================================================

    # Helpers end

    # ==================================================

def accuracy(Y1, Y2):

    """ 

    Y1 and Y2 are 1-D vecs of same dims

    of class labels in numeric representation (whole numbers) 

    """

    # validate

    if len(Y1) != len(Y2): raise Exception("Y1 and Y2 are 1-D vecs of same dims")

        

    true_if_same_else_false  = (Y1 == Y2)

    num_correct              = np.sum(true_if_same_else_false)

    percent_of_correct       = num_correct / len(Y1) 

    

    return percent_of_correct
for k in range(1, 15):



    # 1. initialize

    clf = KNN(k=k, distance_metric=Distances.euclidean)

    

    # 2. Train (simply store)

    clf.fit(X_train, y_train)

    

    # 3. get preds

    y_pred = clf.predict(X_test)

    

    # check results and print

    print(f"k: {k} \tacc: {accuracy(y_pred, y_test)}")
class KNNRegression(KNN):

    """

    Override `__majority_vote` with averages/medians

    """

    @staticmethod

    def __majority_vote(top_k_ys, top_k_wts, n_top=1):

        """

        `top_k_ys` ~ Continous rvs unlike numeric in classification above

        """

        return np.mean(top_k_ys) # use np.median (robust to outliers)
np.random.seed(101)

er = np.random.normal(0, 10, 100)

xs = np.arange(0, 100, 1)

ys = 3*xs + 4 + er



plt.scatter(xs, ys)

plt.show()
X_train, X_test, y_train, y_test = split(xs, ys, test_size=0.33, random_state=42)



X_train  = X_train.reshape(len(X_train), 1)

X_test   = X_test.reshape(len(X_test), 1)
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
for k in range(1, 15):



    # 1. initialize

    clf = KNNRegression(k=k, distance_metric=Distances.euclidean)

    

    # 2. Train (simply store)

    clf.fit(X_train, y_train)

    

    # 3. get preds

    y_pred = clf.predict(X_test)

    

    # check results and print

    print(f"k: {k} \terror: {Distances.mse(y_pred, y_test)}")
# Train w/ k=1 (selected by hyperparam tuning)



clf = KNNRegression(k=2, distance_metric=Distances.euclidean)

clf.fit(X_train, y_train)

y_test_preds = clf.predict(X_test)
plt.figure(figsize=(10, 5))



plt.scatter(np.squeeze(X_train), y_train, label="Train set")

plt.scatter(np.squeeze(X_test), y_test, label="Ground truth")

plt.scatter(np.squeeze(X_test), y_test_preds, label="Predictions")



plt.legend()

plt.show()
class WeightedKNN(KNN):

    """

    Override `__majority_vote`

    """

    @staticmethod

    def __majority_vote(top_k_ys, top_k_wts, n_top=1):

        """

        `top_k_ys` ~ Continous rvs unlike numeric in classification above

        `top_k_wts` as in table above (1/dists)

        """

        # descending (need highest)

        desc_idxs = np.argsort(top_k_wts)[::-1]

        

        return top_k_ys[desc_idxs][:n_top]
iris = datasets.load_iris()

X = iris.data

y = iris.target



X_train, X_test, y_train, y_test = split(X, y, test_size=0.33, random_state=42)
ks     = []

accs   = {

    'train' : [],

    'cv'    : []

}

for k in range(1, 150):

    if (k%5 == 0):

        # 1. initialize

        clf = WeightedKNN(k=k, distance_metric=Distances.euclidean)



        # 2. Train (simply store)

        clf.fit(X_train, y_train)



        # 3. get preds and acc for train and c.v

        # train

        y_pred_train   = clf.predict(X_train)

        acc_train      = accuracy(y_pred_train, y_train)

        # cross-val

        y_pred_cv      = clf.predict(X_test)

        acc_cv         = accuracy(y_pred_cv, y_test)

        

        # check results and print

        print(f"k: {k} \ttrain_acc: {acc_train} \tcv_acc: {acc_cv}")



        # history (for plots)

        ks.append(k)

        accs['train'].append(acc_train)

        accs['cv'].append(acc_cv)
plt.plot(ks, accs['train'], label="Train Acc")

plt.plot(ks, accs['cv'], label="Validation Acc")

plt.plot(

    ks, 

    [1 - acc for acc in accs['train']], 

    label="Train error"

)

plt.plot(

    ks, 

    [1 - acc for acc in accs['cv']], 

    label="Validation error"

)



plt.xlabel("k")

plt.ylabel("accuracy / error")

plt.title("Hyperparameter Tuning")

plt.legend()

plt.grid()

plt.show()