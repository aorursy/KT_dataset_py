import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from plotnine import *



import time



import numpy as np

from abc import ABC, abstractmethod



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier



# import warnings

# warnings.filterwarnings('ignore')
train = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")

test = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")



def split_mnist_data(df):

    df = df[(df.label==0) | (df.label==1)]

    X, y = df.drop("label", axis=1), df['label']

    return X.values,y.values



x_train, y_train = split_mnist_data(train)

x_test, y_test = split_mnist_data(test)
ones_train = x_train[y_train == 1]

zeros_train = x_train[y_train == 0]

N=5

fig, axes = plt.subplots(2, N, sharex=True, sharey=True) #create subplots

for i, ax in enumerate(axes.flatten()): 

    if i >= N: # if image is in the second row

        ax.imshow(zeros_train[i].reshape(28,28))

    else:

        ax.imshow(ones_train[i].reshape(28,28))



plt.suptitle("Drawing {} samples from data".format(N)) #add super title

plt.show()
class Classifier(ABC):



    def score(self, X, y):

        """

        :param X: X data matrix 

        :param y: response vector

        :return: dictionary of different classification scores

        """

        # predict classes

        pred = self.predict(X)

        

        #Positive classified samples

        P = (pred > 0).sum()

        # Negative classified samples

        N = (pred <= 0).sum()

        

        #True Negative

        TN = ((pred <= 0) & (y <= 0)).sum()

        FP = ((pred > 0) & (y <= 0)).sum()



        # True Positive

        TP = ((pred > 0) & (y > 0)).sum()

        

        # True Negative

        FN = ((pred <= 0) & (y > 0)).sum()

    

        acc = (TP + TN) / (P + N) # Accuracy



        return acc



    @abstractmethod

    def predict(self, X):

        pass



    @abstractmethod

    def fit(self, X, y):

        pass


class Perceptron(Classifier):



    def __init__(self):

        self.model = None

        self.name = "Perceptron"



    def fit(self, X, y_in):

        """

        fit the perceptron w vector.

        :param X: X data matrix

        :param y_in: response input vector of 0 and 1

        """

        

        # convert negative labels from 0 to -1 

        y = np.where(y_in > 0, 1, -1)

        

        # number of parameters, observations

        p, m = X.shape[1] + 1, X.shape[0]

        

        # initialize w vector

        self.model = np.zeros(p)



        X = np.column_stack([np.ones(m), X])

        

        # while exist i s.t (y_i*<w, x_i>) <=0

        while True:

            z = (X @ self.model)

            scores = y * z

            if (scores <= 0).any():

                # get the first i s.t. (y_i*<w, x_i>) <=0

                i = np.argmax(scores <= 0)

                # update perceptron

                self.model += y[i] * X[i, :]

            else:

                break



    def predict(self, X):

        # number of parameters, observations

        p, m = X.shape[1] + 1, X.shape[0]

        X = np.column_stack([np.ones(m), X])

        predictions =  np.sign(X @ self.model)

        

        y_hat = np.where(predictions > 0, 1, 0)

        return y_hat

class LDA(Classifier):



    def __init__(self):

        self.name = "LDA"

        self.NEG, self.POS = 0, 1

        self.pr_pos, self.pr_neg = None, None

        self.mu_pos, self.mu_neg = None, None

        self.cov_inv = None



    def fit(self, X, y): 

        """

        calculate LDA parameters by the training data

        :param X: X data matrix

        :param y_in: response input vector

        """

        pr_y = (y > 0).mean()

        self.pr_pos, self.pr_neg = pr_y, 1 - pr_y

       

        self.mu_pos = X[y > 0,:].mean(axis=0)

        self.mu_neg = X[ y <= 0,:].mean(axis=0)

        self.cov_inv = np.linalg.pinv(np.cov(X.T))



    def predict(self, X):

        """

        classify by the LDA decision rule

        :param X: X data matrix

        return: y - predicted values vector

        """

        

        mu = self.mu_pos

        

        d1 = (X @ self.cov_inv @ mu) - 0.5 * (mu.T @ self.cov_inv @ mu) + np.log(self.pr_pos)

        mu = self.mu_neg

        d2 = (X @ self.cov_inv @ mu) - 0.5 * (mu.T @ self.cov_inv @ mu) + np.log(self.pr_neg)

        y = []

        for i in range(len(d1)):

            if d1[i] > d2[i]:

                y.append(self.POS)

            else:

                y.append(self.NEG)

        return np.array(y)



def rearrange_data(X):

    """

    :param X: matrix from size m x N x N

    :return: matrix from shape m x N^2

    """

    m = X.shape[0]

    return X.reshape(m, -1)



def draw_m_random_points(m, X, y):

    """

    :param m: number of observations we draw

    :param X: train matrix X from shape q x N x N

    :param y: np vector from size q x 1    

    :return: X_train_m - reshaped matrix (size m X N^2) of random choice from X

             y_train_m - random choice from y (matched to X_train_m)

    """

    #get random selection of indices without return

    rand = np.random.choice(len(y), m, replace=False)

    

    #slice data by random indices

    X_train_m, y_train_m = X[rand], y[rand]

    

    #check if we have at least one observation from each label

    while (y_train_m == 0).sum() == 0 or (y_train_m == 1).sum() == 0:

        rand = np.random.choice(len(y), 3, replace=False)

        X_train_m, y_train_m = X[rand], y[rand]

    

    #rearranging X random data

    X_train_m = rearrange_data(X_train_m)

    

    return X_train_m, y_train_m



#create classifier objects

logistic, svm, tree, lda, perceptron = LogisticRegression(max_iter=2000), SVC(C=1e-3, kernel='linear'), DecisionTreeClassifier(max_depth=15),  LDA(), Perceptron()



#create classifiers array

classifiers = [logistic, svm, tree, lda, perceptron]



#create classifiers names array

names = ["Logistic", "SVM", "TREE", "LDA", "Perceptron"]



X_test = rearrange_data(x_test)
def classifier_score(c, X_train, y_train, m, X_test,y_test, name,ITER=50):

    """

    This function compute the mean accuracy for a classifier c for ITER iterations

    :param m: number of observations we draw on each iteration

    :param X_train: Train matrix X

    :param y_train: Training labels vector y

    :param X_test: Test matrix X

    :param y_test: Test labels vector y

    :return: mean accuracy of the classifier

    """

    acc_lst = []

    fit_time_lst = []

    score_time_lst = []

    

    for i in range(ITER):

        X_train_m, y_train_m = draw_m_random_points(m, X_train,y_train)

        s_fit = time.time()

        c.fit(X_train_m,y_train_m)

        end_fit = time.time()

        s_score = time.time()

        acc = c.score(X_test, y_test)

        end_score = time.time()

        fit_time = (end_fit-s_fit)

        score_time = (end_score-s_score)   



        acc_lst.append(acc)

        fit_time_lst.append(fit_time)

        score_time_lst.append(score_time)

        try:

            temp = round(np.mean(acc_lst),3)

        except:

            print(acc_lst)



    

    return round(np.mean(acc_lst),6), round(np.mean(fit_time_lst),3), round(np.mean(score_time_lst),3)
# list of m sizes of training set

M = [1000,2000,5000,12000]

# create the final scores dict, later will be used to created dataframe.

final_dict = {}

fit_times = {}

score_times = {}

for m in M:

    classifier_dict = {}

    fit_time_dict = {}

    score_time_dict = {}

    for c in range(len(classifiers)):

        score, fit_time, score_time = classifier_score(classifiers[c], x_train, y_train, m, X_test, y_test, name=names[c])

        classifier_dict[names[c]] = score

        fit_time_dict[names[c]] = fit_time

        score_time_dict[names[c]] = score_time



    final_dict[m] = classifier_dict

    fit_times[m] = fit_time_dict

    score_times[m] = score_time_dict
#create a dataframe of our results

df = pd.DataFrame(final_dict)

df_fit = pd.DataFrame(fit_times)

df_score = pd.DataFrame(score_times)

df_score['type'] = "Prediction"

df_fit['type'] = "Training"





df_times = pd.concat([df_fit, df_score]).reset_index().melt(id_vars=['index', 'type'])





#melt dataframe for easy plotting

melted = df.reset_index().melt(id_vars=['index'])


(ggplot(melted) + geom_line(aes(x='variable',y='value',color='index', group='index')) 

 + labs(x='Number of Train Observations', y='Accuracy', title='Accuracy vs. Train Set Size - Classification') 

 + scale_color_discrete(name='Classifier')).draw();

(ggplot(df_times) + geom_line(aes(x='variable',y='value',color='index', group='index')) + facet_grid("type~", scales='free_y')

 + labs(x='Number of Train Observations', y='TIme (Seconds)', title='Time vs. Test Sample Size - Classification') 

 + scale_color_discrete(name='Classifier') + theme_minimal() 

 + theme(figure_size=(7,6), 

         axis_title_y = element_text(margin={'r': 30}),

        strip_text_y = element_text(size = 14),

        panel_spacing=.5)).draw();
