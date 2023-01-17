import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
FP = "../input/tta/TTA/"
print(os.listdir(FP))
FP = FP + "Journals/"
files = os.listdir(FP)
Journal = pd.read_pickle(FP + files[3])
Journal.tail(10)
Journal.head(10)
Techs = pd.read_pickle("../input/TechXYFullAlt")
print(np.shape(Techs))
Techs.head()
FP = "../input/tta/TTA/NLP/"
summary = pd.read_pickle(FP + "NLP34")
summary.reset_index(drop=True, inplace=True)
d = list(summary.loc[summary["VP"]==0].index) 
# remove all players who resigned since they will have empty "Text" columns later in the game
summary.drop(d, inplace=True)
summary.dropna(inplace=True) # some files were corrupted resuting in empty "Text" columns
print(np.shape(summary))
summary.head()
# bringing in some friends :P
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from sklearn.neural_network import MLPClassifier as MLPC
from scipy.sparse import hstack
from sklearn import svm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neural_network import BernoulliRBM as RBM
from sklearn.preprocessing import StandardScaler
def XY(phases):
    X = []
    d = []
    seg =[]
    for phase in phases:
        x = summary[phase].tolist()
        tfidf = text.TfidfVectorizer(input=x, stop_words="english",
                                    max_df=0.5, min_df=0.05)
        x = tfidf.fit_transform(x)
        X += [x]
        d += list(tfidf.vocabulary_.keys())
        seg += [len(list(tfidf.vocabulary_.keys()))]
    X = hstack(X)  # keeping track of the words we kept
    Y = summary["VP"]/summary["Max VP"]
    Y = Y>Y.median()  
    # this is going to be a simple classifier which predicts
    # whether the % of a player score, with respect to the winner,
    # is among top 50% of all data.
    return X, Y, d, seg
phases = np.arange(6) # get all 6 phases together.
Xnlp, Ynlp, d, seg = XY(phases)
trans = StandardScaler()
X = trans.fit_transform(Xnlp.toarray())
# This rescales the data.
# More importantly, it transform a sparse matrix into 
# a normal matrix, so it won't get rejected by something 
# else down the pipeline

# The rest are optional. I commented them out becuase they did not help.
# But feel free to try them. :P

#trans = TruncatedSVD(n_components=500)
#trans = PCA(n_components=30, whiten=True)
#trans = RBM(n_components=100, learning_rate=0.01)
#X = trans.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Ynlp, test_size=0.2, random_state=10)

#clf = svm.SVC(kernel='linear', probability=False)
clf = MLPC(hidden_layer_sizes=(1,),  
          # more layers did not help, as far as I could tell.
          activation = "identity",
          alpha = 0.001)
clf.fit(X_train, Y_train)

InScore = clf.score(X_train, Y_train)
OutScore = clf.score(X_test, Y_test)
InScore, OutScore
coef = clf.coefs_[0]*clf.coefs_[1] # for MLPC
# a MLPC with 1 neuron in 1 hidden layer is basically 
# a linear classicifier. If you use another linear classifier,
# you need to change the above line into the corresponding 
# coefficients, which should have the same shape.
np.shape(coef)
imp = pd.DataFrame( np.concatenate(( np.swapaxes([d],0,1), 
                                     coef), axis=1), 
                    columns=["word", "weight"] )
phases = sum([ [i]*seg[i] for i in range(6) ], [])
imp["weight"] = imp["weight"].apply(lambda x:float(x))
imp["phase"]=phases
imp.sort_values(["weight"],ascending=False).head(10)
