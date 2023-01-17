import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
# test data
tmpX1 = np.array([int(i.strip()) for i in "2   0   0   0   1   2   3   1  0   0   1   0   2   1   0   0  0   1   0   1   0   2   1   0  1   0   0   2   0   1   0   1  2   0   0   0   1   0   1   3  0   0   1   2   0   0   2   1".split("  ")])
tmpX2 = np.array([int(i.strip()) for i in "0   1   1   0   0   0   1   0  1   2   0   1   0   0   1   1  0   1   1   0   0   2   0   0  0   0   0   0   0   0   0   0  0   0   1   0   1   0   1   0".split("  ")])
X = np.concatenate((tmpX1.reshape(-1,8), tmpX2.reshape(-1,8)), axis=0)
y = np.array([0,0,0,0,0,0,1,1,1,1,1])
X_test = np.array([[2,1,0,0,1,2,0,1],[0,1,1,0,1,0,1,0]])
y_test = np.array([0,1])
print("X and Y shapes\n", X.shape, y.shape)
class MultiNB:
    def __init__(self,alpha=1):
        self.alpha = alpha
    
    def _prior(self): # CHECKED
        """
        Calculates prior for each unique class in y. P(y)
        """
        P = np.zeros((self.n_classes_))
        _, self.dist = np.unique(self.y,return_counts=True)
        for i in range(self.classes_.shape[0]):
            P[i] = self.dist[i] / self.n_samples
        return P
            
    def fit(self, X, y): # CHECKED, matches with sklearn
        """
        Calculates the following things- 
            class_priors_ is list of priors for each y.
            N_yi: 2D array. Contains for each class in y, the number of time each feature i appears under y.
            N_y: 1D array. Contains for each class in y, the number of all features appear under y.
            
        params
        ------
        X: 2D array. shape(n_samples, n_features)
            Multinomial data
        y: 1D array. shape(n_samples,). Labels must be encoded to integers.
        """
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_priors_ = self._prior()
        
        # distinct values in each features
        self.uniques = []
        for i in range(self.n_features):
            tmp = np.unique(X[:,i])
            self.uniques.append( tmp )
            
        self.N_yi = np.zeros((self.n_classes_, self.n_features)) # feature count
        self.N_y = np.zeros((self.n_classes_)) # total count 
        for i in self.classes_: # x axis
            indices = np.argwhere(self.y==i).flatten()
            columnwise_sum = []
            for j in range(self.n_features): # y axis
                columnwise_sum.append(np.sum(X[indices,j]))
                
            self.N_yi[i] = columnwise_sum # 2d
            self.N_y[i] = np.sum(columnwise_sum) # 1d
            
    def _theta(self, x_i, i, h):
        """
        Calculates theta_yi. aka P(xi | y) using eqn(1) in the notebook.
        
        params
        ------
        x_i: int. 
            feature x_i
            
        i: int.
            feature index. 
            
        h: int or string.
            a class in y
        
        returns
        -------
        theta_yi: P(xi | y)
        """
        
        Nyi = self.N_yi[h,i]
        Ny  = self.N_y[h]
        
        numerator = Nyi + self.alpha
        denominator = Ny + (self.alpha * self.n_features)
        
        return  (numerator / denominator)**x_i
    
    def _likelyhood(self, x, h):
        """
        Calculates P(E|H) = P(E1|H) * P(E2|H) .. * P(En|H).
        
        params
        ------
        x: array. shape(n_features,)
            a row of data.
        h: int. 
            a class in y
        """
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self._theta(x[i], i,h))
        
        return np.prod(tmp)
    
    def predict(self, X):
        samples, features = X.shape
        self.predict_proba = np.zeros((samples,self.n_classes_))
        
        for i in range(X.shape[0]):
            joint_likelyhood = np.zeros((self.n_classes_))
            
            for h in range(self.n_classes_):
                joint_likelyhood[h]  = self.class_priors_[h] * self._likelyhood(X[i],h) # P(y) P(X|y) 
                
            denominator = np.sum(joint_likelyhood)
            
            for h in range(self.n_classes_):
                numerator = joint_likelyhood[h]
                self.predict_proba[i,h] = (numerator / denominator)
            
        indices = np.argmax(self.predict_proba,axis=1)
        return self.classes_[indices]
def pipeline(X,y,X_test, y_test, alpha):
    """
    Sklearn Sanity Check
    """
    print("-"*20,'Sklearn',"-"*20)
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X,y)
    sk_y = clf.predict(X_test)
    print("Feature Count \n",clf.feature_count_)
    print("Class Log Prior ",clf.class_log_prior_)
    print('Accuracy ',accuracy_score(y_test, sk_y),sk_y)
    print(clf.predict_proba(X_test))
    print("-"*20,'Custom',"-"*20)
    nb = MultiNB(alpha=alpha)
    nb.fit(X,y)
    yhat = nb.predict(X_test)
    me_score = accuracy_score(y_test, yhat)
    print("Feature Count\n",nb.N_yi)
    print("Class Log Prior ",np.log(nb.class_priors_))
    print('Accuracy ',me_score,yhat)
    print(nb.predict_proba) # my predict proba is only for last test set

pipeline(X,y,X,y, alpha=10)
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
df = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='iso8859_14')
df.drop(labels=df.columns[2:],axis=1,inplace=True)
df.columns=['target','text']
def clean_util(text):
    punc_rmv = [char for char in text if char not in string.punctuation]
    punc_rmv = "".join(punc_rmv)
    stopword_rmv = [w.strip().lower() for w in punc_rmv.split() if w.strip().lower() not in stopwords.words('english')]
    
    return " ".join(stopword_rmv)
df['text'] = df['text'].apply(clean_util)
cv = CountVectorizer()
X = cv.fit_transform(df['text']).toarray()
lb = LabelBinarizer()
y = lb.fit_transform(df['target']).ravel()
print(X.shape,y.shape)
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
sk = MultinomialNB().fit(X_train,y_train)
sk.score(X_test,y_test)
%%time
me = MultiNB()
me.fit(X_train, y_train)
yhat = me.predict(X_test)
print(accuracy_score(y_test,yhat))