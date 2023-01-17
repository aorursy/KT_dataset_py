# Numpy & matplotlib for notebooks 
%pylab inline

# Pandas
import pandas as pd # Data analysis and manipulation 

# Sklearn 
from sklearn.preprocessing import StandardScaler # to standardize features by removing the mean and scaling to unit variance (z=(x-u)/s)
from sklearn.neural_network import MLPClassifier # Multi-layer Perceptron classifier which optimizes the log-loss function using LBFGS or sdg.
from sklearn.model_selection import train_test_split # to split arrays or matrices into random train and test subsets
from sklearn.model_selection import KFold # K-Folds cross-validator providing train/test indices to split data in train/test sets.
from sklearn.decomposition import PCA, TruncatedSVD # Principal component analysis (PCA); dimensionality reduction using truncated SVD.
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB # Naive Bayes classifier for multinomial models
from sklearn.feature_extraction.text import CountVectorizer # Convert a collection of text documents to a matrix of token counts
from sklearn.metrics import roc_auc_score as roc # Compute Area Under the Receiver Operating Characteristic Curve from prediction scores
from sklearn.metrics import roc_curve, auc # Compute ROC; Compute Area Under the Curve (AUC) using the trapezoidal rule

# Matplotlib
import matplotlib # Data visualization
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches  

# Seaborn
import seaborn as sns # Statistical data visualization (based on matplotlib)
# Import the test dataset and create a list of authors
test_data = pd.read_csv("../input/final-dataset/test_data.csv", encoding="utf8")

a_test = []
for author, group in test_data.groupby("author"):
    a_test.append(author)

# Load predictions on validation 

# MLP on doc2vec
x1 = np.load("../input/final-dataset/y_scoremlpClf.npy") #y_D2V-mlpClf.npy

# XGB on countvectorized texts
x2 = np.load("../input/final-dataset/y_predict_XGB.npy")

# MLP on binary countvectorized subreddits
x3 = np.load("../input/final-dataset/y_score_MLPs.npy")

# Load predictions of all models
y = np.load("../input/final-dataset/y_valid.npy") # common validation y of previous steps

# Load predicted test doc2vec
t1 = np.load("../input/final-dataset/y_testD2V.npy")

# Load predicted countvectorized test texts
t2 = np.load("../input/final-dataset/y_predict_testXGBnS.npy") #  #y_testXGBnS.npy

# Load predicted countvectorized test subreddits
t3 = np.load("../input/final-dataset/y_testMLPs.npy")
a = np.vstack((x3,x2,x1))

t = np.vstack((t3,t2,t1))

X = a.T # transpose
T = t.T # transpose
# Plot the test data along the 2 dimensions of largest variance
def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        colors = ['orange','blue']
        if plot:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            orange_patch = mpatches.Patch(color='orange', label='M')
            blue_patch = mpatches.Patch(color='blue', label='F')
            plt.legend(handles=[orange_patch, blue_patch], prop={'size': 20})

fig = plt.figure(figsize=(8, 8))          
plot_LSA(X, y)
plt.show()
# Logistic regression 
lrClf = LogisticRegression(class_weight = "balanced",solver = "saga",C = 0.00005)  #modello

# Model fit
lrClf.fit(X, y)
# Final prediction 
y_scorel = lrClf.predict_proba(T)[:,1]

# Create test dictionary 
test = {'author': a_test,
        'gender': y_scorel
        }

# Create DataFrame
df = pd.DataFrame(test, columns = ['author', 'gender'])


# Create submission csv file
df.to_csv(r'../working/Submission.csv', index = False)