import numpy as np # linear algebra
import pandas as pd # data processing


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('../input/titanic/train.csv')
data.head()
data.info()
data.columns
cols_to_drop= ['PassengerId','Name','Ticket','Cabin','Embarked']
clean_data=data.drop(cols_to_drop,axis=1)
clean_data.head()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
clean_data["Sex"]=le.fit_transform(clean_data["Sex"])
clean_data.head()
clean_data=clean_data.fillna(clean_data['Age'].mean())

clean_data.info()
input_cols=['Pclass','Sex','Age','SibSp','Parch','Fare']
out_cols=['Survived']

X=clean_data[input_cols]
y=clean_data[out_cols]

X.shape,y.shape
def entropy(col):
    data,counts =np.unique(col, return_counts=True)
    ## total items are also needed to find the prob
    N= float(col.shape[0])
    
    ent = 0.0
    
    for count in counts:
        p = count / N               ## predicting the Probability
        ent += p* np.log2(p)
        
    return -ent
    
col = np.array([4,4,3,3,2,2,1,2,2])
entropy(col)
def divide_data(x_data, fkey, fval):
    x_right = pd.DataFrame([], columns=x_data.columns)
    x_left = pd.DataFrame([], columns=x_data.columns)
    
    for xi in range(x_data.shape[0]):
        val = x_data[fkey].iloc[xi]
        
        if val > fval:
            x_right = x_right.append(x_data.loc[xi])
        else:
            x_left = x_left.append(x_data.loc[xi])
            
    return x_left,x_right
# We are making a Binary Tree ,henc spilt node into 2.
# If rain will come or not ,Lets say split this across two probabilities. fkey=Probabilities
# we want to split like prob < 0.5(will come) &prob > 0.5(will not come): fval=0.5

def information_gain(x_data, fkey, fval):
    left, right = divide_data(x_data, fkey, fval)
    
    # %age of examples  in left and right
    l = float(left.shape[0]) / x_data.shape[0]
    r = float(right.shape[0]) / x_data.shape[0]
    
    hs = entropy(x_data.Survived)
    
    igain = hs- (l * entropy(left.Survived) + r * entropy(right.Survived))
    return igain
    
for f in X.columns:
    print(f)
    print(information_gain(clean_data ,f, clean_data[f].mean()))
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
sk_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)
sk_tree.fit(X_train, y_train)
sk_tree.score(X_test, y_test)
sk_tree.predict(X_test[:10])

y_test[:10]
class DecisionTree:
    
    def __init__(self, depth=0, max_depth = 5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth 
        self.depth = depth
        self.target = None
## when I am going to predict at partcular Node , say leaf Node has 50 examples and it has 40 True
## and 10 false then target of this node is False(80%)

    def fit(self, X_train):
        features = ['Pclass','Sex','Age','SibSp','Parch','Fare']
        info_gains = []
        
        for ix in features:
            # calculating the info gain for each feature
            i_gain = information_gain(X_train, ix, X_train[ix].mean())
            info_gains.append(i_gain)
   
        #taking the feature with max IG
        self.fkey = features[np.argmax(info_gains)]
        self.fval = X_train[self.fkey].mean()
          ##print("Choosing the Feature:" self.fkey)
    
    
        # create the data
        # split the data
        data_left , data_right = divide_data(X_train, self.fkey, self.fval)
    
        # reset_index with reset the index again from starting for each Subpart
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)
    
        # reached the Leaf Node    
        if data_left.shape[0] == 0 or data_right.shape[0] == 0:
            if X_train.Survived.mean() >= 0.5:
                self.target = "Survived"
            else:
                self.target =  "Dead"
            return
    
        ## Stop early when depth >= max-depth
        if self.depth >= self.max_depth:
            if X_train.survived.mean():
                self.target = "Survived"
            else:
                self.target = "Dead"
            return
    
        ## Recursion
        self.left = DecisionTree(depth=self.depth + 1)
        self.left.fit(data_left)
    
        self.right = DecisionTree(depth=self.depth + 1)
        self.right.fit(data_left)
                
        
    
    def predict(self, test):
        if test[self.fkey] > self.fval:
            # go to right
            # base case
            if self.right is None:
                return self.target
            # Recursive Case
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)

dt = DecisionTree()
## splitting the data into train and test and then predicting on test data

split = int(0.7*clean_data.shape[0])
train_data = clean_data[:split]
test_data = clean_data[split:]
test_data = test_data.reset_index(drop=True)
dt.fit(train_data)
y_pred = []
for i in range(test_data.shape[0]):
    y_pred.append(dt.predict(test_data.loc[i]))
y_pred[:10]
y_actual = test_data[out_cols]
y_actual[:10]
data[split:][:10]
!pip install --upgrade scikit-learn==0.20.3
import pydotplus
from sklearn.externals.six import StringIO 
from IPython.display import Image
from sklearn.tree import export_graphviz           ## will show me the graph
import graphviz
dot_data = StringIO()
export_graphviz(sk_tree, out_file=dot_data, filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
