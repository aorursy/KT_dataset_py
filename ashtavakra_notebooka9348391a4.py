import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
ds = pd.read_csv('../input/train.csv')

ds.info()
ds.head()#used to represent first 5 columns by default
cols_to_drop = [

    'PassengerId',

    'Name',

    'Ticket',

    'Cabin',

    'Embarked',

]

#  since the above feature is not important so we dropped it 

df = ds.drop(cols_to_drop, axis=1)# along the columns  axis=1

df.head()

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
cols_to_drop={''}





def convert_sex_to_num(s):# data conversion from one value to another

    if s=='male':

        return 0

    elif s=='female':

        return 1

    else:

        return s# this is for none values





df.Sex = df.Sex.map(convert_sex_to_num)

df.head()
data = df.dropna()#any row containg nan value will be dropped

data.describe()# descibe returns all the features like count,mean,std,min
plt.figure()# for plotting a figure

sns.heatmap(data.corr())# there can be two correlations positive as well as negative
input_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

out_cols = ['Survived']



X = data[input_cols]

y = data[out_cols]



#X.head()

print (X.shape, y.shape)
data = data.reset_index(drop=True)# resetting the index.
def divide_data(x_data, fkey, fval):

    x_right = pd.DataFrame([], columns=x_data.columns)

    x_left = pd.DataFrame([], columns=x_data.columns)

    

    

    for ix in range(x_data.shape[0]):

        # Retrieve the current value for the fkey column

        try:

            val = x_data[fkey].loc[ix]

        except:

            print (x_data[fkey])

            val = x_data[fkey].loc[ix]

        # print val

        

        # Check where the row needs to go

        if val > fval:

            # pass the row to right

            x_right = x_right.append(x_data.loc[ix])

        else:

            # pass the row to left

            x_left = x_left.append(x_data.loc[ix])

    

    # return the divided datasets

    return x_left, x_right





def entropy(col):

    p = []

    p.append(col.mean())

    p.append(1-p[0])

    

    ent = 0.0

    for px in p:

        ent += (-1.0 * px * np.log2(px))

    return ent



def information_gain(xdata, fkey, fval):

    left, right = divide_data(xdata, fkey, fval)

    return entropy(left.Survived) + entropy(right.Survived) - entropy(xdata.Survived)
'''for fx in data.columns:

    l, r = divide_data(data, fx, data[fx].mean())

    

    # TODO: check information gain

    print fx,

    print entropy(l.Survived) + entropy(r.Survived) - entropy(data.Survived)'''
class DecisionTree:

    def __init__(self, max_depth=10):

        self.left = None

        self.right = None

        self.fkey = None

        self.fval = None

        self.max_depth = max_depth

    

    def train(self, X_train):

        # Get the best possible feature and division value

        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

        gains = []

        for fx in features:

            gains.append(information_gain(X_train, fx, X_train[fx].mean()))

        

        # store the best feature (using min information gain)

        self.fkey = features[np.argmin(gains)]

        self.fval = X_train[self.fkey].mean()

        

        # divide the dataset

        data_left, data_right = divide_data(X_train, self.fkey, self.fval)

        data_left = data_left.reset_index(drop=True)

        data_right = data_right.reset_index(drop=True)

        

        # Check the shapes

        if data_left.shape[0] == 0 or data_right.shape[0] == 0:

            return

        

        # branch to right

        self.right = DecisionTree(max_depth=self.max_depth-1)

        self.right.train(data_right)

        # branch to left

        self.left = DecisionTree(max_depth=self.max_depth-1)

        self.left.train(data_left)

        

        return
dt = DecisionTree()

dt.train(data)
print (dt.fkey, dt.fval)

print (dt.right.fkey, dt.right.fval)

print (dt)