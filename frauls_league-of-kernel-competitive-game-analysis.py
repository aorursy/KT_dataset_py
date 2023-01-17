# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statistics as stat # this is where I get my statistics tools from
import seaborn as sn # some beautiful plots
import matplotlib.pyplot as plt

# So we first want to load our data and check out some samples.
import os
print(os.listdir("../input"))
data_dir = "../input/LeagueofLegends.csv"
data = pd.read_csv(data_dir)
#This function uses the "eval()" function on every cell so we can convert our stringish list to a real list
def clean_col(data, col_name):
    index = 0
    for row in data[col_name]:
        row = eval(row)
        data.at[index, col_name]= stat.mean(row)
        index = index+1
    return data
data = clean_col(data, 'golddiff')
#We have to convert 'golddiff' to a numeric data type
sample = data.sample(1700)
sample['golddiff'] = pd.to_numeric(sample['golddiff'])
sample[sample['golddiff']<700000].plot.hexbin(x = 'golddiff', y = 'gamelength', gridsize = 18)
#sample[sample['golddiff']<700000].plot.scatter(x = 'golddiff', y = 'gamelength')
#y will be our result vector which we want to predict. Since we are looking at rResult
#If rResult == 1: Red Team wins, else: Blue Team wins
#Now we can drop bResult and rResult too
y = data['rResult']
to_drop = ['Year', 'Season', 'Type', 'League', 'Address', 'bResult', 'rResult']
data.drop(to_drop, inplace = True,axis = 1)
data.head()
#This is to work with lines like bBarons/rBarons
# If the given string == 2 (no Beast has been slain) we put in a 0 to indicate that the team in the given game has not slain a Dragon/Baron/Herald
def replacer(x):
    if(len(x) == 2):
        del x
        return 0
    else:
        del x
        return 1
vector = np.vectorize(replacer)
data.head()
def read_beasts(x):
    if(x == 1):
        return 1
    if(x ==0):
        return 0
beast_reader = np.vectorize(read_beasts)
import warnings; 
warnings.simplefilter('ignore')
#Every team which takes up less then 1.9 percent of games will be categorized as 'Other'
series = data['blueTeamTag'].value_counts()
mask = (series/series.sum() * 100).lt(1.9)
series = data['redTeamTag'].value_counts()
mask = (series/series.sum()*100).lt(1.9)
#We use the np.where() function to change other Teams to 'Other'
#Vectorized functions are cool!
data['blueTeamTag'] = np.where(data['blueTeamTag'].isin(series[mask].index), 'Other', data['blueTeamTag'])
data['redTeamTag'] = np.where(data['redTeamTag'].isin(series[mask].index), 'Other', data['redTeamTag'])


#We just want the information wether or not a team killed a given beast in a game or not
bBaron = vector(data['bBarons'])
rBaron = vector(data['rBarons'])
bDragon = vector(data['bDragons'])
rDragon = vector(data['rDragons'])
bHerald = vector(data['bHeralds'])
rHerald = vector(data['rHeralds'])
beast_frame = pd.DataFrame(data = [bBaron, rBaron, bDragon, rDragon, bHerald, rHerald], index = ['bBaron', 'rBaron', 'bDragon', 'rDragon', 'bHerald', 'rHerald'])
beast_frame = beast_frame.transpose()

#Let's get the inhibs the same way
rInhibs = vector(data['rInhibs'])
bInhibs = vector(data['bInhibs'])
inhib_frame = pd.DataFrame(data=[rInhibs, bInhibs], index =['rInhibs', 'bInhibs'])
inhib_frame = inhib_frame.transpose()

#We find our NaN Values in Teams. Let's drop them
Na_Index = np.where(data['blueTeamTag'].isnull()) 
Na_Index1 = np.where(data['redTeamTag'].isnull()) 
Na_Sum = np.concatenate((Na_Index[0], Na_Index1[0])).tolist()
data.drop(index = Na_Sum, axis = 1, inplace = True)
y.drop(index=Na_Sum, axis = 1, inplace = True)
redTeams = set(data['redTeamTag'])
blueTeams = set(data['blueTeamTag'])

#Create some dummies to get categorical values
dummies = pd.get_dummies(data['redTeamTag'])
dummies1 = pd.get_dummies(data['blueTeamTag'])
dummies.columns = ["rC9","rJAG" ,"rOther" ,"rSKT" ,"rSSG" ,"rTSM"]
dummies1.columns = ["bC9" ,"bDH" ,"bIMG" ,"bJAG" ,"bOther" ,"bSKT" ,"bSSG" ,"bTSM" ,"bas"]

# This is where we add our dummies to our datafram and drop our old TeamTags. ByeBye!
data.drop(['rBarons', 'bBarons', 'rDragons', 'bDragons', 'rHeralds', 'bHeralds'], inplace = True, axis = 1)
data = pd.concat([data, dummies, dummies1], axis=1)
dummies_joined = pd.concat([dummies, dummies1], axis = 1)
data.drop(labels = ['redTeamTag', 'blueTeamTag',], inplace = True, axis = 1)




#If you feel fancy you can use the LabelEncoders instead of dummies
#But there is no reason to so let's move on!
# Now we can clean our team tags to categorical values using one-hot-encoding
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder
#from numpy import argmax
#Encoding to integers
#label_encoder = LabelEncoder()
#integer_encoded = label_encoder.fit_transform(data['blueTeamTag'])
#label_encoder2=LabelEncoder()
#integer_encoded2= label_encoder2.fit_transform(data['redTeamTag'])
#Encoding to binary
#o_h_encoder = OneHotEncoder(sparse = False)
#o_h_encoder2 = OneHotEncoder(sparse = False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#integer_encoded2 = integer_encoded2.reshape(len(integer_encoded), 1)
#one_hot_encoded = o_h_encoder.fit_transform(integer_encoded)
#one_hot_encoded2 = o_h_encoder2.fit_transform(integer_encoded2)
#real_data will be used in our machine learning model
data.drop('rResult', inplace = True, axis=1)
real_data = data.drop(data.iloc[:,2:48], axis=1)
real_data = real_data.merge(beast_frame, left_index = True, right_index = True)
real_data = real_data.merge(inhib_frame, left_index = True, right_index = True)
features = list(real_data.columns)
#we want to scale our data so machine learning models have a good time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sc_X = StandardScaler()
y = y.to_frame()
real_data = sc_X.fit_transform(real_data)


X_train, X_test, y_train, y_test = train_test_split(real_data, y, test_size=0.2, random_state=0)

#Since this is a classifcation problem we can use a Random Forest Classifier to make a decision
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 51, criterion = 'entropy', random_state=4343)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)
#Lets look at a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred, y_test)
#This code snippet will display the results
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative', 'Positive']
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()
from sklearn.tree import export_graphviz
from IPython.display import display, Image

import pydot


tree = classifier.estimators_[5]

export_graphviz(tree, out_file = 'tree.dot', feature_names = features, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('tree.png')
display(Image(filename='tree.png'))
feature_importances = pd.DataFrame(classifier.feature_importances_,
                                   index = features,
                                   columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)