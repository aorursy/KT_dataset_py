import pandas as pd
import numpy as np

data = pd.read_csv('../input/young-people-survey/responses.csv')
pd.set_option('display.max_columns',200)
data.head(2)
data.shape
data.describe()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.rcParams.update({'font.size': 16})

music = data.iloc[0:,2:19].copy()
music.drop(music.columns[[1,2,3,4]], axis=1, inplace=True)
music.drop(music.columns[[7,9,10,12]], axis=1, inplace=True)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
cax = ax.matshow(music.corr())
fig.colorbar(cax,fraction=0.046, pad=0.04)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.set_xticklabels([' ']+music.columns.tolist()  ,rotation=90 )
ax.set_yticklabels([' ']+music.columns.tolist()   )

plt.show()

# separated numerical and categorial variables
numerical_data = data._get_numeric_data()
categorical_data = data.select_dtypes(include=['object'])

# convert to 1 if person answer 4 or 5  (more than the median)
for column in numerical_data:
    numerical_data[column].fillna(0, inplace=True)
    mean = int(np.median(numerical_data[column].unique())+.5)
    numerical_data[column] = numerical_data[column].apply(lambda x : 1 if x > mean else 0)
# get a code per category
for column in categorical_data.columns:
    categorical_data[column] = categorical_data[column].astype('category')
    categorical_data[column] = categorical_data[column].cat.codes
from sklearn.preprocessing import OneHotEncoder

# OneHotEncoder unable negatives
categorical_data = categorical_data.replace(-1,10)

# multiplicate categorial columns to binary columns
print(categorical_data.shape)
categorical_data=OneHotEncoder().fit_transform(categorical_data).toarray()
print(categorical_data.shape)
# concat the data
categorical_data=  pd.DataFrame(categorical_data)
data_prepared = pd.concat([categorical_data,numerical_data],axis=1, join='inner')
print(data_prepared.shape)
music_prepared = data_prepared[music.columns]
demographic_columns= [item for item in data_prepared.columns if item not in music.columns]
demographic_prepared = data_prepared[demographic_columns]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_prepared, music_prepared, test_size=0.33, random_state=42)
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# blabla
def train_clfs( X_train, y_train, X_test, y_test,multilabels=True) :
    clfs = {
        "Knn": KNeighborsClassifier(n_neighbors=10),
        "RandomForest":RandomForestClassifier(n_estimators=50),
        "ID3" : DecisionTreeClassifier(criterion='entropy'),
        "CART" : DecisionTreeClassifier()
    }
    if not multilabels :
        clfs["MLP"] = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
   
    for i in clfs :
        clf = clfs[i]
        clf = clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        print (i," Accuracy Score: ",accuracy_score(y_test, predicted))
train_clfs( X_train, y_train, X_test, y_test)
import json

dict_of_imply = json.load(open('../input/youngpeoplesurvey-implymusicjson/new.json'))

dict_refactor={}
#refactor
for music in dict_of_imply :
    music = music.replace("."," ")
    if music.split(" ")[0] in ["Hiphop","Reggae","Techno","Swing"]:
        music_refactor = music.split(" ")[0]+", " +music.split(" ")[2]
        dict_refactor[music_refactor]=[]  
    else :
        dict_refactor[music]=[]
        music_refactor=music
    music = music.replace(" ",".")
    for element in dict_of_imply[music]:
        element = element.replace("."," ")
        if element in X_train.columns :
            dict_refactor[music_refactor].append(element)    
for music in dict_refactor :
    X_train_music = X_train[dict_refactor[music]]
    X_test_music =  X_test[dict_refactor[music]]
    Y_train_music = y_train[music]
    Y_test_music = y_test[music]
    print(music," : ")
    train_clfs( X_train_music, Y_train_music, X_test_music, Y_test_music,multilabels=False)
    print()
rules = json.load(open('../input/youngpeoplesurvey-implymusic/rules.json'))
