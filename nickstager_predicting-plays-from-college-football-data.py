# Let's take a look at the files
import os
sorted(os.listdir('../input/collegefootballstatistics'))
# We have the root dir, then a bunch of sub folders by year
# Let's explore a particular year
sorted(os.listdir('../input/collegefootballstatistics/cfbstats-com-2013-1-5-20'))
def remove_csv(inp):
    return inp.replace('.csv', '')
# We want to construct a mapping year -> type -> fname
ROOT_DIR = '../input/collegefootballstatistics/'
dirs = filter(lambda i: '__' not in i, os.listdir(ROOT_DIR))
result = dict()
START_YEAR = 2005
for offset, basename in enumerate(sorted(dirs)): # Ignore MACOSX file
    year = START_YEAR + offset
    # Now, for each subkey, add paths
    current = result[year] = dict()
    sub_dir = os.path.join(ROOT_DIR, basename)
    for file in os.listdir(sub_dir):
        if file.rfind('.csv') > -1:
            current[remove_csv(file)] = os.path.join(sub_dir, file)
print(result[2006]['pass'])
import pandas as pd
# Now we have a way to look up stats and years
# Let's play around
# Let's print the keys we can work with again
result[2005].keys()
# Let's look at all kickoff returns for 2009
play2013 = pd.read_csv(result[2013]['play'])
play2012 = pd.read_csv(result[2012]['play'])
play2011 = pd.read_csv(result[2011]['play'])
play2010 = pd.read_csv(result[2010]['play'])
play2009 = pd.read_csv(result[2009]['play'])
play2008 = pd.read_csv(result[2008]['play'])
play2007 = pd.read_csv(result[2007]['play'])
play2006 = pd.read_csv(result[2006]['play'])
play2005 = pd.read_csv(result[2005]['play'])
play2005.head()
frames = [play2005,play2006,play2007,play2008,play2009,play2010,play2011,play2012,play2013]
result = pd.concat(frames)
result = result[pd.notnull(result['Play Type'])]
result = result[result['Play Type'] != 'PENALTY']
result = result[result['Play Type'] != 'KICKOFF']
result = result[result['Play Type'] != 'PUNT']
result = result[result['Play Type'] != 'ATTEMPT']
result = result[result['Play Type'] != 'TIMEOUT']
result = result[result['Play Type'] != 'FIELD_GOAL']
pd.unique(result['Play Type'])

result.shape
result['Play Type'] = result['Play Type'].replace(to_replace = 'RUSH',value = 0)
result['Play Type'] = result['Play Type'].replace(to_replace = 'PASS',value = 1)
result.head()
result = result.fillna(result.mean())

result = result[result['Play Type'] != 'PENALTY']
result['Score Differential'] = result['Offense Points'] - result ['Defense Points']
result.head()
result.shape
train_x = result
train_y = train_x[['Play Type']].copy()
train_x = result.drop(columns=['Play Type'])
train_y.head()
train_x.head()
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.20)
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=10)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))