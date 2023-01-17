from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from scipy.spatial.distance import cdist
import pickle
import array
data = pd.read_csv('../input/doppelganger14500/output.csv')
data = data[:-1].sample(frac=1)
def convert_to_np_array(x):   
    try:
        inter = [int(y) for y in x.split('{"type":"Buffer","data":[')[1].split(']}')[0].split(',')]
        byts = array.array('B', inter).tobytes()
        return pickle.loads(byts).reshape(1,2048)
    except:
        return np.zeros((1,2048))

def convert_to_binary(lst):
    fin = []
    for item in lst:
        fin.append(1 if item >= 1 else 0)
    return fin

def plot_loss(history):
    # "Loss"
    plt.plot(history.history['loss'][2:])
    plt.plot(history.history['val_loss'][2:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
def binary_prob(arr):
    preds = [a[1]**0.5 for a in arr]
    fin = []
    for item in preds:
        if item > 0.5:
            fin.append(1)
        else:
            fin.append(0)
    return fin

def convert_to_binary_array(x):
    return float(x) if x == "1" or x == "0" else 1.0

def select_features(X, features):
    return np.array([X[:,i] for i in features]).T

def divide(a, b):
    quotes = []
    a = a[0]
    b = b[0]
    for i in range(len(a)):
        if a[i] == 0.0 and b[i] == 0.0:
            quotes.append(1.0)
        elif b[i] == 0.0:
            quotes.append(a[i]/0.0000001)
        else:
            quotes.append(a[i]/b[i])
    return quotes
def get_x_and_y(dataset):
    #get ratings
    cols = dataset.columns.tolist()
    cols = ['usernames', 'embedding1', 'embedding2','rating', 'same_gender']
    #filter out unnecessary columns
    dataset = dataset[cols]
    #convert embeddings to np arrays
    dataset['embedding1'] = dataset['embedding1'].apply(convert_to_np_array)
    dataset['embedding2'] = dataset['embedding2'].apply(convert_to_np_array)
    dataset['same_gender'] = dataset['same_gender'].apply(convert_to_binary_array)
    #drop duplicate values (same match, diff reviewer)
    dataset = dataset.groupby('usernames').apply(np.mean)
    dataset = dataset.sample(frac=1)
    #double each datapoint
    scores = np.array([cdist(np.array(dataset['embedding1'][i]).reshape(1,2048), np.array(dataset['embedding2'][i]).reshape(1,2048), 'cosine')[0] for i in range(len(dataset))])
    scores =(1-scores**(2.5-2.5*scores))
    #diff = [list(i[0]) for i in list(dataset["embedding1"] - dataset["embedding2"])]
    diff = [divide(dataset["embedding1"][i], dataset["embedding2"][i]) for i in range(len(dataset["embedding1"]))]
    for i in range(len(diff)):
        diff[i].append(dataset['same_gender'][i])
        diff[i].append(scores[i][0])
    #ratings
    ratings = []
    for i in range(len(dataset)):
        if dataset['same_gender'][i] == 0.0:
            ratings.append(0)
        else:
            ratings.append(dataset["rating"][i])
    X = np.array(diff)
    Y = np.array(convert_to_binary(dataset["rating"]))
    #in_encoder = Normalizer(norm='l2')
    #X = in_encoder.transform(X)
    return X, Y, dataset
X, Y, dataset = get_x_and_y(data[:500])
X, test_X = X[100:], X[:100]
Y, test_Y = Y[100:], Y[:100]
correlations = []
features = []

for i in range(len(X[0])):
    feature = X[:, i]
    cor = np.corrcoef(feature, Y)
    correlations.append(cor[0][1])
    
for i in range(len(correlations)):
    cor = correlations[i]
    if abs(cor) > 0.015:
        print('feature #{}: {}'.format(i, cor))
        features.append(i)
        
plt.plot(np.arange(0,2050), [abs(cor) for cor in correlations])
plt.show()
model = CatBoostClassifier(loss_function='Logloss',
                           iterations=5000,
                           depth=8,
                           learning_rate=0.005,
                           verbose=True)
optimization_dict = {'depth': [6,8,10],
                     'learning_rate': [0.01,0.05,0.1,0.5]}
#model =  GridSearchCV(model, optimization_dict, 
#                      scoring='accuracy', verbose=1)
model.fit(select_features(X[0:200], features), Y, 
            #early_stopping_rounds=20, 
             eval_set=[(select_features(test_X, features), test_Y)])
print(model.best_score_)
print(model.best_params_)
preds = model.predict(select_features(test_X, features))
preds
test_Y[0:100]
test_Y
from sklearn.svm import SVC
model = SVC(kernel='linear', C=0.01, random_state=101)
train_X = select_features(X[0:5000], features)
model.fit(train_X, Y[0:5000])
y_pred = model.predict(select_features(test_X, features))
def get_true_positive(pred, test):
    total_positive = sum(test)
    predicted = sum([1 for i in range(len(pred)) if pred[i] == test[i] and test[i] == 1])
    return predicted/total_positive
get_true_positive(y_pred, test_Y)
test_Y
