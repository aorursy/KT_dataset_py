from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/bbdc2020-features/features.csv')
test_data = pd.read_csv('/kaggle/input/bbdc2020-features/s06t01_test_features.csv')
data = data.drop(['start', 'end', 'Unnamed: 0'], axis = 1)

test_data = test_data.drop(['start', 'end', 'Unnamed: 0', 
                  'la-nothing', 'la-action-change', 'la-object-pick',
                  'la-object-switch-hands', 'la-object-carry', 'la-object-orient',
                  'la-object-place', 'ra-nothing', 'ra-action-change', 'ra-object-pick',
                  'ra-object-switch-hands', 'ra-object-place', 'ra-object-carry',
                  'ra-object-orient'], axis = 1)
data = data.fillna(data.mean())

test_data = test_data.fillna(data.mean())
test_cutoff = len(test_data.index) - len(test_data.index) // 5 * 5
data = data.astype(float)
data = data.iloc[:-4,:]

X = data.iloc[:,:20]
y = data.iloc[:,20:]

X = X.to_numpy()
y = y.to_numpy()

X = X.reshape(6101,5,20)
y = y.reshape(6101,5,14)

test_data = test_data.astype(float)
test_data = test_data.iloc[:-test_cutoff,:]
X_test = test_data.to_numpy()
X_test = X_test.reshape(len(X_test[:,0]) // 5,5,20)
np.shape(X_test)
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(5,20), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(Dense(14, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y, epochs=30, validation_split=0.1)
y_test = model.predict(X_test)
def analyze_timestep(timestep):
    """
    returns a tuple of left and right hand action for a timestep
    """
    left = timestep[:7]
    right = timestep[7:]
    laction_list = ['la-nothing', 'la-action-change', 'la-object-pick',
                    'la-object-switch-hands', 'la-object-carry', 'la-object-orient',
                    'la-object-place']
    raction_list = ['ra-nothing', 'ra-action-change', 'ra-object-pick',
                    'ra-object-switch-hands', 'ra-object-carry', 'ra-object-orient',
                    'ra-object-place']
    return [laction_list[np.argmax(left)], raction_list[np.argmax(right)]]
def translate_predictions(pred):
    """
    takes a numpy array of predictions from rnn and returns a list of 
    human-readable predictions
    """
    predicted_actions = []

    for j in range(np.shape(pred)[0]):
        for i in range(5):
            predicted_actions.append(analyze_timestep(pred[j,i,:]))
    
    #for k in range(len(predicted_actions)):
        #predicted_actions[k] = str(k/10)+','+str((k+1)/10)+' '+predicted_actions[k]
    #    predicted_actions[k] = [k/10, (k+1)/10] + predicted_actions[k]
    return predicted_actions
def make_predictions(who, model):
    """
    takes a subject and a trained model.
    
    Returns predictions for chunks
    """
    test_data = pd.read_csv('/kaggle/input/bbdc2020-features/'+who+'_test_features.csv')
    test_data = test_data.drop(['start', 'end', 'Unnamed: 0', 
                  'la-nothing', 'la-action-change', 'la-object-pick',
                  'la-object-switch-hands', 'la-object-carry', 'la-object-orient',
                  'la-object-place', 'ra-nothing', 'ra-action-change', 'ra-object-pick',
                  'ra-object-switch-hands', 'ra-object-place', 'ra-object-carry',
                  'ra-object-orient'], axis = 1)
    test_cutoff = len(test_data.index) - len(test_data.index) // 5 * 5
    
    test_data = test_data.astype(float)
    
    if test_cutoff != 0:
        test_data = test_data.iloc[:-test_cutoff,:]
    
    X_test = test_data.to_numpy()
    X_test = X_test.reshape(len(X_test[:,0]) // 5,5,20)
    
    y_test = model.predict(X_test)
    
    predictions = translate_predictions(y_test)
    return predictions
def format_predictions(who, predictions):
    """
    takes who and predictions as returned from make_predictions and returns a list of strings
    
    corresponding to lines written in the submission format
    """
    
    left_li = [elem[0] for elem in predictions]
    right_li = [elem[1] for elem in predictions]
    
    for i in range(len(predictions)):
        left_li[i] = [i/10, (i+1)/10] + [left_li[i]]
        right_li[i] = [i/10, (i+1)/10] + [right_li[i]]
        
    new_left_li = [left_li[0]]
    new_right_li = [right_li[0]]
    
    for i in range(len(predictions) -1):
        if left_li[i+1][2] != new_left_li[-1][2]:
                   new_left_li.append(left_li[i+1])
        if right_li[i+1][2] != new_right_li[-1][2]:
                   new_right_li.append(right_li[i+1])
                
    for i in range(len(new_left_li) - 1):
        if new_left_li[i][1] != new_left_li[i+1][0]:
            new_left_li[i][1] = new_left_li[i+1][0]
            
    for i in range(len(new_right_li) - 1):
        if new_right_li[i][1] != new_right_li[i+1][0]:
            new_right_li[i][1] = new_right_li[i+1][0]
    
    left_string_li = [who+'.la,'+str(elem[0])+','+str(elem[1])+','+elem[2] for elem in new_left_li]
    right_string_li = [who+'.ra,'+str(elem[0])+','+str(elem[1])+','+elem[2] for elem in new_right_li]
    
    return left_string_li + right_string_li
    
    
subjects = ['s06t01', 's06t02','s06t03','s06t04','s06t05']

preds = [make_predictions(who, model) for who in subjects]
formatted = [format_predictions(subjects[i],preds[i]) for i in range(len(subjects))]
complete_li = []
for elem in formatted:
    complete_li += elem
with open('predictions.csv', 'w') as f:
    for line in complete_li:
        f.write(line+'\n')