import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, classification_report

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]: #limiting output to 5 files
        print(os.path.join(dirname, filename))

data_dir = '/kaggle/input/particle-collisions/'
# load a random pickle file to get a glimpse of the data structure

pkl_file = open(data_dir+'event112.pkl', 'rb')
eventx = pickle.load(pkl_file)
print(eventx.shape)
particle_types = {11: "electron", 13 : "muon", 211:"pion", 321:"kaon",2212 : "proton"}

X = []
y = []

pkl_data = glob.glob(data_dir+'*.pkl')

for pkl in pkl_data:
    pkl_file = open(pkl, 'rb')
    event1 = pickle.load(pkl_file)
    
    # get the data and target
    data,target = event1[0], event1[1]
    
    X += [d for d in data]
    y += [t for t in target]

np.array(y).shape, np.array(X).shape
np.ndarray.flatten(np.array(X)).shape
event_df = pd.DataFrame({
    'particle_Id':y
})
event_df['class'] = event_df['particle_Id'].map(particle_types)
for i in range(100):
  event_df[str(i)] = [x.flatten()[i] for x in X] # flattening the 10x10 images
event_df.head()
# Let's check the distribution of the target
event_df['class'].value_counts()
# Encoding the target variable ('class')
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
event_df['class'] = encoder.fit_transform(event_df['class'])
# Split the data into train and test data
target = event_df['class']
images = event_df[0:]

X_train, X_test, y_train, y_test = train_test_split(images, target, test_size=0.20, random_state=72)
clf = RandomForestClassifier(n_estimators=350,
                             max_depth=7,
                             random_state=84,
                             n_jobs=-1
                            )
clf.fit(X_train, y_train)
# Log_loss
y_hat = clf.predict_proba(X_test) # log loss requires predictions as probabilities
loss = log_loss(y_test, y_hat)
print(f'Log_loss: {loss}')
# Classification report
# This takes class predictions hence we'll have to use .predict()
y_hat_prime = clf.predict(X_test)
print(f'Classification Report: \n{classification_report(y_test, y_hat_prime, zero_division=False)}')
