import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.calibration import CalibratedClassifierCV

from mlxtend.plotting import plot_decision_regions



female = pd.read_csv("../input/voice-rec-ds/female.csv")

male = pd.read_csv("../input/voice-rec-ds/male.csv")



female.head()
def vectors(ds):

    vec_doc = {}

    vec_doc[0] = []

    for col in ds.columns:

        vec_doc[0].append( round(float(col),4) )

    for el in range(ds.shape[0]):

        vec_doc[el+1] = []

    for col in ds.columns:

        for i in range(ds.shape[0]):

            vec_doc[i+1].append( round(ds[col][i],4) )

    return vec_doc

female_vectors = vectors(female)

male_vectors = vectors(male)

del female_vectors[22]

plt.plot(female_vectors[3])
# female -> 1, male -> 0

df_vec = []

df_tag = []

for el in female_vectors:

    df_vec.append(female_vectors[el])

    df_tag.append(1)

for el in male_vectors:

    df_vec.append(male_vectors[el])

    df_tag.append(0)

sound_df = pd.DataFrame(list(zip(df_vec,df_tag)), 

        columns =['sound_samples','origin'])

sound_df.head()
X_train, X_val, y_train, y_val = train_test_split(sound_df['sound_samples'], sound_df['origin'], test_size=0.3, random_state=0)

X_train = np.array(list(X_train))

y_train = np.array(list(y_train))

X_val = np.array(list(X_val))

y_val = np.array(list(y_val))

'''

parameters = { 

    'gamma': [0.5, 1, 1.5, 1.7, 2, 3, 4, 5], 

    'kernel': ['rbf','poly','sigmoid'], 

    'C': [0.1, 0.5, 1, 1.5, 2, 3],

}

model = GridSearchCV(SVC(), parameters, cv=10, n_jobs=-1).fit(X_train, y_train)

model.cv_results_['params'][model.best_index_]

'''

print('{ C : 1,  gamma : 1.7,  kernel :  sigmoid }')
model = SVC(C = 1, gamma = 1.7, kernel = 'sigmoid')

model.fit(X_train,y_train)

prediction = model.predict(X_val)

print('accuracy:',round(accuracy_score(y_val, prediction),2))
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,3))

#fig.suptitle('pred/reality')

px = [el for el in range(len(prediction))]

ax1.scatter(px,prediction)

ax1.set_title('Prediction')

ax1.set_xlabel('samples')

ax1.set_ylabel('state')

ax2.scatter(px,y_val, color = 'tab:orange')

ax2.set_title('Reality')

ax2.set_xlabel('samples')

ax2.set_ylabel('state')
# female_vectors[3]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(16,5))

# plot1

var = np.array([female_vectors[3]])

ax1.plot(female_vectors[3])

ax1.set_title('FFT (1000 samples)')

ax1.set_xlabel('samples')

ax1.set_ylabel('frequency')

# plot2

clf = CalibratedClassifierCV(model)

clf.fit(X_train,y_train)

var_pred = clf.predict_proba(var)

ax2.bar(['male','female'],list(var_pred[0]), color=['#1f77b4', '#ff7f0e'])

ax2.set_title('Gender')

ax2.set_ylabel('percentage(0-1)')

# plot3

ax3.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1])

ax3.scatter(var_pred[:, 0], var_pred[:, 1], c=model.predict(var), s=50, cmap='autumn')

ax3.set_xlim([0,3])

ax3.set_ylim([0,7])

ax3.set_title('Female sample among support vectors (part)')