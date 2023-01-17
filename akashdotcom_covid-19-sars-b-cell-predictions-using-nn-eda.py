import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.utils import resample

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras.callbacks import EarlyStopping
file_path = '../input/epitope-prediction'

bcell_df = pd.read_csv(f'{file_path}/input_bcell.csv')

covid_df = pd.read_csv(f'{file_path}/input_covid.csv')

sars_df = pd.read_csv(f'{file_path}/input_sars.csv')
bcell_df.head(3)
covid_df.head(3)
sars_df.head(3)
#training data created

frames = [bcell_df, sars_df]

bcell_sars_df = pd.concat(frames, axis=0, ignore_index=True)

bcell_sars_df.head() 
#checking for null values

bcell_sars_df.isna().sum()
#shuffling the dataset

bcell_sars_df = bcell_sars_df.sample(frac=1).reset_index(drop = True)
#Info of the dataset

bcell_sars_df.info()
#Describe the dataset

bcell_sars_df.describe()
#checking the target variable countplot

sns.set_style('darkgrid')

sns.countplot(bcell_sars_df['target'])
idx_train = bcell_sars_df['target'].astype("bool").values

fig, axes = plt.subplots(2, 3,figsize=(16,8))

sns.set_style('darkgrid')

axes = [x for a in axes for x in a]

for i,name in enumerate(["isoelectric_point", "aromaticity", "hydrophobicity", "stability", "parker", "emini"]):

    value = bcell_sars_df[name]

    sns.distplot(value[~idx_train],ax = axes[i], color='red')

    sns.distplot(value[idx_train],ax = axes[i], color = 'blue')

    axes[i].set_xlabel(name,fontsize=12)

    fig.legend(labels = ["target 0","target 1"],loc="right",fontsize=12)
#Corelation Matrix

corr_matrix = bcell_sars_df[['parent_protein_id', 'protein_seq', 'start_position', 'end_position', 

                             'peptide_seq', 'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker', 

                             'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability', 

                             'target']].corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)

mask[np.triu_indices_from(mask)]= True
#corr heatmap

sns.set_style('whitegrid')

f, ax = plt.subplots(figsize=(11, 15)) 

heatmap = sns.heatmap(corr_matrix, 

                      mask = mask,

                      square = True,

                      linewidths = .5,

                      cmap = 'coolwarm',

                      cbar_kws = {'shrink': .4, 

                                'ticks' : [-1, -.5, 0, 0.5, 1]},

                      vmin = -1, 

                      vmax = 1,

                      annot = False,

                      annot_kws = {'size': 12})

#add the column names as labels

ax.set_yticklabels(corr_matrix.columns, rotation = 0)

ax.set_xticklabels(corr_matrix.columns)

sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
#calculating feature importance

X = bcell_sars_df.drop(['target', 'parent_protein_id', 'protein_seq', 'peptide_seq'], axis = 1)

y = bcell_sars_df['target']

forest_clf = ExtraTreesClassifier(n_estimators=250, random_state=420)

forest_clf.fit(X,y)
imp_features = forest_clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest_clf.estimators_], axis = 0)

 

plt.figure(figsize = (15,8))

plt.bar(X.columns, std, color = 'red') 

plt.xlabel('Feature Labels') 

plt.ylabel('Feature Importances') 

plt.title('Comparison of different Feature Importances') 

plt.show()
#train_test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
model = Sequential()

model.add(Dense(units=128,activation='relu'))

model.add(Dropout(0.3))



model.add(Dense(units=64,activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(units=32,activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(units=16,activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')



#Early stopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, 

          y=y_train, 

          epochs=150,

          validation_data=(X_test, y_test), verbose=1,

          callbacks=[early_stop]

          )
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
#predictions

predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions, target_names = ['Covid_Negative','Covid_Positive']))
#confusion matrix

plt.figure(figsize = (10,10))

cm = confusion_matrix(y_test,predictions)

sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Covid_Negative','Covid_Positive'] , yticklabels = ['Covid_Negative','Covid_Positive'])

plt.xlabel("Predicted")

plt.ylabel("Actual")
#Applying PCA

pca = PCA(n_components = 2)



projected = pca.fit_transform(bcell_sars_df[['isoelectric_point', 'aromaticity', 

                                             'start_position', 'end_position', 

                                             'stability', 'hydrophobicity', 

                                             'emini', 'parker']])

plt.figure(figsize=(8,8))

plt.scatter(projected[:, 0], projected[:, 1],

            c=bcell_sars_df.target, edgecolor='none', alpha=0.5,

            cmap=plt.cm.get_cmap('coolwarm', 2))

plt.xlabel('component 1')

plt.ylabel('component 2')

plt.colorbar();
#Prediction for Covid dataset

covid_df_Pred = covid_df.drop(['parent_protein_id', 'protein_seq', 'peptide_seq'], axis = 1)

#transform data

covid_df_Pred = sc.transform(covid_df_Pred)

predictions_covid = model.predict_classes(covid_df_Pred)

predictions_covid
predictions_covid = pd.DataFrame(predictions_covid, columns = ['Predictions'])

#predictions_covid.head()

frames = [covid_df, predictions_covid]

output = pd.concat(frames, axis = 1)

output.head(5)
output['Predictions'].value_counts() #0's are Covid negative, 1's are Covid positive.