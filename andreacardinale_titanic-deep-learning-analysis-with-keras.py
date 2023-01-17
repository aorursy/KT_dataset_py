# Import libraries

# Importo le librerie

%matplotlib inline

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split
# Read csv files

# Leggo i files .csv



df_train = pd.read_csv('../input/train.csv', index_col=0)    #train dataset import. Set first column 'passengerId' as index

df_test = pd.read_csv('../input/test.csv', index_col=0)    #test dataset import. Set first column 'passengerId' as index



# Concatenate two dataframes without 'Survived' column, to better management.

# concateno i due dataframe senza la colonna 'Survived', per una miglior gestione



df_all = pd.concat([df_train, df_test], axis=0, sort=True).drop(['Survived'], axis = 1)

# First 5 train rows

# Prime 5 righe dela df train



df_all.head()
# Plot a heatmap to show how nan are distributed, first on train dataset....

# Plotto una heatmap  per visualizzare come sono distribuiti i Nan, primo sul dataset di train...



sns.heatmap(data=df_all.isnull())
# fill age's nan with the median of age's values grouped by 'Class' and 'Sex'

# Riempio i nan della colonna 'age' con la mediana dei valori di 'age' raggruppati per 'Class' e 'Sex'



df_all['Age'] = df_all.groupby(['Pclass','Sex'], sort=False)['Age'].apply(lambda x: x.fillna(x.median()))

# Count 'Embarked' values...

# Conto i valori della colonna 'Embarked'...



df_all['Embarked'].value_counts(dropna=False).plot(kind='bar', figsize=(6,4), title='Embarked')
# ... and fill Nan with the most frequent value ('S')

# ... e riempi i Nan con il valore più frequente ('S')



df_all['Embarked'] = df_all['Embarked'].fillna('S')
# fill also the unique Nan in 'Fare' test DF's column with the median of the class

# riempi l'unico 'Nan' nella colonna 'Fare' del dataframe DF



df_all['Fare'] = df_all.groupby(['Pclass'], sort=False)['Fare'].apply(lambda x: x.fillna(x.median()))



df_all.info()
# From 'Name' column, Extract titles name

# Estraggo il titolo dalla colonna 'Nome'



df_all['Title'] = df_all['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])



# Try to define the correct title classification for un-classified values ..

# Cerco di definire la corretta classificazione per i titoli..





df_all['Title'] = df_all['Title'].replace(['Mme','Ms'], 'Mrs')

df_all['Title'] = df_all['Title'].replace(['Mlle','Lady'], 'Miss')



# ...and group the less common titles under 'other' label

# ... 



df_all['Title'] = df_all['Title'].replace(['the Countess',

                                               'Capt', 'Col','Don', 

                                               'Dr', 'Major', 'Rev', 

                                               'Sir', 'Jonkheer', 'Dona'], 'Others')



df_all['Title'].value_counts().plot(kind='bar', figsize=(6,4), title='Title')
# The get_dummies converts categorical variables in dummies variables (boolean)

df_all = pd.get_dummies(df_all,columns=['Sex','Embarked','Title'])
# Remove Cabin, Name and Ticket Columns because are unuseful

df_all.drop(['Cabin','Name','Ticket'], axis=1, inplace=True)
# With heatmap we try to find correlation among features

# Con heatmap cerchiamo correlazioni tra le features



plt.figure(figsize= (10,10), dpi=100)

sns.heatmap(df_all.corr(), square=True, annot=True)
# I use my train dataframe to assign values to y (predicted label) and x

y = df_train['Survived']

x = df_all.iloc[:891,1:] # original 891 train rows

# con iloc assegno ad una variabile le colonne dalla 1 (seconda) in poi (i primi : significano tutte le righe)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



bestfeatures = SelectKBest(score_func=chi2, k='all')

fit = bestfeatures.fit(x,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(15,'Score'))  #print 10 best features
# With ExtraTreesClassifier i try to find correlations between features

# Con l'ExtraTreesClassifier cercco di trovare correlazioni tra le features



from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(x,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=x.columns)

feat_importances.nlargest(15).plot(kind='barh')

plt.show()
# With train_test_split i divide my train df in two parts, so i can run my test and show how my predictions works

X_train, X_test, Y_train, Y_test = train_test_split(x.drop(['Fare','Title_Others','Embarked_C','Embarked_S','Embarked_Q'], axis=1), y, test_size=0.3, random_state=42)
# Test with deep learning with 2 model, activation sigmoid, optimizer adam learning rate .1 (0.799 in kaggle )

# Test con deep learning con 2 modelli, activation sigmoid, optimizer adam e learning rate .1 (0.799 in kaggle )



# Test with deep learning



from keras import models

from keras import layers

from keras import optimizers



# initialize the network

model = models.Sequential()



# add nodes to the network

model.add( 

    layers.Dense(1,                    # no. of neurons

                 activation='sigmoid', # activation function

                 input_shape=(14,)      # shape of the input

                ))



model.add( 

    layers.Dense(1,                    # no. of neurons

                 activation='sigmoid', # activation function

                ))



# finalize the network

model.compile(

    optimizers.Adam(lr=.01)

, # lr is the learning rate

    loss='binary_crossentropy',       # loss function

    metrics=['accuracy'] )   # additional quality measure



# train the network

hist = model.fit( x=x, # training examples

           y=y, # desired output

           epochs=200, # number of training epochs 

           verbose=1)

# 0.79425 in kaggle with 2 model, activation sigmoid, optimizer adam learning rate .1

model.evaluate(x, y)
# Plot accuracy and loss

# Plotto accuracy and loss



fig, axes = plt.subplots(figsize=(6,6))



axes.plot(hist.history['loss'], label='Loss')

axes.plot(hist.history['acc'], label='Accuracy')



axes.set_title("Training History", fontsize=18)

axes.set_xlabel("Epochs", fontsize=18)

axes.legend(fontsize=20)



# Final accuracy

print ("Accuracy:", hist.history['acc'][-1])
# write results in a new 'Survived' column df test dataframe. 

# I take only last 418 rows of df_all dataframe, because they represent my test df

df_test['Survived'] = model.predict_classes(df_all.iloc[891:,1:])

df_test['Survived'].to_csv("../prediction.csv", header="PassengerId,Survived")

df_test.drop(['Survived'], axis=1, inplace=True)