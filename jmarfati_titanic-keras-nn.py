import pandas as pd

import numpy as np



from keras import models, layers, metrics

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.models import load_model



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("darkgrid")



# np.random.seed(1)
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_combined = df_test.append(df_train, sort=True)
df_train.loc[df_train.Fare.isnull(),'Fare'] = df_train.Fare.mean()

df_test.loc[df_test.Fare.isnull(),'Fare']   = df_test.Fare.mean()



df_train.loc[df_train.Embarked.isnull(),'Embarked'] = 'S'

df_test.loc[df_test.Embarked.isnull(),'Embarked']   = 'S'





df_train['Cabin_1'] = (df_train['Cabin'].notnull()).astype(int)

df_test['Cabin_1']  = (df_test['Cabin'].notnull()).astype(int)



df_train['Cabin_2'] = df_train['Cabin'].astype(str).str[0]

df_test['Cabin_2']  = df_test['Cabin'].astype(str).str[0]



df_train['family_members'] = df_train['SibSp'] + df_train['Parch']

df_test['family_members']  = df_test['SibSp'] + df_test['Parch']



df_train['is_alone'] =  (df_train['family_members']< 1).astype(int)

df_test['is_alone']  = (df_test['family_members']< 1).astype(int)
guess_ages = np.zeros((2,3))

sex_categories = ['male', 'female']



for i in range(0, 2):

    for j in range(0, 3):

        guess_df = df_combined[(df_combined['Sex'] == sex_categories[i]) & \

                                (df_combined['Pclass'] == j+1)]['Age'].dropna()

          

        age_guess = guess_df.median()

        guess_ages[i,j] = int(age_guess/0.5 + 0.5 ) * 0.5

            

for dataset in [df_train, df_test]:

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == sex_categories[i]) & (dataset.Pclass == j+1),\

                        'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)

    

df_train['Age2'] = df_train.Age **2

df_test['Age2']   = df_test.Age **2
for dataset in [df_train, df_test]:

    dataset.loc[:,'AgeBins'] = '0_2'

    dataset.loc[(dataset['Age'] >= 2) & (dataset['Age'] < 11), 'AgeBins']  = '2_10' 

    dataset.loc[(dataset['Age'] >= 16) & (dataset['Age'] < 25), 'AgeBins'] = '20_25' 

    dataset.loc[(dataset['Age'] >= 25) & (dataset['Age'] < 35), 'AgeBins'] = '25_34' 

    dataset.loc[(dataset['Age'] >= 2) & (dataset['Age'] < 11), 'AgeBins']  = '35_60' 

    dataset.loc[dataset['Age'] >= 60, 'AgeBins'] = '60+'
df_train['title'] = df_train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

df_test['title'] = df_test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



df_train['last_name'] = df_train['Name'].str.split(", ", expand=True)[0]

df_test['last_name'] = df_test['Name'].str.split(", ", expand=True)[0]



df_train['title'] = df_train['title'].replace('Mlle', 'Miss')

df_train['title'] = df_train['title'].replace('Ms', 'Miss')

df_train['title'] = df_train['title'].replace('Mme', 'Mrs')



df_test['title'] = df_test['title'].replace('Mlle', 'Miss')

df_test['title'] = df_test['title'].replace('Ms', 'Miss')

df_test['title'] = df_test['title'].replace('Mme', 'Mrs')





df_train.loc[~df_train.title.isin(['Mr', 'Mrs', 'Miss', 'Master', 'Rev', 'Dr', 'Col', 'Capt', 'Major', 'Lady', 'Sir']),\

             'title'] = 'Unknown'



df_test.loc[~df_test.title.isin(['Mr', 'Mrs', 'Miss', 'Master', 'Rev', 'Dr', 'Col', 'Capt', 'Major', 'Lady', 'Sir']),\

             'title'] = 'Unknown'



df_train.loc[df_train.title.isin(['Lady', 'Sir', 'Capt', 'Major']),\

             'title'] = 'High_Rank'



df_test.loc[df_test.title.isin(['Lady', 'Sir']),\

             'title'] = 'High_Rank'



df_train = df_train[df_train.title!='High_Rank']
categorical_list = ['Sex', 'Embarked', 'title', 'Cabin_2',  'Pclass', 'family_members', 'AgeBins']



continuous_list = [ 'Fare']



iteraction_female_list = ['Cabin_2', 'family_members', 'AgeBins']
train_features = df_train[continuous_list]

test_features = df_test[continuous_list]



for i in categorical_list:

    train_features = train_features.join(pd.get_dummies(df_train[i], prefix= (i + '')))

    test_features = test_features.join(pd.get_dummies(df_test[i], prefix= (i + '')))



for i in range(1, 4):

    train_features['P_class_female_' + str(i) ] = train_features['Pclass_' + str(i)] * train_features['Sex_female']

    test_features['P_class_female_' + str(i) ]  = test_features['Pclass_' + str(i)]  * test_features['Sex_female']

    

    

for i in range(1, 4):

    for j in  df_test['AgeBins'].unique():

        train_features['Age_Pclass_' + str(i) +  str(j)] = train_features['AgeBins_'+ str(j)] * train_features['Pclass_' + str(i) ]

        test_features['Age_Pclass_' + str(i) +  str(j)]  = test_features['AgeBins_'+ str(j)]  * test_features['Pclass_' + str(i) ]



for j in iteraction_female_list:

    for i in df_test[str(j)].unique():

        train_features[j + '_female_' + str(i) ] = train_features[j + '_' + str(i)] * train_features['Sex_female']

        test_features[ j + '_female_' + str(i) ]  = test_features[j + '_' + str(i)]  * test_features['Sex_female']
train_features['Fare_female'] = train_features['Fare'] * train_features['Sex_female']

test_features['Fare_female']  = test_features['Fare']  * test_features['Sex_female']



del(train_features['Cabin_2_T'])
print('Number of Train Features: ' + str(len(train_features.columns)))

train_features.head()
print('Number of Test Features: ' + str(len(test_features.columns)))



test_features.head()
print("Missing values in training set : \n" + str(train_features.isnull().sum().sum()))

print(" ")

print("Missing values in testing set : \n" + str(test_features.isnull().sum().sum()))
'''

combined_features = test_features.append(train_features)

for i in test_features.columns:

    min_1 = combined_features[i].min()

    max_1 = combined_features[i].max()

    train_features[i] = (train_features[i] - min_1)/(max_1- min_1)

    test_features[i]  = (test_features[i] - min_1)/(max_1- min_1)



train_features.head()

'''
train_target = df_train['Survived']

train_features_1, dev_features_1, train_target_1, dev_target_1 = train_test_split(train_features, train_target, \

                                                                                  test_size=0.25,) \

                                                                                  #random_state=2)

network = models.Sequential()

network.add(layers.Dense(units=60, activation='relu', input_shape=(len(train_features_1.columns),)))

network.add(layers.Dropout(0.2))

network.add(layers.Dense(units=30, activation='relu'))

network.add(layers.Dropout(0.2))

network.add(layers.Dense(units=1, activation='sigmoid'))



network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'], ) 



es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=100)

mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=0, save_best_only=True)



history = network.fit(train_features_1, train_target_1, 

            epochs=250, verbose=0, batch_size=100, 

            validation_data=(dev_features_1, dev_target_1), callbacks=[es, mc]) 



saved_model = load_model('best_model.h5')
plt.subplots(figsize=(15, 5))

plt.subplot(1, 2, 1)

plt.title('Loss after each Epoch')

plt.plot(history.epoch[::10], history.history['loss'][::10], label='Train')

plt.plot(history.epoch[::10], history.history['val_loss'][::10], label='Test')

plt.legend(['Train', 'Test'],loc='upper right', title='Sample', facecolor='white',fancybox=True)

plt.xlabel('Loss')

plt.ylabel('Epochs')



plt.subplot(1, 2, 2)

plt.title('Accuracy after each Epoch')

plt.plot(history.epoch[::10], history.history['acc'][::10], label='Train')

plt.plot(history.epoch[::10], history.history['val_acc'][::10], label='Test')

plt.xlabel('Accuracy')

plt.ylabel('Epochs')

plt.legend(['Train', 'Test'], loc='upper left', title='Sample', facecolor='white', fancybox=True)
_, train_acc = saved_model.evaluate(train_features_1, train_target_1, verbose=0)

_, test_acc = saved_model.evaluate(dev_features_1, dev_target_1, verbose=0)



print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))
print(classification_report(train_target_1, saved_model.predict_classes(train_features_1)))

print(confusion_matrix(train_target_1, saved_model.predict_classes(train_features_1)))
print(classification_report(dev_target_1, saved_model.predict_classes(dev_features_1)))

print(confusion_matrix(dev_target_1, saved_model.predict_classes(dev_features_1)))
df_test_check = dev_features_1.join(dev_target_1)

df_test_check['predicted'] = saved_model.predict_classes(dev_features_1)



df_test_check = df_test_check.join(df_train['Age'])

df_test_check = df_test_check.join(df_train['Name'])

df_test_check = df_test_check.join(df_train['Sex'])

df_test_check = df_test_check.join(df_train['Pclass'])

df_test_check = df_test_check.join(df_train['Cabin_2'])

df_test_check = df_test_check.join(df_train['family_members'])



df_test_check[df_test_check['predicted']!= df_test_check['Survived']][['Age', \

                                                        'Name', \

                                                        'family_members',\

                                                        'Sex', \

                                                        'Pclass', \

                                                        'Cabin_2',

                                                        'Survived',\

                                                        'predicted']].sort_values('Survived')
df_test['Survived'] = saved_model.predict_classes(test_features)
df_test[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)
from sklearn.ensemble import GradientBoostingClassifier



clf = GradientBoostingClassifier(n_estimators=60, max_depth=5)



clf.fit(train_features_1, train_target_1) 



train_acc = accuracy_score(train_target_1, clf.predict(train_features_1))

test_acc  = accuracy_score(dev_target_1, clf.predict(dev_features_1))



print('Train Accuracy: %.3f, Test Accuracy: %.3f' % (train_acc, test_acc))
df_test['Survived'] = clf.predict(test_features)

df_test[['PassengerId', 'Survived']].to_csv('submission_gb.csv', index=False)
# pd.DataFrame([train_features.columns, np.round(clf.feature_importances_, 2)]).T.sort_values([1], ascending=False)
# df_train[['Cabin_2', 'Sex', 'Survived']].groupby(['Cabin_2', 'Sex']).mean()