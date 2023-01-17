import itertools

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

#####################################################################

# Function preprocessing_vectorized_cat_features

#  Input : Dataframe, a list of header column with categorical features

#  Output : a DataFrame with vectorized column

#####################################################################

def preprocessing_vectorized_cat_features(cat_features, df):

    categories = list([])

    for cat_feature in cat_features:

        l = sorted(list(df[cat_feature].unique()))

        i = 0

        while i < len(l):

            l[i] = cat_feature + '_' + l[i]

            i = i + 1

        categories.append(l)

    labels = list([])

    for categorie in categories:

        labels += categorie



    dv = DictVectorizer(sparse=False)



    vectorized_df = pd.DataFrame(df[cat_features]).convert_objects(convert_numeric=True)

    vectorized_df = dv.fit_transform(vectorized_df.to_dict(orient='records'))

    vectorized_df = pd.DataFrame(vectorized_df, columns=labels)

    return pd.DataFrame(vectorized_df)



df_data = pd.read_csv("../input/SalesKaggle3.csv")

df_data = df_data.sample(frac=1.0, random_state=3)



#We take only the data with a known output

df_data = df_data[df_data.SoldFlag.isnull() == False]



df_data = df_data.drop(['Order'], axis=1)

df_data = df_data.drop(['File_Type'], axis=1)

df_data = df_data.drop(['ReleaseNumber'], axis=1)

df_data = df_data.drop(['ReleaseYear'], axis=1)

df_data = df_data.drop(['New_Release_Flag'], axis=1)

df_data = df_data.drop(['SoldCount'], axis=1)



#I suppose that no positive prices are outliers

df_data = df_data[df_data['PriceReg'] > 0]

df_data = df_data[df_data['LowUserPrice'] > 0]

df_data = df_data[df_data['LowNetPrice'] > 0]



#Some features engineering

df_data['TypeProduct'] = np.floor(df_data['SKU_number'] / 100000)

df_data = df_data.drop(['SKU_number'], axis=1)



#I compute the log in order to get more normal features

df_data['StrengthFactor'] = np.log(df_data['StrengthFactor'])

df_data['PriceReg'] = np.log(df_data['PriceReg'])

df_data['LowUserPrice'] = np.log(df_data['LowUserPrice'])

df_data['LowNetPrice'] = np.log(df_data['LowNetPrice'])





#I transform categorical features as one hot encoder

cat_features = list([])

cont_features = list([])

for column in df_data.columns:

    #only str features are considered as categorical either we have to set also int or other kind of features

    if df_data[column].apply(lambda x : isinstance(x, str)).any():

        cat_features.append(column)

    else :

        cont_features.append(column)



df_data = pd.concat([df_data[cont_features], preprocessing_vectorized_cat_features(cat_features, df_data)], axis=1)



#I delete na values

df_data = df_data.dropna()



df_data_train, df_data_test = train_test_split(df_data, train_size=0.7, random_state=31)



y_train = df_data_train['SoldFlag'].as_matrix()

X_train = df_data_train.drop(['SoldFlag'], axis=1).as_matrix()



y_test = df_data_test['SoldFlag'].as_matrix()

X_test = df_data_test.drop(['SoldFlag'], axis=1).as_matrix()



classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=23)

classifier.fit(X_train, y_train)





false_positive_rateRF_1, true_positive_rateRF_1, thresholdsRF_1 = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])

roc_aucRF_1 = auc(false_positive_rateRF_1, true_positive_rateRF_1)



plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rateRF_1, true_positive_rateRF_1, 'b', label='AUC = %0.2f' % roc_aucRF_1)

plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



print(accuracy_score(y_test, classifier.predict(X_test)))

cnf_matrix = confusion_matrix(y_test, classifier.predict(X_test))

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=[0, 1],

                      title='Confusion matrix, without normalization')

plt.show()
df_data = pd.read_csv("../input/SalesKaggle3.csv")

df_data = df_data.sample(frac=1.0, random_state=3)



df_data = df_data[df_data.SoldFlag.isnull() == False]



df_data['TypeProduct'] = np.floor(df_data['SKU_number'] / 100000)



df_data = df_data.drop(['SKU_number'], axis=1)

df_data = df_data.drop(['Order'], axis=1)

df_data = df_data.drop(['File_Type'], axis=1)

df_data = df_data.drop(['ReleaseNumber'], axis=1)

df_data = df_data.drop(['ReleaseYear'], axis=1)

df_data = df_data.drop(['New_Release_Flag'], axis=1)



df_data = df_data[df_data['PriceReg'] > 0]

df_data = df_data[df_data['LowUserPrice'] > 0]

df_data = df_data[df_data['LowNetPrice'] > 0]



df_data['StrengthFactor'] = np.log(df_data['StrengthFactor'])

df_data['PriceReg'] = np.log(df_data['PriceReg'])

df_data['LowUserPrice'] = np.log(df_data['LowUserPrice'])

df_data['LowNetPrice'] = np.log(df_data['LowNetPrice'])



cat_features = list([])

cont_features = list([])

for column in df_data.columns:

    #only str features are considered as categorical either we have to set also int or other kind of features

    if df_data[column].apply(lambda x : isinstance(x, str)).any():

        cat_features.append(column)

    else :

        cont_features.append(column)



df_data = pd.concat([df_data[cont_features], preprocessing_vectorized_cat_features(cat_features, df_data)], axis=1)

df_data = df_data.dropna()



df_data_train, df_data_test = train_test_split(df_data, train_size=0.7, random_state=31)

print(len(df_data_train[df_data_train['SoldFlag']==1]))

df_data_forced = pd.DataFrame(columns=df_data.columns)

k = 0

for i, row in df_data_train.iterrows():

    if int(row['SoldCount']) == 0:

        df_data_forced.loc[k] = row

        k = k + 1

    else :

        for j in range(int(row['SoldCount'])*2):

            df_data_forced.loc[k] = row

            k = k + 1



df_data_train = df_data_forced

print(len(df_data_train[df_data_train['SoldFlag']==1]))



df_data_train = df_data_train.drop(['SoldCount'], axis=1)

df_data_test = df_data_test.drop(['SoldCount'], axis=1)



df_data_train_zero = df_data_train[df_data_train['SoldFlag'] == 0]

df_data_train_one = df_data_train[df_data_train['SoldFlag'] == 1]



df_data_train_zero = df_data_train_zero.sample(n=int(len(df_data_train_one)))

df_data_train_one = df_data_train_one.sample(n=int(len(df_data_train_one)*8),replace=True, random_state=12)

df_data_train = pd.concat([df_data_train_one, df_data_train_zero])

df_data_train = df_data_train.sample(frac=1.0, random_state=7)



y_train = df_data_train['SoldFlag'].as_matrix()

X_train = df_data_train.drop(['SoldFlag'], axis=1).as_matrix()

y_test = df_data_test['SoldFlag'].as_matrix()

X_test_base = df_data_test.drop(['SoldFlag'], axis=1).as_matrix()



classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=23)

classifier.fit(X_train, y_train)





false_positive_rateRF_1, true_positive_rateRF_1, thresholdsRF_1 = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])

roc_aucRF_1 = auc(false_positive_rateRF_1, true_positive_rateRF_1)



plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rateRF_1, true_positive_rateRF_1, 'b', label='AUC = %0.2f' % roc_aucRF_1)

plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



print(accuracy_score(y_test, classifier.predict(X_test_base)))

cnf_matrix = confusion_matrix(y_test, classifier.predict(X_test_base))

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=[0, 1],

                      title='Confusion matrix, without normalization')

plt.show()
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D



def preprocessing_pca_features(features, df_train, df_test, n_components=3):

    if n_components <= len(features):

        pca = PCA(n_components=n_components)

        pca.fit(df_train)

        list_col_pca = list([])

        i = 0

        while i < n_components:

            i = i + 1

            list_col_pca.append('pca_comp_' + str(i))

        print(df_train.columns)

        df_test_pca = pd.DataFrame(pca.transform(df_test), columns=list_col_pca)

    return df_test_pca





df_pca_test = preprocessing_pca_features(df_data_test.columns, df_data_train, df_data_test, n_components=3).dropna().as_matrix()



fig = plt.figure()

ax = Axes3D(fig)

plt1 = ax.scatter(df_pca_test[y_test == 0] [:, 0],

           df_pca_test[y_test == 0][:, 1],

           df_pca_test[y_test == 0][:, 2])

plt2 = ax.scatter(df_pca_test[y_test == 1][:, 0],

           df_pca_test[y_test == 1][:, 1],

           df_pca_test[y_test == 1][:, 2])

plt.legend((plt1, plt2), ('unsold', 'sold'))

plt.show()
from keras.models import Sequential

from keras.layers import Dropout

from keras.layers import Input, Dense

from keras.models import Model

from sklearn.preprocessing import StandardScaler

import tensorflow as tf



# np.random.seed(13)

# tf.set_random_seed(13)



print(np.random.get_state())

# print(np.random.get_state)



X = X_train

sc = StandardScaler()

X = sc.fit_transform(X)

print(X.shape)

X_test = sc.transform(X_test_base)



input_img = Input(shape=(X.shape[1],))

encoded = Dense(int(X.shape[1]), activation='tanh')(input_img)

encoded = Dense(int(X.shape[1]), activation='tanh')(encoded)

encoded = Dense(3, activation='tanh')(encoded)



encoder = Model(input_img, encoded)

decoded = Dense(int(X.shape[1]), activation='tanh')(encoded)

decoded = Dense(int(X.shape[1]), activation='tanh')(decoded)

decoded = Dense(int(X.shape[1]), activation='sigmoid')(decoded)



autoencoder = Model(input_img, decoded)



autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(X, X,

                epochs=100,

                batch_size=50,

                shuffle=True,

                validation_data=(X_test, X_test))



encoded_X_test = encoder.predict(X_test)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = Axes3D(fig)

plt1 = ax.scatter(encoded_X_test[y_test == 0][:, 0],

           encoded_X_test[y_test == 0][:, 1],

           encoded_X_test[y_test == 0][:, 2])

plt2 = ax.scatter(encoded_X_test[y_test == 1][:, 0],

           encoded_X_test[y_test == 1][:, 1],

           encoded_X_test[y_test == 1][:, 2])

plt.legend((plt1, plt2), ('unsold', 'sold'))

plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = Axes3D(fig)

plt1 = ax.scatter(encoded_X_test[df_data_test['MarketingType_D'] == 0][:, 0],

           encoded_X_test[df_data_test['MarketingType_D'] == 0][:, 1],

           encoded_X_test[df_data_test['MarketingType_D'] == 0][:, 2])

plt2 = ax.scatter(encoded_X_test[df_data_test['MarketingType_D'] == 1][:, 0],

           encoded_X_test[df_data_test['MarketingType_D'] == 1][:, 1],

           encoded_X_test[df_data_test['MarketingType_D'] == 1][:, 2])

plt.legend((plt1, plt2), ('S', 'D'))

plt.show()
cnf_matrix = confusion_matrix(y_test, classifier.predict_proba(X_test_base)[:, 1] > 0.6)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=[0, 1],

                      title='Confusion matrix, without normalization')

plt.show()