import numpy as np 

import pandas as pd

import os

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

import plotly

import plotly.graph_objects as go

from plotly.subplots import make_subplots



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, accuracy_score, f1_score

from sklearn.manifold import TSNE



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier



import tensorflow as tf

from tensorflow.keras import Sequential, Model

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers

from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



from numpy.random import seed

seed(0)



tf.random.set_seed(0)

df= pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.shape
df.info()
df.isna().sum()
df.describe()
df.drop(["Time"], axis=1, inplace=True)
count_classes = df['Class'].value_counts(normalize=True)



fig= go.Figure(go.Bar(x=count_classes.index,

                      y=count_classes.values,

                     width=0.6,

                     text= count_classes.values.round(decimals=4),

                     textposition='outside',

                     textfont=dict(color='black',

                                  size=15)))



colors=['green','red']



fig.update_layout(title_text='Fraud class histogram',

                 xaxis_title='Class',

                 yaxis_title='Percentage',

                 width=650,

                 height=600,

                titlefont=dict(size=22, color='black')

                  

                 )



fig.update_traces(marker_line_width=1.5,

                 marker_line_color='black',

                 marker_color=colors)

# amount of fraud classes 492 rows.

fraud_df = df.loc[df['Class'] == 1]

non_fraud_df = df.loc[df['Class'] == 0][:492]



normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)



new_df.head()
f, (ax1,ax2)= plt.subplots(2,1,figsize=(24,20))



sns.heatmap(df.corr(), cmap=sns.color_palette("RdBu_r", 10), annot_kws={'size':20}, ax=ax1)

ax1.set_title('Imbalanced Correlation Heatmap', fontdict=dict(fontsize=20))



sns.heatmap(new_df.corr(), cmap=sns.color_palette("RdBu_r", 10), annot_kws={'size':20}, ax=ax2)

ax2.set_title('Balanced Correlation Heatmap', fontdict=dict(fontsize=20))



plt.show()
non_fraudulent_df= df[df.Class==0]

fraudulent_df= df[df.Class==1]
params = {'axes.titlesize':'45'}

matplotlib.rcParams.update(params)



df.hist(bins=50, figsize=(80,60))

plt.suptitle('Imbalanced dataset histogram graphs', fontsize=60)

plt.show()
params = {'axes.titlesize':'45'}

matplotlib.rcParams.update(params)



new_df.hist(bins=50, figsize=(80,60))

plt.suptitle('Balanced dataset histogram graphs', fontsize=60)



plt.show()
columns= df.columns.tolist()



columns=[c for c in columns if c not in ['Class']]



target= 'Class'



X= df[columns]

Y= df[target]



# Whole dataset

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = 0.3, random_state=42)
X_train.loc[:,'Amount']= RobustScaler().fit_transform(X_train.loc[:,'Amount'].values.reshape(-1, 1))

X_test.loc[:,'Amount']= RobustScaler().fit_transform(X_test.loc[:,'Amount'].values.reshape(-1, 1))
X_train_normal = X_train[y_train==0]

X_train_fraud = X_train[y_train==1]
normal_df_sample= non_fraudulent_df.sample(n=3000, random_state=42)

normal_df_sample
df1= fraudulent_df.append(normal_df_sample)
x1 = df1.drop(['Class'], axis = 1).values

y1 = df1["Class"].values
def tsne_plot(x1, y1, name="graph.png"):

    tsne = TSNE(n_components=2, random_state=42)

    X_t = tsne.fit_transform(x1)



    plt.figure(figsize=(12, 8))

    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Non Fraud')

    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Fraud')



    plt.legend(loc='best');

    plt.savefig(name);

    plt.show();

    

tsne_plot(x1, y1, "original.png")
input_layer = Input(shape=(X.shape[1], ))

encoded1 = Dense(29,activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)



latent_view= Dense(15,activation='relu')(encoded1)



decoded1 = Dense(29,activation='relu')(latent_view)



output_layer = Dense(X.shape[1])(decoded1)
autoencoder= Model(input_layer, output_layer)



autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_train_normal,X_train_normal, epochs=20, batch_size=256,

                shuffle = True, validation_split = 0.20)
encoder_all= Sequential([autoencoder.layers[0], autoencoder.layers[1], autoencoder.layers[2]])

enc_X_train= encoder_all.predict(X_train)
norm_hid_rep =encoder_all.predict(X_train_normal[:3000])

fraud_hid_rep= encoder_all.predict(X_train_fraud)
rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)

y_n = np.zeros(norm_hid_rep.shape[0])

y_f = np.ones(fraud_hid_rep.shape[0])

rep_y = np.append(y_n, y_f)

tsne_plot(rep_x, rep_y, "latent_representation.png")
knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(enc_X_train,y_train)
knn_predicted= knn_model.predict(encoder_all.predict(X_test))
pd.Series(knn_predicted).value_counts()
params = {'figure.figsize': (15, 10),

         'axes.titlesize':20}



matplotlib.rcParams.update(params)
conf_matrix = confusion_matrix(y_test,knn_predicted)



ax=plt.subplot()



sns.heatmap(conf_matrix,annot=True,ax=ax,fmt='g')#annot=True to annotate cells, fmt='g' numbers not scientific form

ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')

ax.set_title('KNN Confusion Matrix'); 

ax.xaxis.set_ticklabels(['Normal', 'Fraud']); ax.yaxis.set_ticklabels(['Normal', 'Fraud']);

ax.set(yticks=[0, 2], 

       xticks=[0.5, 1.5])





ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
print(classification_report(y_test, knn_predicted))
accuracy= round(accuracy_score(y_test,knn_predicted)*100, 2)

print(f'Accuracy Score for KNN: {accuracy}%')
input_layer = Input(shape=(enc_X_train.shape[1], ))

layer2 = Dense(8,activation='relu')(input_layer)

layer3 = Dense(4,activation='relu')(layer2)

layer4 = Dense(1,activation='sigmoid')(layer3)



model= Model(input_layer, layer4)
model.compile(optimizer=Adam(learning_rate= 0.005), loss='binary_crossentropy', metrics='Recall')
early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=1e-5) #stop training if loss does not decrease with at least 0.00001

reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, min_delta=1e-5, factor=0.2) #reduce learning rate (divide it by 5 = multiply it by 0.2) if loss does not decrease with at least 0.00001



callbacks = [early_stopping, reduce_lr]



model.fit(enc_X_train,y_train,batch_size=128, epochs=30, validation_data=[encoder_all.predict(X_test), y_test])
neural_network_prediction_train= model.predict(encoder_all.predict(X_train))
mse = np.mean(np.power(y_train.values.reshape(-1,1) - neural_network_prediction_train, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y_train})

error_df.groupby('true_class').describe()
normal_mean= error_df.groupby('true_class').describe().loc[0,'reconstruction_error']['mean']

normal_std= error_df.groupby('true_class').describe().loc[0,'reconstruction_error']['std']
test_predictions= model.predict(encoder_all.predict(X_test))

mse = np.mean(np.power(y_test.values.reshape(-1,1) - test_predictions, 2), axis=1)

y_pred=[(lambda er: 1 if er>=(normal_mean+ 3*normal_std)  else 0)(er) for er in mse]
pd.Series(y_pred).value_counts()
pd.Series(y_test).value_counts()
conf_matrix_nn = confusion_matrix(y_test,y_pred)



ax=plt.subplot()

sns.heatmap(conf_matrix_nn,annot=True,ax=ax,fmt='g')#annot=True to annotate cells, fmt='g' numbers not scientific form

ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')

ax.set_title('Neural Network Confusion Matrix'); 

ax.xaxis.set_ticklabels(['Normal', 'Fraud']); ax.yaxis.set_ticklabels(['Normal', 'Fraud']);

ax.set(yticks=[0, 2], 

       xticks=[0.5, 1.5])

ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
print(classification_report(y_test,y_pred))
accuracy= round(accuracy_score(y_test,y_pred)*100, 2)

print(f'Accuracy Score for Simple Neural Network: {accuracy}%')
clf= XGBClassifier(random_state=42)
clf= XGBClassifier()

clf.fit(enc_X_train, y_train)
XGB_prediction= clf.predict(encoder_all.predict(X_test))
pd.Series(XGB_prediction).value_counts()
conf_matrix_nn = confusion_matrix(y_test,XGB_prediction)



ax=plt.subplot()

sns.heatmap(conf_matrix_nn,annot=True,ax=ax,fmt='g')#annot=True to annotate cells, fmt='g' numbers not scientific form

ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')

ax.set_title('XGB Confusion Matrix'); 

ax.xaxis.set_ticklabels(['Normal', 'Fraud']); ax.yaxis.set_ticklabels(['Normal', 'Fraud']);

ax.set(yticks=[0, 2], 

       xticks=[0.5, 1.5])

ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
print(classification_report(y_test,XGB_prediction))
accuracy= round(accuracy_score(y_test,XGB_prediction)*100, 2)

print(f'Accuracy Score for XGBClassifier: {accuracy}%')
X_oversampled_train, Y_oversampled_train= SMOTE(sampling_strategy='minority').fit_sample(enc_X_train,y_train)
clf= XGBClassifier()

clf.fit(X_oversampled_train, Y_oversampled_train)
X_oversampled_train.shape
XGB_over_prediction= clf.predict(encoder_all.predict(X_test))
pd.Series(XGB_over_prediction).value_counts()
conf_matrix_nn = confusion_matrix(y_test,XGB_over_prediction)



ax=plt.subplot()

sns.heatmap(conf_matrix_nn,annot=True,ax=ax,fmt='g')#annot=True to annotate cells, fmt='g' numbers not scientific form

ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')

ax.set_title('Oversampled XGB Confusion Matrix'); 

ax.xaxis.set_ticklabels(['Normal', 'Fraud']); ax.yaxis.set_ticklabels(['Normal', 'Fraud']);

ax.set(yticks=[0, 2], 

       xticks=[0.5, 1.5])

ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
print(classification_report(y_test,XGB_over_prediction))