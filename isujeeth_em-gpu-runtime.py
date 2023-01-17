from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.decomposition import FastICA
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import silhouette_score
from sklearn.random_projection import GaussianRandomProjection


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


%matplotlib inline
data=pd.read_csv("../input/sgemm/sgemm_product.csv")
data.head()
#take average of 4 run
data["run_avg"]=np.mean(data.iloc[:,14:18],axis=1)

mean_run=np.mean(data["run_avg"])
print(mean_run)

#Binary Classification run_avg>mean_run
data["run_class"]=np.where(data['run_avg']>=mean_run, 1, 0)
data.groupby("run_class").size()

data.describe()
sgemm_df=data.drop(columns=['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)','run_avg'])
sgemm_df.to_csv(r'segmm_product_classification.csv')
sgemm_df.head()
#data info
sgemm_df.info()
#No null values in the data
#checking for NULL values
sgemm_df.isnull().sum() #no NULL values


df_test=sgemm_df.iloc[:,0:14]


#checking variable distribution
for index in range(10):
    df_test.iloc[:,index] = (df_test.iloc[:,index]-df_test.iloc[:,index].mean()) / df_test.iloc[:,index].std();
df_test.hist(figsize= (14,16));
plt.figure(figsize=(14,14))
sns.set(font_scale=1)
sns.heatmap(df_test.corr(),cmap='GnBu_r',annot=True, square = True ,linewidths=.5);
plt.title('Variable Correlation')
#Varibale and predictor
y=np.array(sgemm_df["run_class"])

X=np.array(sgemm_df.iloc[:,0:14])



sc = StandardScaler()
cluster_data = sc.fit_transform(X)

cluster_data[:10]
def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]


# fit model
model = GaussianMixture(n_components=4,covariance_type='tied',random_state=42)
model.fit(X)
# predict latent values
yhat = model.predict(X)
s = silhouette_score(X, yhat)
print('Silhouette Score:', s)
# Create a PCA instance: pca
pca = PCA(n_components=14)
principalComponents = pca.fit_transform(X)# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
print(pca.explained_variance_ratio_)
# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

PCA_components.iloc[:,:3]
# fit model
model = GaussianMixture(n_components=4,covariance_type='tied',random_state=42)
model.fit(PCA_components.iloc[:,:3])
# predict latent values
yhat_pca = model.predict(PCA_components.iloc[:,:3])
yhat_prob_pca=model.predict_proba(PCA_components.iloc[:,:3])
s = silhouette_score(PCA_components.iloc[:,:3], yhat_pca)
print('Silhouette Score:', s)
import plotly.express as px
#df = px.data.iris()
fig = px.scatter_3d(PCA_components.iloc[:,:3], x=0, y=1, z=2,
            color=yhat_pca)
fig.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train, y_train)
boolvec=sel.get_support()
boolvec.astype(bool)
boolvec
input_file=sgemm_df.iloc[:,0:14]
#X_RF=input_file.loc[:, sel.get_support()]
#input_file=sgemm_df.loc[:, sel.get_support()].head()
selected_feat= input_file.columns[(sel.get_support())]
#selected_feat = np.where(boolvec[:,None], X_train,X_train)
len(selected_feat)
print(selected_feat)
#sgemm_df

X_RF=input_file.loc[:, sel.get_support()]

# fit model
model = GaussianMixture(n_components=4,covariance_type='tied',random_state=42)
model.fit(X_RF)
# predict latent values
yhat_rf = model.predict(X_RF)
yhat_prob_rf = model.predict(X_RF)

#labels=gmm.predict(X)
sil=silhouette_score(X_RF, yhat_rf, metric='euclidean')
print('Silhouette Score:', s)
rca = GaussianRandomProjection(n_components=2, eps=0.1, random_state=42)
X_rca=rca.fit_transform(X)
# fit model
model = GaussianMixture(n_components=4,covariance_type='tied',random_state=42)
model.fit(X_rca)
# predict latent values
yhat_rca = model.predict(X_rca)
yhat_prob_rca = model.predict(X_rca)
#labels=gmm.predict(X)
sil=silhouette_score(X_rca, yhat_rca, metric='euclidean')
print('Silhouette Score:', s)
plt.scatter(X_rca[:, 0], X_rca[:, 1], c=yhat_rca, s=50, cmap='viridis')
plt.title('EM - RCA')

ICA = FastICA(n_components=2, random_state=42) 
X_ica=ICA.fit_transform(X)


# fit model
model = GaussianMixture(n_components=4,covariance_type='tied',random_state=42)
model.fit(X_ica)
# predict latent values
yhat_ica = model.predict(X_ica)
yhat_prob_ica = model.predict(X_ica)
#labels=gmm.predict(X)
sil=silhouette_score(X_rca, yhat_ica, metric='euclidean')
print('Silhouette Score:', s)
plt.scatter(X_ica[:, 0], X_ica[:, 1], c=yhat_ica, s=50, cmap='viridis')
plt.title('EM - ICA')
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score, precision_score, recall_score,f1_score
from sklearn.metrics import make_scorer, accuracy_score
import time
yhat_prob_pca

yhat_pca
X=np.array(yhat_prob_pca)
y=np.array(yhat_pca)
#Train Test Validation Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


y_train_enc = to_categorical(y_train)
y_test_enc = to_categorical(y_test)
print(y_train_enc.shape)




# define the model
#get number of columns in training data
n_cols=X_train.shape[1]

# define model 2 layers
model = Sequential()
model.add(Dense(100, input_dim=n_cols, activation='relu'))
model.add(Dense(50,  activation='relu'))
model.add(Dense(4, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs= 200
start=time.time()
#fit model
hist=model.fit(X_train, y_train_enc,  validation_split=0.2, epochs=epochs,batch_size=100, verbose=1)
end=time.time()
print("Elapsed Time: ", end-start)
# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=1)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
#precision = precision_score(y_test, yhat_classes)
#print('Precision: %f' % precision)
# recall: tp / (tp + fn)
#recall = recall_score(y_test, yhat_classes)
#print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(y_test, yhat_classes)
#print('F1 score: %f' % f1)


# ROC AUC
auc = roc_auc_score(y_test_enc, yhat_probs)
print('ROC AUC: %f' % auc)




# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print(matrix)




# plot loss during training
plt.figure(1, figsize=(10,12))
plt.subplot(211)
plt.title('Loss')
plt.plot(hist.history['loss'], label='Train')
plt.plot(hist.history['val_loss'], label='Validation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(hist.history['accuracy'], label='Train')
plt.plot(hist.history['val_accuracy'], label='Validation')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

