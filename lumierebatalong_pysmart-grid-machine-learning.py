# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from scipy.stats import normaltest, norm

import scipy as sp

import holoviews as hv

from holoviews import opts

hv.extension('bokeh')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Perceptron, LogisticRegression

from xgboost import XGBRegressor, XGBRFClassifier

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, RocCurveDisplay,confusion_matrix

from sklearn.metrics import plot_roc_curve, roc_auc_score, classification_report, accuracy_score, f1_score

from sklearn.metrics import recall_score, plot_confusion_matrix, precision_score, plot_precision_recall_curve, classification_report

    

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
sns.set()

pd.plotting.register_matplotlib_converters()

%matplotlib inline

print("Setup Complete")
file = '/kaggle/input/smart-grid/Data_for_UCI_named.csv'
uci_data = pd.read_csv(file)
uci_data.tail()
uci_data.info()
uci_data.describe()
stabf_count = uci_data.stabf.value_counts()#

stabf_count
print('stabf target have {}% for unstable and {}% for stable.'.format(round(100*(stabf_count[0]/stabf_count.sum())),

                                                                  round(100*(stabf_count[1]/stabf_count.sum()))))
plt.figure(dpi=100)

sns.countplot(uci_data.stabf)

plt.title(r'Stabf categorical count.')

plt.show()
# we plot also stab

fig, axes = plt.subplots(ncols=2, figsize=(17, 5), dpi=100)

plt.tight_layout()



sns.barplot(x='stabf', y='stab', data=uci_data, ax=axes[0])

sns.boxplot(x='stabf', y='stab', data=uci_data, ax=axes[1])

axes[0].set_title('Stab Bar')

axes[1].set_title('Stabf Box')

plt.show()
fig, axis = plt.subplots(ncols=2, figsize=(17, 5), dpi=100)

plt.tight_layout()



sns.boxplot(uci_data.stab, ax = axis[0])

sns.distplot(uci_data['stab'], fit=norm, kde=False,  ax = axis[1], norm_hist=True)



axis[0].set_title('Stab box')

axis[1].set_title('Stab distplot')

plt.show()
#test if stab is normal

# the null hypothesis is data come from normal distribution if p > alpha the null hypo. cannot rejected .

normaltest(uci_data.stab)
cols = list(set(uci_data.columns) - set(['stab', 'stabf']))
cols = sorted(cols)
def distplot_multi(data):

    """ plot multi distplot"""

        

    from scipy.stats import norm

    cols = []

        

    #Feature that is int64 or float64 type 

    for i in data.columns:

        if data[i].dtypes == "float64" or data[i].dtypes == 'int64':

                cols.append(i)

        

    gp = plt.figure(figsize=(20,20))

    gp.subplots_adjust(wspace=0.4, hspace=0.4)

    for i in range(1, len(cols)+1):

        ax = gp.add_subplot(3,4,i)

        sns.distplot(data[cols[i-1]], fit=norm, kde=False)

        ax.set_title('{} max. likelihood gaussian'.format(cols[i-1]))
def boxplot_multi(data):

        

    """ plot multi box plot

        hue for plotting categorical data

    """

    

    cols = []

    for i in data.columns:

        if data[i].dtypes == "float64" or data[i].dtypes == 'int64':

            cols.append(i)

    

    gp = plt.figure(figsize=(20,20))

    gp.subplots_adjust(wspace=0.4, hspace=0.4)

    for i in range(1, len(cols)+1):

        ax = gp.add_subplot(3, 4, i)

        sns.boxplot(x = cols[i-1], data=data)

        ax.set_title('Boxplot for {}'.format(cols[i-1]))
def correlation_plot(data, vrs= 'stab', vsr='stabf'):

    

    """

    This function plot only a variable that are correlated with a target  

        

        data: array m_observation x n_feature

        vrs:  target feature (n_observation, )

        cols: interested features

    """

    cols = data.columns # we take all feature

                

    feat = list(set(cols) - set([vrs, vsr]))

    

    fig = plt.figure(figsize=(20, 20))

    fig.subplots_adjust(wspace = 0.3, hspace = 0.3)

    

    for i in range(1,len(feat)+1):

        

        ax = fig.add_subplot(2, 2, i)     

        sns.scatterplot(x=data[feat[i-1]], y=data[vrs], data=data, hue=vsr, ax=ax)   

        ax.set_xlabel(feat[i-1])

        ax.set_ylabel(vrs)

        ax.set_title('Plotting data {0} vs {1}'.format(vrs, feat[i-1]))

        ax.legend(loc='best')
distplot_multi(uci_data[cols])
#we check if p1 comes from normal distribution

normaltest(uci_data.p1)
boxplot_multi(uci_data[cols])
uci_data.corr(method='spearman')
plt.figure(dpi=100, figsize=(15,5))

sns.heatmap(uci_data.corr(method='spearman'), robust=True, annot=True)

plt.show()
correlation_plot(uci_data[['tau1', 'tau2', 'tau3', 'tau4', 'stab', 'stabf']])
#for gamma

correlation_plot(uci_data[['g1', 'g2', 'g3', 'g4', 'stab', 'stabf']])
correlation_plot(uci_data[['p2', 'p3', 'p4', 'p1', 'stabf']], vrs='p1')
uci_data['stabf'] = uci_data['stabf'].astype('category')

uci_data['stabf'].cat.categories = [0, 1] # 0 for stable, 1 for unstable

uci_data['stabf'] = uci_data['stabf'].astype('int')
uci_data.tail()
#we  define data, reg_target, clas_target

data = uci_data.drop(columns=['p1', 'stab', 'stabf'])
reg_target = uci_data['stab'] #target for regression

clas_target = uci_data['stabf']#target fot classification
# regression

rtrain, rtest, rytrain, rytest = train_test_split(data, reg_target, test_size=0.2, random_state=42)
print('Regression: xtrain: {}, xtest: {}, ytrain: {}, ytest: {}'.format(rtrain.shape, rtest.shape,

                                                                        rytrain.shape, rytest.shape))
# classification

ctrain, ctest, cytrain, cytest = train_test_split(data, clas_target, stratify=clas_target, test_size=0.2,

                                                 random_state=42)
print('Classification: xtrain: {}, xtest: {}, ytrain: {}, ytest: {}'.format(ctrain.shape, ctest.shape,

                                                                        cytrain.shape, cytest.shape))
class toolsGrid:

    """

        This class contains all function for classification and regresssion

    """

    

    def __init__(self, xtrain=None, ytrain=None):

        

        self.xtrain = xtrain # train data

        self.ytrain = ytrain # train target data

        

        # list of different learner for regression

        self.reg_model = {'LinearRegression': LinearRegression(), 

                'KNeighborsRegression': KNeighborsRegressor(),

                'RandomForestRegression': RandomForestRegressor(),

                'GradientBoostingRegression': GradientBoostingRegressor(),

                'XGBoostRegression': XGBRegressor(),

                'AdaboostRegression': AdaBoostRegressor(),

                'ExtraTreesRegressor': ExtraTreesRegressor()}

        

        # list of different learner for classification

        self.clas_model = {'LogisticRegression': LogisticRegression(), 

                'KNeighborsClassifier': KNeighborsClassifier(),

                'RandomForestClassifier': RandomForestClassifier(),

                'GradientBoostingClassifier': GradientBoostingClassifier(),

                'XGBoostClassifier': XGBRFClassifier(),

                'AdaboostClassifier': AdaBoostClassifier(),

                'ExtraTreesClassifier': ExtraTreesClassifier(),

                'Perceptron': Perceptron()}

        

    

    def split_data(self, label= True):

        """

        This function splits data to train set and target set

        

        data: matrix feature n_observation x n_feature dimension

        

        return Xtrain, Xvalid, Ytrain, Yvalid

        """

        if label:

            Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(self.xtrain, self.ytrain, random_state=42,

                                                          test_size=0.2, shuffle=True, stratify=self.ytrain)

        else:

            Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(self.xtrain, self.ytrain, random_state=42,

                                                          test_size=0.2, shuffle=True)

        

        return Xtrain, Xvalid, Ytrain, Yvalid

    

    

    def regression_learner_selection(self):



        """

            This function compute differents score measure like cross validation,

            r2, root mean squared error and mean absolute error.

            reg_model: dictionary type containing different model algorithm.     

        """ 

    

        result = {}

        

        #x, _, y, _ = self.split_data() # take only xtrain and ytrain

    

        # we take each regression model

        for cm in list(self.reg_model.items()):

        

            name = cm[0] #name of learner

            model = cm[1] # learner

        

            cvs = cross_val_score(model, self.xtrain, self.ytrain, cv=10).mean() #mean of cv score

            ypred = cross_val_predict(model, self.xtrain, self.ytrain, cv=10) #prediction cv

            r2 = r2_score(self.ytrain, ypred)

            mse = mean_squared_error(self.ytrain, ypred)

            mae = mean_absolute_error(self.ytrain, ypred)

            rmse = np.sqrt(mse)

        

            result[name] = {'cross_val_score': cvs, 'rmse': rmse, 'mae': mae, 'r2': r2}

        

            print('{} model done !!!'.format(name))

            

        return pd.DataFrame(result)

            

    

    def classification_learner_selection(self):



        """

            This function compute differents score measure like cross validation,

            auc, accuracy, recall, precision and f1.

            reg_model: dictionary type containing different model algorithm.     

        """ 

    

        result = {}

        matrix = []

        

        #

    

        # we take each classification model

        for cm in list(self.clas_model.items()):

        

            name = cm[0] #name of learner

            model = cm[1] # learner

        

            cvs = cross_val_score(model, self.xtrain, self.ytrain, cv=10).mean() #mean of cv score

            ypred = cross_val_predict(model, self.xtrain, self.ytrain, cv=10) #prediction cv

            auc = roc_auc_score(self.ytrain, ypred)

            acc = accuracy_score(self.ytrain, ypred)

            recall = recall_score(self.ytrain, ypred)

            precision = precision_score(self.ytrain, ypred)

            f1 = f1_score(self.ytrain, ypred)

        

            result[name] = {'cross_val_score': cvs, 'auc': auc, 'acc': acc, 'precision': precision,

                           'recall':recall, 'f1': f1}

        

            print('{} model done !!!'.format(name))

            

        return pd.DataFrame(result)

    

    

    def confusion_matrix(self):

        """

            plot confusion matrix

        """

        xtrain, xvalid, ytrain, yvalid = self.split_data() # take only xtrain and ytrain

        

        

        feat = list(self.clas_model.keys()) # we take all learner

        

        fig = plt.figure(figsize=(20, 20))

        fig.subplots_adjust(wspace = 0.4, hspace = 0.4)

    

        for i in range(1,len(feat)+1):

        

            ax = fig.add_subplot(2, 4, i)   

            learner = self.clas_model[feat[i-1]]

            

            plot_confusion_matrix(learner.fit(xtrain, ytrain), xvalid, yvalid,

                                  labels=[0,1], ax=ax)  

           

            ax.set_title('{} Conf. Matrix'.format(feat[i-1]))

            plt.grid(False) 

            

    

    def roc_auc(self):

        """

            plot roc_auc

        """

        xtrain, xvalid, ytrain, yvalid = self.split_data() # take only xtrain and ytrain

        

        feat = list(self.clas_model.keys()) # we take all learner

        

        fig = plt.figure(figsize=(20, 20))

        fig.subplots_adjust(wspace = 0.4, hspace = 0.4)

    

        for i in range(1,len(feat)+1):

        

            ax = fig.add_subplot(2, 4, i)   

            learner = self.clas_model[feat[i-1]]

            

            plot_roc_curve(learner.fit(xtrain, ytrain), xvalid, yvalid, ax=ax)  

           

            ax.set_title('{} ROC'.format(feat[i-1]))

            plt.grid(False) 

            

    def precision_recall(self):

        """

            plot precision recall

        """

        xtrain, xvalid, ytrain, yvalid = self.split_data() # take only xtrain and ytrain

        

        feat = list(self.clas_model.keys()) # we take all learner

        

        fig = plt.figure(figsize=(20, 20))

        fig.subplots_adjust(wspace = 0.4, hspace = 0.4)

    

        for i in range(1,len(feat)+1):

        

            ax = fig.add_subplot(2, 4, i)   

            learner = self.clas_model[feat[i-1]]

            

            plot_precision_recall_curve(learner.fit(xtrain, ytrain), xvalid, yvalid, ax=ax)  

           

            ax.set_title('{} PR'.format(feat[i-1]))

            plt.grid(False) 

        

            

            
electrical = toolsGrid(xtrain=ctrain, ytrain=cytrain)
result = electrical.classification_learner_selection()
result
electrical.confusion_matrix()
electrical.roc_auc()
electrical.precision_recall()
model = toolsGrid(xtrain=rtrain, ytrain=rytrain)
model.regression_learner_selection()
#create neural network function



def neural_network(output_activation=None, data=None, n=None):

    

    """

        neural network function

        output_activation is for the last dense

        n number of output unit

        

    """

    inputs = keras.Input(shape=(data.shape[1], ), name = 'Electrical_Grid') #input for data

    

    # first hidden dense 100 neurons with dropout layers

    x = layers.Dense(units=100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02),

                    name='dense_1')(inputs)

    x = layers.Dropout(0.2)(x)

    

    # second hidden dense 100 neurons with dropout layers

    x = layers.Dense(units=100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02),

                    name='dense_2')(x)

    x = layers.Dropout(0.2)(x)

    

    #output dense

    outputs = layers.Dense(units=n, activation=output_activation, name='prediction')(x) 

    

    model = keras.Model(inputs=inputs, outputs=outputs) # create model

    

    return model
#we start

model_classifier = neural_network(output_activation='sigmoid', data=ctrain, n=1)
model_classifier.summary()
#we can see my model

keras.utils.plot_model(model_classifier, "multi_input_and_output_model.png", show_shapes=True)
model_classifier.compile(loss='binary_crossentropy', metrics=['AUC','accuracy','Precision', 'Recall'])
# Load the TensorBoard notebook extension

%load_ext tensorboard
tensorboard_callback = keras.callbacks.TensorBoard( log_dir="/logs/classification",

    histogram_freq=1,  # How often to log histogram visualizations

    embeddings_freq=0,  # How often to log embedding visualizations

    update_freq="epoch",

)  # How often to write logs (default: once per epoch)



history_classifier = model_classifier.fit(ctrain, cytrain, epochs=200, callbacks=[tensorboard_callback], verbose=0, batch_size=128,

                     validation_split=0.2)
hist_class = pd.DataFrame(history_classifier.history).rolling(window=16).mean()
lister = list(history_classifier.history.keys())
fig = plt.figure(figsize=(20,20))

fig.subplots_adjust(hspace=0.2, wspace=0.2)



for i in range(5):

    ax = fig.add_subplot(2,3,i+1)

    ax.plot(hist_class[lister[i]])

    ax.plot(hist_class[lister[i+5]])

    ax.set_title(lister[i] + ' vs '+ lister[i+5])

    ax.set_xlabel('epoch')

    ax.legend([lister[i], lister[i+5]], loc='upper left')
regression_model = neural_network(output_activation='linear', data=rtrain, n=1)
def coef_r2(y_true, y_pred):

    ## we campute a coef. determination

    

    y_true = tf.cast(y_true, dtype=tf.float64)

    y_pred = tf.cast(y_true, dtype=tf.float64)

    ss_res = keras.backend.sum(keras.backend.square(y_true - y_pred))

    ss_tot = keras.backend.sum(keras.backend.square(y_true - keras.backend.mean(y_true)))

    

    return (1 - ss_res/(ss_tot))
regression_model.compile(loss='mse', metrics=[coef_r2, 'mse','mae'])
board_callback = keras.callbacks.TensorBoard( log_dir="logs",

    histogram_freq=1,  # How often to log histogram visualizations

    embeddings_freq=1,  # How often to log embedding visualizations

    update_freq="epoch",

)  # How often to write logs (default: once per epoch)



history = regression_model.fit(rtrain, rytrain, epochs=200, callbacks=[board_callback], verbose=0, batch_size=128,

                     validation_split=0.2)
story = pd.DataFrame(history.history)
st = list(story.keys())
ig = plt.figure(figsize=(20,20))

ig.subplots_adjust(hspace=0.2, wspace=0.2)



for i in range(4):

    ax = ig.add_subplot(2,2,i+1)

    ax.plot(story[st[i]])

    ax.plot(story[st[i+3]])

    ax.set_title(st[i] + ' vs '+ st[i+4])

    ax.set_xlabel('epoch')

    ax.legend([st[i], st[i+4]], loc='upper left')
tree = toolsGrid(xtrain=ctrain, ytrain=cytrain)
#we split our data

X_train, X_valid, Y_train, Y_valid = tree.split_data()
print('X_train: {}, X_valid: {}'.format(X_train.shape, X_valid.shape))
extratree = ExtraTreesClassifier(n_estimators=2000, criterion='entropy', random_state=42, n_jobs=-1)
extratree.fit(X_train, Y_train)
Y_pred = extratree.predict(X_valid)
#we compute auc, acc, precision, recal and f1

print(classification_report(Y_valid, Y_pred)) # 0 for stable and 1 for unstable
#AUC

print('AUC for ExtraTreeClassifier: {}'.format(roc_auc_score(Y_valid, Y_pred)))
xgboost = toolsGrid(xtrain=rtrain, ytrain=rytrain)
#we split data

x_train, x_valid, y_train, y_valid = xgboost.split_data(label=False)
print('x_train: {}, x_valid: {}'.format(x_train.shape, x_valid.shape))
xgb = XGBRegressor(n_estimators=1000,learning_rate=0.100000012, importance_type='entropy', random_state=42)
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_valid)
print('r2: {}, mae: {}, rmse: {}'.format(r2_score(y_valid, y_pred), mean_absolute_error(y_valid, y_pred),

                                        np.sqrt(mean_squared_error(y_valid, y_pred))) )
model_classifier.evaluate(ctest, cytest)
proba_class = model_classifier.predict(ctest)
rypred = xgb.predict(rtest)
print('r2: {}, mae: {}, rmse: {}'.format(r2_score(rytest, rypred), mean_absolute_error(rytest, rypred),

                                        np.sqrt(mean_squared_error(rytest, rypred))))
prediction = pd.DataFrame()
# Classification

prediction['stabf'] = cytest.values

prediction['nn_proba'] = proba_class

prediction['stab'] = rytest.values

prediction['xgboost'] = rypred
prediction.head(10) #classification