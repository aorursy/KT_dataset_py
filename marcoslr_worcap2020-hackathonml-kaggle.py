!cp ../input/desafio-worcap-2020/*.csv
import pandas as pd

train = pd.read_csv('treino.csv')
train = train.set_index('id')
#train.info()
train.describe(include='all')
train.label.value_counts()
#train.label.unique()
test = pd.read_csv('teste.csv')
test = test.set_index('id')
test.info()
# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from sklearn.naive_bayes import GaussianNB

#------------------------------------------------------------
# Simulate some data
np.random.seed(0)
mu1 = [1, 1]
cov1 = 0.3 * np.eye(2)

mu2 = [5, 3]
cov2 = np.eye(2) * np.array([0.4, 0.1])

X = np.concatenate([np.random.multivariate_normal(mu1, cov1, 100),
                    np.random.multivariate_normal(mu2, cov2, 100)])

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5), dpi= 80, facecolor='w',
                               edgecolor='k')
ax1.plot(X)
ax1.legend(('mu1=[1,1]','mu2=[5,3]'))
ax1.set_title('Numpy random multivariate normal distribution')

y = np.zeros(200)
y[100:] = 1

#------------------------------------------------------------
# Fit the Naive Bayes classifier
clf = GaussianNB()
clf.fit(X, y)

# predict the classification probabilities on a grid
xlim = (-1, 8)
ylim = (-1, 5)
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 71),
                     np.linspace(ylim[0], ylim[1], 81))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)

#------------------------------------------------------------
# Plot the results
ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary, edgecolors= 'k', zorder=2)

ax2.contour(xx, yy, Z, [0.5], colors='k')

ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')

fig.show()
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import pandas as pd
import os

from sklearn.naive_bayes import GaussianNB

import seaborn as sns
sns.set(style="ticks")


# Load SNR dataset
pathDataFrame = './'
df_SNR = pd.read_csv(os.path.join(pathDataFrame,'treino.csv'))

# Creating train and test arrays
df_SNR = df_SNR.loc[:,['b1','b2','b3','b4','b5','b6','b7','b8','b9','label']]

# Creating train and test arrays
from sklearn.model_selection import train_test_split
dados_mod = df_SNR.loc[:,df_SNR.columns != 'label']
dados_label = df_SNR.label

X_train, X_test, y_train, y_test = train_test_split(
    dados_mod,
    dados_label,
    test_size=0.33, 
    shuffle=True,
    random_state=42,
)
print("Size Train data (features) and Train labels:",X_train.shape,y_train.shape)
print("Example featues:\n",X_train.loc[0])
print('Result label:',y_train.loc[0])


#------------------------------------------------------------
# Fit the Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

#Return the mean accuracy on the given test data and labels.
print('Mean accuracy of training:',clf.score(X_train, y_train))

prob_pos_clf = clf.predict_proba(X_test)


# Create and Plot confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels

y_pred = clf.predict(X_test)
y_true = y_test
class_names = np.array(['s ', 'd ', 'o ', 'h '])

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, fig =None, ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
      fig, ax = plt.subplots()
      
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    #ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
          # ... and label them with the respective list entries
          xticklabels=classes, yticklabels=classes,
          title=title,
          ylabel='True label',
          xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

fig2, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5), dpi= 80, facecolor='w',
                              edgecolor='k')

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization',fig=fig2,
                    ax=ax1)

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',fig=fig2,
                     ax=ax2)

fig2.show()
#fig2.savefig(pathDataFrame+'SNR_full_ConfusionMatrix.png',bbox_inches='tight',dpi=150)

print('Accuracy:',accuracy_score(y_test, y_pred))
# Load SNR dataset
pathDataFrame = './'
df_SNR = pd.read_csv(os.path.join(pathDataFrame,'treino.csv'))
df_SNR = df_SNR.set_index('id')

# Creating train and test arrays
df_SNR = df_SNR.loc[:,['b1','b2','b3','b4','b5','b6','b7','b8','b9','label']]


X_train = df_SNR.loc[:,df_SNR.columns != 'label']
y_train = df_SNR.label

#------------------------------------------------------------
# Fit the Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train, y_train)

#Return the mean accuracy on the given test data and labels.
print('Mean accuracy of training:',clf.score(X_train, y_train))
# Load SNR dataset
pathDataFrame = './'
df_SNR_test = pd.read_csv(os.path.join(pathDataFrame,'teste.csv'))

# Creating test array
X_test = df_SNR_test.loc[:,['b1','b2','b3','b4','b5','b6','b7','b8','b9']]

y_pred = clf.predict(X_test)
data = {'id': df_SNR_test.id ,
        'label': y_pred}

df = pd.DataFrame (data, columns = ['id','label'])
df.head()
import os
os.chdir(r'/kaggle/working')

df.to_csv("submission.csv", index = False)
from IPython.display import FileLink
FileLink(r'submission.csv')
# ANOVA feature selection for numeric input and categorical output
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Load SNR dataset
pathDataFrame = './'
df_SNR = pd.read_csv(os.path.join(pathDataFrame,'treino.csv'))
df_SNR = df_SNR.set_index('id')

X_train = df_SNR.loc[:,df_SNR.columns != 'label']
y_train = df_SNR.label

results=[]
for k in range(1,(X_train.shape[1]+1)):
  # define feature selection
  fs = SelectKBest(score_func=f_classif, k=k)

  # Run score function on (X, y) and get the appropriate features
  features = fs.fit(X_train, y_train)

  # apply feature selection
  X_train_selected= X_train.loc[:,features.get_support()]

  #------------------------------------------------------------
  # Fit the Naive Bayes classifier
  clf = GaussianNB()
  clf.fit(X_train_selected, y_train)

  results.append(clf.score(X_train_selected, y_train)) 

fig, ax = plt.subplots() 
  
ax.plot(range(1,len(results)+1),results)
ax.set_xlabel('K features selected',  
               fontweight ='bold')
ax.set_ylabel('Training accurancy',  
               fontweight ='bold')  
ax.grid(True)  
ax.set_title('ANOVA feature selection', fontsize = 14, fontweight ='bold') 
plt.show()

print('The best number for feature selection K=',(np.argmax(results) + 1),'maximum accurancy achieved:',results[np.argmax(results)])


# define feature selection
fs = SelectKBest(score_func=f_classif, k=(np.argmax(results) + 1))

# Run score function on (X, y) and get the appropriate features
features = fs.fit(X_train, y_train)

# apply feature selection
X_train_selected= X_train.loc[:,features.get_support()]

#------------------------------------------------------------
# Fit the Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train_selected, y_train)

#Return the mean accuracy on the given test data and labels.
print('Mean accuracy of training:',clf.score(X_train_selected, y_train))


# Load SNR dataset
pathDataFrame = './'
df_SNR_test = pd.read_csv(os.path.join(pathDataFrame,'teste.csv'))
df_SNR_test = df_SNR_test.set_index('id')

# Creating test array
X_test_selected = df_SNR_test.loc[:,features.get_support()]
X_test_selected.head()
y_pred = clf.predict(X_test_selected)

data = {'id': df_SNR_test.index ,
        'label': y_pred}

df = pd.DataFrame (data, columns = ['id','label'])
df.head()
df.to_csv("submission_feature_selection.csv", index = False)
from IPython.display import FileLink
FileLink(r'submission_feature_selection.csv')
"""Since we were mainly interested in reflected spectral information rather than thermal information,
   only the green (0.52–0.60μm), red (0.63–0.69μm) and near-infrared(NIR) (0.76–0.86μm) bands from each
   image were used for analysis (total of nine bands, 15 m spatial resolution). Colour infrared images
   from the three dates are shown in figure 1 (Johnson et al., 2012)."""


# Load SNR dataset
pathDataFrame = './'
df_SNR = pd.read_csv(os.path.join(pathDataFrame,'treino.csv'))
df_SNR = df_SNR.set_index('id')

X_train = df_SNR.loc[:,df_SNR.columns != 'label']
y_train = df_SNR.label

#b1 (green), b2 (red) and b3 (NIR) at 26 September 2010
#b4 (green), b5 (red) and b6 (NIR) at 19 March 2011
#b7 (green), b8 (red) and b9 (NIR) at 8 May 2011

 
"""Satellite maps of vegetation show the density of plant growth over the entire 
   globe. The most common measurement is called the Normalized Difference Vegetation
   Index (NDVI). Very low values of NDVI (0.1 and below) correspond to barren areas 
   of rock, sand, or snow. Moderate values represent shrub and grassland (0.2 to 0.3), 
   while high values indicate temperate and tropical rainforests (0.6 to 0.8).
   Source:  https://earthobservatory.nasa.gov/Features/MeasuringVegetation"""

#################################################################
# Normalizaded Difference Vegetation Index (Rouse et al.,1973):
X_train['NDVI1'] =  (X_train.b3-X_train.b2)/(X_train.b3+X_train.b2)
X_train['NDVI2'] = (X_train.b6-X_train.b5)/(X_train.b6+X_train.b5)
X_train['NDVI3'] = (X_train.b9-X_train.b8)/(X_train.b9+X_train.b8)

"""The Enhanced Vegetation Index (EVI) presents a more accurate vegetation 
   response than NDVI due to reduced atmospheric effects and soil background
   response (Matsushita et al., 2007)"""

#################################################################
# Enhanced Vegetation Index:
X_train['EVI1'] = 2.5*(X_train.b3-X_train.b2)/(X_train.b3+X_train.b2+1)
X_train['EVI2'] = 2.5*(X_train.b6-X_train.b5)/(X_train.b6+X_train.b5+1)
X_train['EVI3'] = 2.5*(X_train.b9-X_train.b8)/(X_train.b9+X_train.b8+1)
results=[]
for k in range(1,(X_train.shape[1]+1)):
  # define feature selection
  fs = SelectKBest(score_func=f_classif, k=k)

  # Run score function on (X, y) and get the appropriate features
  features = fs.fit(X_train, y_train)

  # apply feature selection
  X_train_selected= X_train.loc[:,features.get_support()]

  #------------------------------------------------------------
  # Fit the Naive Bayes classifier
  clf = GaussianNB()
  clf.fit(X_train_selected, y_train)

  results.append(clf.score(X_train_selected, y_train)) 

fig, ax = plt.subplots() 
  
ax.plot(range(1,len(results)+1),results)
ax.set_xlabel('K features selected',  
               fontweight ='bold')
ax.set_ylabel('Training accurancy',  
               fontweight ='bold')  
ax.grid(True)  
ax.set_title('ANOVA feature selection', fontsize = 14, fontweight ='bold') 
plt.show()

print('The best number for feature selection K=',(np.argmax(results) + 1),'maximum accurancy achieved:',results[np.argmax(results)])
# define feature selection
fs = SelectKBest(score_func=f_classif, k=(np.argmax(results) + 1))

# Run score function on (X, y) and get the appropriate features
features = fs.fit(X_train, y_train)

# apply feature selection
X_train_selected= X_train.loc[:,features.get_support()]

#------------------------------------------------------------
# Fit the Naive Bayes classifier
clf = GaussianNB()
clf.fit(X_train_selected, y_train)

#Return the mean accuracy on the given test data and labels.
print('Mean accuracy of training:',clf.score(X_train_selected, y_train))
# Load SNR dataset
pathDataFrame = './'
df_SNR_test = pd.read_csv(os.path.join(pathDataFrame,'teste.csv'))
df_SNR_test = df_SNR_test.set_index('id')

#################################################################
# Normalizaded Difference Vegetation Index (Rouse et al.,1973):
df_SNR_test['NDVI1'] =  (df_SNR_test.b3-df_SNR_test.b2)/(df_SNR_test.b3+df_SNR_test.b2)
df_SNR_test['NDVI2'] = (df_SNR_test.b6-df_SNR_test.b5)/(df_SNR_test.b6+df_SNR_test.b5)
df_SNR_test['NDVI3'] = (df_SNR_test.b9-df_SNR_test.b8)/(df_SNR_test.b9+df_SNR_test.b8)

#################################################################
# Enhanced Vegetation Index:
df_SNR_test['EVI1'] = 2.5*(df_SNR_test.b3-df_SNR_test.b2)/(df_SNR_test.b3+df_SNR_test.b2+1)
df_SNR_test['EVI2'] = 2.5*(df_SNR_test.b6-df_SNR_test.b5)/(df_SNR_test.b6+df_SNR_test.b5+1)
df_SNR_test['EVI3'] = 2.5*(df_SNR_test.b9-df_SNR_test.b8)/(df_SNR_test.b9+df_SNR_test.b8+1)


# Creating test array
X_test_selected = df_SNR_test.loc[:,features.get_support()]
X_test_selected.head()
y_pred = clf.predict(X_test_selected)

data = {'id': df_SNR_test.index ,
        'label': y_pred}

df = pd.DataFrame (data, columns = ['id','label'])
df.head()
df.to_csv("submission_VI2_feature_selection.csv", index = False)
from IPython.display import FileLink
FileLink(r'submission_VI2_feature_selection.csv')
from sklearn.ensemble import RandomForestClassifier
import os

# Load SNR dataset
pathDataFrame = './'
df_SNR = pd.read_csv(os.path.join(pathDataFrame,'treino.csv'))
df_SNR = df_SNR.set_index('id')

# Separate input features (X) and target variable (y)
#y = df_SNR.label
#X = df_SNR.drop('label', axis=1)
y = train.label
X = train.drop('label', axis=1)
 
# Train model
clf = RandomForestClassifier()
clf.fit(X, y)
 
#Return the mean accuracy on the given test data and labels.
print('Mean accuracy of training:',clf.score(X, y))
# Load SNR dataset
pathDataFrame = './'
df_SNR_test = pd.read_csv(os.path.join(pathDataFrame,'teste.csv'))
df_SNR_test = df_SNR_test.set_index('id')

# Creating test array
y_pred = clf.predict(df_SNR_test)

data = {'id': df_SNR_test.index ,
        'label': y_pred}

df = pd.DataFrame (data, columns = ['id','label'])
df.head()
df.to_csv("final.csv", index = False)
from IPython.display import FileLink
FileLink(r'final.csv')