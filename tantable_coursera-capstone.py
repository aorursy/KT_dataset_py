# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#here we are combining three sets of data, to make one dataset containing all the UK accidents between 2005 and 2012

data_2005 = pd.read_csv("../input/uk-data-three-files/accidents_2005_to_2007.csv/accidents_2005_to_2007.csv")
data_2009 = pd.read_csv("../input/uk-data-three-files/accidents_2009_to_2011.csv/accidents_2009_to_2011.csv")
data_2012 = pd.read_csv("../input/uk-data-three-files/accidents_2012_to_2014.csv/accidents_2012_to_2014.csv")

frames = [data_2005, data_2009, data_2012]
df_complete = pd.concat(frames)
df_complete.head()

# lets take an inital looka the data
df_complete["Urban_or_Rural_Area"].value_counts()
#lets make a new DF that contains just contain columns that could indicate the severity result or could 
#influcence the severity of an accident


df_select = df_complete[['Longitude', 'Latitude', 'Accident_Severity',
       'Number_of_Vehicles', 'Number_of_Casualties', 'Date', 'Day_of_Week',
       'Time','Road_Type', 'Speed_limit','Light_Conditions',
       'Weather_Conditions', 'Road_Surface_Conditions',
       'Special_Conditions_at_Site', 'Carriageway_Hazards',
       'Urban_or_Rural_Area']]
df_select.head()

#lets have a look at the data in each column

def list_values (columns, df):
    for i in columns:
        print(i)
        print(df[i].value_counts())
        print("number of NaN's ", df[i].isnull().sum(axis = 0))
        print()
        

    
list_values(df_select.columns, df_select)

df_select["Weather_Conditions"] = df_select["Weather_Conditions"].replace({'Unknown' : np.NaN, 'Other': np.NaN})
df_select["Road_Type"] = df_select["Road_Type"].replace({'Unknown' : np.NaN})
# i think we can tidy up weather conditions to have more bollean options and make the details more precise
# fine | yes or no | 1 is 
# high winds | 1 is high windes
# rain: yes or no 1 is rain
# snow: yes or no 1 is snow 0 no snow

#so lets make three new columss that we will populare with the new values

df_select["rain"] = df_select["Weather_Conditions"]
df_select["high_winds"] = df_select["Weather_Conditions"]
df_select["fine"] = df_select["Weather_Conditions"]
df_select["snow"] = df_select["Weather_Conditions"]


df_select["rain"].value_counts()
# changing the vlaue of the columns clumn to be boolean representing their state
# its a bit like manual one hot encoding


df_select["rain"] = df_select["rain"].replace(
    {'Fine without high winds' : 0,
     'Raining without high winds' : 1,
     'Raining with high winds' : 1,
     'Fine with high winds': 0,
     'Snowing without high winds': 0,
     'Fog or mist' : 0,
     'Snowing with high winds' : 0}

) 

df_select["fine"] = df_select["fine"].replace(
    {'Fine without high winds' : 1,
     'Raining without high winds' : 0,
     'Raining with high winds' : 0,
     'Fine with high winds': 1,
     'Snowing without high winds': 0,
     'Fog or mist' : 0,
     'Snowing with high winds' : 0}

) 

df_select["high_winds"] = df_select["high_winds"].replace(
    {'Fine without high winds' : 0,
     'Raining without high winds' : 0,
     'Raining with high winds' : 1,
     'Fine with high winds': 1,
     'Snowing without high winds': 0,
     'Fog or mist' : 0,
     'Snowing with high winds' : 1}

) 

df_select["snow"] = df_select["snow"].replace(
    {'Fine without high winds' : 0,
     'Raining without high winds' : 0,
     'Raining with high winds' : 0,
     'Fine with high winds': 0,
     'Snowing without high winds': 0,
     'Fog or mist' : 0,
     'Snowing with high winds' : 1}

) 

#now we will drop the weatahter_conditions column
df_select = df_select.drop(["Weather_Conditions"], axis = 'columns')
# now we need to look at road type, according to the metadata of the dataset the data should just contain 1 or 2 values, but we have 132 insrances of 3
#we will convert all the 3's to NaN

df_select["Urban_or_Rural_Area"] = df_select["Urban_or_Rural_Area"].replace({3 : np.NaN})
#we will now drop all the NaN's

df_select.dropna(inplace = True)
df_select.info()

# now we need to balance the data

from imblearn.under_sampling import RandomUnderSampler

X = df_select[['Longitude', 'Latitude', 'Day_of_Week', 'Road_Type', 'Speed_limit',
       'Light_Conditions', 'Road_Surface_Conditions', 'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Urban_or_Rural_Area',
       'rain', 'high_winds', 'fine','snow']]
y = df_select[['Accident_Severity']]

sampler = RandomUnderSampler(random_state=5555)
X_bal, y_bal = sampler.fit_resample(X, y)


df_select = X_bal
df_select['Accident_Severity'] = y_bal

sns.countplot(x="Accident_Severity", data=df_select)

fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(50,24))
plt.subplots_adjust(hspace = 0.8, wspace=0.8)

colData = df_select


sns.countplot(y="Day_of_Week", hue="Accident_Severity", data=colData, ax=axs[0, 0])
sns.countplot(y="Road_Type", hue="Accident_Severity", data=colData, ax=axs[0, 1])
sns.countplot(y="Speed_limit", hue="Accident_Severity", data=colData, ax=axs[0, 2])
sns.countplot(y="Light_Conditions", hue="Accident_Severity", data=colData, ax=axs[1, 0])
sns.countplot(y="Road_Surface_Conditions", hue="Accident_Severity", data=colData, ax=axs[1, 1])
#sns.countplot(y="Special_Conditions_at_Site", hue="Accident_Severity", data=colData, ax=axs[1, 2])
sns.countplot(y="Carriageway_Hazards", hue="Accident_Severity", data=colData, ax=axs[2, 0])
sns.countplot(y="rain", hue="Accident_Severity", data=colData, ax=axs[2,1])
sns.countplot(y="snow", hue="Accident_Severity", data=colData, ax=axs[2, 2])
sns.countplot(y="high_winds", hue="Accident_Severity", data=colData, ax=axs[3, 0])
sns.countplot(y="fine", hue="Accident_Severity", data=colData, ax=axs[3, 1])
#from the above charts we can see that right away we can drop Special_Conditions_at_Site, Carriageway_Hazards, snow
#we can also drop rain as it is just the inverse of fine and their correlation is almost perfect negative

df_select.drop(['Special_Conditions_at_Site', 'Carriageway_Hazards', 'snow','rain'], axis = 'columns', inplace = True)

#we will also drop the above from our X data set

X.drop(['Special_Conditions_at_Site', 'Carriageway_Hazards', 'snow','rain', 'Longitude', 'Latitude'], axis = 'columns', inplace = True)

X.head()


# example of mutual information feature selection for categorical data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

# prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc
 
# prepare target
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc
 
# feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def calc_entropy(X, y, n):
    scores = []
    result = np.zeros(len(X.columns))
    
    for i in range(n):
        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        # prepare input data
        X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
        # prepare output data
        y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
        # feature selection
        X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
        # what are scores for the features
        result = fs.scores_ + result
        
    return result / n


X = X.astype(str)


# Change the iteration number if you want to average the results
iterations = 1

result = calc_entropy(X, y, iterations)

for i in range(len(result)):
    print('Feature %s: %f' % (X.columns[i], result[i]))
# Plot the scores

sns.barplot(y=X.columns, x=result)
#plt.savefig('fig/feature_extraction.png',dpi=300, bbox_inches = "tight")
df_select.head()
df_select.dropna(inplace = True)
print(df_select.shape)
print(feature_set.shape)
# import folium
# from folium import plugins

# latitude = df_select.Latitude.mean()
# longitude = df_select.Longitude.mean()

# # create map and display it
# map_uk = folium.Map(location=[latitude, longitude], zoom_start=12)

# # instantiate a mark cluster object for the incidents in the dataframe
# collisions = plugins.MarkerCluster().add_to(map_uk)

# # loop through the dataframe and add each data point to the mark cluster
# for lat, lng, label, in zip(df_select.Latitude, df_select.Longitude, df_select.Accident_Severity):
#     folium.Marker(
#         location=[lat, lng],
#         icon=None,
#         popup=label,
#     ).add_to(collisions)


# map_uk


#define the feature set
feature_set = df_select[['Speed_limit', 'Urban_or_Rural_Area', 'Light_Conditions', 'Road_Type', 'fine','Road_Surface_Conditions']]
feature_set
feature_set.head()


# generate binary values using get_dummies
dum_df = pd.get_dummies(feature_set[["Urban_or_Rural_Area","fine", "Light_Conditions",'Road_Type','Road_Surface_Conditions']])
# merge with main df bridge_df on key values

X1 = feature_set[['Speed_limit']].join(dum_df)
X1.info()


#now we normalise the data

from sklearn.preprocessing import MinMaxScaler

# create a scaler object
scaler = MinMaxScaler()
# fit and transform the data
X1_norm = pd.DataFrame(scaler.fit_transform(X1), columns=X1.columns)

X1_norm
X1_norm.info()
y1 = df_select['Accident_Severity']
y1.shape
X = X1
y = y1
print(X.shape)
print(y.shape)
#lets split the data into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=6665)
print ('Train Set:', X_train.shape,  y_train.shape)
print ('Test Set:', X_test.shape,  y_test.shape)
X = X.astype('Int64')
X.info()

#KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Ks = 12
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])


print("k values", mean_acc)
print("best k accuracy value", mean_acc.max())

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


#testing KNN model on the train set data
yhat_train = neigh.predict(X_train)
#lets do some nice plotting of KNN points

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions




def knn_comparison(data_knn, k):
    x = data_knn[['weather_cat', 'roadCond_cat']].values
    y_ = data_knn['severity'].astype(int).values
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(x, y_)
    
# Plotting decision region
    plot_decision_regions(x, y_, clf=clf, legend=2)
    
# Adding axis annotations
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Knn with K="+ str(k))
    plt.show()
    


data_knn = df_bal_cat
for i in [2,8,12]:
    knn_comparison(data_knn, i)



#decision trees
from sklearn.tree import DecisionTreeClassifier

crashTreeEntrophy = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
crashTreeGini = DecisionTreeClassifier(criterion="gini", max_depth = 4)


print("crashTreeEntrophy",crashTreeEntrophy) # it shows the default parameters
print("crashTreeGini", crashTreeGini)
crashTreeEntrophy.fit(X_train,y_train)
crashTreeGini.fit(X_train,y_train)
predictionCrashEntrophy = crashTreeEntrophy.predict(X_test)
predictionCrashGini = crashTreeGini.predict(X_test)

DT_yhat_train = crashTreeEntrophy.predict(X_train)
print("entrophy model")
print("")
print ("predicted value",predictionCrashEntrophy [0:100])
print ("actual value", y_test [0:100])
print("")
print("")


print("gini model")
print("")
print ("predicted value",predictionCrashGini [0:10])
print ("actual value", y_test [0:10])



#evaluating the decision tree
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy Entrophy Model: ", metrics.accuracy_score(y_test, predictionCrashEntrophy))
print("DecisionTrees's Accuracy Entrophy Model: ", metrics.accuracy_score(y_test, predictionCrashGini))
# !conda install -c conda-forge pydotplus -y
# !conda install -c conda-forge python-graphviz -y

#we will reduce the size of the data  now so it is faster to run in SVM

from sklearn.utils import resample
df_majority = df[df.severity==1]
df_minority = df[df.severity==2]

 
#Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=20000,  # to match minority class
                                 random_state=5555) # reproducible results
 
# Combine minority class with downsampled majority class
df_balanced_small = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_balanced_small["severity"].value_counts()
#we will reduce the size of the data now so it is faster to run in SVM

from sklearn.utils import resample

df_majority = df_balanced_small[df_balanced_small.severity==2]
df_minority = df_balanced_small[df_balanced_small.severity==1]

 
#Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=20000,  # to match minority class
                                 random_state=5555) # reproducible results
 
# Combine minority class with downsampled majority class
df_balanced_small = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_balanced_small["severity"].value_counts()
df_balanced_small.columns
df_balanced_small.head()
# encode the small data


df_balanced_small[['weather', "roadCond","lightCond", "speeding", "underInfluence", "inattention"]] = df_balanced[['weather', "roadCond","lightCond", "speeding", "underInfluence", "inattention"]].astype('category')

df_balanced_small['weather_cat'] = df_balanced_small['weather'].cat.codes
df_balanced_small['roadCond_cat'] = df_balanced_small['roadCond'].cat.codes
df_balanced_small['lightCond_cat'] = df_balanced_small['lightCond'].cat.codes
df_balanced_small['speeding_cat'] = df_balanced_small['speeding'].cat.codes

df_balanced_small['underInfluence_cat'] = df_balanced_small['underInfluence'].cat.codes
df_balanced_small['inattention_cat'] = df_balanced_small['inattention'].cat.codes
df_balanced_small.head()

# lets create a new df with just the categorical values in and just the fieleds we wiull use
# we will remove the columns for other severity indicators. 


df_bal_cat_small = df_balanced_small[['severity', 'weather_cat', 'roadCond_cat', 'lightCond_cat',
       'speeding_cat', 'underInfluence_cat', 'inattention_cat']]

df_bal_cat_small.head()
df_bal_cat_small["severity"].value_counts()
#now we have a much smaller Dataset 20000 for each severity
#now we will creatre our x and y sets

X_small = df_bal_cat_small[['inattention_cat', 'underInfluence_cat', 'speeding_cat', 'weather_cat', 'roadCond_cat','lightCond_cat']].values
y_small = df_bal_cat_small["severity"].values

X_small = preprocessing.StandardScaler().fit(X_small).transform(X_small.astype(float))

#lets split the data into test and train

from sklearn.model_selection import train_test_split

X_train_small, X_test_small, y_train_small, y_test_small = train_test_split( X_small, y_small, test_size=0.35, random_state=6665)
print ('Small Train Set:', X_train_small.shape,  y_train_small.shape)
print ('Small Test Set:', X_test_small.shape,  y_test_small.shape)


from sklearn import svm

#adding the linear kernels
clf_lin= svm.SVC(kernel="linear")
# training the data using linear kernels
clf_lin.fit(X_train_small, y_train_small)



#adding polynomial svm kernel
clf_pol= svm.SVC(kernel='poly')
# training the polynomial kernel
clf_pol.fit(X_train_small, y_train_small)
#adding the sigmod svm kernel
clf_sig= svm.SVC(kernel='sigmoid')
#training the sigmod kernel
clf_sig.fit(X_train_small, y_train_small)


#adding the rbf svm kernel
clf_rbf= svm.SVC(kernel='rbf')
#training the rbf svm kernel
clf_rbf.fit(X_train_small, y_train_small)

#making preductions based on the above fitted models
yhat_svm_lin = clf_lin.predict(X_test_small)
yhat_svm_pol = clf_pol.predict(X_test_small)
yhat_svm_rbf = clf_rbf.predict(X_test_small)
yhat_svm_sig = clf_sig.predict(X_test_small)


range_ = "0:10"
print("actual y values", y_test_small[0:10])

print('yhat_svm_lin', yhat_svm_lin[0:10])
print("yhat_svm_pol", yhat_svm_pol[0:10])
print("yhat_svm_rbf", yhat_svm_rbf[0:10])
print("yhat_svm_sig", yhat_svm_sig[0:10])


#lets show the diffetnt values of yhat based on the type of SCVM kernel
pandas_df = [[]]
pandas_df["actual"] = pd.DataFrame(y_test_small)
pandas_df["yhat_lin"] = pd.DataFrame(yhat_svm_lin)
pandas_df["yhat_pol"] = pd.DataFrame(yhat_svm_lin)
pandas_df["yhat_rbf"] = pd.DataFrame(yhat_svm_lin)
pandas_df["yhat_sig"] = pd.DataFrame(yhat_svm_lin)

pandas_df_of_svm = pandas_df[['actual', 'yhat_lin','yhat_pol', 'yhat_rbf', 'yhat_sig']]
pandas_df_of_svm.head(50)
yhat_svm_pol_train = clf_pol.predict(X_train_small)
from sklearn.metrics import classification_report, confusion_matrix
import itertools
#the code for confusion matrix

def plot_confusion_matrix_svm(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print("")
        print("")
    else:
        print('Confusion matrix, without normalization')
        print("")
        print("")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual Value')
    plt.xlabel('Predicted Value')
# Compute confusion matrix


cnf_matrix = confusion_matrix(y_test_small, yhat_svm_lin, labels=[1,2])
np.set_printoptions(precision=5)

print ("classification_report LIN")
print(classification_report(y_test_small, yhat_svm_lin))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix_svm(cnf_matrix, classes=['serious(2)','less serious(1)'],normalize= False,  title='Confusion matrix LIN')
print("")
print("")

#_________________________-_-_-_-_-_----

cnf_matrix = confusion_matrix(y_test_small, yhat_svm_pol, labels=[1,2])
np.set_printoptions(precision=5)

print ("classification_report Polly")
print(classification_report(y_test_small, yhat_svm_pol))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix_svm(cnf_matrix, classes=['serious(2)','less serious(1)'],normalize= False,  title='Confusion matrix Polly')
print("")
print("")

#_________________________-_-_-_-_-_----


cnf_matrix = confusion_matrix(y_test_small, yhat_svm_rbf, labels=[1,2])
np.set_printoptions(precision=5)

print ("classification_report RBF")
print(classification_report(y_test_small, yhat_svm_rbf))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix_svm(cnf_matrix, classes=['serious(2)','less serious(1)'],normalize= False,  title='Confusion matrix RBF')
print("")
print("")

#_________________________-_-_-_-_-_----



cnf_matrix = confusion_matrix(y_test_small, yhat_svm_sig, labels=[1,2])
np.set_printoptions(precision=5)

print ("classification_report SIG")
print(classification_report(y_test_small, yhat_svm_sig))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix_svm(cnf_matrix, classes=['severity = 1','severity =2'],normalize= False,  title='Confusion matrix SIG')
print("")
print("")

#lets take the best version of SVM and include the accuracy here

#RBF seems to be the most accurate

from sklearn.metrics import f1_score

print("f1 score for RBF model")
f1_score(y_test_small, yhat_svm_rbf, average='weighted') 


from sklearn.metrics import jaccard_score as jss
print("Jaccard Similarity Score")
jss(y_test_small, yhat_svm_rbf)
import scipy.optimize as opt
import pylab as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR
yhat_LR = LR.predict(X_test)
yhat_LR_train = LR.predict(X_train)
yhat_LR
yhat_prob_LR = LR.predict_proba(X_test)
yhat_prob_LR
#EVALUATION OF LR
from sklearn.metrics import jaccard_score

LR_jac_score = jaccard_score(y_test, yhat_prob_LR)
print("LR jaccard score :", LR_jac_score)
from sklearn.metrics import jaccard_score
print("LR jaccard score :")
jaccard_score(y_test, yhat_LR)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat_LR, labels=[1,2])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix_svm(cnf_matrix, classes=['severity = 1','severity =2'],normalize= False,  title='Confusion matrix')
print (classification_report(y_test, yhat_LR))
yhat_prob_LR_train = LR.predict_proba(X_train)
#Summary report

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# getting a summaty of the accuraccy on test data
knn_prec = precision_score(y_test, yhat, average='weighted')
DT_prec = precision_score(y_test, predictionCrashGini, average='weighted')
LR_prec = precision_score(y_test, yhat_LR, average='weighted')
SVM_prec = precision_score(y_test_small, yhat_svm_pol, average='weighted')

knn_rec = recall_score(y_test, yhat, average='weighted')
DT_rec = recall_score(y_test, predictionCrashGini, average='weighted')
LR_rec = recall_score(y_test, yhat_LR, average='weighted')
SVM_rec = recall_score(y_test_small, yhat_svm_pol, average='weighted')

knn_f1 = f1_score(y_test, yhat, average='weighted')
DT_f1 = f1_score(y_test, predictionCrashGini, average='weighted')
LR_f1 = f1_score(y_test, yhat_LR, average='weighted')
SVM_f1 = f1_score(y_test_small, yhat_svm_pol, average='weighted')

knn_acc = metrics.accuracy_score(y_test, yhat)
DT_acc = metrics.accuracy_score(y_test, predictionCrashGini)
LR_acc = metrics.accuracy_score(y_test, yhat_LR)
SVM_acc = metrics.accuracy_score(y_test_small, yhat_svm_pol)

# getting a summaty of the accuraccy on the train data

knn_prec_train = precision_score(y_train, yhat_train, average='weighted')
DT_prec_train = precision_score(y_train, DT_yhat_train, average='weighted')
LR_prec_train = precision_score(y_train, yhat_LR_train, average='weighted')
SVM_prec_train = precision_score(y_train_small, yhat_svm_pol_train, average='weighted')

knn_rec_train = recall_score(y_train, yhat_train, average='weighted')
DT_rec_train = recall_score(y_train, DT_yhat_train, average='weighted')
LR_rec_train = recall_score(y_train, yhat_LR_train, average='weighted')
SVM_rec_train = recall_score(y_train_small, yhat_svm_pol_train, average='weighted')

knn_f1_train = f1_score(y_train, yhat_train, average='weighted')
DT_f1_train = f1_score(y_train, DT_yhat_train, average='weighted')
LR_f1_train = f1_score(y_train, yhat_LR_train, average='weighted')
SVM_f1_train = f1_score(y_train_small, yhat_svm_pol_train, average='weighted')

knn_acc_train = metrics.accuracy_score(y_train, yhat_train)
DT_acc_train = metrics.accuracy_score(y_train, DT_yhat_train)
LR_acc_train = metrics.accuracy_score(y_train, yhat_LR_train)
SVM_acc_train = metrics.accuracy_score(y_train_small, yhat_svm_pol_train)





import pandas as pd

# initialize list of lists 
data = [['KNN', knn_prec,knn_prec, knn_f1, knn_acc], 
        ['Decision Tree', DT_prec,DT_rec,DT_f1, DT_acc], 
        ['Log Reg', LR_prec, LR_rec,LR_f1,LR_acc], 
        ['SVM (Polly)', SVM_prec,SVM_rec,SVM_f1,SVM_acc]] 

# initialize list of lists 
data_train = [['KNN', knn_prec_train,knn_prec_train, knn_f1_train, knn_acc_train], 
        ['Decision Tree', DT_prec_train,DT_rec_train,DT_f1_train, DT_acc_train], 
        ['Log Reg', LR_prec_train, LR_rec_train,LR_f1_train,LR_acc_train], 
        ['SVM (Polly)', SVM_prec_train,SVM_rec_train,SVM_f1_train,SVM_acc_train]] 

  
# Create the pandas DataFrame 
results_df = pd.DataFrame(data, columns = ['model','Precision', 'Recall', 'f1 score', 'accuracy']) 
results_df_train = pd.DataFrame(data_train, columns = ['model','Precision', 'Recall', 'f1 score', 'accuracy']) 


results_df = results_df.set_index('model')
results_df_train = results_df_train.set_index('model')

print("results on test data")
print(results_df.head(10))
print("")
print("")


print("results on train data")
print(results_df_train.head(10))


results_plot = results_df.plot
results_plot()
x = ('model')


results_plot_train_data = results_df_train.plot
results_plot_train_data()
x = ('model')







