import numpy as np 
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('../input/land_train.csv')    #Separating out the  Features and Target

features = list(dataset.columns[:12])            
target = dataset.columns[12]                      

print('Features:',features)
print('Target:',target)

# store feature matrix in "X"
X = dataset.iloc[:,:12]                          # slicing: all rows and 1 to 12 cols

# store response vector in "y"
Y = dataset.iloc[:,12]                            # slicing: all rows and 13th col

total_features = X.shape[1]
#check for any null value in the code.
X.isnull().any().describe() 
# Number of instances belonging to each class
dataset.groupby('target').size()
#statistics to understand the distribution of data
pd.set_option('display.max_columns', None)
print(X.describe())
#correlation graphs

cols=dataset.columns  #get the names of all the columns
data_corr = dataset.corr()  # Calculates pearson co-efficient for all combinations
threshold = 0.5 # Set the threshold to select only only highly correlated attributes
corr_list = [] # List of pairs along with correlation above threshold

for i in range(0,total_features):   #Search for the highly correlated pairs
    for j in range(i+1,total_features): # avoiding repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))    #Sort to show higher ones first   



# Scatter plot of only the highly correlated pairs
total_plots = 5

#Print correlations and column names
for v,i,j in s_corr_list[:total_plots]:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
    sns.pairplot(dataset, hue="target", size=9, x_vars=cols[i],y_vars=cols[j],markers=['P','*','|','v'] )
    plt.show()

#normalize
def normalize(df):
   return (df-df.mean())/df.std()

X = normalize(X)


#Data Split
X_res, X_test, Y_res, Y_test = train_test_split(X, Y, test_size = 0.1,random_state=1)
X_train, X_val,Y_train,Y_val = train_test_split(X_res, Y_res,test_size =0.1,random_state=1)
total_features = X_train.shape[1]

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
#SMOTE ALGO for resampling
def foo(x):
    return {3:22000,4:14000} #dictionary of number of examples required for each data set.

rus = SMOTE(random_state=1, sampling_strategy= foo)
X_train, Y_train = rus.fit_sample(X_train, Y_train)
from sklearn.linear_model import LogisticRegression


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
logreg.predict([range(total_features)]) 

Y_pred = logreg.predict(X_test)

print(Y_pred)


accuracy = accuracy_score(Y_test, Y_pred)

print("accuracy with logistic regression : ",accuracy)




from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
#one-hot encode
Y_train = pd.get_dummies(Y_train)
Y_val = pd.get_dummies(Y_val)
Y_test = pd.get_dummies(Y_test)

model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(total_features,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(500, activation='relu', input_shape=(50,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(500, activation='relu', input_shape=(500,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(Dense(500, activation='relu', input_shape=(500,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu', input_shape=(500,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu', input_shape=(500,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(4, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 30
batch_size = 128
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
          verbose = 2)




def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
   
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluate(X, Y,subset):
    Y_pred = model.predict(X)

    # Convert  one hot vectors to predictions classes.
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    Y_true = np.argmax(Y.values,axis = 1)

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(4)) 
    acc = accuracy_score(np.array(Y_true), 
                                     np.array(Y_pred_classes))
    print('%s accuracy : %f' %(subset,acc ))
evaluate(X_train,Y_train,'training')
evaluate(X_val,Y_val,'validation')
evaluate(X_test,Y_test,'test')
unknown_dataset = pd.read_csv('../input/land_test.csv')

unknown_dataset = normalize(unknown_dataset)

unknown_pred = model.predict(unknown_dataset)
unknown_pred_classes = np.argmax(unknown_pred, axis=1)

unknown_dataset['target'] = unknown_pred_classes
unknown_dataset.to_csv('land_test_Final_submission.csv')
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectFromModel

# clf = RandomForestClassifier(n_estimators=50,n_jobs=-1)
# clf.fit(X_train,Y_train)



# for feature in zip(features, clf.feature_importances_):
#     print(feature)
# # Y_pred = clf.predict(X_test)

# # # View The Accuracy Of Our Full Feature (4 Features) Model
# # accuracy_score(Y_test, Y_pred)

# # Create a selector object that will use the random forest classifier to identify
# # features that have an importance of more than threshold
# sfm = SelectFromModel(clf, threshold=0.05)

# # Train the selector
# sfm.fit(X_train, Y_train)
# # Print the names of the most important features
# for feature_list_index in sfm.get_support(indices=True):
#     print(features[feature_list_index])
# X_train = sfm.transform(X_train)
# X_val = sfm.transform(X_val)
# X_test = sfm.transform(X_test)

# total_features = X_train.shape[1]
# print('before :',X_train.shape)
# def foo(x):
#     return {4:18000 ,1:30000,2:10000,3:20000}
# rus =  RandomUnderSampler(sampling_strategy = 'not minority',random_state=1)
# X_train, Y_train = rus.fit_sample(X_train, Y_train)
# print('After : ',X_train.shape)
# rus =  RandomOverSampler(sampling_strategy = 'not majority',random_state=1)
# X_train, Y_train = rus.fit_sample(X_train, Y_train)






