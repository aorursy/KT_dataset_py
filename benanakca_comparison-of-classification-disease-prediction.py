import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop,Nadam,Adadelta,Adam

from tensorflow.keras.layers import BatchNormalization,LeakyReLU

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import seaborn as sns

import scipy.stats as stats

import sklearn

import os





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_raw = pd.read_csv("/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv", sep=";")

data_raw.head()
data_raw.info()
data_raw.drop("id",axis=1,inplace=True)

print("There is {} duplicated values in data frame".format(data_raw.duplicated().sum()))
duplicated = data_raw[data_raw.duplicated(keep=False)]

duplicated = duplicated.sort_values(by=['age', "gender", "height"], ascending= False) 

# I sorted the values to see duplication clearly



duplicated.head(2) # Show us just 1 duplication of 24
data_raw.drop_duplicates(inplace=True)

print("There is {} duplicated values in data frame".format(data_raw.duplicated().sum()))
print("There is {} missing values in data frame".format(data_raw.isnull().sum().sum()))
x = data_raw.copy(deep=True)

x.describe()
s_list = ["age", "height", "weight", "ap_hi", "ap_lo"]

def standartization(x):

    x_std = x.copy(deep=True)

    for column in s_list:

        x_std[column] = (x_std[column]-x_std[column].mean())/x_std[column].std()

    return x_std 

x_std=standartization(x)

x_std.head()
x_melted = pd.melt(frame=x_std, id_vars="cardio", value_vars=s_list, var_name="features", value_name="value", col_level=None)

x_melted
plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", hue="cardio", data=x_melted)

plt.xticks(rotation=90)
ap_list = ["ap_hi", "ap_lo"]

boundary = pd.DataFrame(index=["lower_bound","upper_bound"]) # We created an empty dataframe

for each in ap_list:

    Q1 = x[each].quantile(0.25)

    Q3 = x[each].quantile(0.75)

    IQR = Q3 - Q1



    lower_bound = Q1- 1.5*IQR

    upper_bound = Q3 + 1.5*IQR

    boundary[each] = [lower_bound, upper_bound ]

boundary
ap_hi_filter = (x["ap_hi"] > boundary["ap_hi"][1])

ap_lo_filter = (x["ap_lo"] > boundary["ap_lo"][1])                                                           

outlier_filter = (ap_hi_filter | ap_lo_filter)

x_outliers = x[outlier_filter]

x_outliers["cardio"].value_counts()

sns.countplot(x='cardio',data=x_outliers,linewidth=2,edgecolor=sns.color_palette("dark", 1))
out_filter = ((x["ap_hi"]>250) | (x["ap_lo"]>200) )

print("There is {} outlier".format(x[out_filter]["cardio"].count()))

x = x[~out_filter]

corr = x.corr()

f, ax = plt.subplots(figsize = (15,15))

sns.heatmap(corr, annot=True, fmt=".3f", linewidths=0.5, ax=ax)
def bmi_calc(w, h):

    return w/(h**2)
x["bmi"] = x["weight"]/ (x["height"]/100)**2
x.head()
a = x[x["gender"]==1]["height"].mean()

b = x[x["gender"]==2]["height"].mean()

if a > b:

    gender = "male"

    gender2 = "female"

else:

    gender = "female"

    gender2 = "male"

print("Gender:1 is "+ gender +" & Gender:2 is " + gender2)
x["gender"] = x["gender"] % 2
from sklearn.preprocessing import StandardScaler

x_std = standartization(x)



data = pd.melt(x_std,id_vars="cardio",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="gender", y="bmi", hue="cardio", data=x,split=True, inner="quart")

plt.xticks(rotation=90)
y = x["cardio"]

y.shape
x.drop("cardio", axis=1,inplace=True)

x.head()
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import normalize

x_train = normalize(x_train)

x_test = normalize(x_test)

x = normalize(x)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



dec = DecisionTreeClassifier()

ran = RandomForestClassifier(n_estimators=100)

knn = KNeighborsClassifier(n_neighbors=100)

svm = SVC(random_state=1)

naive = GaussianNB()



models = {"Decision tree" : dec,

          "Random forest" : ran,

          "KNN" : knn,

          "SVM" : svm,

          "Naive bayes" : naive}

scores= { }



for key, value in models.items():    

    model = value

    model.fit(x_train, y_train)

    scores[key] = model.score(x_test, y_test)

  

scores_frame = pd.DataFrame(scores, index=["Accuracy Score"]).T

scores_frame.sort_values(by=["Accuracy Score"], axis=0 ,ascending=False, inplace=True)

scores_frame
plt.figure(figsize=(5,5))

sns.barplot(x=scores_frame.index,y=scores_frame["Accuracy Score"])

plt.xticks(rotation=45) # Rotation of Country names...
from sklearn.model_selection import cross_val_score

accuracies_random_forest = cross_val_score(estimator=ran, X=x_train, y=y_train, cv=10)

accuracies_knn = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10)
print("Random Forest Average accuracy: ", accuracies_random_forest.mean())

print("Random Forest Standart Deviation: ", accuracies_random_forest.std())

print("KNN Average accuracy: ", accuracies_knn.mean())

print("KNN Standart Deviation: ", accuracies_knn.std())
# grid search cross validation with 1 hyperparameter

from sklearn.model_selection import GridSearchCV



grid = {"n_estimators" : np.arange(10,150,10)}



ran_cv = GridSearchCV(ran, grid, cv=3) # GridSearchCV

ran_cv.fit(x_train,y_train)# Fit



# Print hyperparameter

print("Tuned hyperparameter n_estimators: {}".format(ran_cv.best_params_)) 

print("Best score: {}".format(ran_cv.best_score_))
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver="liblinear", max_iter=200)

grid = {"penalty" : ["l1", "l2"],

         "C" : np.arange(60,80,2)} # (60,62,64 ... 78)

log_reg_cv = GridSearchCV(log_reg, grid, cv=3)

log_reg_cv.fit(x_train, y_train)



# Print hyperparameter

print("Tuned hyperparameter n_estimators: {}".format(log_reg_cv.best_params_)) 

print("Best score: {}".format(log_reg_cv.best_score_))
logreg_best = LogisticRegression(C=74, penalty="l1", solver="liblinear")

logreg_best.fit(x_train, y_train)

print("Test accuracy: ",logreg_best.score(x_test, y_test))
y_true = y_test

y_pred = logreg_best.predict(x_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.2, linecolor="purple", ax=ax)

plt.xlabel("Predicted")

plt.ylabel("Grand Truth")

plt.show()
TN = cm[0,0]

TP = cm[1,1]

FN = cm[1,0]

FP = cm[0,1]

Precision = TP/(TP+FP)

Recall = TP/(TP+FN)

F1_Score = 2*(Recall * Precision) / (Recall + Precision)

pd.DataFrame([[Precision, Recall, F1_Score]],columns=["Precision", "Recall", "F1 Score"], index=["Results"])
x.shape
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(6, input_dim=12, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
optimizer = RMSprop(learning_rate=0.002)

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau( 

    monitor='val_loss',    # Quantity to be monitored.

    factor=0.1,       # Factor by which the learning rate will be reduced. new_lr = lr * factor

    patience=50,        # The number of epochs with no improvement after which learning rate will be reduced.

    verbose=1,         # 0: quiet - 1: update messages.

    mode="auto",       # {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; 

                       # in the max mode it will be reduced when the quantity monitored has stopped increasing; 

                       # in auto mode, the direction is automatically inferred from the name of the monitored quantity.

    min_delta=0.0001,  # threshold for measuring the new optimum, to only focus on significant changes.

    cooldown=0,        # number of epochs to wait before resuming normal operation after learning rate (lr) has been reduced.

    min_lr=0.00001     # lower bound on the learning rate.

    )



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=400, restore_best_weights=True)

history = model.fit(x=x_train, y=y_train.values,

                    batch_size=1024, epochs=1500,

                    verbose=0,validation_data=(x_test,y_test.values),

                    callbacks=[learning_rate_reduction, es],

                    shuffle=True)



model.evaluate(x_test, y_test.values, verbose=2)



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()