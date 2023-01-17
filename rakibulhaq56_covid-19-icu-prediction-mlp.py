

import numpy as np 

import pandas as pd 

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import seaborn as sns

sns.set(style="whitegrid")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_excel("../input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")





print(df.shape)



df.head(5)
df.dtypes
df.select_dtypes(include=['object']).head()
print(df.select_dtypes(include=['object']).isnull().sum())
df.select_dtypes(exclude=['object']).isnull().sum()
df.dropna().shape
df_cat = df.select_dtypes(include=['object'])

df_numeric = df.select_dtypes(exclude=['object'])



imp = SimpleImputer(missing_values=np.nan, strategy='mean')



idf = pd.DataFrame(imp.fit_transform(df_numeric))

idf.columns = df_numeric.columns

idf.index = df_numeric.index





idf.isnull().sum()

def perc_to_int(percentile):

    #print(percentile, ''.join(filter(str.isdigit, percentile)))

    return(int(''.join(filter(str.isdigit, percentile))))

   

def wdw_to_int(window):

    if window == "ABOVE_12":

        window = "ABOVE-13"

    #print(window, ''.join(filter(str.isdigit, window.split("-")[1])))

    return(int(''.join(filter(str.isdigit, window.split("-")[1]))))
df_cat['AGE_PERCENTILE'] = df_cat.AGE_PERCENTIL.apply(lambda x: perc_to_int(x))

df_cat['WINDOW'] = df_cat.WINDOW.apply(lambda x: wdw_to_int(x))

df_cat['AGE_PERCENTILE']=df_cat['AGE_PERCENTILE'].astype("float64")

df_cat['WINDOW']=df_cat['WINDOW'].astype("float64")

print(df_cat.head())
df_cat=df_cat.drop(columns=["AGE_PERCENTIL"])

df_cat.head()
def min_max_convert_wdw(x):

    return (((x-2) *2) / 11) -1

def min_max_convert_age(x):

    return (((x-10)*2) /80) -1

df_cat['AGE_PERCENTILE'] = df_cat.AGE_PERCENTILE.apply(lambda x: min_max_convert_age(x))

df_cat['WINDOW'] = df_cat.WINDOW.apply(lambda x:min_max_convert_wdw(x))

    
idf.drop(["PATIENT_VISIT_IDENTIFIER"],1)

idf = pd.concat([idf,df_cat ], axis=1)

idf.head()


cor = idf.corr(method="pearson")

cor_target = abs(cor["ICU"])

relevant_features = cor_target[cor_target>0.3]

relevant_features.index
data = idf[relevant_features.index]
data.shape




data.ICU.value_counts()
plt.figure(figsize=(15,7))

percentile = age = sns.countplot(sorted(idf.AGE_PERCENTILE), hue='ICU', data=idf)

plt.xticks(rotation=40)

plt.xlabel("Age Percentile")

plt.ylabel("Patient Count")

plt.title("COVID-19 ICU Admissions by Age Percentile")

plt.legend(title = "ICU Admission",labels=['Not Admitted', 'Admitted'], loc = 0)






data.ICU = data.ICU.astype(int)



data.head()
y = data.ICU

X = data.drop("ICU", 1)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42, shuffle = True)
X_train.shape
y_train
from keras.models import Sequential

from keras.layers import Dense, Dropout

import numpy

import numpy as np
model = Sequential()



model.add(Dense(512, input_dim=16,activation='relu'))



model.add(Dense(250,activation="relu"))



model.add(Dense(124,activation="relu"))



model.add(Dense(54,activation="relu"))



model.add(Dense(1, activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])



model.summary()







history = model.fit(X_train, y_train, epochs=250, batch_size=16, validation_split=0.1)
f = plt.figure(1)

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

#plt.show()

# summarize history for loss

g = plt.figure(2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

#plt.show()





# evaluate the model

scores = model.evaluate(X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_pred = model.predict(X_test)
def mapin(x):

    if x>0.4:

        return 1

    else:

        return 0

def array_for(x):

    return np.array([mapin(xi) for xi in x])
y_pred=array_for(y_pred)
from sklearn.metrics import auc

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True, fmt = 'g', cmap = 'Blues')



print(classification_report(y_test, y_pred))

print("AUC = ",roc_auc_score(y_test, y_pred))



y_pred_keras = model.predict(X_test).ravel()

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

auc_keras = auc(fpr_keras, tpr_keras)



plt.figure(3)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')