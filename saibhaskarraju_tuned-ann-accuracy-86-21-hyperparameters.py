import pandas as pd

import numpy as np



df = pd.read_csv("../input/deep-learning-az-ann/Churn_Modelling.csv")
# Visualizing the Dataset

df.head()
# Data columns type

df.info()
df=df.drop(['RowNumber','CustomerId','Surname'],axis='columns')

df.describe()
# 1. Check out the missing values 

df.isnull().sum()



# As you see below, we dont have any missing values, so we are moving forward and least bothered 

#about correcting the column values
# Correlation heatmap

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

sns.heatmap(df.corr(), cmap='BuGn',annot=True)
#splitting data into Train, DEV, test

from sklearn.model_selection import train_test_split

y=df.Exited # pulling values into another array so that we can drop

X=df.drop(['Exited'],axis='columns')

X_train, X_Dev, y_train, y_Dev = train_test_split(X,y,test_size=0.3,random_state=0,shuffle=False)

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2,random_state=0,shuffle=False)

#[Train] divide train data into categories , numerical and binary



binary_columns=["HasCrCard","IsActiveMember"]

binary_df=pd.DataFrame(X_train[binary_columns])



numerical_columns =["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]

numerical_df=pd.DataFrame(X_train[numerical_columns])



category_columns=['Geography','Gender']

category_df=pd.DataFrame(X_train[category_columns])

#[TRAIN] Encode Categorical Data



category_df['Geography'] = category_df['Geography'].astype('category')

category_df['Gender'] = category_df['Gender'].astype('category')

category_df_Final = pd.get_dummies(category_df)
#[TRAIN] feature scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_df_train_mean=numerical_df.mean()

numerical_df_train_std=numerical_df.std(axis=0)

numerical_df_scale =pd.DataFrame(scaler.fit_transform(numerical_df),columns=numerical_columns)
# [TRAIN] Concatenate Columns

X_train = pd.concat([numerical_df_scale, category_df_Final,binary_df], axis=1)
#[DEV] dividing data into binary, number and category

binary_columns=["HasCrCard","IsActiveMember"]

binary_df=pd.DataFrame(X_Dev[binary_columns])



numerical_columns =["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]

numerical_df=pd.DataFrame(X_Dev[numerical_columns])



category_columns=['Geography','Gender']

category_df=pd.DataFrame(X_Dev[category_columns])



# [DEV] Encode Categorical Data

category_df['Geography'] = category_df['Geography'].astype('category')

category_df['Gender'] = category_df['Gender'].astype('category')

category_df_Final = pd.get_dummies(category_df)



# [DEV] feature scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_df["CreditScore"]=(numerical_df["CreditScore"]-numerical_df_train_mean["CreditScore"]).div(numerical_df_train_std["CreditScore"])

numerical_df["Age"]=(numerical_df["Age"]-numerical_df_train_mean["Age"]).div(numerical_df_train_std["Age"])

numerical_df["Tenure"]=(numerical_df["Tenure"]-numerical_df_train_mean["Tenure"]).div(numerical_df_train_std["Tenure"])

numerical_df["Balance"]=(numerical_df["Balance"]-numerical_df_train_mean["Balance"]).div(numerical_df_train_std["Balance"])

numerical_df["NumOfProducts"]=(numerical_df["NumOfProducts"]-numerical_df_train_mean["NumOfProducts"]).div(numerical_df_train_std["NumOfProducts"])

numerical_df["EstimatedSalary"]=(numerical_df["EstimatedSalary"]-numerical_df_train_mean["EstimatedSalary"]).div(numerical_df_train_std["EstimatedSalary"])



#[DEV] Concatenate Columns

X_Dev = pd.concat([numerical_df, category_df_Final,binary_df], axis=1)
# [TEST] dividing data into binary, number and category

binary_columns=["HasCrCard","IsActiveMember"]

binary_df=pd.DataFrame(X_test[binary_columns])



numerical_columns =["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]

numerical_df=pd.DataFrame(X_test[numerical_columns])



category_columns=['Geography','Gender']

category_df=pd.DataFrame(X_test[category_columns])



# [TEST] Encode Categorical Data

category_df['Geography'] = category_df['Geography'].astype('category')

category_df['Gender'] = category_df['Gender'].astype('category')

category_df_Final = pd.get_dummies(category_df)



# [TEST] feature scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_df["CreditScore"]=(numerical_df["CreditScore"]-numerical_df_train_mean["CreditScore"]).div(numerical_df_train_std["CreditScore"])

numerical_df["Age"]=(numerical_df["Age"]-numerical_df_train_mean["Age"]).div(numerical_df_train_std["Age"])

numerical_df["Tenure"]=(numerical_df["Tenure"]-numerical_df_train_mean["Tenure"]).div(numerical_df_train_std["Tenure"])

numerical_df["Balance"]=(numerical_df["Balance"]-numerical_df_train_mean["Balance"]).div(numerical_df_train_std["Balance"])

numerical_df["NumOfProducts"]=(numerical_df["NumOfProducts"]-numerical_df_train_mean["NumOfProducts"]).div(numerical_df_train_std["NumOfProducts"])

numerical_df["EstimatedSalary"]=(numerical_df["EstimatedSalary"]-numerical_df_train_mean["EstimatedSalary"]).div(numerical_df_train_std["EstimatedSalary"])



# [TEST] Concatenate Columns

X_test = pd.concat([numerical_df, category_df_Final,binary_df], axis=1)
# assigning NULL to unused variables

df=None

X=None

y=None

binary_columns=None

binary_df=None

category_df=None

category_columns=None

category_df_Final=None

numerical_df=None

numerical_columns=None

numerical_df_train_mean=None

scaler=None

numerical_df_train_std=None

numerical_df_scale=None

null_columns=None
#defining and compiling model

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.optimizers import SGD



def deep_model():

    classifier = Sequential()

    classifier.add(Dense(units=3, kernel_initializer='he_uniform',

                bias_initializer='ones', activation='tanh', input_dim=13))

    classifier.add(Dense(units=3, kernel_initializer='he_uniform',

                bias_initializer='ones', activation='tanh'))

    classifier.add(Dense(units=2, kernel_initializer='he_uniform',

                bias_initializer='ones', activation='tanh'))

    #classifier.add(Dense(units=3, kernel_initializer='he_uniform',

                #bias_initializer='ones', activation='tanh'))

    #classifier.add(Dense(units=2, kernel_initializer='he_uniform',

                #bias_initializer='ones', activation='relu'))

    classifier.add(Dense(units=1,  kernel_initializer='he_uniform',

                bias_initializer='ones', activation='sigmoid'))

    classifier.compile(optimizer=Adam(learning_rate=0.01, amsgrad=False), 

    #classifier.compile(optimizer=SGD(learning_rate=0.001, momentum=0.8, nesterov=False), 

    loss='binary_crossentropy', 

    metrics=['accuracy','mae'])

    return classifier
# fitting the data 

from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint

classifier = deep_model()

# Set callback functions to early stop training and save the best model so far

callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

output=classifier.fit(X_train, y_train, batch_size=32,callbacks=callbacks ,epochs=100,validation_data=(X_Dev,y_Dev),shuffle=False)

#plotting

print(output.history.keys())

import matplotlib.pyplot as plt

# summarize history for accuracy

plt.plot(output.history['accuracy'])

plt.plot(output.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'Validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(output.history['loss'])

plt.plot(output.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'Validation'], loc='upper left')

plt.show()
#Calculating Errors

from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_absolute_error





#Confusion Matric Accuracy

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)*1

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

print("Confusion Matrix Accuracy: "+ str(accuracy*100)+"%")



#F1 score

recall=(cm[0][0])/(cm[0][0]+cm[0][1])

precision=(cm[0][0])/(cm[0][0]+cm[1][0])

F1=(2*recall*precision)/(precision+recall)

print("F1 Score:"+str(F1))



#MAE

mae=mean_absolute_error(y_test, y_pred)

print("MAE:"+str(mae))