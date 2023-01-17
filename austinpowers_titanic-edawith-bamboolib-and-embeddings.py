#This may take a while. 
#Please wait until the kernel is idle before you continue.
!pip install --upgrade bamboolib>=1.4.1

#then RELOAD the page
# RUN ME AFTER YOU HAVE RELOADED THE BROWSER PAGE

# Uncomment and run lines below
import bamboolib as bam

import tensorflow as tf
tf.__version__
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import re
sns.set_style('whitegrid')


train_original = pd.read_csv('../input/train.csv')
test_original = pd.read_csv('../input/test.csv')
submission_example = pd.read_csv('../input/gender_submission.csv')
submission_example.head()
train_original.head()
test_original.head()
train = train_original.copy()
test = test_original.copy()
bam.glimpse(train)
bam.glimpse(test)
data = pd.concat([train.drop(['Survived'], axis=1), test])
data
# bamboolib live code export
data_age = data.groupby(['Pclass', 'Sex']).agg({'Age': ['min', 'max', 'mean', 'median']})
data_age.columns = ['_'.join(multi_index) for multi_index in data_age.columns.ravel()]
data_age = data_age.reset_index()
data_age
train
# bamboolib live code export
train.loc[((train['Age'].isna()) & (train['Pclass'] == 1)) & (train['Sex'].isin(['male'])), 'Age'] = 42
train.loc[((train['Age'].isna()) & (train['Pclass'] == 1)) & (train['Sex'].isin(['female'])), 'Age'] = 36
train.loc[((train['Age'].isna()) & (train['Pclass'] == 2)) & (train['Sex'].isin(['male'])), 'Age'] = 29.5
train.loc[((train['Age'].isna()) & (train['Pclass'] == 2)) & (train['Sex'].isin(['female'])), 'Age'] = 28
train.loc[((train['Age'].isna()) & (train['Pclass'] == 3)) & (train['Sex'].isin(['male'])), 'Age'] = 25
train.loc[((train['Age'].isna()) & (train['Pclass'] == 3)) & (train['Sex'].isin(['female'])), 'Age'] = 22
train

test
# bamboolib live code export
test.loc[((test['Age'].isna()) & (test['Pclass'] == 1)) & (test['Sex'].isin(['male'])), 'Age'] = 42
test.loc[((test['Age'].isna()) & (test['Pclass'] == 1)) & (test['Sex'].isin(['female'])), 'Age'] = 36
test.loc[((test['Age'].isna()) & (test['Pclass'] == 2)) & (test['Sex'].isin(['male'])), 'Age'] = 29.5
test.loc[((test['Age'].isna()) & (test['Pclass'] == 2)) & (test['Sex'].isin(['female'])), 'Age'] = 28
test.loc[((test['Age'].isna()) & (test['Pclass'] == 3)) & (test['Sex'].isin(['male'])), 'Age'] = 25
test.loc[((test['Age'].isna()) & (test['Pclass'] == 3)) & (test['Sex'].isin(['female'])), 'Age'] = 22
test
train


# bamboolib live code export
split_df = train["Name"].str.split(',', expand=True)
split_df.columns = [f"Name_{id_}" for id_ in range(len(split_df.columns))]
train = pd.merge(train, split_df, how="left", left_index=True, right_index=True)
split_df = train["Name_1"].str.split('.', expand=True)
split_df.columns = [f"Name_1_{id_}" for id_ in range(len(split_df.columns))]
train = pd.merge(train, split_df, how="left", left_index=True, right_index=True)
train = train.drop(columns=['PassengerId', 'Name', 'Name_0', 'Name_1', 'Name_1_1', 'Name_1_2'])
train
# bamboolib live code export
train = train.drop(columns=['Cabin'])
split_df = train["Ticket"].str.split(' ', expand=True)
split_df.columns = [f"Ticket_{id_}" for id_ in range(len(split_df.columns))]
train = pd.merge(train, split_df, how="left", left_index=True, right_index=True)
train = train.drop(columns=['Ticket', 'Ticket_1', 'Ticket_2'])
train
# bamboolib live code export
train = train.rename(columns={'Name_1_0': 'Title'})
train = train.rename(columns={'Ticket_0': 'Ticket'})
train
# bamboolib live code export
train["Title"] = train["Title"].str.strip()
train
# bamboolib live code export
train = pd.get_dummies(train, columns=['Sex'], drop_first=True, dummy_na=False)
train
# bamboolib live code export
train = train.loc[~(train['Embarked'].isna())]
train
test
# bamboolib live code export
split_df = test["Name"].str.split(',', expand=True)
split_df.columns = [f"Name_{id_}" for id_ in range(len(split_df.columns))]
test = pd.merge(test, split_df, how="left", left_index=True, right_index=True)
split_df = test["Name_1"].str.split('.', expand=True)
split_df.columns = [f"Name_1_{id_}" for id_ in range(len(split_df.columns))]
test = pd.merge(test, split_df, how="left", left_index=True, right_index=True)
test = test.drop(columns=['PassengerId', 'Name', 'Cabin', 'Name_0', 'Name_1', 'Name_1_1'])
split_df = test["Ticket"].str.split(' ', expand=True)
split_df.columns = [f"Ticket_{id_}" for id_ in range(len(split_df.columns))]
test = pd.merge(test, split_df, how="left", left_index=True, right_index=True)
test = test.drop(columns=['Ticket', 'Ticket_1', 'Ticket_2'])
test = test.rename(columns={'Ticket_0': 'Ticket'})
test = test.rename(columns={'Name_1_0': 'Title'})
test
# bamboolib live code export
test = pd.get_dummies(test, columns=['Sex'], drop_first=True, dummy_na=False)
test
# bamboolib live code export
test["Title"] = test["Title"].str.strip()
test
test['Fare'].fillna(value=7)
test
# bamboolib live code export
test.loc[test['Fare'].isna(), 'Fare'] = 7
test
data_Title = pd.concat([train, test])
data_Title
print("train df legth :{}".format(len(train)))
print("test df legth :{}".format(len(test)))
print("data_Title df legth :{}".format(len(data_Title)))
data_Title['Title'].nunique(), data_Title['Title'].unique()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_Title['Title']= le.fit_transform(data_Title['Title'])
data_Title
# bamboolib live code export
data_Title = data_Title.drop(columns=['Ticket'])
data_Title
data_Title['Embarked'] = data_Title['Embarked'].astype(str, errors='raise')
em = LabelEncoder()
data_Title['Embarked']= em.fit_transform(data_Title['Embarked'])
data_Title.info()
train = data_Title.iloc[:889,:]
test = data_Title.iloc[889:,:]
train
test
# bamboolib live code export
test = test.drop(columns=['Survived'])
test
X = train.drop('Survived',axis=1)
y = train['Survived']
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X,y,test_size=0.15,
                                                  random_state=101)
x_test = test
x_train.keys()
categorical_vars = ['Pclass','Sex_male','SibSp','Parch','Embarked','Title']
numerical_vars=['Age','Fare']
for c in train[categorical_vars]:
    
    print(c +" : {}".format(train[c].nunique()))
X_train = [] #create empty arrays
X_val = []
X_test = []
for cat in categorical_vars:# append the data from every single category
    X_train.append(x_train[cat].values)
    X_val.append(x_val[cat].values)
    X_test.append(x_test[cat].values)
#standardize the numerical variables
from sklearn.preprocessing import StandardScaler


ss = StandardScaler()

x_train_num = x_train[numerical_vars].astype('float32').values
x_val_num = x_val[numerical_vars].astype('float32').values
x_test_num = x_test[numerical_vars].astype('float32').values

X_scaled_train = ss.fit_transform(x_train_num)
X_scaled_val = ss.transform(x_val_num)
X_scaled_test = ss.transform(x_test_num)
X_train.append(X_scaled_train) #append continues values
X_val.append(X_scaled_val)
X_test.append(X_scaled_test)
X_train
#size of embeddings
cat_sizes = {}#empty list for category size and embedding size
cat_embsizes = {}
for cat in categorical_vars:
    cat_sizes[cat] = x_train[cat].nunique()
    cat_embsizes[cat] = min(50, cat_sizes[cat]//2+1)#rule for embedding
#imports
from keras.layers import Dense, Dropout, Embedding, Input, Reshape, Concatenate
from keras.models import Model
inputs = []
concat = []
#create inputs and concatenate for categorical vars
for cat in categorical_vars:
    x = Input((1,), name=cat)
    inputs.append(x)
    x = Embedding(cat_sizes[cat]+1, cat_embsizes[cat], input_length=1)(x)
    x = Reshape((cat_embsizes[cat],))(x)
    concat.append(x)
#create inputs and concatenate for continuous variables
numerical_inputs = Input((len(numerical_vars),), name='continuous_vars')
inputs.append(numerical_inputs)
concat.append(numerical_inputs)
y = Concatenate()(concat)
#add the deep NN model and compile
p=0.2
y = Dense(128, activation= 'relu')(y)
y = Dropout(p)(y)
y = Dense(64, activation= 'relu')(y)
y = Dropout(p)(y)
y = Dense(32, activation= 'relu')(y)
y = Dropout(p)(y)

y = Dense(1, activation='sigmoid')(y)
model = Model(inputs=inputs, outputs=y)
model.compile(optimizer='adam', loss='binary_crossentropy')
#callbacks earlystopping and checkpoint
from keras.callbacks import EarlyStopping,ModelCheckpoint

#checkpoint = ModelCheckpoint('best_test_embed.h5', monitor='val_loss',
                             #save_best_only=True, verbose=2)

early_stopping = EarlyStopping(monitor='loss',mode='min', patience=10)
from keras.utils import plot_model
plot_model(model,to_file='model.png',
           show_shapes=True,
           show_layer_names=True,
           rankdir='TB')
from IPython.display import Image
Image('../working/model.png')
model.fit(X_train, y_train, epochs=1000), 
          #validation_data=[X_val, y_val])#,
          #callbacks=[early_stopping])#,checkpoint
X_train






sns.countplot(x='SibSp', data=train, palette='OrRd_r')
train['Fare'].hist(bins=50, figsize=(12,6), color='darkred')
import cufflinks as cf
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=50, color='darkred',title='Fare')
plt.figure(figsize=(12,8))
sns.boxplot(x='Pclass',y='Age',data=train, palette='OrRd_r')
train[train['Pclass']==1]['Age'].mean()
train[train['Pclass']==2]['Age'].mean()
train[train['Pclass']==3]['Age'].mean()
# create a function

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 38
        
        elif Pclass == 2:
            return 30
        
        elif Pclass == 3:
            return 25
        
    else:
        return Age
        
train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='OrRd')
sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='OrRd_r')
train.drop('Cabin', axis=1,inplace=True)
test.drop('Cabin', axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='OrRd')
sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='OrRd_r')
train['Fare'].mean()
test[test['Fare'].isnull()] = 32
sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='OrRd_r')
train.dropna(inplace=True)
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='OrRd')
sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='OrRd_r')
train.head()
test.head()
# Store target variable of training data in a safe place
survived_train = train.Survived
survived_train.head()
# Concatenate training and test sets
data = pd.concat([train.drop(['Survived'], axis=1), test])
data.info()
data.Name

# Extract Title from Name, store in column and plot barplot
#data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
#re.search
#sns.countplot(x='Title', data=data);
#plt.xticks(rotation=45);

pd.get_dummies(train['Sex'], drop_first=True).head()
sex_train = pd.get_dummies(train['Sex'], drop_first=True)
sex_train.head()
pd.get_dummies(test['Sex'], drop_first=True).head()
sex_test = pd.get_dummies(test['Sex'], drop_first=True)
sex_test.head()
embark_train = pd.get_dummies(train['Embarked'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)

embark_train.head()
embark_test.head()
train = pd.concat([train,sex_train,embark_train], axis=1)
test = pd.concat([test,sex_test,embark_test], axis=1)

train.head()
sns.violinplot('Pclass','Age', hue='Survived', data=train,split=True)
train.drop(['PassengerId','Name','Sex','Ticket','Embarked'], axis=1, inplace=True)
train.head()
test.drop(['PassengerId','Name','Sex','Ticket','Embarked'], axis=1, inplace=True)
test.head()
test.drop(['female','C'], axis=1, inplace=True)
test.head()
X = train.drop('Survived',axis=1)
y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11 )
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions_logmodel = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions_logmodel))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions_logmodel)
sns.heatmap(confusion_matrix(y_test,predictions_logmodel),cmap='OrRd_r',annot=True)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
prediction_d3 = dtree.predict(X_test)
print(classification_report(y_test,prediction_d3))
print(confusion_matrix(y_test,prediction_d3))
sns.heatmap(confusion_matrix(y_test,prediction_d3),cmap='OrRd_r',annot=True)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
prediction_rfc = rfc.predict(X_test)
print(classification_report(y_test,prediction_rfc))
print(confusion_matrix(y_test,prediction_rfc))
sns.heatmap(confusion_matrix(y_test,prediction_rfc),cmap='OrRd_r',annot=True)
error_rate = []
index_error = []

for i in range(1,200):
    
    rfc = RandomForestClassifier(n_estimators=i)
    
    rfc.fit(X_train,y_train)
    pred_i = rfc.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    index_error.append(i)
error_rate_df = pd.DataFrame({'n_estimators': index_error,
                                     'error_rate': error_rate})
error_rate_df.head()
error_rate_df[error_rate_df['error_rate'].min()]['n_estimators']
df[df['A']==5].index.item()
plt.figure(figsize=(10,6))
plt.plot(range(1,200),error_rate,color='orange', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. n_estimators')
plt.xlabel('n_estimators')
plt.ylabel('Error Rate')
rfc = RandomForestClassifier(n_estimators=39)
rfc.fit(X_train,y_train)
prediction_rfc = rfc.predict(X_test)
print(classification_report(y_test,prediction_rfc))
print(confusion_matrix(y_test,prediction_rfc))
sns.heatmap(confusion_matrix(y_test,prediction_rfc),cmap='OrRd_r',annot=True)
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X,y)
prediction_rfc_kaggle = rfc.predict(test)
kaggle_submission = pd.DataFrame({"PassengerId": test_original["PassengerId"],
                                     "Survived": prediction_rfc_kaggle})
kaggle_submission.head()
kaggle_submission.to_csv("kaggle_submission.csv", index=False)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
from keras.optimizers import SGD
import graphviz
# Creating the model
model = Sequential()

# Inputing the first layer with input dimensions
model.add(Dense(64, 
                activation='relu',  
                input_dim=8,
                kernel_initializer='uniform'))

# Adding an Dropout layer to previne from overfitting
model.add(Dropout(0.50))

#adding second hidden layer 
model.add(Dense(32,
                kernel_initializer='uniform',
                activation='relu'))

# Adding another Dropout layer
model.add(Dropout(0.50))

# adding the output layer that is binary [0,1]
model.add(Dense(1,
                kernel_initializer='uniform',
                activation='sigmoid'))
#With such a scalar sigmoid output on a binary classification problem, the loss
#function you should use is binary_crossentropy

#Visualizing the model
model.summary()
X_train.head()
#Creating an Stochastic Gradient Descent
sgd = SGD(lr = 0.01, momentum = 0.9)

# Compiling our model
model.compile(optimizer = sgd, 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
#optimizers list
#optimizers['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

# Fitting the ANN to the Training set
model.fit(X_train, y_train, 
               batch_size = 60, 
               epochs = 300, verbose=1)
prediction_NN = model.predict(X_test)
def positive_sol(x):
    if x>0.5:
        return 1
    else:
        return 0
    
pred = prediction_NN.apply(lambda x: positive_sol )
print(classification_report(y_test,prediction_NN))
print(confusion_matrix(y_test,prediction_NN))
