import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import time
%matplotlib inline
train_df = pd.read_csv("../input/adopt-a-buddy/train.csv")
test_df=pd.read_csv("../input/adopt-a-buddy/test.csv")

train_df.head()

train_df.isnull().sum()
train_df.head()
train_df.pet_category.unique()

#train_df['pet_category'].replace(4,3,inplace=True)
train_df.color_type.unique()
train_df.pet_category.unique()
train_df.breed_category.unique()
train_df.breed_category=train_df.breed_category.astype(int)
# train_df.pet_category=train_df.pet_category.astype(str)
train_df.condition=train_df.condition.astype('Int64')

train_df.color_type = pd.Categorical(train_df.color_type)
train_df.color_type=train_df.color_type.cat.codes
test_df.color_type = pd.Categorical(test_df.color_type)
test_df.color_type=test_df.color_type.cat.codes
train_df.head()
#train_df.dtypes

train_df.condition.unique()
##All NaN belongs to particular breed category
train_df[train_df['breed_category']==2].condition.isnull().sum()
##All NaN belongs to particular breed category
train_df[train_df['breed_category']==2].condition.unique()
##Fill NaN values with 3 
train_df['condition'].fillna(3,inplace=True)
test_df['condition'].fillna(3,inplace=True)
train_df[train_df['breed_category']==2].condition.unique()
train_df.breed_category=train_df.breed_category.astype(int)
# train_df.pet_category=train_df.pet_category.astype(str)
train_df.condition=train_df.condition.astype(int)
train_df.color_type=train_df.condition.astype(int)

test_df.head()
sns.distplot(train_df['length(m)'])
print(len(train_df[train_df['length(m)'] == 0]))
print(len(test_df[test_df['length(m)']==0]))
train_df['length(cm)'] = train_df['length(m)'].apply(lambda x: x*100)
test_df['length(cm)'] = test_df['length(m)'].apply(lambda x: x*100)
train_df.drop('length(m)', axis=1, inplace=True)
test_df.drop('length(m)', axis=1, inplace=True)
train_df[train_df['length(cm)']==0].groupby(['length(cm)','pet_category']).size()
test_df['length(cm)'].mean()
val = train_df['length(cm)'].mean()
train_df['length(cm)'] = train_df['length(cm)'].replace(to_replace=0, value=val)
test_df['length(cm)'] = test_df['length(cm)'].replace(to_replace=0, value=val)
train_df[['length(cm)','height(cm)']].describe()
train_df['ratio_len_height'] = train_df['length(cm)']/train_df['height(cm)']
sns.boxplot(x='pet_category',y='length(cm)',data=train_df)
sns.distplot(test_df['length(cm)'])
sns.distplot(train_df['height(cm)'])
print(train_df.groupby(['breed_category']).size())
print(train_df.groupby(['pet_category']).size())
print(train_df.groupby(['breed_category','pet_category']).size())

matrix=np.tril(train_df.corr())
sns.heatmap(train_df.corr(),annot=True,fmt='.1g' ,mask=matrix)
test_df
# separate minority and majority classes
breedcat0 = train_df[train_df.breed_category==0]
breedcat1 = train_df[train_df.breed_category==1]
breedcat2= train_df[train_df.breed_category==2]

# upsample minority
pos_upsampled = resample(breedcat2,
 replace=True, # sample with replacement
 n_samples=len(breedcat1), # match number in majority class
 random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled_breed = pd.concat([breedcat0,breedcat1, pos_upsampled])
upsampled_breed.breed_category.value_counts()
# separate minority and majority classes
petcat0 = upsampled_breed[upsampled_breed.pet_category==0]
petcat1 = upsampled_breed[upsampled_breed.pet_category==1]
petcat2 = upsampled_breed[upsampled_breed.pet_category==2]
petcat4 = upsampled_breed[upsampled_breed.pet_category==4]


# upsample minority
pos_upsampled_0 = resample(petcat0,
 replace=True, # sample with replacement
 n_samples=len(petcat1), # match number in majority class
 random_state=27) # reproducible results
pos_upsampled_4 = resample(petcat4,
 replace=True, # sample with replacement
 n_samples=len(petcat1), # match number in majority class
 random_state=27) # reproducible results

# combine majority and upsampled minority
train_upsampled = pd.concat([pos_upsampled_0,petcat1, petcat2, petcat4])
train_upsampled.pet_category.value_counts()
train_upsampled


categorical_columns_test = ['condition', 'color_type', 'X1', 'X2']
categorical_columns = ['condition', 'color_type', 'X1', 'X2','breed_category']
numerical_columns = ['height(cm)','length(cm)']
output1 = ['breed_category']
output2=['pet_category']
#train_cat_df_1=pd.DataFrame(columns=categorical_columns1)
for category in categorical_columns:
    train_upsampled[category] = train_upsampled[category].astype('category')
    train_upsampled[category]=train_upsampled[category].cat.codes

#test_cat_df_1=pd.DataFrame(columns=categorical_columns1)
for category in categorical_columns_test:
    test_df[category] = test_df[category].astype('category')
    test_df[category]=test_df[category].cat.codes
train_upsampled.dtypes

X_train=train_upsampled.iloc[:,[3,4,5,6,7,10]]
X_train_pet=train_upsampled.iloc[:,[3,4,5,6,7,8,10]]
X_test=test_df.iloc[:,[3,4,5,6,7,8]]
#y_train=train_upsampled.iloc[:,[8,9]]
y_train_breed=train_upsampled["breed_category"]
y_train_pet=train_upsampled["pet_category"]

y_train_pet
y_train_pet=y_train_pet.replace(4,3)

y_train_pet
X_train.dtypes
y_train_breed_enc=to_categorical(y_train_breed)
y_train_breed_enc.shape
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)
# define the model
#get number of columns in training data
n_cols=X_train_scaled.shape[1]
n_out=y_train_breed_enc.shape[1]
# define model 2 layers
model_breed = Sequential()
model_breed.add(Dense(100, input_dim=n_cols, activation='relu'))
model_breed.add(Dense(50, activation='relu'))

model_breed.add(Dense(n_out, activation='softmax'))
# compile model
model_breed.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs= 10
start=time.time()
#fit model
hist=model_breed.fit(X_train_scaled, y_train_breed_enc,  validation_split=0.1, epochs=epochs,batch_size=100, verbose=1)
end=time.time()
print("Elapsed Time: ", end-start)
# predict probabilities for test set
yhat_probs = model_breed.predict(X_test_scaled, verbose=0)
# predict crisp classes for test set
yhat_classes = model_breed.predict_classes(X_test_scaled, verbose=1)
#yhat=pd.DataFrame(yhat_classes,name='pet_category')
#X_test_breed=pd.concat([X_test,yhat])
#X_test_breed
X_test['breed_category']=yhat_classes

X_test['breed_category']=X_test['breed_category'].astype('category')
X_test['breed_category']=X_test['breed_category'].cat.codes
X_test
y_train_pet_enc=to_categorical(y_train_pet)
y_train_pet_enc
#raname 4 to 3
sc = StandardScaler()
X_train_scaled_pet = sc.fit_transform(X_train_pet)
X_test_scaled_pet = sc.fit_transform(X_test)
X_train_scaled_pet
X_train_scaled_pet.shape
# define the model
#get number of columns in training data
n_cols=X_train_scaled_pet.shape[1]
n_out=y_train_pet_enc.shape[1]
# define model 2 layers
model_pet = Sequential()
model_pet.add(Dense(32, input_dim=n_cols, activation='relu'))
model_pet.add(Dense(16, activation='relu'))
model_pet.add(Dense(8, activation='relu'))

model_pet.add(Dense(n_out, activation='softmax'))
# compile model
model_pet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs= 100
start=time.time()
#fit model
hist=model_pet.fit(X_train_scaled_pet, y_train_pet_enc,  validation_split=0.1, epochs=epochs,batch_size=200, verbose=1)
end=time.time()
print("Elapsed Time: ", end-start)
# predict probabilities for test set
yhat_probs_final = model_pet.predict(X_test_scaled_pet, verbose=0)
# predict crisp classes for test set
yhat_classes_final = model_pet.predict_classes(X_test_scaled_pet, verbose=1)
X_test['pet_category']=yhat_classes_final
X_test
new_data=pd.merge(df_pred.pet_id,X_test[['breed_category','pet_category']], how = 'left', left_index = True, right_index = True)
new_data.to_csv('predictions_nn.csv')
df_pred=pd.DataFrame(test_df.pet_id)
df_pred.head()
X_test
test_df
pd.DataFrame(y_hat, index=df_pred.pet_id, columns=['breed_category','pet_category'], dtype=None, copy=False).to_csv('predictions4.csv')
