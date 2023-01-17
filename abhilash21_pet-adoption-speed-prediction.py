import numpy as np
import pandas as pd
from matplotlib import image
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy import stats
import cv2
from keras.utils import Sequence
#load and display the 1st 10 rows of the train data
df = pd.read_csv("../input/training/input.csv", index_col=0)
df.head(5)
# dataset dimension
n_rows = df.shape[0]
n_cols = df.shape[1]

print("\nNumber of rows - ", n_rows, "\nNumber of columns - ", n_cols)

# dataset datatypes - #int cols, #string cols
# would be useful during one-hot encoding
n_stringTypes = 0 

for i in df.iloc[0]:
    if type(i)==str:
        n_stringTypes+=1
print("Number of string type columns - ", n_stringTypes) 
filepath = '../input/petfinder-adoption-prediction/PetFinder-BreedLabels.csv'
df_breedLabels = pd.read_csv(filepath)
df_breedLabels.head(5)
# dataset dim
n_df_breedRows = df_breedLabels.shape[0]
n_df_breedCols = df_breedLabels.shape[1]
print("Dataset shape - ", df_breedLabels.shape)
# different number of breeds for dog [type=1] and cat [type=2]

grp1, grp2 = df_breedLabels.groupby('Type').apply(lambda ser: ser['BreedName'].unique())
print("\nDog Breeds - ", grp1[:10], "....","\nNumber of Dog Breeds - ", len(grp1))
print("\nCat Breeds - ", grp2[:10], "....","\nNumber of Cat Breeds - ", len(grp2))
filepath = '../input/petfinder-adoption-prediction/PetFinder-ColorLabels.csv'
df_ColorLabels = pd.read_csv(filepath)
print(df_ColorLabels)
print("Shape - ", df_ColorLabels.shape)
print("Number of unique color for pet dataset - ", len(df_ColorLabels))
filepath = '../input/petfinder-adoption-prediction/PetFinder-StateLabels.csv'
df_StateLabels = pd.read_csv(filepath)
print(df_StateLabels)
print("Shape - ", df_StateLabels.shape)
print("Number of unique states for pet dataset - ", len(df_StateLabels))
# Loading pet images corresponding to top 3 rows of the dataframe. This is just for visualization
# Not loading all images at once, as it would take up a lot of memory
def plot_image(filename):
    data = image.imread(filename)
    print("\nData type - ",data.dtype, "\nData shape - ", data.shape)
    # display image
    plt.imshow(data)
    plt.show()
    
img_path = "../input/petfinder-adoption-prediction/train_images/"
dff = df.head(2)
for i in range(0,len(dff)):
    print("\nName - ", dff.loc[i,'Name'], "\nPetID - ",dff.loc[i,'PetID'])
    plot_image(img_path+dff.loc[i,'PetID']+'-'+'1.jpg')
    
df.loc[:,'PetID'].duplicated().any() #if False, no duplicate values present
def checkIfNull():
    for i in df.columns:
        if df[i].isnull().any():
            print('Column','"',i,'"',' has missing values')
        else:
            continue
checkIfNull()
#Let's print the column Names
names = df.loc[:,'Name']
names[:28]
# The NaN values are visible
df.loc[:,'Name'] = df.loc[:,'Name'].fillna('Unknown')
for i,name in enumerate(names):
    if re.search("Name", name):
        df.loc[:,'Name'][i] = "Unknown"
print(df.loc[:,'Name'][:15])
print(df.loc[:,'Name'].isnull().any())
dtypes = {}
for i,k in enumerate(df.iloc[0]):
    dtypes[df.columns[i]] = type(k)
print(dtypes)
# from the output it can be stated that the data types of each col match their content
#Set Membership 

# Gender
print('Gender - ', df.loc[:,'Gender'].isin([1,2,3]).all())
#Fur length
print('Fur length - ', df.loc[:,'FurLength'].isin([0,1,2,3]).all())
# MaturitySize
print('Maturity - ', df.loc[:,'MaturitySize'].isin([0,1,2,4,3]).all())
# Vaccinated
print('Vaccinated - ', df.loc[:,'Vaccinated'].isin([1,2,3]).all())
# Dewormed
print('Dewormed - ', df.loc[:,'Dewormed'].isin([1,2,3]).all())
# Sterilized
print('Sterilized - ', df.loc[:,'Sterilized'].isin([1,2,3]).all())
# Health
print('Health - ', df.loc[:,'Health'].isin([0,1,2,3]).all())
# Foreign Key membership - Breed, color, state

#breed
print('Breed 1 - ', df.loc[:,'Breed1'].isin(df_breedLabels.loc[:,'BreedID']).all())
print('Breed 2 - ', df.loc[:,'Breed2'].isin(df_breedLabels.loc[:,'BreedID']).all())

#color
print('Color 1 - ', df.loc[:,'Color1'].isin(df_ColorLabels.loc[:,'ColorID']).all())
print('Color 2 - ', df.loc[:,'Color2'].isin(df_ColorLabels.loc[:,'ColorID']).all())
print('Color 3 - ', df.loc[:,'Color3'].isin(df_ColorLabels.loc[:,'ColorID']).all())

#State
print('State - ', df.loc[:,'State'].isin(df_StateLabels.loc[:,'StateID']).all())
#Cols - Breed1, Breed2, Color2, Color3 needs to be checked
def getIndices(col,colname):
    indices = col[col==False].index[:]
    ls = [df.loc[i,colname] for i in indices]
    return ls, indices
        
breed1 = df.loc[:,'Breed1'].isin(df_breedLabels.loc[:,'BreedID'])
breed2 = df.loc[:,'Breed2'].isin(df_breedLabels.loc[:,'BreedID'])
color2 = df.loc[:,'Color2'].isin(df_ColorLabels.loc[:,'ColorID'])
color3 = df.loc[:,'Color3'].isin(df_ColorLabels.loc[:,'ColorID'])

b1,ind = getIndices(breed1, 'Breed1')
b2,ind2 = getIndices(breed2, 'Breed2')
c2,ind3 = getIndices(color2, 'Color2')
c3,ind4 = getIndices(color3, 'Color3')
print('\nb1 values (anomaly) -', b1,' at indices', ind,'\nb2 no of anomalous values - ',len(ind2))
print('c2 no of anomalous values - ', len(ind3), '\nc3 no of anomalous values - ', len(ind4))
fig, axes = plt.subplots(1,2, figsize=(14,5))
axes[0].hist(df.loc[:,'Breed2'], color='blue')
axes[0].set_xlabel('Breed2 values')
axes[0].set_ylabel('Frequency')
axes[0].set_title('No of pets per Breed2')

axes[1].hist(df.loc[:,'Color3'], bins = df_ColorLabels.shape[0],histtype='barstacked')
axes[1].set_xlabel('Color3 values')
axes[1].set_ylabel('Frequency')
axes[1].set_title('No of pets per Color3')
plt.show()
# Dropping the Breed2 and Color3 columns
df = df.drop(['Breed2', 'Color3'], axis=1)
df.head(5)
fig, axes = plt.subplots(1,3, figsize=(17,4))

axes[0].hist(df.loc[:,'FurLength'], bins = range(0,6,1))
axes[0].set_xlabel('FurLength values')
axes[0].set_ylabel('Frequency')
axes[0].set_title('# of Pets per Fur length')

axes[1].hist(df.loc[:,'MaturitySize'], bins = range(0,6,1),color='g')
axes[1].set_xlabel('MaturitySize values')
axes[1].set_ylabel('Frequency')
axes[1].set_title('# of Pets vs MaturitySize')

axes[2].hist(df.loc[:,'Health'], bins = range(0,5,1),color='y')
axes[2].set_xlabel('Health values')
axes[2].set_ylabel('Frequency')
axes[2].set_title('# of Pets vs Health Values')

plt.show()
# non-categorical attributes - Age, Quantity,Fee,VideoAmt,PhotoAmt. Calculating range of values

age_min = df.loc[:,'Age'].min() #age is in months
age_max = df.loc[:,'Age'].max()
print("Pet ages range - ", age_max-age_min, "(max -", age_max,",min - ", age_min,")")
print("Average age of pets - ", df.loc[:,'Age'].mean())

quantity_min = df.loc[:,'Quantity'].min()
quantity_max = df.loc[:,'Quantity'].max()
print("\nMin and Max no. pets in a profile - ", quantity_min,",",quantity_max)

fee_min = df.loc[:,'Fee'].min()
fee_max = df.loc[:,'Fee'].max()
print("\nMin and Max fee - ", fee_min,",",fee_max)
print("Average adoption fee of pets - ", df.loc[:,'Fee'].mean())

video_avg = df.loc[:,'VideoAmt'].mean()
print("\nMean no of videos uploaded for each pet - ", video_avg)

photo_avg = df.loc[:,'PhotoAmt'].mean()
print("Mean no of photos uploaded for each pet - ", photo_avg)
#AdoptionSpeed vs Distribution of VideoAmt, PhotoAmt
def adoptionSpeedDistribution():
    adoption0 = np.where(df.loc[:,'AdoptionSpeed']==0)
    adoption1 = np.where(df.loc[:,'AdoptionSpeed']==1)
    adoption2 = np.where(df.loc[:,'AdoptionSpeed']==2)
    adoption3 = np.where(df.loc[:,'AdoptionSpeed']==3)
    adoption4 = np.where(df.loc[:,'AdoptionSpeed']==4)
    adoption_ = [adoption0, adoption1, adoption2, adoption3, adoption4]
    return adoption_  
adoption_ = adoptionSpeedDistribution()
n_pets = []
video_amt = []
for i in range(5):
    n_pets += [i]* len(adoption_[i][0])
    video_amt += [i]* df.loc[adoption_[i][0],'VideoAmt'].sum()

fig, axes = plt.subplots(1,2, figsize=(14,5))

#Mean is not plotted for videoAmt, instead the sum is plotted, since the mean is 0 at every adoptionSpeed 
#target. Plotting the sum (which too is less as can be seen in the graph) clearly shows that VideoAmt has 
# 0 or very less impact on the target

axes[0].hist([n_pets,video_amt],bins = range(0,6,1), color = ['r','y'], label=['pets', 'videos'])
axes[1].plot(range(0,101),df.loc[:100, 'VideoAmt'], label=['VideoAmt'])
axes[1].plot(range(0,101),df.loc[:100, 'AdoptionSpeed'], label=['AdoptionSpeed'])

axes[0].set_xlabel('Adoption Speed - 0,1,2,3,4 days')
axes[0].set_ylabel('Number of pets, videos')

axes[1].set_xlabel('Pet# (Only till 100)')
axes[1].set_ylabel('Number of videos')
axes[1].legend()
axes[0].legend()
axes[0].set_title('AdoptionSpeed Distribution wrt VideoAmt_sum')
axes[1].set_title('Distribution VideoAmt')
plt.show()
adoption_p = adoptionSpeedDistribution()
n_pets_ = []
photo_amt = []
for i in range(5):
    n_pets_ += [i]* int(len(adoption_p[i][0])/100)
    photo_amt += [i]* int(df.loc[adoption_p[i][0],'PhotoAmt'].astype('int64').mean())

fig, axes = plt.subplots(1,2, figsize=(14,5))

axes[0].hist([n_pets_,photo_amt],bins = range(0,6,1), color = ['r','y'], label=['pets', 'photos'])
axes[1].plot(range(0,101),df.loc[:100, 'PhotoAmt'], label=['PhotoAmt'])
axes[1].plot(range(0,101),df.loc[:100, 'AdoptionSpeed'], label=['AdoptionSpeed'])

axes[0].set_xlabel('Adoption Speed - 0,1,2,3,4 days')
axes[0].set_ylabel('Number of pets (1/100), photos')

axes[1].set_xlabel('Pet# (Only till 100)')
axes[1].set_ylabel('Number of photos')
axes[1].legend()
axes[0].legend()
axes[0].set_title('AdoptionSpeed Distribution wrt PhotoAmt_mean')
axes[1].set_title('Distribution PhotoAmt')
plt.show()
df = df.drop(['VideoAmt'], axis=1)
df.shape
# box-plot for Age and adoption price - to check for outliers
fig, axes = plt.subplots(2,2,figsize=(14,6), sharex=True)

sns.set(style="whitegrid")
sns.boxplot(x=df.loc[:,'Age'], ax = axes[0,0])
axes[0,0].set_title('Age Box-plot')

sns.boxplot(x=df.loc[:,'Fee'], ax = axes[0,1])
axes[0,1].set_title('Adoption Fee Box-plot')

axes[1,0].scatter(range(0,14993), df.loc[:,'Age'])
axes[1,0].set_xlabel('Sample#')
axes[1,0].set_ylabel('Age')
axes[1,0].set_title('Pet Ages vs sample#')

axes[1,1].scatter(range(0,14993), df.loc[:,'Fee'])
axes[1,1].set_xlabel('Sample#')
axes[1,1].set_ylabel('Adoption Fee')
axes[1,1].set_title('Adoption fee vs sample#')
plt.setp(axes, yticks=[])
plt.tight_layout()

# Checking adoption speed of cats vs dogs

adoption_ = adoptionSpeedDistribution()
cats = []
dogs = []
for i in range(len(adoption_)):
    cats += [i]*(np.where(df.loc[adoption_[i][0],'Type']==2)[0].shape[0])
    dogs += [i]*(np.where(df.loc[adoption_[i][0],'Type']==1)[0].shape[0])

fig, axes = plt.subplots()
plt.hist([cats,dogs],bins = range(0,6,1), color = ['b','g'], label=['cats', 'dogs'])
plt.xlabel('Adoption Speed - 0,1,2,3,4 days')
plt.ylabel('Number of pets')
plt.legend()
plt.title('Cats and Dogs Adoption Speed')
plt.show()
# Label Encoding 'Name', 'RescuerID', 'State'
def labelEncode(attr):
    enc = LabelEncoder()
    attr_ = list(attr)
    enc.fit(attr_)
    return enc.transform(attr_)

#Not encoding PetID, since it represents the image, which would be seperated out as a different dataset
df.loc[:,'Name'] = labelEncode(df.loc[:,'Name'])
df.loc[:,'RescuerID'] = labelEncode(df.loc[:,'RescuerID'])
df.loc[:,'State'] = labelEncode(df.loc[:,'State'].astype(str))
df.head(5)
df.loc[:,'State'].max() #number of unique state values
# Correlation matrix of the dataset

corr = df.corr(method='pearson')
adoptionSpeed = corr['AdoptionSpeed'][:-1]
#correlation of attributes with adoptionspeed
adoptionSpeed
x = np.where(adoptionSpeed<0)
col = adoptionSpeed[x[0]].idxmax() #column to delete
col
df = df.drop(['RescuerID'],axis=1)
df.shape
#correlation matrix
corr
def combineFeatures(dff, col1, col2, col3):
    x,y,z = df.loc[:,col1], df.loc[:,col2], df.loc[:,col3] 
    a = np.hstack((np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)))
    a = np.hstack((np.array(a), np.array(z).reshape(-1,1)))
    col_new = [int(stats.mode(a[i])[0]) for i in range(a.shape[0])]
    print('Original dataframe - ', dff.shape)
    dff = df.drop([col1, col2, col3],axis=1)
    # Let the new column name be Vaccinated
    dff['Vaccinated'] = col_new
    print('New Dataframe - ', dff.shape)
    return dff

df = combineFeatures(df, 'Vaccinated', 'Dewormed', 'Sterilized')
plt.hist(df.loc[:,'Fee'])
df = df.drop(['Fee'], axis=1)
print('Current Dataframe - ', df.shape)
from skimage.color import gray2rgb
from skimage.transform import resize

mean = [0.485,0.456,0.406] # standard values, based on ImageNet data
std = [0.229,0.224,0.225] # standard values, based on ImageNet data

def read_image(path):
    """
    resizing image into size 224x224x3 to feed into ResNet50
    """
    default_path = '../input/petfinder-adoption-prediction/train_images/86e1089a3-1.jpg'
    try:
        img = image.imread(path)
        img = img/255.0
        
    except FileNotFoundError:
        img = image.imread(default_path) #read this default img (randomly selected to fill missing data)
        img = img/255.0
    
    return gray2rgb(resize(img, (160,160)))

def normalize_image(img):
    
    img[:,:,0] -= mean[0]
    img[:,:,0] /= std[0]
        
    img[:,:,1] -= mean[1]
    img[:,:,1] /= std[1]
        
    img[:,:,2] -= mean[2]
    img[:,:,2] /= std[2]
        
    return img
from sklearn.model_selection import train_test_split

# seperate the target and features
target = df.loc[:, 'AdoptionSpeed'].to_numpy()
features = df.drop(['AdoptionSpeed'], axis=1).to_numpy()

train_x, val_x, train_y, val_y = train_test_split(features, target, test_size=0.10, random_state=42)
# remove the image col from the numpy arrays (index 12 in a row)
train_images = train_x[:, 12]
val_images = val_x[:,12]
train_x = np.delete(train_x, 12, 1)
val_x = np.delete(val_x, 12, 1)
# One Hot Encoding the labels
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
y_train = enc.fit_transform(train_y.reshape(-1,1))
y_val = enc.fit_transform(val_y.reshape(-1,1))
print('Train X - ', train_x.shape, '\nVal X - ', val_x.shape, '\nTrain Y - ', y_train.shape, '\nVal Y - '
     , y_val.shape, '\nTrain images - ', train_images.shape, '\nVal images - ', val_images.shape)
# get the output of a neural network layer. Would be useful for data generator
def get_layer_output(model, data, layer = 'dense_4'):
    
    network_output = model.get_layer(layer).output
    feature_extraction_model = Model(model.input, network_output)
    prediction = feature_extraction_model.predict(np.asarray(data).astype(np.float32))
    #print(type(prediction))
    #print(prediction.shape)
    return np.asarray(prediction).astype(np.float32)
# Generator to read and preprocess data in batches for model training
class DataGenerator(Sequence) :
     
    def __init__(self, trainX, train_imgs, y_train, model, densenet, batch_size) :
        self.trainx = trainX
        self.imgs = train_imgs
        self.labels = y_train
        self.model_1 = model
        self.densenet = densenet
        self.batch_size = batch_size
    
    def __len__(self) :
        return (np.ceil(len(self.imgs) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx) :
        batch_x_imgs = self.imgs[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x = self.trainx[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        
        NN_1_data = get_layer_output(self.model_1, batch_x)
        images = []
        for file in batch_x_imgs:
            images.append(normalize_image(read_image(img_path+file+'-1.jpg')))
        
        DenseNet_data = get_layer_output(self.densenet, np.array(images), layer='avg_pool')
        
        return np.concatenate((NN_1_data, DenseNet_data), axis=1), np.array(batch_y)
def plot_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Activation, Input, Dropout, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.applications import DenseNet169
from keras.models import load_model
# NN-1 model
inputs = Input(shape=(14,))
x = Dense(32, activation='sigmoid')(inputs)
#x = Dropout(0.2)(x)
x = Dense(64, activation='sigmoid')(x)
#x = Dense(128)(x) ## Deeper networks degrade the model. Doesn't fit the data well
#x = Dense(256, activation='tanh')(x)
#x = Dropout(0.25)(x)
#x = Dense(512, activation='sigmoid')(x)
#x = Dense(20, activation='tanh')(x)
#x = Dropout(0.25)(x)
out = Dense(5, activation='softmax')(x)

model = Model(inputs=inputs, outputs=out)
model.summary()
# NN-1 model
lr = 0.001
checkpt = ModelCheckpoint(filepath='../input/output/models/best_model.h5',monitor='val_acc',save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='min')
model.compile(optimizer=Adam(lr=lr),loss='categorical_crossentropy',metrics=['accuracy'])
# EXPERIMENTAL
# scaling down the name column of training data
def scaleData(X):
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_std * (10)
train_x[:,3] = scaleData(train_x[:, 3])
train_x[:,1] = scaleData(train_x[:, 1])

# After scaling, all values remain same, except training accuracy increases by 8%
# NN-1 model
history = model.fit(np.asarray(train_x).astype(np.float32),np.asarray(y_train),epochs=200,batch_size=32,shuffle=True,
          validation_data=(np.asarray(val_x).astype(np.float32),np.asarray(y_val)),callbacks=[checkpt])
plot_history(history) #an averaged out curve might look less noisy
model.evaluate(np.array(val_x).astype(np.float32), np.array(y_val), batch_size=32)
model.save('best_model_latest.h5')
#model = load_model('../input/models-files/best_model_latest.h5')
# NN-2 model
model_densenet = DenseNet169(include_top=True, weights="imagenet")
model_densenet.summary()
# NN-2 model
batch_size = 64

train_gen = DataGenerator(train_x, train_images, y_train, model, model_densenet, batch_size)
val_gen = DataGenerator(val_x, val_images, y_val, model, model_densenet,batch_size)
# NN-2 model
# Model architecture - input : (1728,)
inputs = Input(shape=(1728,))

x = Dense(1024, activation='sigmoid')(inputs)
x = Dropout(0.2)(x)

x = Dense(512, activation='sigmoid')(x)

x = Dense(256, activation='tanh')(x)
x = Dropout(0.25)(x)

x = Dense(256, activation='sigmoid')(x)

x = Dense(64, activation='tanh')(x)
x = Dropout(0.25)(x)

out = Dense(5, activation='softmax')(x)

model_2 = Model(inputs=inputs, outputs=out)
model_2.summary()
# NN-2 model
lr = 0.001
checkpt = ModelCheckpoint(filepath='../input/output/models/best_model_Part2.h5',monitor='val_acc',save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='min')
model_2.compile(optimizer=Adam(lr=lr),loss='categorical_crossentropy',metrics=['accuracy'])
# NN-2 model
history = model_2.fit_generator(generator=train_gen,steps_per_epoch = int(13493 // batch_size),epochs = 20,
                   verbose = 1,validation_data = val_gen,validation_steps = int(1500 // batch_size))
model_2.save('final_model_latest.h5')
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
# load the trained model
model_2 = load_model('final_model_latest.h5')
# evaluate model
model_2.evaluate_generator(val_gen, 1500)
y_pred = model_2.predict_generator(val_gen, 1500)
y_pred.shape
# Converting predictions into suitable format to feed into sklearn libraries
def process_predictions():
    prediction = []
    for i in range(len(y_pred)):
        t = np.zeros(5)
        t[np.argmax(y_pred[i])] = 1
        prediction.append(t)
    return np.array(prediction)

prediction = process_predictions()
# Classification Report
print(classification_report(y_val, prediction))
# Confusion Matrix
print(multilabel_confusion_matrix(y_val,prediction,labels=[0,1,2,3,4]))
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
clf1 = AdaBoostClassifier(n_estimators=100, random_state=0)
clf2 = GradientBoostingClassifier(min_samples_split = 8, random_state=0)
# transform One hot encoded data to labelled data (as in labelEncoded data)
def decode(y):
    ys = []
    for i in range(len(y)):
        ys.append(np.argmax(y[i]))
    return np.array(ys)
y_train = decode(y_train)
y_val = decode(y_val)
y_pred = decode(prediction) # NN model predictions
clf1.fit(train_x, y_train) # fit the classifier
clf2.fit(train_x, y_train) # fit the classifier
pred1 = clf1.predict(val_x) # AdaBoost predictions
pred2 = clf2.predict(val_x) # GB predictions
#combining both models' predictions
def combinePreds(pred, y_pred):
    pred_combined = []
    for i in range(len(pred)):
        if pred[i]==y_pred[i]:
            pred_combined.append(pred[i])
        else:
            pred_combined.append(int((pred[i]+y_pred[i])//2))
    
    return pred_combined

pred_combined1 = combinePreds(pred1, y_pred)
pred_combined2 = combinePreds(pred2, y_pred)
# GradientBoosting Classifier classification report 
print(classification_report(y_val, pred_combined2))