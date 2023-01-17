#import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import output_notebook, figure, show
from bokeh.layouts import row
output_notebook()

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


#import models
import tensorflow as tf
from keras import models, layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
train = pd.read_csv('../input/downloaded-minst-data/train.csv',index_col=0)
train.head(5)
test = pd.read_csv('../input/downloaded-minst-data/test.csv',index_col=0)
test.head(5)
# Plot digits from the data
class_id = train['label'].sort_values().unique()
i=0

plt.figure(figsize=(8,9))
for c in class_id:
    dig_mtx = train[train['label']==c]
    rdm_index = np.random.randint(0,len(dig_mtx),10)
    for r in rdm_index:
        pic=dig_mtx.iloc[r,1:]
        pic_np=np.array(pic).reshape(28,28)
        plt.subplot(10,10,i+1)
        plt.imshow(pic_np,cmap='binary_r')
        plt.axis('off')
        plt.title(c)
        i+=1
plt.tight_layout()
# Defining X & Y variables 
x = train.drop(columns=['label'])
y = train['label']

# splitting train and test data sets
train_x, valid_x, train_y, valid_y = train_test_split(x,y,test_size=0.1, random_state=0)

#Normalize data
tr_x=np.array(train_x)/255
va_x=np.array(valid_x)/255

# Convert y-variables to categorical
train_y_cat = np.array(to_categorical(train_y, num_classes=10))
valid_y_cat = np.array(to_categorical(valid_y, num_classes=10))


# Create text x and y variables
test_x= np.array(test.drop(columns=['label']))/255
test_y = test['label']

def create_plot(train_accuracy,val_accuracy, train_loss, val_loss):
    x_plt = np.arange(0,len(train_accuracy),1)

    #Accuracy Plot
    p=figure(plot_width=700, plot_height=350, title ='Training & Validation Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
    p.line(x=x_plt, y=train_accuracy, legend_label='Training Accuracy', line_width=3, color ='black')
    p.line(x=x_plt, y=val_accuracy, legend_label='Validation Accuracy', color ='red', line_width=3)
    p.legend.location='bottom_right'
    
    #loss Plot
    q=figure(plot_width=600, plot_height=350, title ='Training & Validation Loss', x_axis_label='Epoch', y_axis_label='Accuracy')
    q.line(x=x_plt, y=train_loss, legend_label='Training Loss', line_width=3, color ='black')
    q.line(x=x_plt, y=val_loss, legend_label='Validation Loss', color ='red', line_width=3)
    q.legend.location='bottom_right'
    
    show(row(p,q))
# defining input and output sizes
input_size = 784
output_size = 10
# create NN model
m1 = models.Sequential()
m1.add(layers.Dense(1, activation ='relu', input_shape =(1, input_size)))
m1.add(layers.Dense(output_size, activation='softmax'))
m1.summary()
# compile NN model
m1.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Early Stopping 
es=EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=25)

# Fit the model 
m1history = m1.fit(tr_x, train_y_cat, validation_data=(va_x,valid_y_cat),epochs=100,callbacks=es, batch_size=128, verbose=0)
#Summarizing results
train_loss = m1history.history['loss']
val_loss =m1history.history['val_loss']
train_accuracy = m1history.history['accuracy']
val_accuracy = m1history.history['val_accuracy']

#Evaluate on validation data set
y_pred=m1.predict_classes(test_x)

#Evaluate on validation data set
y_pred=m1.predict_classes(test_x)
diff =test_y - y_pred
diff=pd.DataFrame(diff)
test_accuracy = (diff[diff==0].count()/len(diff))[0]
#Create Plot
print('Test Accuracy is: ', np.round(test_accuracy*100,2),'%')
create_plot(train_accuracy,val_accuracy, train_loss, val_loss)

#Confusion Matrix
cm = confusion_matrix(y_pred, test_y)
cm_df=pd.DataFrame(cm, columns=[x for x in range(10)])
cm_df.index=[x for x in range(10)]

#Print Confusion Matrix
cm_df

plt.matshow(cm, cmap=plt.cm.gray)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")
plt.show()
#Classification report
cr=classification_report(y_pred,test_y, target_names=[x for x in range(10)], output_dict=True)
cr_df =pd.DataFrame(cr).transpose()
cr_df=np.round(cr_df,2)
cr_df
test_accuracy=cr['accuracy']
#visualization of items not predicted properly

test['predicted']=y_pred

# Plot digits from the data
class_id = test['label'].sort_values().unique()
i=0

plt.figure(figsize=(20,10))

for c in class_id:
    dig_mtx = test[(test['label']==c) & (test['predicted']!=c)]
    
    for r in range(10):
        rdm_index = np.random.choice(dig_mtx.index)
        pic=dig_mtx.loc[rdm_index]
        pic_np=np.array(pic.iloc[0:784]).reshape(28,28)
        plt.subplot(10,10,i+1)
        plt.imshow(pic_np,cmap='binary_r')
        plt.axis('off')
        title='ACT:'+str(test.loc[rdm_index,'label'])+',PRED:'+str(test.loc[rdm_index,'predicted'])
        plt.title(title)
        i+=1
plt.tight_layout()

# Extracts the outputs of the 2 layers:
layer_outputs = [layer.output for layer in m1.layers]

# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=m1.input, outputs=layer_outputs)
# concatenating train and validation sets

x_train_norm = np.concatenate((tr_x, va_x), axis=0)

# Get the outputs of all the hidden nodes for each of the 60000 training images

activations = activation_model.predict(x_train_norm)
hidden_layer1_activation = activations[0]
output_layer_activations = activations[1]
#Get the dataframe of all the node values

pred_classes=m1.predict_classes(x_train_norm)


activation_data = {'pred_class':pred_classes}
for k in range(0,1): 
    activation_data[f"act_val_{k}"] = hidden_layer1_activation[:,k]

activation_df = pd.DataFrame(activation_data)
activation_df.head()
# To see how closely the hidden node activation values correlate with the class predictions
plt.figure(figsize=(9,8))
bplot = sns.boxplot(y='act_val_0', x='pred_class', 
                 data=activation_df[['act_val_0','pred_class']], 
                 width=0.75,
                 palette="colorblind")
plt.title('1 Hidden layer Plot', fontsize=15);
# defining input and output sizes
input_size = 784
output_size = 10
# create NN model - m2
m2 = models.Sequential()
m2.add(layers.Dense(2, activation ='relu', input_shape =(1, input_size)))
m2.add(layers.Dense(output_size, activation='softmax'))
m2.summary()
# compile NN model
m2.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Early Stopping 
es=EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=25)

# Fit the model 
m2history = m2.fit(tr_x, train_y_cat, validation_data=(va_x,valid_y_cat),epochs=100,callbacks=es, batch_size=128, verbose=0)
#Summarizing results
train_loss = m2history.history['loss']
val_loss =m2history.history['val_loss']
train_accuracy = m2history.history['accuracy']
val_accuracy = m2history.history['val_accuracy']


#Evaluate on validation data set
y_pred=m2.predict_classes(test_x)
diff =test_y - y_pred
diff=pd.DataFrame(diff)
test_accuracy = (diff[diff==0].count()/len(diff))[0]
#Create Plot
print('Test Accuracy is: ', np.round(test_accuracy*100,2),'%')
create_plot(train_accuracy,val_accuracy, train_loss, val_loss)

#Confusion Matrix
cm = confusion_matrix(y_pred, test_y)
cm_df=pd.DataFrame(cm, columns=[x for x in range(10)])
cm_df.index=[x for x in range(10)]

#Print Confusion Matrix
cm_df
plt.matshow(cm, cmap=plt.cm.gray)
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")
plt.show()
#Classification report
cr=classification_report(y_pred,test_y, target_names=[x for x in range(10)], output_dict=True)
cr_df =pd.DataFrame(cr).transpose()
cr_df=np.round(cr_df,2)
cr_df
test_accuracy=cr['accuracy']
# Extracts the outputs of the 2 layers:
layer_outputs = [layer.output for layer in m2.layers]

# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=m2.input, outputs=layer_outputs)
# Get the outputs of all the hidden nodes for each of the 60000 training images

x_train_norm = np.concatenate((tr_x, va_x), axis=0)
activations = activation_model.predict(x_train_norm)
hidden_node1_activation=activations[0][:,0]
hidden_node2_activation=activations[0][:,1]
pred_class=m2.predict_classes(x_train_norm)

results=pd.DataFrame({'pred_class':pred_class,'hidden_node1_activation':hidden_node1_activation,'hidden_node2_activation': hidden_node2_activation})
plt.figure(figsize=(15,10))
sns.scatterplot(data=results, x='hidden_node1_activation', y='hidden_node2_activation', hue='pred_class', palette = 'deep')
plt.title('2 Hidden Nodes Scatter Plot',fontsize=20);
n=[1]
mod = [m1]
train_acc =[train_accuracy]
val_acc =[val_accuracy]
tr_loss = [train_loss]
v_loss = [val_loss]
test_acc = [test_accuracy]

# create NN models

for nodes in [2,4,8,16,32, 64, 128, 256, 512, 1024]:
    m = models.Sequential()
    m.add(layers.Dense(nodes, activation ='relu', input_shape =(1, input_size)))
    m.add(layers.Dense(output_size, activation='softmax'))

    # compile NN model
    m.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])


    #Early Stopping 
    es=EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=5)

    # Fit the model 
    mhistory = m.fit(tr_x, train_y_cat, validation_data=(va_x,valid_y_cat),epochs=100,callbacks=es, batch_size=128, verbose=0)

    #Summarizing results
    train_loss = mhistory.history['loss']
    val_loss =mhistory.history['val_loss']
    train_accuracy = mhistory.history['accuracy']
    val_accuracy = mhistory.history['val_accuracy']

    
    #Evaluate on validation data set
    y_pred=m.predict_classes(test_x)
    diff =test_y - y_pred
    diff=pd.DataFrame(diff)
    test_accuracy = (diff[diff==0].count()/len(diff))[0]

    #Save results
    n.append(nodes)
    mod.append(m)
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    tr_loss.append(train_loss)
    v_loss.append(val_loss)
    test_acc.append(test_accuracy)
np.array(test_acc)*100
#Results summary for multiple nodes 

results3=pd.DataFrame({'Number of nodes':n,'Test Accuracy in %':np.array(test_acc)*100})
results3

p = figure(plot_width=700, plot_height=400, title='Test Accuracy', x_axis_label='Number of Nodes', y_axis_label='Accuracy')
p.line(x=n, y=test_acc, legend_label='Test Accuracy', line_width=3, color ='black',)
p.legend.location ='bottom_right'
show(p)

#PCA variables
pca = PCA(n_components=154)


pca_train_x=pca.fit_transform(tr_x)
pca_valid_x=pca.transform(va_x)
pca_test_x =pca.transform(test_x)
plt.plot(pca.explained_variance_ratio_.cumsum(),color ='red');
plt.title('Explained Variance based on the number of features',fontsize=15);
#clearing the data for previous models

K.clear_session()
del m
# run the best model from earlier with or without PCA and compare

m = models.Sequential()
m.add(layers.Dense(64, activation ='relu', input_shape =(1, 154)))
m.add(layers.Dense(output_size, activation='softmax'))

m.summary()
# compile NN model
m.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])


#Early Stopping 
es=EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=25)

# Fit the model 
mhistory = m.fit(pca_train_x, train_y_cat, validation_data=(pca_valid_x,valid_y_cat),epochs=100,callbacks=es,batch_size=128, verbose=0)

#Summarizing results
train_loss = mhistory.history['loss']
val_loss =mhistory.history['val_loss']
train_accuracy = mhistory.history['accuracy']
val_accuracy = mhistory.history['val_accuracy']


#Evaluate on validation data set
y_pred=m.predict_classes(pca_test_x)
diff =test_y - y_pred
diff=pd.DataFrame(diff)
test_accuracy = (diff[diff==0].count()/len(diff))[0]
print('Test Accuracy is: ', np.round(test_accuracy*100,2),'%')
create_plot(train_accuracy, val_accuracy, train_loss, val_loss)
# fit a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x,y)
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = 'hot',
               interpolation="nearest")
    plt.axis("off")

plot_digit(rf.feature_importances_)
cbar = plt.colorbar(ticks=[rf.feature_importances_.min(), rf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])
plt.show()
#Analyzing the number of pixels that hold the most importance

features_imp = np.sort(rf.feature_importances_)
plt.figure(figsize=(20,10));
plt.bar(np.arange(0,784,1),features_imp[::-1]);
plt.grid()
plt.title('Pixel contribution towards final classification-Random Forest',fontsize=15);
# https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
n_features = 70
imp_arr = rf.feature_importances_
idx = (-imp_arr).argsort()[:n_features]
# Create training and test images using just the 70 pixel locations obtained above
train_images_sm = tr_x[:,idx]
test_images_sm = test_x[:,idx]
train_images_sm.shape, test_images_sm.shape # the reduced images have dimension 70

# Convert y-variables to categorical
#rm_y = np.array(to_categorical(train_y_cat, num_classes=10))
# Random Forest NN model 

# create NN model
rm = models.Sequential()
rm.add(layers.Dense(16, activation ='relu', input_shape =(1, n_features)))
rm.add(layers.Dense(output_size, activation='softmax'))
rm.summary()
# compile NN model
rm.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Early Stopping 
es=EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=25)

# Fit the model 
rmhistory = rm.fit(train_images_sm, train_y_cat,validation_split=0.1,epochs=100,callbacks=es, batch_size=128, verbose=0)

#Summarizing results
train_loss = rmhistory.history['loss']
val_loss =rmhistory.history['val_loss']
train_accuracy = rmhistory.history['accuracy']
val_accuracy = rmhistory.history['val_accuracy']


#Evaluate on validation data set
y_pred=rm.predict_classes(test_images_sm)
diff =test_y - y_pred
diff=pd.DataFrame(diff)
test_accuracy = (diff[diff==0].count()/len(diff))[0]
#Create Plot
print('Test Accuracy is: ', np.round(test_accuracy*100,2),'%')
create_plot(train_accuracy,val_accuracy, train_loss, val_loss)
# to convert an index n, 0<= n < 784
def pair(n_feature,size):
    x = n_feature//size 
    y = n_feature%size
    return x,y
plt.figure(figsize=(20,3))
for ch in range(5):
    plt.subplot(1,5,ch+1)
    plt.imshow(x_train_norm[ch].reshape(28,28),cmap='binary')
    x, y = np.array([pair(k,28) for k in idx]).T
    plt.scatter(x,y,color='red',s=20);
# defining input and output sizes
input_size = 784
output_size = 10

# create NN model
m6 = models.Sequential()
m6.add(layers.Dense(32, activation ='relu', input_shape =(1, input_size)))
m6.add(layers.Dense(64, activation ='relu'))
m6.add(layers.Dense(128, activation ='relu'))
m6.add(layers.Dense(output_size, activation='softmax'))
m6.summary()
# compile NN model
m6.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])


#Early Stopping 
es=EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=25)

# Fit the model 
m6history = m6.fit(tr_x, train_y_cat, validation_data=(va_x,valid_y_cat),epochs=100,callbacks=es, batch_size=128, verbose=0)

#Summarizing results
train_loss = m6history.history['loss']
val_loss =m6history.history['val_loss']
train_accuracy = m6history.history['accuracy']
val_accuracy = m6history.history['val_accuracy']

#Evaluate on validation data set
y_pred=m6.predict_classes(test_x)
diff =test_y - y_pred
diff=pd.DataFrame(diff)
test_accuracy = (diff[diff==0].count()/len(diff))[0]
#Create Plot
print('Test Accuracy is: ', np.round(test_accuracy*100,2),'%')
create_plot(train_accuracy,val_accuracy, train_loss, val_loss)