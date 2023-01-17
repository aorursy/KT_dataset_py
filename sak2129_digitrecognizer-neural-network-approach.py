# import base modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import output_notebook, figure, show, ColumnDataSource
from bokeh.layouts import row
from bokeh.transform import factor_cmap, factor_mark
output_notebook()

# Other modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# import models
import tensorflow as tf
from keras import models, layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
# Import data
(train_x, train_y), (test_x, test_y)= tf.keras.datasets.mnist.load_data();
train = pd.DataFrame(train_x.reshape(60000,784));
train.columns = ['pixel'+str(x) for x in range(784)]
train['label']=train_y
train.head(3)
test = pd.DataFrame(test_x.reshape(10000,784))
test.columns = ['pixel'+str(x) for x in range(784)]
test['label']=test_y
test.head(3)
# Draw out a few numbers
class_id = train['label'].sort_values().unique()
i=0
plt.figure(figsize=(8,8))
for c in class_id:
    dig_mtx = train[train['label']==c]
    rdm_index = np.random.randint(0,len(dig_mtx),10)
    for r in rdm_index:
        pic = dig_mtx.iloc[r,1:]
        pic_np = np.array(pic).reshape(28,28)
        plt.subplot(10,10,i+1)
        plt.imshow(pic_np,cmap='binary')
        plt.xticks([])
        plt.yticks([])
        #plt.title(c)
        i+=1
        plt.tight_layout()
# Create X and y variables
X = train.drop(columns=['label'])
y = train['label']

# Split train variable in train and test sets
tr_x, valid_x, tr_y, valid_y = train_test_split(X,y,test_size=0.2)

# Normalize data
tr_x = np.array(tr_x)/255
valid_x = np.array(valid_x)/255

# Convert y-variables to categorical
tr_y_encoded = np.array(to_categorical(tr_y, num_classes=10))
valid_y_encoded = np.array(to_categorical(valid_y, num_classes=10))

# Test Data
test_x = test_x.reshape(10000,784)/255
# Create a plot of training/validation accuracy and loss from the resultant neural network
def create_plot(train_accuracy, val_accuracy, train_loss, val_loss):
    
    x_plot = np.arange(0,len(train_accuracy),1)
    
    # Accuracy Plot
    p = figure(plot_width=500, plot_height=350, title='Training & Validation Accuracy',x_axis_label='Epoch', y_axis_label='Accuracy')
    p.line(x=x_plot,y=train_accuracy, legend_label='Training Accuracy', line_width=3)
    p.line(x=x_plot,y=val_accuracy, legend_label='Validation Accuracy', color='green', line_width=3)
    p.legend.location = 'bottom_right'

    # Loss Plot
    q = figure(plot_width=500, plot_height=350, title='Training & Validation Loss',x_axis_label='Epoch', y_axis_label='Loss')
    q.line(x=x_plot,y=train_loss, legend_label='Training Loss', line_width=3)
    q.line(x=x_plot,y=val_loss, legend_label='Validation Loss', color='green', line_width=3)
    q.legend.location = 'top_right'
    
    show(row(p,q))
# Indicate input and output sizes
input_size = 784
output_size = 10
# Create an NN model
m1 = models.Sequential()
m1.add(layers.Dense(1, activation='relu', input_shape=(1,input_size)))
m1.add(layers.Dense(output_size, activation='softmax'))
m1.summary()
# Compile NN model
m1.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
m1.summary()
# Fit the model
es = EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=20)
m1history = m1.fit(tr_x, tr_y_encoded, validation_data=(valid_x, valid_y_encoded), epochs=100, callbacks=es, batch_size=128, verbose=0)
# Saving result parameters and creating plots
train_loss = m1history.history['loss']
val_loss = m1history.history['val_loss']
train_accuracy = m1history.history['accuracy']
val_accuracy = m1history.history['val_accuracy']
# Create plots
create_plot(train_accuracy, val_accuracy, train_loss, val_loss)
# Evaluate on un-seen data
y_pred = m1.predict_classes(test_x)
#results = pd.DataFrame({'y_pred':y_pred, 'y_actual':test_y})
cm = confusion_matrix(y_pred, test_y)
cm_df = pd.DataFrame(cm, columns=[x for x in range(10)])
cm_df.index=[x for x in range(10)]

# Print confusion matrix
cm_df
cm
cr = classification_report(y_pred, test_y, target_names=[x for x in range(10)], output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df = np.round(cr_df,2)
cr_df
test_accuracy = cr['accuracy']
# Visualize the confusion matrix
sns_df = cm_df

## To visualize the discrepancies, we will remove all diagonal values
#for i in range(10):
#    sns_df.iloc[i,i]= 0

plt.figure(figsize=(10,6))
sns.heatmap(sns_df, cmap='Blues', annot=True, fmt='g');
# Visualize items not predicted properly
test['predicted']=y_pred

# Draw out a few numbers
class_id = test['label'].sort_values().unique()
i=0
plt.figure(figsize=(20,10))
for c in class_id:
    dig_mtx = test[(test['label']==c) & (test['predicted']!=c)]
    
    for r in range(10):
        rdm_index = np.random.choice(dig_mtx.index)
        pic = dig_mtx.loc[rdm_index]
        pic_np = np.array(pic.iloc[0:784]).reshape(28,28)
        plt.subplot(10,10,i+1)
        plt.imshow(pic_np,cmap='binary')
        plt.axis('off')
        actual_label = test.loc[rdm_index,'label']
        predicted_label = test.loc[rdm_index,'predicted']
        title = 'ACT:'+str(actual_label)+', PRED:'+str(predicted_label)
        plt.title(title)
        i+=1
plt.tight_layout()
# Extract the outputs
layer_outputs = [layer.output for layer in m1.layers]

# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=m1.input, outputs=layer_outputs)
activation_model.summary()
# Get the outputs of all the hidden nodes for each of the 60000 training images
training_x_data = np.concatenate((tr_x, valid_x), axis=0)
activations = activation_model.predict(training_x_data)
hidden_layer1_activation = activations[0]
output_layer_activation = activations[1]
# Get the dataframe of all the node values
pred_classes = m1.predict_classes(training_x_data)

activation_data = {'pred_class':pred_classes}
for k in range(0,1): 
    activation_data[f"act_val_{k}"] = hidden_layer1_activation[:,k]

activation_df = pd.DataFrame(activation_data)
activation_df.head()
# Comparing activation values with class predictions
plt.figure(figsize=(20,8))
bplot = sns.boxplot(y='act_val_0', x='pred_class', 
                 data=activation_df, 
                 width=0.5,
                 palette="bright")
plt.title('Activation Values from one hidden layer', fontsize=20)
plt.xlabel('Predicted Class', fontsize=16)
plt.ylabel('Activation Value', fontsize=16);
#K.clear_session()
#del m2
# Indicate input and output sizes
input_size = 784
output_size = 10

# Create an NN model
m2 = models.Sequential()
m2.add(layers.Dense(2, activation='relu', input_shape=(1,input_size)))
m2.add(layers.Dense(output_size, activation='softmax'))

# Compile NN model
m2.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
m2.summary()
# Fit the model
es = EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=20)
m2history = m2.fit(tr_x, tr_y_encoded, validation_data=(valid_x, valid_y_encoded), epochs=100, batch_size=128, verbose=0, callbacks=es)
# Saving result parameters and creating plots
train_loss = m2history.history['loss']
val_loss = m2history.history['val_loss']
train_accuracy = m2history.history['accuracy']
val_accuracy = m2history.history['val_accuracy']
# Create plots
create_plot(train_accuracy, val_accuracy, train_loss, val_loss)
# Evaluate on un-seen data
y_pred = m2.predict_classes(test_x)
#results = pd.DataFrame({'y_pred':y_pred, 'y_actual':test_y})
cm2 = confusion_matrix(y_pred, test_y)
cm_df2 = pd.DataFrame(cm2, columns=[x for x in range(10)])
cm_df2.index=[x for x in range(10)]
# Print confusion matrix
cm_df2
cr2 = classification_report(y_pred, test_y, target_names=[x for x in range(10)], output_dict=True)
cr_df2 = pd.DataFrame(cr2).transpose()
cr_df2 = np.round(cr_df2,2)
cr_df2
test_accuracy2 = cr['accuracy']
# Visualize the confusion matrix
sns_df2 = cm_df2

plt.figure(figsize=(10,6))
sns.heatmap(sns_df2, cmap='Blues', annot=True, fmt='g');
# Extract the outputs
layer_outputs = [layer.output for layer in m2.layers]

# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=m2.input, outputs=layer_outputs)

# Get the outputs of all the hidden nodes for each of the 60000 training images
training_x_data = np.concatenate((tr_x, valid_x), axis=0)
activations = activation_model.predict(training_x_data)
hidden_layer1_activation = activations[0]
output_layer_activation = activations[1]

# Get the dataframe of all the node values
pred_classes = m2.predict_classes(training_x_data)

activation_data = {'pred_class':pred_classes}
for k in range(0,2): 
    activation_data[f"act_val_{k}"] = hidden_layer1_activation[:,k]

activation_df = pd.DataFrame(activation_data)
plt.figure(figsize=(17,7))
plt.title('Activation values from the 2 nodes in the hidden layer', fontsize=20)
plt.xlabel('Activation Values: Node 1', fontsize=16)
plt.ylabel('Activation Values: Node 2', fontsize=16)
plt.legend(fontsize=16)
sns.scatterplot(data=activation_df, x='act_val_0', y='act_val_1', hue='pred_class', palette="deep");
n = [1]
mod = [m1]
train_acc = [train_accuracy]
val_acc = [val_accuracy]
tr_loss = [train_loss]
v_loss = [val_loss]
test_acc = [test_accuracy]
# Test out models with different nodes in the hidden layer
for nodes in [2,4,8,16, 32, 64, 128, 256, 512, 1024]:
    m = models.Sequential()
    m.add(layers.Dense(nodes, activation='relu', input_shape=(1,input_size)))
    m.add(layers.Dense(output_size, activation='softmax'))
    
    # Compile NN model
    m.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    # Fit the model
    es = EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=5)
    mhistory = m.fit(tr_x, tr_y_encoded, validation_data=(valid_x, valid_y_encoded), epochs=100, callbacks=es, batch_size=128, verbose=0)

    # Saving result parameters and creating plots
    train_loss = mhistory.history['loss']
    val_loss = mhistory.history['val_loss']
    train_accuracy = mhistory.history['accuracy']
    val_accuracy = mhistory.history['val_accuracy']

    # Evaluate on un-seen data
    y_pred = m.predict_classes(test_x)
    diff = test_y - y_pred
    diff = pd.DataFrame(diff)
    test_accuracy = (diff[diff==0].count()/len(diff))[0]
    
    # Save results
    n.append(nodes)
    mod.append(m)
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    tr_loss.append(train_loss)
    v_loss.append(val_loss)
    test_acc.append(test_accuracy)
    del m
# Plot test accuracy for models with different nodes
p = figure(plot_width=400, plot_height=400, title='Test Accuracy for models with various Nodes options',x_axis_label='Number of Nodes', y_axis_label='Accuracy')
p.line(x=n,y=test_acc, legend_label='Test Accuracy', line_width=3)
p.legend.location = 'bottom_right'

q = figure(plot_width=400, plot_height=400, title='Test Accuracy for models with various Nodes options',x_axis_label='Number of Nodes', y_axis_label='Accuracy')
q.line(x=n[0:6],y=test_acc[0:6], legend_label='Test Accuracy', line_width=3)
q.legend.location = 'bottom_right'

show(row(p,q))
# Indicate input and output sizes
input_size = 784
output_size = 10

# Create an NN model
m16 = models.Sequential()
m16.add(layers.Dense(16, activation='relu', input_shape=(1,input_size)))
m16.add(layers.Dense(output_size, activation='softmax'))

# Compile NN model
m16.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
m16.summary()
# Fit the model
es = EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=20)
m16history = m16.fit(tr_x, tr_y_encoded, validation_data=(valid_x, valid_y_encoded), epochs=100, batch_size=128, verbose=0, callbacks=es)
# Saving result parameters and creating plots
train_loss = m16history.history['loss']
val_loss = m16history.history['val_loss']
train_accuracy = m16history.history['accuracy']
val_accuracy = m16history.history['val_accuracy']
# Create plots
#create_plot(train_accuracy, val_accuracy, train_loss, val_loss)
# Evaluate on un-seen data
y_pred = m16.predict_classes(test_x)
#results = pd.DataFrame({'y_pred':y_pred, 'y_actual':test_y})
cm16 = confusion_matrix(y_pred, test_y)
cm_df16 = pd.DataFrame(cm16, columns=[x for x in range(10)])
cm_df16.index=[x for x in range(10)]
cr16 = classification_report(y_pred, test_y, target_names=[x for x in range(10)], output_dict=True)
cr_df16 = pd.DataFrame(cr16).transpose()
cr_df16 = np.round(cr_df16,2)
cr_df16
# Visualize the confusion matrix
sns_df16 = cm_df16

plt.figure(figsize=(10,6))
sns.heatmap(sns_df16, cmap='Blues', annot=True, fmt='g');
# Create PCA variables
pca = PCA(n_components=154)

pca_train_x = pca.fit_transform(tr_x)
pca_valid_x = pca.transform(valid_x)
pca_test_x = pca.transform(test_x)
# Plot the explained variance ratio
p = figure(plot_width=500, plot_height=300, title='Cumulative Explained Variance',x_axis_label='Number of Nodes', y_axis_label='Explained Variance')
ex_var = pca.explained_variance_ratio_.cumsum()
p.line(x=np.arange(0,len(ex_var),1), y=ex_var, line_width=3)
show(p)
# Run the best model from earlier with and without PCA and compare

m = models.Sequential()
m.add(layers.Dense(16, activation='relu', input_shape=(1,154)))
m.add(layers.Dense(output_size, activation='softmax'))

# Compile NN model
m.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
es = EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=5)
mhistory = m.fit(pca_train_x, tr_y_encoded, validation_data=(pca_valid_x, valid_y_encoded), epochs=100, callbacks=es, batch_size=128, verbose=0)

# Saving result parameters and creating plots
train_loss = mhistory.history['loss']
val_loss = mhistory.history['val_loss']
train_accuracy = mhistory.history['accuracy']
val_accuracy = mhistory.history['val_accuracy']

# Evaluate on un-seen data
y_pred = m.predict_classes(pca_test_x)
diff = test_y - y_pred
diff = pd.DataFrame(diff)
test_accuracy = (diff[diff==0].count()/len(diff))[0]
print('Test Accuracy is:', np.round(test_accuracy*100,2),'%')
create_plot(train_accuracy, val_accuracy, train_loss, val_loss)
# Compare the PCA accuracy with the full model accuracy
print('For nodes:',n[4], 'the test accuracy was:', np.round(test_acc[4]*100,2),'%')
create_plot(train_acc[4], val_acc[4], tr_loss[4], v_loss[4])
# Fit a random forest classifier on the data and identify most important features
rf = RandomForestClassifier(n_estimators=100);
rf.fit(X,y);
# Identifying the important features
feature_imp = pd.DataFrame({'pixel':X.columns, 'importance':rf.feature_importances_})
sorted_features = feature_imp.sort_values(by='importance', ascending=False)
sorted_features_cumsum = sorted_features['importance'].cumsum()
# Plot the feature importances
p = figure(plot_width=900, plot_height=400, title='Feature Importances', y_axis_label='Importance', x_axis_label='Feature')
p.line(x=np.arange(0,784,1),y=sorted_features_cumsum, line_width=3);
show(p)
# Creating a prediction using random forest classifier to see how it performs against the best model
rf_pred = rf.predict(test_x)
diff = test_y - rf_pred
diff = pd.DataFrame(diff)
test_accuracy = (diff[diff==0].count()/len(diff))[0]
print('Test Accuracy is:', np.round(test_accuracy*100,2),'%')
# Create new input variables using the important features
imp_columns = np.array(sorted_features.reset_index(drop=True).loc[0:399,'pixel'])
rf_train = np.array(X[imp_columns])

# Split train variable in train and test sets
rf_train_x, rf_valid_x, tr_y, valid_y = train_test_split(rf_train,y,test_size=0.2)

# Normalize data
rf_train_x = np.array(rf_train_x)/255
rf_valid_x = np.array(rf_valid_x)/255

# Convert y-variables to categorical
tr_y_encoded = np.array(to_categorical(tr_y, num_classes=10))
valid_y_encoded = np.array(to_categorical(valid_y, num_classes=10))

# Test Data
rf_test = test.drop(columns=['predicted','label'])
rf_test_x = rf_test[imp_columns]
rf_test_x = np.array(rf_test_x).reshape(10000,400)/255
np_features = rf.feature_importances_.reshape(28,28)
plt.figure(figsize=(7,7))
plt.imshow(np_features, cmap='inferno_r');
cbar = plt.colorbar(ticks=[rf.feature_importances_.min(), rf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important']);
#plt.xticks([]), plt.yticks([])
# Run the best model with random forest features

m_rf = models.Sequential()
m_rf.add(layers.Dense(16, activation='relu', input_shape=(1,400)))
m_rf.add(layers.Dense(output_size, activation='softmax'))

# Compile NN model
m_rf.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# Fit the model
es = EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=20)
m_rfhistory = m_rf.fit(rf_train_x, tr_y_encoded, validation_data=(rf_valid_x, valid_y_encoded), epochs=100, callbacks=es, batch_size=128, verbose=0)

# Saving result parameters and creating plots
train_loss = m_rfhistory.history['loss']
val_loss = m_rfhistory.history['val_loss']
train_accuracy = m_rfhistory.history['accuracy']
val_accuracy = m_rfhistory.history['val_accuracy']

# Evaluate on un-seen data
y_pred = m_rf.predict_classes(rf_test_x)
diff = test_y - y_pred
diff = pd.DataFrame(diff)
test_accuracy = (diff[diff==0].count()/len(diff))[0]
print('Test Accuracy is:', np.round(test_accuracy*100,2),'%')
create_plot(train_accuracy, val_accuracy, train_loss, val_loss)
# Initialize input and output sizes
input_size = 784
output_size = 10
# Result variables
n = []
mod = []
train_acc = []
val_acc = []
tr_loss = []
v_loss = []
test_acc = []
# Test out models with different nodes in the hidden layer
for nodes in [1, 2,4,8,16, 32, 64, 128, 256, 512, 1024]:
    m = models.Sequential()
    m.add(layers.Dense(16, activation='relu', input_shape=(1,input_size)))
    m.add(layers.Dense(nodes, activation='relu'))
    m.add(layers.Dense(output_size, activation='softmax'))
    
    # Compile NN model
    m.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    # Fit the model
    es = EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=5)
    mhistory = m.fit(tr_x, tr_y_encoded, validation_data=(valid_x, valid_y_encoded), epochs=100, callbacks=es, batch_size=128, verbose=0)

    # Saving result parameters and creating plots
    train_loss = mhistory.history['loss']
    val_loss = mhistory.history['val_loss']
    train_accuracy = mhistory.history['accuracy']
    val_accuracy = mhistory.history['val_accuracy']

    # Evaluate on un-seen data
    y_pred = m.predict_classes(test_x)
    diff = test_y - y_pred
    diff = pd.DataFrame(diff)
    test_accuracy = (diff[diff==0].count()/len(diff))[0]
    
    # Save results
    n.append(nodes)
    mod.append(m)
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    tr_loss.append(train_loss)
    v_loss.append(val_loss)
    test_acc.append(test_accuracy)
    del m
# Plot test accuracy for models with different nodes
p = figure(plot_width=400, plot_height=400, title='Test Accuracy for models with various Nodes options',x_axis_label='Number of Nodes', y_axis_label='Accuracy')
p.line(x=n,y=test_acc, legend_label='Test Accuracy', line_width=3)
p.legend.location = 'bottom_right'

q = figure(plot_width=400, plot_height=400, title='Test Accuracy for models with various Nodes options',x_axis_label='Number of Nodes', y_axis_label='Accuracy')
q.line(x=n[0:6],y=test_acc[0:6], legend_label='Test Accuracy', line_width=3)
q.legend.location = 'bottom_right'

show(row(p,q))
# Initialize input and output sizes
input_size = 784
output_size = 10
# Result variables
n = []
mod = []
train_acc = []
val_acc = []
tr_loss = []
v_loss = []
test_acc = []
# Test out models with different nodes in the hidden layer
for activation_function in ['relu', 'sigmoid', 'tanh']: 
    for nodes in [1,2,4,8,16,32]:
        m = models.Sequential()
        m.add(layers.Dense(nodes, activation=activation_function, input_shape=(1,input_size)))
        m.add(layers.Dense(output_size, activation='softmax'))

        # Compile NN model
        m.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

        # Fit the model
        es = EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=5)
        mhistory = m.fit(tr_x, tr_y_encoded, validation_data=(valid_x, valid_y_encoded), epochs=100, callbacks=es, batch_size=128, verbose=0)

        # Saving result parameters and creating plots
        train_loss = mhistory.history['loss']
        val_loss = mhistory.history['val_loss']
        train_accuracy = mhistory.history['accuracy']
        val_accuracy = mhistory.history['val_accuracy']

        # Evaluate on un-seen data
        y_pred = m.predict_classes(test_x)
        diff = test_y - y_pred
        diff = pd.DataFrame(diff)
        test_accuracy = (diff[diff==0].count()/len(diff))[0]

        # Save results
        n.append(nodes)
        mod.append(m)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
        tr_loss.append(train_loss)
        v_loss.append(val_loss)
        test_acc.append(test_accuracy)
        del m
# Plot test accuracy for models with different nodes
p = figure(plot_width=500, plot_height=400, title='Test Accuracy for models with different activation functions',x_axis_label='Number of Nodes', y_axis_label='Accuracy')
p.line(x=n[0:6],y=test_acc[0:6], legend_label='RELU', line_width=3)
p.line(x=n[6:12],y=test_acc[6:12], legend_label='SIGMOID', line_width=3, color='green')
p.line(x=n[12:18],y=test_acc[12:18], legend_label='TANH', line_width=3, color='orange')
p.legend.location = 'bottom_right'

show(p)
# Initialize input and output sizes
input_size = 784
output_size = 10
# Result variables
n = []
mod = []
train_acc = []
val_acc = []
tr_loss = []
v_loss = []
test_acc = []
# Test out models with different nodes in the hidden layer
for opt in ['rmsprop', 'sgd', 'adam', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']: 
    for nodes in [1,2,4,8,16,32]:
        m = models.Sequential()
        m.add(layers.Dense(nodes, activation='relu', input_shape=(1,input_size)))
        m.add(layers.Dense(output_size, activation='softmax'))

        # Compile NN model
        m.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

        # Fit the model
        es = EarlyStopping(monitor='val_accuracy', mode='average', verbose=0, patience=5)
        mhistory = m.fit(tr_x, tr_y_encoded, validation_data=(valid_x, valid_y_encoded), epochs=100, callbacks=es, batch_size=128, verbose=0)

        # Saving result parameters and creating plots
        train_loss = mhistory.history['loss']
        val_loss = mhistory.history['val_loss']
        train_accuracy = mhistory.history['accuracy']
        val_accuracy = mhistory.history['val_accuracy']

        # Evaluate on un-seen data
        y_pred = m.predict_classes(test_x)
        diff = test_y - y_pred
        diff = pd.DataFrame(diff)
        test_accuracy = (diff[diff==0].count()/len(diff))[0]

        # Save results
        n.append(nodes)
        mod.append(m)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
        tr_loss.append(train_loss)
        v_loss.append(val_loss)
        test_acc.append(test_accuracy)
        del m
# Plot test accuracy for models with different nodes
p = figure(plot_width=700, plot_height=400, title='Test Accuracy for models with different optimizers',x_axis_label='Number of Nodes', y_axis_label='Accuracy')
colors=['cadetblue', 'crimson', 'darkgreen', 'darkorange', 'dodgerblue', 'hotpink', 'lightseagreen', 'teal']
i=0
for opt in ['rmsprop', 'sgd', 'adam', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']:
    p.line(x=n[i*6:i*6+6],y=test_acc[i*6:i*6+6], legend_label=opt, line_width=3, color=colors[i])
    i+=1
p.legend.location = 'bottom_right'

show(p)