# import packages
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# load the dataset
os.chdir(r'C:\Users\YZD\Desktop\NUS\MSBA\Semster 2 - 2020\BT5152 - Foundation in Data Analytics II\Assignments\Assignment 2\default-payment')
df_payment = pd.read_csv('pre_processed_data_v2.csv',index_col=0)
print(df_payment.shape)
print(df_payment.head())

# Remove columns 
df_payment_new = df_payment.drop(columns=['BILL_AMT4_rt', 'BILL_AMT6_rt', 'BILL_AMT1_sq', 'BILL_AMT3_rt','BILL_AMT1_rt','BILL_AMT5_rt',
'BILL_AMT4_sq','BILL_AMT6_sq','BILL_AMT4_rt','BILL_AMT6_rt','BILL_AMT2_rt','BILL_AMT3_rt','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5',
'BILL_AMT6','May_leftover_credit','LIMIT_BAL_sq','Aug_leftover_credit','LIMIT_BAL_rt','LIMIT_BAL','avg_bill','PAY_AMT2_sq','PAY_6','sum_pay','sum_exp'])

print(df_payment_new.shape)
print(df_payment_new.head())


y = df_payment_new['def_payment_nm']
print(y.shape)
print(type(y))
x = df_payment_new.drop(['def_payment_nm'], axis=1)

# Fix Nan and Infinity values in X features, fill in NaN with mean
x[x==np.inf]=np.nan
x.fillna(x.mean(), inplace=True)

print(x.shape)
print(type(x))

y_val = y.values
x_val = x.values
x_val = x_val.astype('float32')

# Normalize some features to [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
x_nom = scaler.fit_transform(x_val)
print(x_nom.shape)

# Split data into train test
X_train, X_test, y_train, y_test = train_test_split(x_nom, y_val, test_size=0.2, random_state=42)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=71, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=500, batch_size=5)

# predict probabilities for test set
predict_y_probs = model.predict(X_test, verbose=1)
# predict classes for test set
predict_y_classes = model.predict_classes(X_test, verbose=1)

# reduce the result to 1d array
predict_y_probs = predict_y_probs[:, 0]
predict_y_classes = predict_y_classes[:, 0]

# evaluation the model and results
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
matrix = confusion_matrix(y_test, predict_y_classes)
print(matrix)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, predict_y_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, predict_y_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, predict_y_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, predict_y_classes)
print('F1 score: %f' % f1)



# plot normalized confusion matrix
def plot_confusion_matrix(y_test, predict_test, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_test, predict_test)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="red" if cm[i, j] > thresh else "green")
    fig.tight_layout()
    return ax

plot_confusion_matrix(y_test, predict_y_classes, classes=["0","1"], normalize=True)