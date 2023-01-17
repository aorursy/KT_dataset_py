import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_style('whitegrid')
%matplotlib inline
hr_data = pd.read_csv("../input/HR_comma_sep.csv")
hr_data.info()
axes_ind = 0
for col in ['satisfaction_level','last_evaluation','average_montly_hours']:
    
    if axes_ind > 7: break
    
    df1 = hr_data[hr_data['left']==0][col]
    df2 = hr_data[hr_data['left']==1][col]
    max_col = max(hr_data[col])
    min_col = min(hr_data[col])

    plt.hist([df1, df2], 
                 bins = 20,
                 edgecolor='black',
                 range=(min_col, max_col), 
                 stacked=True)

    plt.legend(('Stayed','Left'), loc='best')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Count')
    
    axes_ind += 1
    plt.show()

for col in ['number_project','Work_accident','salary','promotion_last_5years','sales']:
    cat_xt = pd.crosstab(hr_data[col], hr_data['left'])
    cat_xt.plot(kind='bar', stacked=True, title= col)
    plt.xlabel(col)
    plt.legend(('Stayed','Left'), loc='best')
    plt.xticks(rotation=60)
    plt.ylabel('count')
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.optimizers import Adam

dataset =hr_data
# transform columns in sequantial
salary = dataset['salary'].unique()
salary_mapping =dict(zip(salary, range(0, len(salary) + 1)))
dataset['salary_int'] = dataset['salary'].map(salary_mapping).astype(int)
depart = dataset['sales'].unique()
depart_mapping =dict(zip(depart, range(0, len(depart) + 1)))
dataset['depart_int'] = dataset['sales'].map(depart_mapping).astype(int)
dataset= dataset.drop(["salary","sales"],axis=1)

# slpit dataset in train and test
train,test=train_test_split(dataset, test_size=0.2)

# split train and test into input (X) and output (Y) variables
X_TRAIN = train.drop(["left"],axis=1)
Y_TRAIN= train["left"]
X_TEST = test.drop(["left"],axis=1)
Y_TEST=  test["left"]

print (salary_mapping)
print (depart_mapping)

print (train.shape)




def create_model(neurons=16, dropout_rate=0.4, weight_constraint=3,activation='relu',learn_rate=0.01, momentum=0.8):
    #create model
    model2 = Sequential()
    model2.add(Dense(neurons, input_dim=9, kernel_initializer='uniform',
                        activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    model2.add(Dense(neurons, kernel_initializer='uniform', activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    model2.add(Dense(neurons, kernel_initializer='uniform', activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    model2.add(Dense(neurons, kernel_initializer='uniform', activation=activation, kernel_constraint=maxnorm(weight_constraint)))
    model2.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model2
best_model = create_model()
history_best = best_model.fit(X_TRAIN, Y_TRAIN, epochs=50, batch_size=20, validation_split=0.2,verbose=1)

                                                    
# list all data in history
print(history_best.history.keys())
# summarize history for accuracy
plt.plot(history_best.history['acc'])
plt.plot(history_best.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_best.history['loss'])
plt.plot(history_best.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
def create_model_easy():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=9, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0,7))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0,7))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0,7))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(decay=0.001)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Fit the model
my_model = create_model_easy()
history = my_model.fit(X_TRAIN, Y_TRAIN, epochs=500, batch_size=30, validation_split=0.2,verbose=1)

print("Model finished")

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
scores = my_model.evaluate(X_TEST, Y_TEST)
print("\n%s: %.2f%%" % (my_model.metrics_names[0], scores[0]*100))
print("\n%s: %.2f%%" % (my_model.metrics_names[1], scores[1]*100))
