# Importing necessary libraries and .csv dataset



from keras.models import Sequential

from keras.layers import Dense, Activation

import pandas as pd

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import numpy as np

import time

import seaborn as sns

import matplotlib.pyplot as plt

from copy import deepcopy



data = pd.read_csv('/kaggle/input/ranked_data_formated.csv')
def data_summary(d):

    print('Dataset entries: ',d.shape[0])

    print('Dataset variables: ',d.shape[1])

    print('-'*10)

    print('Data-type of each column: \n')

    print(d.dtypes)

    print('-'*10)

    print('missing rows in each column: \n')

    c=d.isnull().sum()

    print(c[c>0])

data_summary(data)
# Data as a table summary:

data.head()
sns.countplot(data['blueWins'],label="Sum")



plt.show()
# Drop the "blueBaronDifference" column - it is not possible to get baron in League Of Legends before minute 20

# Also, drop the "gameId" column, since it has no influence on the game outcome



data = data.drop(["blueBarons","redBarons","gameId"], axis=1)
corr = data.corr()



plt.figure(figsize=(20, 20))

sns.heatmap(corr, annot=True, cmap="coolwarm")
# From the Correlation Matrix, we will look into variables that are highly correlated with each other >0.90 or < -0.90

# With all variables included, we can see that both red and blue AverageLevel and TotalExperience are highly correlated

# This makes sense, since level depends on the total experience

# Because of that, one of them will be dropped. It will be the "average level" column, since it has lower correlation than "total experience" with "blue wins" final outcome



data = data.drop(["redAvgLevel","blueAvgLevel"], axis=1)
# Save the dataset with all factors for later

data_original = data.copy()
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



# Create a copy of data

pca_data = data.copy()



# Separate target (blueWins) from rest of data

pca_data_y = pca_data['blueWins']

pca_data_x = pca_data.drop(['blueWins'], axis=1)



# Standardize the features for PCA

pca_data_x = StandardScaler().fit_transform(pca_data_x)



n_components = 21

pca = PCA(n_components)

principal_components = pca.fit_transform(pca_data_x)



pca_columns = []

for i in range(1,n_components+1):

    pca_columns.append('principal component '+str(i))



principalDf = pd.DataFrame(data = principal_components

             ,columns=pca_columns)



pca_data = pd.concat([principalDf, pca_data_y], axis = 1)





# Find how many Principal Components are necessary to explain 95% of the variance

sum = 0

i=0

while(sum<0.99):

    sum += pca.explained_variance_ratio_[i]

    i+=1

print(i-1)
# First 14 Principal components explain 99% of the variance

# We can drop the remaining to reduce the dimensions of the set



pca_99 = pd.concat([pca_data_y, principalDf], axis = 1)

pca_99 = pca_99.drop(pca_columns[14:], axis=1)

print(pca_99)
# Plot the PCA results into a graph

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = [0,1]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = pca_data['blueWins'] == target

    ax.scatter(pca_data.loc[indicesToKeep, 'principal component 1']

               , pca_data.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
# Biplot





def biplot(score,coeff,labels=None):

    plt.rcParams["figure.figsize"] = (20,20)

    xs = score[:,0]

    ys = score[:,1]

    n = coeff.shape[0]

    scalex = 1.0/(xs.max() - xs.min())

    scaley = 1.0/(ys.max() - ys.min())

    plt.scatter(xs * scalex,ys * scaley, c = pca_data_y)

    for i in range(n):

        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)

        if labels is None:

            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center', fontsize=13)

        else:

            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center', fontsize=13)

    plt.xlim(-1,1)

    plt.ylim(-1,1)

    plt.xlabel("PC{}".format(1))

    plt.ylabel("PC{}".format(2))

    plt.grid()



#Call the function. Use only the 2 PCs.

biplot(principal_components[:,0:2],np.transpose(pca.components_[0:2, :]), data.columns[1:])

#plt.show()
# As a first step instead, a number of new factors showing the difference between some of the metrics will help to reduce the dimension on the model and get rid of redundant variables.



# Function that adds a difference factor and drops original columns



def add_difference_factor(data, column1, column2):

    temp_array = []

    for index, row in data.iterrows():

        temp_array.append(row[column1]-row[column2])

    temp_df = pd.DataFrame({(column1+"Difference"):temp_array})

    data = data.join(temp_df)

    data = data.drop([column1, column2], axis=1)

    return data
# Function for further plotting of loss and accuracy plots

def plot_history(history):

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]

    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]

    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]



    if len(loss_list) == 0:

        print('Loss is missing in history')

        return



        ## As loss always exists

    epochs = range(1, len(history.history[loss_list[0]]) + 1)



    ## Loss

    plt.figure(1)

    for l in loss_list:

        plt.plot(epochs, history.history[l], 'b',

                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    for l in val_loss_list:

        plt.plot(epochs, history.history[l], 'g',

                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))



    plt.title('Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()



    ## Accuracy

    plt.figure(2)

    for l in acc_list:

        plt.plot(epochs, history.history[l], 'b',

                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    for l in val_acc_list:

        plt.plot(epochs, history.history[l], 'g',

                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')



    plt.title('Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()
# Create a backup copy of the original data set



data_original = data.copy()
# Refactor all shared variables to difference variables instead



data = add_difference_factor(data, 'blueTotalExperience', 'redTotalExperience')

data = add_difference_factor(data, 'blueCSPerMin', 'redCSPerMin')

data = add_difference_factor(data, 'blueGoldPerMin', 'redGoldPerMin')

data = add_difference_factor(data, 'blueKills', 'redKills')

data = add_difference_factor(data, 'blueDeaths', 'redDeaths')

data = add_difference_factor(data, 'blueAssists', 'redAssists')

data = add_difference_factor(data, 'bluedragons', 'reddragons')

data = add_difference_factor(data, 'blueEliteMonsters', 'redEliteMonsters')

data = add_difference_factor(data, 'blueHeralds', 'redHeralds')

data = add_difference_factor(data, 'blueTotalMastery', 'redTotalMastery')
# Display a summary of the new dataset

data_summary(data)
# Class that creates a model given a number of hidden layers and builder factors

# builder factor determines the number of neurons in the following hidden layer

# e.g with builder factor 0.5, if the first layer has 22 neurons, the following layer will have 11 neurons

# First layer and hidden layers will use 'relu' activation function

# Output layer consists of a single neuron with sigmoid activation function, which will result in the final classification

# Model will use binary crossentropy as its loss function ( perfect for binary classification problems )

# Model uses the most common, standard 'adam' optimizer and 'accuracy' as its score metric



class ModelBuilder:

    last_num_of_neurons = 0



    builder_factor = 0



    def __init__(self, num_of_layers, builder_factor, input_neurons):

        self.num_of_layers = num_of_layers

        self.builder_factor = builder_factor

        self.input_neurons = input_neurons

        self.last_num_of_neurons = input_neurons



    def create_model(self):

        model = Sequential()

        model.add(Dense(self.num_of_layers, input_dim=self.input_neurons, activation='relu'))

        for x in range(0, self.num_of_layers-1):

            model.add(Dense(int(self.builder_factor * self.last_num_of_neurons), activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        # Compile model

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

# This method will conduct training of a given set of data using k=10 k-folds Cross-Validation

# Individual models are saved in clf_models array for later validation



def estimate_model(build_fn, X_data, Y_data):

    

    kf = StratifiedKFold(n_splits=10, shuffle=True)



    #MLP classifier

    clf = KerasClassifier(build_fn=build_fn, epochs=100, batch_size=128, verbose=0)

    best_model = None



    # keep in mind your X and y should be indexed same here

    kf.get_n_splits(X_data)

    fold_score = []

    for train_index, test_index in kf.split(X_data, Y_data):

        X_train, X_test = X_data[train_index], X_data[test_index]

        y_train, y_test = Y_data[train_index], Y_data[test_index]

        tmp_clf = deepcopy(clf)

        history = tmp_clf.fit(X_train, y_train, validation_split=0.1)

        score = tmp_clf.score(X_test, y_test)

        fold_score.append(score)

        if not best_model:

            best_model = (tmp_clf, score, history)

        elif score > best_model[1]:

            best_model = (tmp_clf, score, history)

    print("Mean accuracy: %.6f%% (+/- %.2f%%)" % (np.mean(fold_score), np.std(fold_score)))



    return best_model
# This method will train and evaluate different models with different number of layers and builder factors in order to find model with the highest accuracy



def try_different_models(data):



    x_data = data[0:, 1:]

    x_data = StandardScaler().fit_transform(x_data)

    y_data = data[0:, 0]



    builder_factors = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

    number_of_layers = 4

    best_models = []

    # Create model and add layers

    for builder_factor in builder_factors:

        for x in range(0, number_of_layers):

            timestamp = time.time()

            print("Model with " + str(x + 1) + " layers and factor: " + str(builder_factor))

            model_builder = ModelBuilder(x + 1, builder_factor, np.ma.size(x_data, axis=1))

            best_models.append(estimate_model(model_builder.create_model, x_data, y_data))

            print("Time elapsed: " + str(time.time() - timestamp) + " seconds")



    models_initial = pd.DataFrame({

        'Number of layers': [1, 2, 3, 4],

        '0.5': [best_models[0][1], best_models[1][1], best_models[2][1], best_models[3][1]],

        '0.75': [best_models[4][1], best_models[5][1], best_models[6][1], best_models[7][1]],

        '1': [best_models[8][1], best_models[9][1], best_models[10][1], best_models[11][1]],

        '1.25': [best_models[12][1], best_models[13][1], best_models[14][1], best_models[15][1]],

        '1.5': [best_models[16][1], best_models[17][1], best_models[18][1], best_models[19][1]],

        '1.75': [best_models[20][1], best_models[21][1], best_models[22][1], best_models[23][1]],

        '2': [best_models[24][1], best_models[25][1], best_models[26][1], best_models[27][1]]

    }, columns=['Number of layers', '0.5', '0.75', '1', '1.25', '1.5', '1.75', '2'])



    models_initial.sort_values(by='Number of layers', ascending=True)

    max_score = best_models[0][1]

    the_best_model = best_models[0]

    for model in best_models:

        if model[1] > max_score:

            the_best_model = model

    plot_history(the_best_model[2])



data_summary(data)
data = data.to_numpy()

data_original = data_original.to_numpy()

pca_99 = pca_99.to_numpy()

try_different_models(data)

print("~~~~~~~~~~~~~~^ DATA ^~~~~~~~~~~~~~~")

try_different_models(data_original)

print("~~~~~~~~~~~~~~^ DATA ORIGINAL ^~~~~~~~~~~~~~~")

try_different_models(pca_99)

print("~~~~~~~~~~~~~~^ DATA PCA_99 ^~~~~~~~~~~~~~~")