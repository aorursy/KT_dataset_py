from typing import Tuple

import pandas as pd
from pandas import DataFrame

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras import metrics
seed = 7
np.random.seed(seed)
df = pd.read_csv("../input/data.csv")
df.head()
df.shape # shows the shape of data stored in matrix
df.info()
print(df["diagnosis"].unique())
B, M = df["diagnosis"].value_counts()
ax = sns.countplot(x="diagnosis", data=df)

print("Benign count: {}".format(B))
print("Malignat count: {}".format(M))

plt.show()
df = df.drop(["id", "Unnamed: 32"], 1)
df.isnull().values.any()
X = StandardScaler().fit_transform(df.iloc[:,1:31])
y = LabelEncoder().fit_transform(df.diagnosis.values)
def create_logistic_model(input_features_dimension) -> Sequential:
    model = Sequential()
    model.add(Dense(1, input_dim=input_features_dimension, activation='sigmoid')) 
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[metrics.mae, metrics.mse, 'acc'])
    return model

def create_model1(input_features_dimension) -> Sequential: 
    model = Sequential()
        
    model.add(Dense(64, input_dim=input_features_dimension))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[metrics.mae, metrics.mse, 'acc'])
    
    return model

def create_model2(input_features_dimension) -> Sequential: 
    model = Sequential()
    
    model = Sequential()
    model.add(Dense(64, activation="relu", input_dim=input_features_dimension))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.mae, metrics.mse, 'acc'])
    #model.summary()
    
    return model

def create_model3(input_features_dimension) -> Sequential: 
    model = Sequential()
        
    model.add(Dense(32, input_dim=input_features_dimension))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[metrics.mae, metrics.mse, 'acc'])
    
    return model

def create_model4(input_features_dimension) -> Sequential: 
    model = Sequential()
        
    model.add(Dense(32, input_dim=input_features_dimension))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[metrics.mae, metrics.mse, 'acc'])
    
    return model

def create_model5(input_features_dimension) -> Sequential: 
    model = Sequential()
        
    model.add(Dense(64, input_dim=input_features_dimension))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[metrics.mae, metrics.mse, 'acc'])
    
    return model

def create_model6(input_features_dimension) -> Sequential: 
    model = Sequential()
        
    model.add(Dense(64, input_dim=input_features_dimension))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=[metrics.mae, metrics.mse, 'acc'])
    
    return model
def crossvalidation(X, y, model_creator, folds_number = 10, epochs = 100, batch_size = 64) -> str:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    
    for train, test in kfold.split(X, y):
        model = model_creator(input_features_dimension=X[train].shape[1])
        model.fit(X[train], y[train], epochs=epochs, batch_size=batch_size, verbose=0)
        scores = model.evaluate(X[test], y[test], verbose=0)
        cvscores.append(scores[3] * 100)    
        
    return [np.mean(cvscores), np.std(cvscores)]
def perform_tests(
        models_creators = {"logistic model": create_logistic_model, 
                           "model1": create_model1, 
                           "model2": create_model2, 
                           "model3": create_model3, 
                           "model4": create_model4,
                           "model5": create_model5,
                           "model6": create_model6},
        epochs_to_test = [200, 150, 100, 50, 25], 
        batch_sizes_to_test = [256, 128, 64, 32, 16, 8],
        X = X,
        y = y):
    results = [] 

    for model_creator in models_creators:
        for epochs in epochs_to_test:
            for batch_size in batch_sizes_to_test:
                validation_result = (
                    crossvalidation(
                        X, 
                        y, 
                        models_creators[model_creator], 
                        epochs=epochs, 
                        batch_size=batch_size)
                )
                
                final_results = [model_creator]
                final_results.extend(validation_result)
                final_results.extend([epochs, batch_size])
                
                results.append(final_results)
                
                print("Crossvalidation of {} for {} batch size and {} epochs completed.".format(model_creator, batch_size, epochs))

    for result in results: 
        print(result)
        
    return results
  
# Run this to perform tests
# perform_tests()
models_results = pd.DataFrame(
    columns=["model_name", "accuracy", "deviation", "epochs", "batch_size"],
    data = [
        ("logistic model", 95.08, 1.73, 200, 256),
        ("logistic model", 97.00, 1.77, 200, 128),
        ("logistic model", 97.89, 1.31, 200, 64),
        ("logistic model", 98.07, 1.46, 200, 32),
        ("logistic model", 97.71, 1.12, 200, 16),
        ("logistic model", 97.54, 1.15, 200, 8),
        ("logistic model", 94.56, 2.77, 150, 256),
        ("logistic model", 95.61, 1.79, 150, 128),
        ("logistic model", 97.70, 1.60, 150, 64),
        ("logistic model", 97.89, 1.32, 150, 32),
        ("logistic model", 98.07, 1.23, 150, 16),
        ("logistic model", 97.72, 1.36, 150, 8),
        ("logistic model", 90.15, 5.92, 100, 256),
        ("logistic model", 95.07, 2.83, 100, 128),
        ("logistic model", 97.53, 1.79, 100, 64),
        ("logistic model", 97.19, 1.79, 100, 32),
        ("logistic model", 98.24, 1.11, 100, 16),
        ("logistic model", 98.42, 0.95, 100, 8),
        ("logistic model", 82.64, 6.59, 50, 256),
        ("logistic model", 86.62, 7.13, 50, 128),
        ("logistic model", 95.78, 2.28, 50, 64),
        ("logistic model", 96.13, 1.31, 50, 32),
        ("logistic model", 97.18, 1.61, 50, 16),
        ("logistic model", 98.06, 1.65, 50, 8),
        ("logistic model", 59.25, 21.56, 25, 256),
        ("logistic model", 83.40, 14.65, 25, 128),
        ("logistic model", 88.06, 6.26, 25, 64),
        ("logistic model", 94.02, 3.10, 25, 32),
        ("logistic model", 96.13, 1.73, 25, 16),
        ("logistic model", 97.71, 1.58, 25, 8),
        ("model1", 98.07, 1.46, 200, 256),
        ("model1", 97.89, 1.53, 200, 128),
        ("model1", 97.71, 1.58, 200, 64),
        ("model1", 97.71, 1.37, 200, 32),
        ("model1", 97.71, 1.77, 200, 16),
        ("model1", 97.18, 1.97, 200, 8),
        ("model1", 97.71, 1.13, 150, 256),
        ("model1", 98.06, 0.94, 150, 128),
        ("model1", 97.71, 1.13, 150, 64),
        ("model1", 97.53, 1.41, 150, 32),
        ("model1", 97.71, 1.58, 150, 16),
        ("model1", 96.83, 1.89, 150, 8),
        ("model1", 97.89, 1.06, 100, 256),
        ("model1", 97.53, 1.41, 100, 128),
        ("model1", 97.89, 1.06, 100, 64),
        ("model1", 97.88, 1.32, 100, 32),
        ("model1", 97.88, 1.53, 100, 16),
        ("model1", 97.71, 1.58, 100, 8),
        ("model1", 95.95, 2.39, 50, 256),
        ("model1", 97.01, 1.12, 50, 128),
        ("model1", 98.24, 1.37, 50, 64),
        ("model1", 97.71, 1.13, 50, 32),
        ("model1", 97.53, 1.17, 50, 16),
        ("model1", 97.71, 1.58, 50, 8),
        ("model1", 95.08, 1.53, 25, 256),
        ("model1", 96.31, 1.83, 25, 128),
        ("model1", 96.66, 2.16, 25, 64),
        ("model1", 97.88, 1.53, 25, 32),
        ("model1", 97.72, 1.37, 25, 16),
        ("model1", 97.53, 1.41, 25, 8),
        ("model2", 97.71, 1.58, 200, 256),
        ("model2", 97.71, 1.58, 200, 128),
        ("model2", 97.71, 1.77, 200, 64),
        ("model2", 97.71, 1.58, 200, 32),
        ("model2", 97.19, 1.80, 200, 16),
        ("model2", 97.54, 1.60, 200, 8),
        ("model2", 97.54, 1.16, 150, 256),
        ("model2", 97.71, 1.58, 150, 128),
        ("model2", 97.89, 1.31, 150, 64),
        ("model2", 97.36, 1.96, 150, 32),
        ("model2", 97.71, 1.77, 150, 16),
        ("model2", 97.54, 2.25, 150, 8),
        ("model2", 97.71, 1.13, 100, 256),
        ("model2", 97.71, 1.12, 100, 128),
        ("model2", 97.89, 1.52, 100, 64),
        ("model2", 97.19, 1.79, 100, 32),
        ("model2", 97.54, 1.61, 100, 16),
        ("model2", 97.71, 1.38, 100, 8),
        ("model2", 96.83, 2.23, 50, 256),
        ("model2", 97.01, 1.59, 50, 128),
        ("model2", 97.71, 1.38, 50, 64),
        ("model2", 97.90, 1.87, 50, 32),
        ("model2", 97.89, 1.54, 50, 16),
        ("model2", 97.72, 1.58, 50, 8),
        ("model2", 94.71, 2.26, 25, 256),
        ("model2", 96.83, 1.56, 25, 128),
        ("model2", 97.37, 1.41, 25, 64),
        ("model2", 97.72, 1.36, 25, 32),
        ("model2", 97.71, 1.58, 25, 16),
        ("model2", 97.19, 1.61, 25, 8),
        ("model3", 97.89, 1.06, 200, 256),
        ("model3", 97.71, 1.37, 200, 128),
        ("model3", 98.24, 1.36, 200, 64),
        ("model3", 97.71, 1.58, 200, 32),
        ("model3", 98.06, 1.24, 200, 16),
        ("model3", 97.54, 1.39, 200, 8),
        ("model3", 97.18, 1.61, 150, 256),
        ("model3", 97.71, 1.13, 150, 128),
        ("model3", 97.89, 1.32, 150, 64),
        ("model3", 97.88, 1.32, 150, 32),
        ("model3", 97.53, 1.41, 150, 16),
        ("model3", 97.36, 1.42, 150, 8),
        ("model3", 97.71, 1.37, 100, 256),
        ("model3", 98.24, 1.11, 100, 128),
        ("model3", 97.89, 1.31, 100, 64),
        ("model3", 97.89, 1.32, 100, 32),
        ("model3", 97.71, 1.58, 100, 16),
        ("model3", 97.88, 1.32, 100, 8),
        ("model3", 94.91, 2.01, 50, 256),
        ("model3", 96.84, 1.90, 50, 128),
        ("model3", 97.01, 2.22, 50, 64),
        ("model3", 98.06, 1.24, 50, 32),
        ("model3", 97.71, 1.76, 50, 16),
        ("model3", 97.89, 1.32, 50, 8),
        ("model3", 92.10, 3.85, 25, 256),
        ("model3", 95.09, 3.00, 25, 128),
        ("model3", 95.07, 1.36, 25, 64),
        ("model3", 96.31, 1.68, 25, 32),
        ("model3", 97.18, 1.43, 25, 16),
        ("model3", 96.31, 2.28, 25, 8),
        ("model4", 97.36, 1.60, 200, 256),
        ("model4", 97.72, 1.93, 200, 128),
        ("model4", 97.71, 1.58, 200, 64),
        ("model4", 97.88, 1.32, 200, 32),
        ("model4", 97.71, 1.58, 200, 16),
        ("model4", 97.18, 1.40, 200, 8),
        ("model4", 97.01, 1.56, 150, 256),
        ("model4", 97.36, 1.80, 150, 128),
        ("model4", 97.37, 1.41, 150, 64),
        ("model4", 97.36, 1.62, 150, 32),
        ("model4", 98.23, 1.12, 150, 16),
        ("model4", 97.88, 1.32, 150, 8),
        ("model4", 97.54, 1.61, 100, 256),
        ("model4", 98.06, 0.95, 100, 128),
        ("model4", 97.89, 1.05, 100, 64),
        ("model4", 97.89, 1.53, 100, 32),
        ("model4", 98.06, 1.24, 100, 16),
        ("model4", 97.88, 1.53, 100, 8),
        ("model4", 96.31, 1.85, 50, 256),
        ("model4", 97.19, 0.86, 50, 128),
        ("model4", 97.54, 1.16, 50, 64),
        ("model4", 97.53, 1.79, 50, 32),
        ("model4", 97.88, 1.32, 50, 16),
        ("model4", 97.54, 1.16, 50, 8),
        ("model4", 95.78, 1.80, 25, 256),
        ("model4", 96.48, 2.48, 25, 128),
        ("model4", 97.19, 1.15, 25, 64),
        ("model4", 97.36, 1.65, 25, 32),
        ("model4", 97.53, 1.18, 25, 16),
        ("model4", 96.66, 1.46, 25, 8),
        ("model5", 97.71, 1.12, 200, 256),
        ("model5", 98.07, 1.46, 200, 128),
        ("model5", 97.54, 1.79, 200, 64),
        ("model5", 98.06, 1.46, 200, 32),
        ("model5", 97.53, 1.60, 200, 16),
        ("model5", 96.84, 1.31, 200, 8),
        ("model5", 97.53, 1.18, 150, 256),
        ("model5", 97.89, 1.32, 150, 128),
        ("model5", 98.07, 1.65, 150, 64),
        ("model5", 98.24, 1.36, 150, 32),
        ("model5", 97.88, 1.32, 150, 16),
        ("model5", 97.19, 1.40, 150, 8),
        ("model5", 98.06, 0.94, 100, 256),
        ("model5", 97.53, 1.17, 100, 128),
        ("model5", 97.89, 1.05, 100, 64),
        ("model5", 97.89, 1.05, 100, 32),
        ("model5", 98.41, 1.24, 100, 16),
        ("model5", 97.36, 1.80, 100, 8),
        ("model5", 96.83, 2.23, 50, 256),
        ("model5", 97.19, 1.41, 50, 128),
        ("model5", 97.88, 1.32, 50, 64),
        ("model5", 98.06, 1.24, 50, 32),
        ("model5", 97.89, 1.06, 50, 16),
        ("model5", 97.53, 1.41, 50, 8),
        ("model5", 94.73, 3.60, 25, 256),
        ("model5", 95.96, 1.92, 25, 128),
        ("model5", 97.36, 1.18, 25, 64),
        ("model5", 97.36, 1.18, 25, 32),
        ("model5", 97.19, 1.16, 25, 16),
        ("model5", 97.18, 1.95, 25, 8),
        ("model6", 97.72, 1.76, 200, 256),
        ("model6", 98.07, 1.99, 200, 128),
        ("model6", 97.89, 1.72, 200, 64),
        ("model6", 97.54, 1.41, 200, 32),
        ("model6", 97.36, 1.80, 200, 16),
        ("model6", 97.36, 1.42, 200, 8),
        ("model6", 98.59, 1.05, 150, 256),
        ("model6", 98.06, 0.94, 150, 128),
        ("model6", 97.71, 1.58, 150, 64),
        ("model6", 97.89, 1.72, 150, 32),
        ("model6", 97.71, 1.77, 150, 16),
        ("model6", 97.71, 2.09, 150, 8),
        ("model6", 98.07, 1.22, 100, 256),
        ("model6", 98.24, 1.11, 100, 128),
        ("model6", 97.54, 1.40, 100, 64),
        ("model6", 97.71, 1.13, 100, 32),
        ("model6", 98.41, 1.24, 100, 16),
        ("model6", 97.53, 1.41, 100, 8),
        ("model6", 97.54, 0.87, 50, 256),
        ("model6", 97.71, 1.58, 50, 128),
        ("model6", 97.54, 1.16, 50, 64),
        ("model6", 97.01, 1.57, 50, 32),
        ("model6", 97.89, 1.72, 50, 16),
        ("model6", 98.06, 1.66, 50, 8),
        ("model6", 95.42, 1.98, 25, 256),
        ("model6", 96.83, 1.56, 25, 128),
        ("model6", 96.84, 1.71, 25, 64),
        ("model6", 97.72, 1.58, 25, 32),
        ("model6", 97.89, 1.31, 25, 16),
        ("model6", 97.36, 1.17, 25, 8)
    ]
)
def draw_heatmaps(
        models_results_df: DataFrame,
        model_name_column = "model_name", 
        batch_size_column = "batch_size", 
        epochs_column = "epochs",
        accuracy_column = "accuracy"):
    for model in models_results_df[model_name_column].unique():
        plt.figure(figsize=(6,2))

        current_data = (
            models_results_df[models_results_df[model_name_column] == model]
                .groupby([batch_size_column, epochs_column])[accuracy_column]
                .sum()
                .unstack()
        )

        sns.heatmap(data=current_data, annot=True, fmt='.2f', cmap="coolwarm")

        plt.suptitle(model)
        plt.show()
    
draw_heatmaps(models_results)
models_results.loc[models_results.accuracy.idxmax()]