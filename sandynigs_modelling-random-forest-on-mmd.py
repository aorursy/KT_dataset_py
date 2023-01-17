#All imports her

import pandas as pd

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
data_asm_byte_final = pd.read_csv("../input/data_asm_byte_final.csv", index_col = 0)

data_asm_byte_final.head()
final_y = data_asm_byte_final["Class"]

data_asm_byte_final = data_asm_byte_final.drop("Class", axis=1)
#Let's normalize the data.

def normalize(dataframe):

    #print("Here")

    test = dataframe.copy()

    for col in tqdm(test.columns):

        if(col != "Id" and col !="Class"):

            max_val = max(dataframe[col])

            min_val = min(dataframe[col])

            test[col] = (dataframe[col] - min_val) / (max_val-min_val)

    return test
data_asm_byte_final = normalize(data_asm_byte_final)
data_asm_byte_final.head()
data_y = final_y

# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]

x_train, x_test, y_train, y_test = train_test_split(data_asm_byte_final.drop(['Id'], axis=1), data_y,stratify=data_y,test_size=0.20)

# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]

x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train,stratify=y_train,test_size=0.20)
def perform_hyperparam_tuning(list_of_hyperparam, model_name,  x_train, y_train, x_cv, y_cv):

    cv_log_error_array = []

    for i in tqdm(list_of_hyperparam):

        if(model_name == "rf"):

            model = RandomForestClassifier(n_estimators=i,random_state=42,n_jobs=-1, class_weight='balanced')

        model.fit(x_train, y_train)

        caliberated_model = CalibratedClassifierCV(model, method = "sigmoid")

        caliberated_model.fit(x_train, y_train)

        predict_y = caliberated_model.predict_proba(x_cv)

        cv_log_error_array.append(log_loss(y_cv, predict_y))

    for i in range(len(cv_log_error_array)):

        print ('log_loss for hyper_parameter = ',list_of_hyperparam[i],'is',cv_log_error_array[i])

    return cv_log_error_array

       

def get_best_hyperparam(list_of_hyperparam, cv_log_error_array):

    index = np.argmin(cv_log_error_array)

    best_hyperparameter = list_of_hyperparam[index]

    return best_hyperparameter





def perform_on_best_hyperparam(model_name, best_hyperparameter, cv_log_error_array,x_train,y_train,x_cv,y_cv,x_test,y_test):

    

    if(model_name == "rf"):

            model = RandomForestClassifier(n_estimators = best_hyperparameter,random_state = 42,n_jobs = -1,  class_weight='balanced')

    model.fit(x_train, y_train)

    

    caliberated_model = CalibratedClassifierCV(model, method = "sigmoid")

    caliberated_model.fit(x_train, y_train)



    predicted_y = caliberated_model.predict_proba(x_train)

    print("The training log-loss for best hyperparameter is", log_loss(y_train, predicted_y))

    predicted_y = caliberated_model.predict_proba(x_cv)

    print("The cv log-loss for best hyperparameter is", log_loss(y_cv, predicted_y))

    predicted_y = caliberated_model.predict_proba(x_test)

    print("The test log-loss for best hyperparameter is", log_loss(y_test, predicted_y))



    predicted_y = caliberated_model.predict(x_test)

    #plot_confusion_matrix(y_test, predicted_y)

    



def plot_cv_error(list_of_hyperparam, cv_log_error_array):

    fig, ax = plt.subplots()

    ax.plot(list_of_hyperparam, cv_log_error_array,c='g')

    for i, txt in enumerate(np.round(cv_log_error_array,3)):

        ax.annotate((list_of_hyperparam[i],np.round(txt,3)), (list_of_hyperparam[i],cv_log_error_array[i]))

    plt.grid()

    plt.title("Cross Validation Error for each hyperparameter")

    plt.xlabel("Hyperparameter")

    plt.ylabel("Error measure")

    plt.show()

list_of_hyperparam = [10,50,100,500,1000,2000,3000]

model_name = "rf"

cv_log_error_array = perform_hyperparam_tuning(list_of_hyperparam, model_name,  x_train, y_train, x_cv, y_cv)
best_hyperparameter = get_best_hyperparam(list_of_hyperparam, cv_log_error_array)

perform_on_best_hyperparam(model_name, best_hyperparameter, cv_log_error_array,x_train,y_train,x_cv,y_cv,x_test,y_test)