"""Generate a raw data preview report"""

import pandas as pd

pd.set_option('display.max_columns',None)  #show all columns

pd.set_option('display.width',None)

pd.set_option('display.unicode.ambiguous_as_wide', True)

pd.set_option('display.unicode.east_asian_width', True)



def read_file(file_path):

    """read_file according to file format"""

    if file_path.endswith("csv"):

        data = pd.read_csv(file_path)

    elif file_path.endswith("xls"):

        data = pd.read_excel(file_path)

    elif file_path.endswith("xlsx"):

        data = pd.read_excel(file_path)

    return data



def preview(file_path,class_name): #transfer into file_path and the data's class/label name (in this case, class name is ACTIVITY)

    data = read_file(file_path)  



    """1.show data shape"""

    data_shape = data.shape

    print("1..data shape:", data_shape)

    print("*******"*20)

    #

    """2.show first 10 rows and last 10 rows of data"""

    head_10 = data.head(10)

    print("2.First 10 rows:")

    print(head_10)

    print("*******"*20)

    print("3.Last 10 rows:")

    tail_10 = data.tail(10)

    print(tail_10)

    print("*******"*20)



    """3.show basic information of the raw data"""

    print("4.Basic information of the raw data:")

    data_info = data.info()

    print(data_info)

    print("*******"*20)



    """4.Show more detailed information of the data..."""

    data_describe = data.describe(include = "all")

    print("5.Detailed information of the raw data:")

    print(data_describe)

    print("*******"*20)



    """5.Examine null"""

    print("6.Situation of null value of raw data:")

    data_null = data.isnull().sum()

    print(data_null)

    print("*******" * 20)



    """6.Gain feature name(column name) list"""

    print("7.feature_name_list：")

    features_list = data.columns.values.tolist()

    print(features_list)

    print("*******" * 20)



    """7.Gain the distribution of class value"""

    print("8.Distribution of class_value {}:".format(repr(class_name)))

    class_values_distribution = data[class_name].value_counts()

    print(class_values_distribution)



#use function 'preview'

preview(r'../input/falldeteciton.csv','ACTIVITY')
import pandas as pd



data = pd.read_csv(r"../input/falldeteciton.csv")



#1.change the order of features to make "ACTIVITY" to be the last column

data = data[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION','ACTIVITY' ]]



#2. change the class value from numeric to nominal

dict_ = {0:"Standing",1:"Walking",2:'Sitting',3:"Falling",4:"Cramps",5:"Running"}

data["ACTIVITY"] = data["ACTIVITY"].map(dict_)



"""3.min-max normalization"""

# from sklearn.preprocessing import MinMaxScaler

# minmax = MinMaxScaler()

# data[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']] = minmax.fit_transform(data[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']])



# data.to_csv("prepcessed_minmax.csv")



"""4.z-score normalization"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(data[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']]) 

data[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']] = scaler.fit_transform(data[['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']])

# data.to_csv("preprocessed_zscore.csv")
import pandas as pd

import numpy as np



# file_path_zscore = r"xxxxxxxxxxxxx\preprocessed_zscore.csv"  #read preprocessed file

# data = pd.read_csv(file_path_zscore,engine='python')





features = ['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']                       #get features list

label = ['ACTIVITY']                                                                 #get lable name

class_value = ["Standing","Walking","Sitting","Falling","Cramps","Running"]     #get class values list

features_data = data[features]                                                         #get features' Dataframe

label_data = data[label]                                                              #get class data



#split the data into training data and test data

from sklearn.model_selection import train_test_split  #导入split包

train_features, test_features,train_label, test_label = train_test_split(features_data, label_data, test_size=0.33, random_state=33)



#feature selection part

from sklearn.feature_selection import RFE

from sklearn.tree import DecisionTreeClassifier

#import an external estimator

clf = DecisionTreeClassifier(criterion="entropy")

#↓ n_features_to_select:4

selector = RFE(estimator=clf,n_features_to_select=4,step=1)

selector.fit(features_data,np.ravel(label_data))

selected_features = [features[i] for i in list(selector.get_support(indices=True))]

print(selected_features)  # show selected features



########################################################################

#L1-based feature selection

from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC

lsvc = LinearSVC(C=0.005, penalty="l1", dual=False).fit(features_data, np.ravel(label_data))

selector = SelectFromModel(lsvc,prefit=True)

selected_features = [features[i] for i in list(selector.get_support(indices=True))]

print(selected_features)

import pandas as pd

import numpy as np



# file_path_zscore = r"xxxxxxxxxxxxx\preprocessed_zscore.csv"  #read preprocessed file

# data = pd.read_csv(file_path_zscore,engine='python')





features = ['TIME', 'SL', 'EEG', 'BP', 'HR', 'CIRCLUATION']                       #get features list

label = ['ACTIVITY']                                                                 #get lable name

class_value = ["Standing","Walking","Sitting","Falling","Cramps","Running"]     #get class values list

features_data = data[features]                                                         #get features' Dataframe

label_data = data[label]                                                              #get class data



#split the data into training data and test data

from sklearn.model_selection import train_test_split  #导入split包

train_features, test_features,train_label, test_label = train_test_split(features_data, label_data, test_size=0.33, random_state=33)



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import GridSearchCV

# parameters = {'n_estimators':np.arange(110,150,3)}

# clf = GridSearchCV(estimator=ExtraTreesClassifier(),n_jobs=-1,cv=5,param_grid=parameters)      #get the best n_estimators is 125

# clf.fit(features_data,np.ravel(label_data))

# print(clf.best_params_)

# print(clf.best_score_)

clf = ExtraTreesClassifier(n_estimators=125,n_jobs=-1)



# fit the data

clf.fit(train_features,train_label)

predict_label = clf.predict(test_features)



#####################################################################################################################

# I create a evaluation pack, following is a simplified edition

import pandas as pd

def evaluate(predict_label,test_label,class_value):

    print("***********The evaluation of split test data.*************")

    from sklearn.metrics import accuracy_score

    print("Accuracy-Test data:", accuracy_score(y_pred=predict_label, y_true=test_label))

    print('****'*30)

    

    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(test_label, predict_label)

    print("Kappa:", kappa)

    print('****' * 30)



    from sklearn.metrics import confusion_matrix

    confustion = confusion_matrix(y_pred=predict_label, y_true=test_label, labels=class_value)

    confustion_matrix_df = pd.DataFrame(confustion, columns=class_value, index=class_value)

    print(confustion_matrix_df)

    print('****'*30)



    from sklearn.metrics import classification_report

    report_dict = classification_report(y_true=test_label, y_pred=predict_label, output_dict=False)

    print(report_dict)

##################################################################################################

evaluate(predict_label,test_label,class_value)