import numpy as np
def fit(x_train,y_train):
    result={}
    class_values=set(y_train) 
    for current_class in class_values:
        result[current_class]={}
        result["total_data"]=len(y_train)
        current_class_rows=(y_train==current_class)
        x_train_current=x_train[current_class_rows]
        y_train_current=y_train[current_class_rows]
        result[current_class]["total_count"]=len(y_train_current)
        num_features=x_train.shape[1]
        for j in range(1,num_features+1):
            result[current_class][j]={}
            all_possible_values=set(x_train[:,j-1])
            for current_values in all_possible_values:
                result[current_class][j][current_values]=(x_train_current[:,j-1]==current_values).sum()
    return result                 
def probability(dictionary,x,current_class):
    output=np.log(dictionary[current_class]["total_count"])-np.log(dictionary["total_data"])
    num_features=len(dictionary[current_class].keys())-1
    for j in range(1,num_features+1):
        xj=x[j-1]
        count_curr_class_with_value_xj=dictionary[current_class][j][xj]+1
        count_curr_class=dictionary[current_class]["total_count"]+len(dictionary[current_class][j].keys())
        current_xj_probability=np.log(count_curr_class_with_value_xj)-np.log(count_curr_class)
        output=output+current_xj_probability
    return output    
def predictSinglePoint(x,dictionary):
    classess = dictionary.keys()
    best_p=-1000
    best_class=-1
    first_run=True
    for current_class in classess:
        if(current_class=="total_data"):
            continue
        p_current_class=probability(dictionary,x,current_class)
        if(first_run or p_current_class >=best_p):
            best_p=p_current_class
            best_class=current_class
        first_run =False
    return best_class    
def predict(dictionary,x_test):
    y_pred=[]
    for x in x_test:
        x_class=predictSinglePoint(x,dictionary)
        y_pred.append(x_class)
    return y_pred    
def marklabelled(column):
    second_limit=column.mean()
    first_limit=0.5*second_limit
    third_limit=1.5*second_limit
    for i in range(0,len(column)):
        if(column[i]<first_limit):
            column[i]=0
        elif(column[i]>=first_limit and column[i]<second_limit):
            column[i]=1
        elif(column[i]>=second_limit and column[i]<third_limit):
            column[i]=2
        else:
            column[i]=3
    return column    
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target
columns=x.shape[1]
for i in range(0,columns):
    x[:,i]=marklabelled(x[:,i])  
from sklearn import model_selection
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)
algl=fit(x_train,y_train)
y_pred=predict(algl,x_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
