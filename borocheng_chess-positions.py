# Helper libraries

import os

import gc

import time

import random

import sklearn

import platform

import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix



print("version of Python:", platform.python_version())

print("version of numpy used: ",np.__version__)

print("version of pandas used: ",pd.__version__)

print("version of PIL used: ",Image.__version__)

print("version of sklearn used: ",sklearn.__version__)
#labels

EmptyGrid = 0

blackPawn = 1

whitePawn = 2

blackBishop = 3

whiteBishop = 4

blackRock = 5

whiteRock = 6

blackKnight = 7

whiteKnight = 8

blackQueen = 9

whiteQueen = 10

blackKing = 11

whiteKing = 12



#dict

chess_dict = {'p' : blackPawn, 'P' : whitePawn,

              'b' : blackBishop, 'B' : whiteBishop,

              'r' : blackRock, 'R' : whiteRock,

              'n' : blackKnight, 'N' : whiteKnight,

              'q' : blackQueen, 'Q' : whiteQueen,

              'k' : blackKing, 'K' : whiteKing,

              '0' : EmptyGrid}



# reserse dict 

chess_dict2={0:'0',

             1:'p',2:'P',

             3:'b',4:'B',

             5:'r',6:'R',

             7:'n',8:'N',

             9:'q',10:'Q',

             11:'k',12:'K',

             }



#for visualisation

labels_list=["e0","p1","P2","b3","B4","r5","R6","n7","N8","q9","Q10","k11","K12"]
from PIL import Image

import glob

import matplotlib.pyplot as plt

import os

import numpy as np



def load_data(path, images_list, start, end, mode):

    loading_train_set = True

    if mode == 'test':

        loading_train_set = False

    

    # grid height and width

    grid_h = 30

    grid_w = 30

    

    # size

    size = end - start

    

    features_df=[]

    labels=[]

    

    count=0

    

    for i in range(start, end):

        image_name = images_list[i]

        # load one image and convert it into gray scaled. Then rescale the size to 240 x 240 

        image = Image.open(path + image_name).convert('L').resize((240,240),Image.ANTIALIAS)

        filename = os.path.basename(path + image_name)

        # extract chess location information from the file name

        for i in range(1,9):

            filename = filename.replace(str(i), '0'*i)

        label_table = filename.split('.')[0].split('-')

        

        # width and height of the rescaled chess plate

        width, height = image.size



        

        

        # used to select one dark and one bright empty grid as empty grid samples when loading training set

        bright_empty_grid_selected = False

        dark_empty_grid_selected = False

        

        # cut the entire chess plate into 64 pieces

        for h in range(0,height,grid_h):

            for w in range(0,width, grid_w):

                

                # determine the label based on the file name

                y = int( h / grid_h)

                x = int( w / grid_w)

                grid_type = chess_dict[label_table[y][x]]

                

                # to select one dark and one bright empty grids as empty grid samples when loading train set

                if grid_type == '0' and loading_train_set:

                    if bright_empty_grid_selected and dark_empty_grid_selected:

                        continue

                    is_dark = (((y % 2) + (x % 2)) % 2)

                    if is_dark == 1:

                        if not bright_empty_grid_selected:

                            bright_empty_grid_selected = True

                        else:

                            continue

                    else:

                        if not dark_empty_grid_selected:

                            dark_empty_grid_selected = True

                        else:

                            continue

                

                # get the pixel map of one grid

                box = (w, h, w + grid_w, h + grid_h)

                grid_pix_map = image.crop(box)

                

                # np version of the pixel map of the grid

                np_pixel_map = np.asarray(grid_pix_map)

                # convert the pixel map into one dimensional feature list

                flat_pixels = list(np_pixel_map.flatten().astype(np.int))

                # add one row into the dataframe

                features_df.append(flat_pixels)

                # add one label into the label list

                labels.append(float(grid_type))

                

        count=count+1

        if count % 100 == 0:

            # show progress

            print("Loading progress:", float(count / size) * 100, "%")

    

    features_df = np.array(features_df)

    #return the dataset and corresponding label list 

    return features_df, np.array(labels)

#DATA_PATH='../input/chess-positions'

#train_path=os.path.join(DATA_PATH, 'train')

#test_path=os.path.join(DATA_PATH, 'test')



train_path='../input/chess-positions/train/'



train_size = 5000

images_names_train = os.listdir(train_path)



random.shuffle(images_names_train)

images_names_train = images_names_train[:train_size]

train_data,train_label=load_data(train_path, images_names_train, 0, train_size, 'train')

train_data = train_data/255

pca = PCA(n_components=0.95)

pca.fit(train_data)

X_train=pca.transform(train_data)
def data_dimensions(x,y):

    print('Data Dimension:')

    print('Number of Records:', x.shape[0])

    print('Number of Features:', x.shape[1])

    print("classes: ",np.unique(list(y)))

    



data_dimensions(X_train, train_label)

# clean memory

train_data = 0

gc.collect()
test_path='../input/chess-positions/test/'



test_size = 5000

images_names_test = os.listdir(test_path)



random.shuffle(images_names_test)

images_names_test = images_names_test[:test_size]

test_data, test_label=load_data(test_path, images_names_test, 0, test_size, 'test')
test_data = test_data/255

X_test = pca.transform(test_data)
data_dimensions(X_test, test_label)
# clean memory

test_data = 0

gc.collect()
y_test = test_label

y_train = train_label



# split the samples into two sets, one part to train and tune hyperparameter while the other to validate the performance/

X_validation = X_train[0: int(X_train.shape[0]/2)]

X_train = X_train[int(X_train.shape[0]/2):]



y_validation = y_train[0:int(y_train.shape[0]/2)]

y_train = y_train[int(y_train.shape[0]/2):]
## write a function to plot the result of prediction

def graph_confusion_matrix(confusion_matrix, labels_name, title):

    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]   

    plt.imshow(confusion_matrix, interpolation='nearest')    

    plt.title(title)    

    plt.colorbar()

    num_local = np.array(range(len(labels_name)))    

    plt.xticks(num_local, labels_name, rotation=0)  

    plt.yticks(num_local, labels_name)   

    plt.ylabel('X')    

    plt.xlabel('Y')



def calculate_accuracy(testlabel,predlabel):

    accu=0

    error=0

    for i in range (len(testlabel)):

        if testlabel[i]==predlabel[i]:



            pass

        else:



            pass



        if testlabel[i]==predlabel[i]:

            accu+=1

        else:

            error+=1

        

    accu_rate=accu/(len(list(testlabel)))



    return accu, accu_rate

labels_list=["e0","p1","P2","b3","B4","r5","R6","n7","N8","q9","Q10","k11","K12"]



from sklearn.neighbors import KNeighborsClassifier

model_knn=KNeighborsClassifier() 



n_neighbors=list(range(1,20))

parameters_knn = [

    {

        'n_neighbors': n_neighbors,

    }

]

sk_knn = GridSearchCV(model_knn, parameters_knn, n_jobs=-1, cv=10, verbose = 10)

sk_knn.fit(X_train, y_train)

print("\n")

result=sk_knn.cv_results_

print(" KNN model: \n",sk_knn)

print("-------------------------------------------------------------")

print("\n Best Estimator for KNN: \n",sk_knn.best_estimator_)

print("-------------------------------------------------------------")

print(" Best Parameters for KNN: ",sk_knn.best_params_)

print("-------------------------------------------------------------")

# plot accuracy vs parameters

plt.figure(figsize=(15,8))

plt.plot(n_neighbors,list(result["mean_test_score"]),label="KNN",marker='o', mec='r', mfc='w')

for x, y in zip(n_neighbors, list(result["mean_test_score"])):

    plt.text(x, y+0.001, "%.1f%%"  %(y*100), ha='center', va='bottom', fontsize=10.5)

plt.xlabel("K")

plt.ylabel("Accuracy")

plt.title("Model knn vs k")

plt.legend()  

plt.yticks([x/100 for x in list(range(95,100,1))])







# accuracy of model knn

accu_knn=sk_knn.best_score_ 

print("\n accuracy for KNN:","%.1f%%"  %(sk_knn.best_score_*100))

print("-------------------------------------------------------------")
import time



# classification report



start_time = time.time()

y_pred_knn=sk_knn.predict(X_validation)

end_time = time.time()







class_report_knn=sklearn.metrics.classification_report(y_validation, y_pred_knn,digits=2, target_names=labels_list,output_dict=True)

print("\n classification report for KNN\n:",sklearn.metrics.classification_report(y_validation, y_pred_knn,digits=2, target_names=labels_list,output_dict=False))

print("-------------------------------------------------------------")



# F1 Score

print("\n F1 Score for KNN:",sklearn.metrics.f1_score(y_validation, y_pred_knn, labels=None, average='macro',pos_label=1, sample_weight=None))

print("-------------------------------------------------------------")



# confusion matrix

print("\n Confusion_matrix for KNN: \n",confusion_matrix(y_validation, y_pred_knn))

cm=confusion_matrix(y_validation, y_pred_knn)

graph_confusion_matrix(cm,labels_list ,"confusion_matrix")

print("-------------------------------------------------------------")



print("\n Time consumed in prediction: ", end_time - start_time, "sec")
from sklearn.linear_model import LogisticRegression

model_LogisticRegression = LogisticRegression(multi_class="multinomial",solver="saga")



parameter_lr1=[x/10 for x in list(range(1,50,5))]

parameters_lr = [{   "C" : parameter_lr1    }]



sk_lr = GridSearchCV(model_LogisticRegression, parameters_lr, n_jobs=-1,cv=10, verbose = 10)

sk_lr.fit(X_train, y_train)





result_lr=sk_lr.cv_results_



# model function 

print("-------------------------------------------------------------")

print("\n model for LR: \n",sk_lr)

print("-------------------------------------------------------------")

print("\n Best Estimator for LR: \n",sk_lr.best_estimator_)

print("-------------------------------------------------------------")

print("\n Best Parameters for LR: ",sk_lr.best_params_)



# accuracy of model lr

accu_lr=sk_lr.best_score_ # accuracy

print("\n Best accuracy for LR:","%.1f%%"  %(sk_lr.best_score_*100))

print("-------------------------------------------------------------")



# plot accuracy vs parameters

plt.figure(figsize=(15,8))

plt.plot(parameter_lr1,list(result_lr["mean_test_score"]),label="LR",marker='o', mec='r', mfc='w')

for x, y in zip(parameter_lr1, list(result_lr["mean_test_score"])):

    plt.text(x, y+0.001, "%.1f%%"  %(y*100), ha='center', va='bottom', fontsize=10.5)

plt.xlabel("C")

plt.ylabel("Accuracy")

plt.title("Model LR vs C")

plt.legend()  

plt.yticks([x/100 for x in list(range(95,100,1))])



# classification report



start_time = time.time()

y_pred_lr=sk_lr.predict(X_validation)

end_time = time.time()





class_report_lr=sklearn.metrics.classification_report(y_validation, y_pred_lr,digits=2, target_names=labels_list,output_dict=True)

print("\n Report for LR \n:",sklearn.metrics.classification_report(y_validation, y_pred_lr,digits=2, target_names=labels_list))

print("-------------------------------------------------------------")



# F1 Score

print("\n F1 Score:",sklearn.metrics.f1_score(y_validation, y_pred_lr, labels=None, average='macro',pos_label=1, sample_weight=None))

print("-------------------------------------------------------------")



# confusion matrix

print("\n Confusion_matrix for LR: \n",confusion_matrix(y_validation, y_pred_lr))

plt.figure(figsize=(9,9))

cm=confusion_matrix(y_validation, y_pred_lr)

graph_confusion_matrix(cm,labels_list ,"Confusion_matrix")

print("-------------------------------------------------------------")



print("\n Time consumed in training: ", sk_lr.refit_time_, "sec")

print("\n Time consumed in prediction: ", end_time - start_time, "sec")



from sklearn.ensemble import RandomForestClassifier

model_randomforest=RandomForestClassifier()



p_rf1=list(range(10,115,20))

p_rf2=list(range(3,8,2))

parameters_rf =[

        {         

             'n_estimators':p_rf1,

             'max_features':p_rf2

             }

        ]

sk_rf = GridSearchCV(model_randomforest,parameters_rf, n_jobs=-1, verbose = 10,cv=10)

sk_rf.fit(X_train, y_train)





result_rf=sk_rf.cv_results_



print("\n")

# plot accuracy vs parameters



result_rf_dic=dict()

for i in  range(len(p_rf1)*len(p_rf2)):

  if result_rf['params'][i].get("max_features") not in result_rf_dic:

    result_rf_dic[result_rf['params'][i].get("max_features")] = [list(result_rf["mean_test_score"])[i]]

  else:

    result_rf_dic[result_rf['params'][i].get("max_features")].append(list(result_rf["mean_test_score"])[i])





plt.figure(figsize=(15,8))

for g in range(len(p_rf2)):

    plt.plot(p_rf1,result_rf_dic[p_rf2[g]],label="max_features={}".format(p_rf2[g]),marker='o', mec='r', mfc='w')

    for x, y in zip(p_rf1, result_rf_dic[p_rf2[g]]):

        plt.text(x, y+0.001, "%.1f%%"  %(y*100), ha='center', va='bottom', fontsize=10.5)

    plt.xlabel("value of n_estimators ")

    plt.ylabel("crossValidation accuracy")

    plt.title("Model randomforest ")

    plt.legend()  

    plt.yticks([x/100 for x in list(range(95,100,1))])



# model function 

print("-------------------------------------------------------------")

print(" model for RF: \n",sk_rf)

print("-------------------------------------------------------------")

print(" Best Estimator for RF: ",sk_rf.best_estimator_)

print("-------------------------------------------------------------")

print(" Best Parameters for RF: ",sk_rf.best_params_)







# accuracy 

accu_rf=sk_rf.best_score_ # accuracy

print(" Best Accuracy for RF:","%.1f%%"  %(sk_rf.best_score_*100))

print("-------------------------------------------------------------")


# classification report



start_time = time.time()

y_pred_rf=sk_rf.predict(X_validation)

end_time = time.time()



class_report_rf=sklearn.metrics.classification_report(y_validation, y_pred_rf,digits=2, target_names=labels_list,output_dict=True)

print(" classification report for RF \n:",sklearn.metrics.classification_report(y_validation, y_pred_rf,digits=2, target_names=labels_list))

print("-------------------------------------------------------------")



# F1 Score

print(" F1 Score for RF:",sklearn.metrics.f1_score(y_validation, y_pred_rf, labels=None, average='macro',pos_label=1, sample_weight=None))

print("-------------------------------------------------------------")



# confusion matrix

print(" Confusion_matrix for RF: \n",confusion_matrix(y_validation, y_pred_rf))

plt.figure(figsize=(9,8))

cm=confusion_matrix(y_validation, y_pred_rf)

graph_confusion_matrix(cm,labels_list ,"confusion_matrix")

print("-------------------------------------------------------------")



print("\n Time consumed in training: ", sk_rf.refit_time_, "sec")

print("\n Time consumed in prediction: ", end_time - start_time, "sec")
#convert prediction result into Forsyth-Edwards Notation

def ypred_convert_filename(y_prediction):

    def convert(chess64):

        result=[]

        for i in range(64):

            result.append(chess_dict2[chess64[i]])

            if (i+1)%8==0 and i!=63:

                result.append("-")

                

        result1="".join(result)

        for i in [8,7,6,5,4,3,2,1]:        

            result1=result1.replace(i*"0",str(i))

        return result1

    

    y_pred_convert=[]

    i=0

    while i+64<=len(y_prediction):

        chess64=y_prediction[i:i+64]

        y_pred_convert.append(convert(chess64))

        i=i+64

    return y_pred_convert



y_pred_converted_names_rf=ypred_convert_filename(y_pred_rf)
###  Evaluation method 

evaluation_method_dic=dict({"precision":0,"recall":1,"f1":2})



def evaluation1(evaluation_method,methodlist,methodnames,categorys_count):



    df_classification_report=pd.DataFrame()

    for model in methodlist:

        df_classification_report=df_classification_report.append(pd.DataFrame(model).iloc[evaluation_method_dic[evaluation_method],list(range(len(categorys_count)))+[len(categorys_count)+2]] )

    df_classification_report.index=methodnames

    

    print(df_classification_report.T)

    df_classification_report.T.plot.bar(alpha=0.7,rot=0,legend=False)

    plt.title("comparision of {}".format(evaluation_method))

    plt.legend(bbox_to_anchor=(1.05,1.0),borderaxespad = -0.2)  

    plt.yticks([x/100 for x in list(range(0,105,10))])



    return df_classification_report

    

methodlist=[class_report_knn,class_report_lr,class_report_rf]

methodnames=["knn","lr","rf"]

categorys_count=np.unique(y_train)



# class_report_recall=evaluation1("recall",methodlist,methodnames,categorys_count)

# class_report_f1=evaluation1("f1",methodlist,methodnames,categorys_count)

# class_report_precision=evaluation1("precision",methodlist,methodnames,categorys_count)



def combine_cll_report(evaluation_method,methodlist,methodnames,categorys_count):



    df_classification_report=pd.DataFrame()

    for model in methodlist:

        df_classification_report=df_classification_report.append(pd.DataFrame(model).iloc[evaluation_method_dic[evaluation_method],list(range(len(categorys_count)))+[len(categorys_count)+2]] )

    df_classification_report.index=methodnames

    

    return df_classification_report



def plot_combine_report(df_classification_report,titile):

    print(df_classification_report.T)

    df_classification_report.T.plot.bar(alpha=0.7,rot=0,legend=False)

    plt.title("comparision of {}".format(titile))

    plt.legend(bbox_to_anchor=(1.05,1.0),borderaxespad = -0.2)  

    plt.yticks([x/100 for x in list(range(0,105,10))])



    

from sklearn.model_selection import KFold



# split the validation set into 10 folds

kf = KFold(n_splits=10)

i=0







# create container to store prediction result

y_pred_converted_names_knn=ypred_convert_filename(y_pred_knn)

y_pred_converted_names_lr=ypred_convert_filename(y_pred_lr)



class_report_recall_combine=pd.DataFrame()

class_report_f1_combine=pd.DataFrame()

class_report_precision_combine=pd.DataFrame()





class_accuracy_combine=pd.DataFrame()



for train_index, test_index in kf.split(X_validation):

    i=i+1

    print(i)

    X_trainf, y_trainf = X_validation[train_index], y_validation[train_index]

    X_testf, y_testf = X_validation[test_index], y_validation[test_index]



    sk_lr1 = sk_lr.best_estimator_

    sk_lr1.fit(X_trainf, y_trainf)

    

    sk_rf1 = sk_rf.best_estimator_

    sk_rf1.fit(X_trainf, y_trainf)

    

    sk_knn1 = sk_knn.best_estimator_

    sk_knn1.fit(X_trainf, y_trainf)

    

    y_pred_lr=sk_lr1.predict(X_testf)

    y_pred_rf=sk_rf1.predict(X_testf)

    y_pred_knn=sk_knn1.predict(X_testf)

    

    class_report_lr=sklearn.metrics.classification_report(y_testf, y_pred_lr,digits=2, target_names=labels_list,output_dict=True)

    class_report_rf=sklearn.metrics.classification_report(y_testf, y_pred_rf,digits=2, target_names=labels_list,output_dict=True)

    class_report_knn=sklearn.metrics.classification_report(y_testf, y_pred_knn,digits=2, target_names=labels_list,output_dict=True)



    methodlist=[class_report_knn,class_report_lr, class_report_rf]

    methodnames=["knn","lr","rf"]

    categorys_count=np.unique(y_trainf)

       

    class_report_recall=combine_cll_report("recall",methodlist,methodnames,categorys_count)

    class_report_f1=combine_cll_report("f1",methodlist,methodnames,categorys_count)

    class_report_precision=combine_cll_report("precision",methodlist,methodnames,categorys_count)

   

    class_report_recall_combine=class_report_recall_combine.append(class_report_recall)

    class_report_f1_combine=class_report_f1_combine.append(class_report_f1)

    class_report_precision_combine=class_report_precision_combine.append(class_report_precision)

    

    y_pred_rf_filename=ypred_convert_filename(y_pred_rf)

    y_pred_knn_filename=ypred_convert_filename(y_pred_knn)

    y_pred_lr_filename=ypred_convert_filename(y_pred_lr)

    

    y_test_filename=ypred_convert_filename(y_testf)

    

    accuracy_rf=calculate_accuracy(y_test_filename,y_pred_rf_filename)[1]

    accuracy_knn=calculate_accuracy(y_test_filename,y_pred_knn_filename)[1]

    accuracy_lr=calculate_accuracy(y_test_filename,y_pred_lr_filename)[1]

    

    class_accuracy_combine['accu {}'.format(i)]=[accuracy_knn, accuracy_lr, accuracy_rf]

# combine the results generates in multiple iterations for visualisation

class_report_recall_combine['group_index']=class_report_recall_combine.index

class_report_f1_combine['group_index']=class_report_f1_combine.index

class_report_precision_combine['group_index']=class_report_precision_combine.index



class_report_recall_combine_mean=class_report_recall_combine.groupby('group_index').mean()

class_report_f1_combine_mean=class_report_f1_combine.groupby('group_index').mean()

class_report_precision_combine_mean=class_report_precision_combine.groupby('group_index').mean()



plot_combine_report(class_report_recall_combine_mean,"recall rate for 10 folds")

plot_combine_report(class_report_f1_combine_mean,"F1 score for 10 folds")

plot_combine_report(class_report_precision_combine_mean,"precision for 10 folds")



class_accuracy_combine.index=methodnames

plot_combine_report(class_accuracy_combine,"accuracy for 10 folds")
def performance_best_model(model_best):

    start=time.time()

    X_train_all=np.vstack((X_validation,X_train))

    y_train_all=np.array(list(y_validation)+list(y_train))

    

    y_testlabel=ypred_convert_filename(y_test)

    

    model_best.fit(X_train_all,y_train_all)



    y_pred_best_model=model_best.predict(X_test)

    y_pred_converted_names_best_model=ypred_convert_filename(y_pred_best_model)



    accuracy_best_model=calculate_accuracy(y_testlabel,y_pred_converted_names_best_model)

    print('Time used:{} s'.format(time.time()-start))

    return accuracy_best_model





model_best=sk_rf.best_estimator_

accu_best_model=performance_best_model(model_best)

print('accuracy of best_model: {}%'.format(accu_best_model[1]*100))