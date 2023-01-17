import os

import pathlib

import pandas as pd

from sklearn.metrics import classification_report

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

os.environ["CUDA_VISIBLE_DEVICES"]="0"; 
!pip install ktrain

import ktrain

from ktrain import text
#check if the paths for the input data is valid.

train_path="../input/sentimentdatasets/trainGithub.csv"

test_path="../input/sentimentdatasets/testGithub.csv"

tr_path= pathlib.Path(train_path)

te_path=pathlib.Path(test_path)

if tr_path.exists ():

    print("Train data path set.")

else: 

    raise SystemExit("Train data path does not exist.")

     

if te_path.exists ():

    print("Test data path set.")

else: 

    raise SystemExit("Test data path does not exist.")

     
#showing the first 5 lines of the train data

train_df=pd.read_csv(train_path, sep=';', header=0)

train_df.head()

#showing the first 5 lines of the test data

test_df=pd.read_csv(test_path, sep=';', header=0)

test_df.head()
(x_train, y_train), (x_test, y_test), preproc =  text.texts_from_array(train_df["Text"], train_df["Polarity"],  x_test=test_df["Text"], y_test=test_df["Polarity"], 

                                                                     maxlen=500, preprocess_mode='bert')

                                                                     

                  
model = text.text_classifier('bert', (x_train, y_train) , preproc=preproc)
learner = ktrain.get_learner(model, 

                             train_data=(x_train, y_train), 

                             val_data=(x_test, y_test), 

                             batch_size=6)
learner.lr_find()
learner.lr_plot()
# 2e-5 is one of the LRs  recommended by Google and is consistent with the plot above.

learner.autofit(2e-5, early_stopping=5)
model.save("github_model.h5")
predictor = ktrain.get_predictor(learner.model, preproc)
data=test_df["Text"].tolist()

label=test_df["Polarity"].tolist()
i=0

correct=0

wrong=0

total=len(data)

true_lab=[]

pred_lab=[]

text=[]

for dt in data:

    result=predictor.predict(dt)

    if not result== label[i]:

        text.append(dt)

        pred_lab.append(result)

        true_lab.append(label[i])

        wrong+=1

    else:

        correct+=1

    

    i+=1



name_dict = {

            'Name': text,

            'Gold Label' : true_lab,

            'Predicted Label': pred_lab

          }



wrong_data= pd.DataFrame(name_dict)



wrong_data.to_csv("wrong_results.csv", sep=';')   

    

    
names = ['negative', 'neutral', 'positive']

y_pred = predictor.predict(data)

y_true= test_df["Polarity"]

print(classification_report(y_true, y_pred, target_names=names))
print("Correct: ", correct,"/",total,"\nWrong: ", wrong,"/",total)