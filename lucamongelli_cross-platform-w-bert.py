import os

import pathlib

import pandas as pd

from sklearn.metrics import classification_report

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

os.environ["CUDA_VISIBLE_DEVICES"]="0"; 
!pip install ktrain

import ktrain

from ktrain import text
#SETTINGS 

#check the commented line for info



train_path="../input/sentimentdatasets/dummy.csv" 

tr_path=pathlib.Path(train_path)

if tr_path.exists ():

    train_df=pd.read_csv(tr_path, header=0) 

    print("Dummy path set.")

else: 

    raise SystemExit("Dummy path does not exist.")   

    

    

model_path="../input/models/stackoverflow_model.h5" #insert the path for the model

mo_path=pathlib.Path(model_path)



if mo_path.exists ():

    print("Model path set.")

else: 

    raise SystemExit("Model path does not exist.")  

    



data_path="../input/sentimentdatasets/testGithub.csv" #insert the path of the data for the prediction

da_path=pathlib.Path(data_path)



#set the dataframe 

#parameters for 

#jiradataset - - - data_path, encoding="utf-16", header=None - - - remember to set test_df[2]=test_df[2].astype(str) due to a dataset bug

#stackoverflowdataset - - - data_path, encoding='utf-16',sep=';', header=None

#githubdataset - - - data_path, sep=';', header=0

if da_path.exists ():

    test_df=pd.read_csv( data_path, sep=';', header=0) 

    data_text=test_df["Text"] #set 2 for StackOverFlow and Jira and "Text" for Github

    data_label=test_df["Polarity"] #set 1 for StackOverFlow and Jira and "Polarity" for Github

    print("Data path set.")

else: 

    raise SystemExit("Data path does not exist.")    

    

  

 

train_df.head()
test_df.head()
#parameters for 

#jiradataset - - - train_df["Text"], train_df["Polarity"], x_test=test_df[2], y_test=test_df[1], maxlen=500, preprocess_mode='bert', lang="en"

#stackoverflowdataset - - - train_df["Text"], train_df["Polarity"], x_test=test_df[2], y_test=test_df[1], maxlen=500, preprocess_mode='bert', lang="en"

#githubdataset - - - train_df["Text"], train_df["Polarity"], x_test=test_df["Text"], y_test=test_df["Polarity"], maxlen=500, preprocess_mode='bert', lang="en"

(x_train, y_train),(x_test, y_test), preproc=text.texts_from_array(train_df["Text"], train_df["Polarity"], x_test=test_df["Text"], y_test=test_df["Polarity"], maxlen=500, preprocess_mode='bert', lang="en")
learner = ktrain.get_learner(text.text_classifier('bert', (x_train, y_train) , preproc=preproc), 

                             train_data=(x_train, y_train), 

                             val_data=(x_test, y_test), 

                             batch_size=6)
learner.load_model(model_path)

print("model loaded successfully")
predictor = ktrain.get_predictor(learner.model, preproc)
#get the texts and the labels from the test dataset

data=data_text.tolist()

label=data_label.tolist()

#generating a csv result file containing the wrong predictions

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

y_true= label

print(classification_report(y_true, y_pred, target_names=names))
print("Correct: ", correct,"/",total,"\nWrong: ", wrong,"/",total)