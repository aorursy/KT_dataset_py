!pip install jovian -q
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import jovian
from tqdm import tqdm_notebook
import warnings
warnings.simplefilter("ignore")
chem_df = pd.read_csv("../input/drug-drug-similarity-dataset/chem_Jacarrd_sim.csv",index_col=0)
indication_df = pd.read_csv("../input/drug-drug-similarity-dataset/enzyme_Jacarrd_sim.csv",index_col=0)
target_df = pd.read_csv("../input/drug-drug-similarity-dataset/target_Jacarrd_sim.csv",index_col=0)
transporter_df = pd.read_csv("../input/drug-drug-similarity-dataset/transporter_Jacarrd_sim.csv",index_col=0)
#------------------ Using only 4 datassets for the model--------------------

drug_drug_matrix = pd.read_csv("../input/drug-drug-similarity-dataset/drug_drug_matrix.csv",index_col=0) #--> Labels
#---------------------------------------------------------------------------

enzyme_df = pd.read_csv("../input/drug-drug-similarity-dataset/enzyme_Jacarrd_sim.csv",index_col=0)
pathway_df = pd.read_csv("../input/drug-drug-similarity-dataset/pathway_Jacarrd_sim.csv",index_col=0)
offsideeffect_df = pd.read_csv("../input/drug-drug-similarity-dataset/offsideeffect_Jacarrd_sim.csv",index_col=0)
sideeffect_df = pd.read_csv("../input/drug-drug-similarity-dataset/sideeffect_Jacarrd_sim.csv",index_col=0)


complete_df = chem_df+indication_df+target_df+transporter_df+enzyme_df+pathway_df+offsideeffect_df+sideeffect_df
integrated_df = pd.DataFrame(np.zeros((149878, 1096)))
integrated_target = pd.DataFrame(np.zeros((149878,1)))
ind=0
flag=0
for i in tqdm_notebook(range(515)):
    for j in range(i,548):
        a = complete_df.iloc[i].to_numpy().reshape(1,-1).tolist()[0]
        b = complete_df.iloc[j].to_numpy().reshape(1,-1).tolist()[0]
        try:
            integrated_df.iloc[ind] = a+b
            integrated_target.iloc[ind] = drug_drug_matrix.iloc[i,j]
            ind+=1
        except:
            flag=1
            break
    if flag:
        break
chem_df1 = pd.DataFrame(np.zeros((149878, 1096)))
indication_df1 = pd.DataFrame(np.zeros((149878, 1096)))
target_df1 = pd.DataFrame(np.zeros((149878, 1096)))
transporter_df1 = pd.DataFrame(np.zeros((149878, 1096)))
enzyme_df1 = pd.DataFrame(np.zeros((149878, 1096)))
pathway_df1 = pd.DataFrame(np.zeros((149878, 1096)))
offsideeffect_df1 = pd.DataFrame(np.zeros((149878, 1096)))
sideeffect_df1 = pd.DataFrame(np.zeros((149878, 1096)))
ind=0
flag=0
for i in tqdm_notebook(range(515)):
    for j in range(i,548):
        a = chem_df.iloc[i].to_numpy().reshape(1,-1).tolist()[0]
        b = chem_df.iloc[j].to_numpy().reshape(1,-1).tolist()[0]
        try:
            chem_df1.iloc[ind] = a+b
            ind+=1
        except:
            flag=1
            break
    if flag:
        break
for i in tqdm_notebook(range(515)):
    for j in range(i,548):
        a = indication_df.iloc[i].to_numpy().reshape(1,-1).tolist()[0]
        b = indication_df.iloc[j].to_numpy().reshape(1,-1).tolist()[0]
        try:
            indication_df1.iloc[ind] = a+b
            ind+=1
        except:
            flag=1
            break
    if flag:
        breakind=0
flag=0
for i in tqdm_notebook(range(515)):
    for j in range(i,548):
        a = target_df.iloc[i].to_numpy().reshape(1,-1).tolist()[0]
        b = target_df.iloc[j].to_numpy().reshape(1,-1).tolist()[0]
        try:
            target_df1.iloc[ind] = a+b
            ind+=1
        except:
            flag=1
            break
    if flag:
        breakind=0
flag=0
for i in tqdm_notebook(range(515)):
    for j in range(i,548):
        a = transporter_df.iloc[i].to_numpy().reshape(1,-1).tolist()[0]
        b = transporter_df.iloc[j].to_numpy().reshape(1,-1).tolist()[0]
        try:
            transporter_df1.iloc[ind] = a+b
            ind+=1
        except:
            flag=1
            break
    if flag:
        breakind=0
flag=0
for i in tqdm_notebook(range(515)):
    for j in range(i,548):
        a = enzyme_df.iloc[i].to_numpy().reshape(1,-1).tolist()[0]
        b = enzyme_df.iloc[j].to_numpy().reshape(1,-1).tolist()[0]
        try:
            enzyme_df1.iloc[ind] = a+b
            ind+=1
        except:
            flag=1
            break
    if flag:
        breakind=0
flag=0
for i in tqdm_notebook(range(515)):
    for j in range(i,548):
        a = pathway_df.iloc[i].to_numpy().reshape(1,-1).tolist()[0]
        b = pathway_df.iloc[j].to_numpy().reshape(1,-1).tolist()[0]
        try:
            pathway_df1.iloc[ind] = a+b
            ind+=1
        except:
            flag=1
            break
    if flag:
        breakind=0
flag=0
for i in tqdm_notebook(range(515)):
    for j in range(i,548):
        a = offsideeffect_df.iloc[i].to_numpy().reshape(1,-1).tolist()[0]
        b = offsideeffect_df.iloc[j].to_numpy().reshape(1,-1).tolist()[0]
        try:
            offsideeffect_df1.iloc[ind] = a+b
            ind+=1
        except:
            flag=1
            break
    if flag:
        breakind=0
flag=0
for i in tqdm_notebook(range(515)):
    for j in range(i,548):
        a = sideeffect_df.iloc[i].to_numpy().reshape(1,-1).tolist()[0]
        b = sideeffect_df.iloc[j].to_numpy().reshape(1,-1).tolist()[0]
        try:
            sideeffect_df1.iloc[ind] = a+b
            ind+=1
        except:
            flag=1
            break
    if flag:
        break
def Model(lr=0.0001):
    model = tf.keras.Sequential([Dense(300,input_shape=[1,1096],activation="relu"),
                                 Dropout(0.5),
                                 Dense(400,activation="relu"),
                                 Dropout(0.5),
                                 Dense(1,activation="sigmoid")])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss="binary_crossentropy",metrics=["accuracy"])
    return model
model = Model()
model.summary()
X_train,X_test,Y_train,Y_test = train_test_split(integrated_df,integrated_target,stratify=integrated_target,test_size=0.1,random_state=0)
logs = model.fit(X_train,Y_train,epochs=1000,validation_split=0.01, batch_size = 20480)
plt.figure(figsize=(10,10))
plt.plot(logs.history["loss"],label="Training Loss")
plt.plot(logs.history["val_loss"],label="Validation Loss")
plt.legend(fontsize=13)
plt.show()
plt.figure(figsize=(10,10))
plt.plot(logs.history["accuracy"],label="Training Accuracy")
plt.plot(logs.history["val_accuracy"],label="Validation Accuracy")
plt.legend(fontsize=13)
plt.show()
preds = model.predict(X_train)
preds = pd.DataFrame(preds)
preds = preds.apply(pd.cut,bins=2,labels=[0,1])
print(f"Accuracy: {accuracy_score(Y_train,preds)}")
print(f"Precision Score: {precision_score(Y_train,preds)}")
print(f"Recall Score: {recall_score(Y_train,preds)}")
print(f"F1-Score: {f1_score(Y_train,preds)}")
print(f"ROC AUC Score: {roc_auc_score(Y_train,preds)}")
X_test.shape
preds = model.predict(X_test)
preds = pd.DataFrame(preds)
preds = preds.apply(pd.cut,bins=2,labels=[0,1])
print(f"Accuracy: {accuracy_score(Y_test,preds)}")
print(f"Precision Score: {precision_score(Y_test,preds)}")
print(f"Recall Score: {recall_score(Y_test,preds)}")
print(f"F1-Score: {f1_score(Y_test,preds)}")
print(f"ROC AUC Score: {roc_auc_score(Y_test,preds)}")
preds = model.predict(integrated_df)
preds = pd.DataFrame(preds)
preds = preds.apply(pd.cut,bins=2,labels=[0,1])
print(f"\tAccuracy: {accuracy_score(integrated_target,preds)}")
print(f"\tPrecision Score: {precision_score(integrated_target,preds)}")
print(f"\tRecall Score: {recall_score(integrated_target,preds)}")
print(f"\tF1-Score: {f1_score(integrated_target,preds)}")
print(f"\tROC AUC Score: {roc_auc_score(integrated_target,preds)}")
jovian.commit(project="ddi using ANNs 2",environment=None)
"""{'chem': 0.899, #2
 'target': 0.787, #4
 'transporter': 0.945, #1
 'enzyme': 0.734,
 'pathway': 0.767, 
 'indication': 0.802, #3
 'sideeffect': 0.778,
 'offsideeffect': 0.782}"""
model.predict(X_train.iloc[0,:].to_numpy().reshape(1,-1))
for i in range(Y_train.shape[0]):
    if Y_train.iloc[i][0]==1:
        print(i)
model.predict(X_train.iloc[2602,:].to_numpy().reshape(1,-1))
