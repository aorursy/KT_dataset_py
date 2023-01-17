import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/epitope-prediction/input_bcell.csv")
df.head()
df.info()
df.describe()
sns.countplot(x = "target" ,data = df)
df["protein_seq_len"] = df["protein_seq"].apply(len)
df["peptide_seq_len"] = df["peptide_seq"].apply(len)
df.head()


df = pd.get_dummies(df,columns = ["target"],drop_first=True)
df.head()
df["parent_protein_id"].nunique()
df["start_position"].nunique()
plt.figure(figsize=(18,12))

sns.heatmap(df.corr(),linewidth = 1.5, annot = True)
df.drop(columns = ["peptide_seq"], axis =1 ,inplace =True)
df.head()
df[df["parent_protein_id"] == "A2T3T0"].head()
df.drop(columns = ["protein_seq"], axis = 1, inplace = True)
df.head()
df = pd.get_dummies(df,columns=["parent_protein_id"],drop_first = True)
df.head()
df.describe()










df.head()
df.info()

from sklearn.model_selection import train_test_split
X = df.drop(["target_1"], axis =1).values
y = df["target_1"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

early_stop=EarlyStopping(monitor="val_loss",mode="min",patience=25,verbose=1)
model = Sequential()

# Choose whatever number of layers/neurons you want.
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
#Binary Classification
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam")
model.fit(X_train,y_train,epochs=150,validation_data=(X_test,y_test),batch_size=256,callbacks=[early_stop])
history=pd.DataFrame(model.history.history)

history.plot()
pred=model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
