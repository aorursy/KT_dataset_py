import pandas as pd

# Load full data
df = pd.read_csv("../input/titanic-ful-data/T_full.csv")
df.head()
# Keep the original copy
df_original=df.copy()
#OHE
import category_encoders as ce
# Create an object of the OHE
OHE = ce.OneHotEncoder(cols=['Sex','Cabin','Embarked','Title'],use_cat_names=True)
# Encode the Categorical Variable
df = OHE.fit_transform(df)
#Scalte the data
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create an Object of the StandardScaler
scaler = StandardScaler()

# Fit with the ITem_Mrp
scaler.fit(np.array(df.Age).reshape(-1,1))
scaler.fit(np.array(df.Fare).reshape(-1,1))
scaler.fit(np.array(df.Ticket).reshape(-1,1))

# Transform the data
df.Age = scaler.transform(np.array(df.Age).reshape(-1,1))
df.Fare = scaler.transform(np.array(df.Fare).reshape(-1,1))
df.Ticket = scaler.transform(np.array(df.Ticket).reshape(-1,1))
#data separation
train = df.loc[df.train_or_test.isin(['train'])]
test = df.loc[df.train_or_test.isin(['test'])]
#Drop the columns which are not required
train=train.drop(['train_or_test','PassengerId','Name'],axis=1) 
test=test.drop(['train_or_test','PassengerId','Name','Survived'],axis=1)
X = train.drop('Survived',1) 
y = train.Survived
# Check the shape of the train & test data
train.shape,test.shape

# Model Building - 1
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
#Create the model object
model = LogisticRegression()

#train the model
model.fit(x_train,y_train)

#predict on validation data
pred_cv = model.predict(x_cv)

#check accuracy
accuracy_score(y_cv,pred_cv)
#final prediction
pred_test = model.predict(test)
## Logstic Regression with StratifiedKFold

from sklearn.model_selection import StratifiedKFold

i=1 
kf = StratifiedKFold(n_splits=8,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]
    
    model = LogisticRegression(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test2 = model.predict(test) 
    pred=model.predict_proba(xvl)[:,1]
