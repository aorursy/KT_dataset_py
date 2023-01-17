import pandas as pd







from sklearn.metrics import roc_auc_score



from sklearn.ensemble import RandomForestRegressor
x=pd.read_csv("../input/train.csv")





y=x.pop("Survived")#stored survived by taking it from training data

x.describe()



x["Age"].fillna(x.Age.mean(),inplace=True)#science age is missing we take mean and replace it



x.describe()#describing after replacing it
numeric_variables=list(x.dtypes[x.dtypes !="object"].index)# i am just taking numeric data types by omitting object ones



x[numeric_variables].head()#just looking at head parts of data to verify only numerics
model=RandomForestRegressor(n_estimators=100,oob_score=True,random_state=42)#to notice out of bag score true

model.fit(x[numeric_variables],y)#applying random forest #randomstate taken random
model.oob_score_#after fitting model underscore are available#it produces r^2 value

y_oob=model.oob_prediction_



from sklearn.metrics import roc_auc_score



print ("c-stat:",roc_auc_score(y,y_oob))#important to notice cstat score





#just by doing sample random forest by imputing age 
y_oob #predictions of out of bag
def describe_categorical(X):

    from IPython.display import display,HTML

    display(HTML(X[X.columns[X.dtypes=="object"]].describe().to_html()))

    
describe_categorical(x)

x.drop(["Name","Ticket","PassengerId"],axis=1,inplace=True)
def clean_cabin(X):

    try:

        return X[0]

    except TypeError:

        return "None"

    

x["Cabin"]=x.Cabin.apply(clean_cabin)
x.Cabin
categorical_variables=["Sex","Cabin","Embarked"]

for variable in categorical_variables:

        x[variable].fillna("Missing",inplace=True)

        

        dummy=pd.get_dummies(x[variable],prefix=variable)

        

        x=pd.concat([x,dummy],axis=1)

        

        x.drop([variable],axis=1,inplace=True)#dropping original above categorical variable
x
def printall(x,max_rows=10):   #this function to show all coloumns

    from IPython.display import display,HTML

    display(HTML(x.to_html(max_rows=max_rows)))

    

printall(x)
model=RandomForestRegressor(100,oob_score=True,n_jobs=-1,random_state=42)

model.fit(x,y)

print ("modified c-stat:",roc_auc_score(y,model.oob_prediction_))
model.feature_importances_

model=RandomForestRegressor(n_estimators=1000,oob_score=True,n_jobs=-1,random_state=42,max_features="auto",min_samples_leaf=5)

model.fit(x,y)

print ("modified c-stat:",roc_auc_score(y,model.oob_prediction_))#finalmodel