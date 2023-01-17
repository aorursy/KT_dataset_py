import pandas as pd
mydata_path = '../input/simple.csv'
mydata = pd.read_csv(mydata_path)
mydata.head()
#We will use a,b,c as features
features=['a','b','c']
X = mydata[features]
# just have a look and make sure everything is the way you want
# because training may take some time ...
X.head()
#to predict y
y = mydata.y
from sklearn.tree import DecisionTreeRegressor
my_model = DecisionTreeRegressor()
#X contains the fetaures that will be used to predict y
my_model.fit(X, y)
X.describe()
#We create 5 test conditions
case_one =[1,1,1] 
case_two =[1,0,1] 
case_three =[0,0,0] 
case_four =[1,0,0] 
case_five =[0,1,0] 
y_test=[case_one,case_two,case_three,case_four,case_five]
print(my_model.predict(y_test))