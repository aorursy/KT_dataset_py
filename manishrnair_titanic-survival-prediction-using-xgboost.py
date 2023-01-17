# import all the required modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score,accuracy_score

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.ensemble import RandomForestClassifier

#from google.colab import files

from xgboost import XGBClassifier

from xgboost import plot_importance



sns.set()
#Load the training and testing data 

url_train = "../input/train.csv"

url_test = "../input/test.csv"



df_train = pd.read_csv(url_train)

df_test = pd.read_csv(url_test)

# take a look in to head to see if the data has been loaded correctly.

df_train.head(3)
df_train.info()
df_train.describe()
survival = pd.DataFrame(df_train.Survived.value_counts())

survival["ratio"] = survival.Survived/survival.Survived.sum()

survival.sort_index(inplace=True)

survival.index=["Not Survived","Survived"]

survival.reset_index(inplace=True)

survival.columns =["status","count","ratio"]





survival
def plot_pie_chart(data,title,label_col):

    color_list = np.array(['r','g', 'b','c','y'])

    plt.pie(data["count"],colors=color_list[:data.shape[0]],autopct='%.2f')

    plt.axis('equal')

    plt.title(title)

    plt.legend(data[label_col])

    plt.show()
plot_pie_chart(survival,"Survived vs Not Survived",'status')
passenger_class = pd.DataFrame(df_train.Pclass.value_counts())

passenger_class["ratio"] = passenger_class.Pclass/passenger_class.Pclass.sum()

passenger_class.columns = ['count','ratio']

passenger_class.sort_index(ascending=True,inplace=True)

#passenger_class.index=['1st','2nd','3rd']

passenger_class.reset_index(inplace=True)

passenger_class.columns = ['Pclass','count','ratio']

passenger_class
plot_pie_chart(passenger_class,'Classwise Breakup of Passengers','Pclass')
passenger_gender = pd.DataFrame(df_train.Sex.value_counts())

passenger_gender["ratio"] = passenger_gender.Sex/passenger_gender.Sex.sum()

passenger_gender.sort_index(ascending=True)

passenger_gender.columns=['count','ratio']

passenger_gender.reset_index(inplace=True)

passenger_gender.columns=['Sex','count','ratio']

passenger_gender
plot_pie_chart(passenger_gender,"Genderwise Breakup",'Sex')
print ("Observations with missing age:",df_train.Age.isna().sum())
sample_name="Conlon, Mr. Thomas Henry"

def get_salutation(full_name):

    '''

    This method takes in the name and spits out the caption from the name

    

    '''

    start = str(full_name).find(',')

    end= str(full_name).find('.')

    salutation = full_name[start+1:end]

    return salutation



df_train["salutation"] = df_train.Name.apply(get_salutation)

df_train.salutation = df_train.salutation.str.strip()

df_train.salutation = df_train.salutation.str.lower()
mean_ages = df_train.groupby(["salutation","Pclass"]).agg({"Age":"mean"}).reset_index()

age_unknown = df_train[df_train.Age.isna()]

imputed_age = pd.merge(age_unknown,mean_ages,on=['salutation','Pclass'],how='left')[["PassengerId","Age_y"]]

data = pd.merge(df_train,imputed_age,on=["PassengerId"], how='left')

data.Age.fillna(0,inplace=True)

data.Age_y.fillna(0,inplace=True)

data.Age = data.Age+data.Age_y

df_train =data

df_train.Age = df_train.Age.round()
df_train.Age.describe()
df_train.Age.plot(kind='kde',color='c')

plt.xlabel("Age")

plt.title("Passenger Age Distribution")

plt.show()
plt.hist(df_train.Age,cumulative=1,density=True,histtype='step',color='c')

plt.show()
print ("Unique passengers:",len(df_train.PassengerId.unique()))

print ("Unique tickets:",len(df_train.Ticket.unique()))

plt.title("Fare histogram")

df_train.Fare.hist(color='c')

plt.xlabel("Fare")

plt.ylabel("Frequency")

plt.xticks(np.arange(0,500,50))

plt.show()
df_train.Fare.plot(kind='kde',color='c')

plt.show()
df_train.Fare.describe()
print ("passengers with zero fare: ",(df_train.Fare == 0).sum())
df_train["Parch"].describe()
df_train["Parch"].plot(kind='kde',color='c')

plt.title("Parch Distribution")

plt.xlabel("Parent/Children")

plt.show()
df_train["SibSp"].describe()
df_train["SibSp"].plot(kind='kde',color='c')

plt.title("SibSp Distribution")

plt.xlabel("Sibling/Spouse")

plt.show()
df_train["family_size"]=df_train["SibSp"]+df_train["Parch"]
df_train.family_size.describe()
df_train.family_size.plot(kind='kde')

plt.xlabel("Family Size")

plt.show()
survivors = df_train[df_train["Survived"]==1]

non_survivors = df_train[df_train["Survived"]==0]

survivors.family_size.plot(kind='kde',color='g')

non_survivors.family_size.plot(kind='kde',color='r')

plt.legend(["Survivor","Non-survivor"])

plt.xlabel("family_size")

plt.title("Passenger family_size Distribution")

plt.xticks(np.arange(0,10,1))

plt.show()
#impute the missing values for Embarked as mode of the column

df_train.loc[df_train["Embarked"].isna(),"Embarked"] = 'S'
#Assert if null values have been removed

assert df_train["Embarked"].isna().sum() == 0
embarked_data =pd.DataFrame(df_train["Embarked"].value_counts())

embarked_data["ratio"] = embarked_data.Embarked/embarked_data.Embarked.sum()



embarked_data.reset_index(inplace=True)

embarked_data.columns=['Embarked','count','ratio']

embarked_data

plot_pie_chart(embarked_data,"Port of Embarkation","Embarked")
(float(df_train["Cabin"].isna().sum())/df_train["Cabin"].shape[0])*100
df_train["has_cabin"] = 1

df_train.loc[df_train["Cabin"].isna(),"has_cabin" ] = 0
cabin = pd.DataFrame(df_train.has_cabin.value_counts()).reset_index()

cabin.columns=["cabin","count"]

cabin.loc[cabin.cabin == 0 , "cabin"] = "don't have"

cabin.loc[cabin.cabin == 1 , "cabin"] = "have" 

plot_pie_chart(cabin,"Cabinwise Passenger","cabin")
pass_class_survival = df_train.groupby(['Pclass','Survived']).agg({"PassengerId":"count"})

pass_class_survival.reset_index(inplace=True)

pass_class_survival = pd.merge(pass_class_survival,passenger_class,on=["Pclass"],how='left')

pass_class_survival["ratio"] = pass_class_survival["PassengerId"]/pass_class_survival["count"]

pass_class_survival= pass_class_survival[pass_class_survival.Survived==1][['Pclass','ratio']]

pass_class_survival.columns=['Pclass','prob_survival']

pass_class_survival
df_train[["Pclass","Fare"]].corr()
gender_survival = df_train.groupby(["Sex","Survived"]).agg({"PassengerId":"count"})

gender_survival.reset_index(inplace=True)

gender_survival = pd.merge(gender_survival,passenger_gender,on=['Sex'], how ='left')

gender_survival['prob_survival']=gender_survival['PassengerId']/gender_survival['count']

gender_survival = gender_survival[gender_survival.Survived==1][["Sex","prob_survival"]]

gender_survival
print("Correaltion between Age and Survival")

df_train[["Age","Survived"]].corr()
sns.boxplot(y='Age',x='Survived',data=df_train)

plt.title("Age and Survival")

plt.plot()
survivors = df_train[df_train["Survived"]==1]

non_survivors = df_train[df_train["Survived"]==0]

survivors.Age.plot(kind='kde',color='g')

non_survivors.Age.plot(kind='kde',color='r')

plt.legend(["Survivor","Non-survivor"])

plt.xlabel("Age")

plt.title("Survivors vs Non Survivors Age Distribution")

plt.show()


def get_age_group(age):

   if(age<=18):

      return 'child'

   elif((age>18)& (age<=35)):

     return 'adults'

   elif((age>35)):

         return 'senior'

#    elif((age>45) &(age<=60)):

#          return '45-60'

#    elif((age>60)):

#       return '60+'

        

      #ge_labels = ['15-','15-35','35-45','40-60','60+']

df_train["Age_grp"] = df_train.Age.apply(get_age_group)
df_train["Age_grp"].value_counts()
age_grp_survival = df_train.groupby(["Age_grp","Survived"]).agg({"PassengerId":"count"}).reset_index()

age_grp_passengers = df_train.groupby(["Age_grp"]).agg({"PassengerId":"count"}).reset_index()



age_grp_survival = pd.merge(age_grp_survival,age_grp_passengers,on=["Age_grp"], how="left")

age_grp_survival["prob_survival"] = age_grp_survival["PassengerId_x"]/age_grp_survival["PassengerId_y"]

age_grp_survival[age_grp_survival.Survived == 1][["Age_grp","prob_survival"]]
#### Fare and Survived
survivors = df_train[df_train["Survived"]==1]

non_survivors = df_train[df_train["Survived"]==0]

survivors.Fare.plot(kind='kde',color='g')

non_survivors.Fare.plot(kind='kde',color='r')

plt.legend(["Survivor","Non-survivor"])

plt.xlabel("Fare")

plt.title("Passenger Fare Distribution")

plt.xlim(0,200)

plt.xticks(np.arange(0,200,10))

plt.show()
sns.boxplot(y='Fare',x='Pclass',data=df_train)

plt.title("Class and Fare")

plt.plot()
df_train.groupby(["Pclass"]).agg({"Fare":"mean"})
def group_fare(fare):

  if fare <= 30:

    return "0-30"

  elif fare >30 and fare<=100:

    return "30-100"

  elif fare >100 and fare < 150:

    return "100-150"

  elif fare > 150:

    return "150+"
df_train["fare_group"] = df_train["Fare"].apply(group_fare)

df_train[["Survived","Parch"]].corr()
df_train[["Survived","SibSp"]].corr()
df_train["family_size"] = df_train["SibSp"]+df_train["Parch"]
print ("Distribution of Family Size:")

print (df_train["family_size"].describe())
df_train[["family_size","Survived"]].corr()
df_train[df_train["Survived"]==1]["family_size"].plot(kind='kde',color='g')

df_train[df_train["Survived"]==0]["family_size"].plot(kind='kde',color='r')

plt.xticks(np.arange(0,15,1))

plt.show()
def group_family_size(size):

  if size == 0:

    return "single"

  elif size >0 and size<=3:

    return "small"

  elif size >3 :

    return "medium"

 
df_train["family"] = df_train["SibSp"]+df_train["Parch"]



df_train["family_size"] = df_train["family"].apply(group_family_size)
port_survival = df_train.groupby(["Embarked","Survived"]).agg({"PassengerId":"count"})

port_survival.reset_index(inplace=True)

port_survival = pd.merge(port_survival,embarked_data,on=["Embarked"],how="left")

port_survival.columns=["Embarked","Survived","count","total_count","prob_survival"]

port_survival["prob_survival"] = port_survival["count"]/port_survival["total_count"]

port_survival[port_survival["Survived"] == 1][["Embarked","prob_survival"]]


sns.heatmap(df_train.corr())

plt.show()
class_sex_survivor_data = df_train.groupby(["Pclass","Sex","Survived"]).agg({"PassengerId":"count"}).reset_index()

class_sex_data = class_sex_survivor_data.groupby(["Pclass","Sex"]).agg({"PassengerId":"sum"}).reset_index()

class_sex_survivor_data = pd.merge(class_sex_survivor_data,class_sex_data,on=["Pclass","Sex"],how="left")

class_sex_survivor_data["prob_survival"] = class_sex_survivor_data["PassengerId_x"]/class_sex_survivor_data["PassengerId_y"]

survivors_class_gender = class_sex_survivor_data[class_sex_survivor_data["Survived"]==1][["Pclass","Sex","prob_survival"]]
male_data = survivors_class_gender[survivors_class_gender["Sex"] == "male"]["prob_survival"]

female_data = survivors_class_gender[survivors_class_gender["Sex"] == "female"]["prob_survival"]



ind = np.arange(3) 

width = 0.35       

plt.bar(ind, male_data, width, label='Men')

plt.bar(ind + width, female_data, width,label='Women')



plt.xlabel("Class")

plt.ylabel('Prob of Survival')

plt.title('Survival Probability by Class and Gender')



plt.xticks(ind + width / 2, ('1', '2', '3'))

plt.legend(loc='best')

plt.show()


df_test["salutation"] = df_test.Name.apply(get_salutation)

df_test.salutation = df_test.salutation.str.strip()

df_test.salutation = df_test.salutation.str.lower()
mean_ages = df_test.groupby(["salutation","Pclass"]).agg({"Age":"mean"}).reset_index()

age_unknown = df_test[df_test.Age.isna()]

imputed_age = pd.merge(age_unknown,mean_ages,on=['salutation','Pclass'],how='left')[["PassengerId","Age_y"]]

data = pd.merge(df_test,imputed_age,on=["PassengerId"], how='left')

data.Age.fillna(0,inplace=True)

data.Age_y.fillna(0,inplace=True)

data.Age = data.Age+data.Age_y

df_test =data

df_test.Age = df_test.Age.round()
df_test["Age"].describe()
df_test["Age_grp"] = df_test.Age.apply(get_age_group)
df_test["family"] = df_test["SibSp"]+df_test["Parch"]

df_test["family_size"] = df_test["family"].apply(group_family_size)

df_test["fare_group"] = df_test["Fare"].apply(group_fare)

    

  
def get_predictor_target_data(data,features,target):

    predictor_variables=features

    target_variable=target

    df_training_data = data[predictor_variables]

    try:

      df_target_data = data[target_variable]

    except:

      df_target_data=None

    

    return df_training_data, df_target_data

def get_sub_mission_files():

    df_submission= pd.DataFrame(df_test["PassengerId"])

    df_submission["Survived"]=predictions

    df_submission.to_csv("submission.csv",index=False)

    print("submissions saved")

    #files.download("submission.csv")

    



def filter_cols(colname,filters):

  if colname.startswith('salutation'):

    if colname.endswith(tuple(np.intersect1d(np.array(test_OHE.columns),np.array(train_data_OHE.columns)))):

      return True

    else:

      return False

  elif(colname in filters):

    return False

  else:

        return True
df_test.head(3)
# Kaggle Score : 0.80382.

#shuffle the dataframe

feature_filter=['Age_grp_child','salutation_miss','Age_grp_adults','fare_group_100-150','family_size_single','Embarked_C']

df_train["family"] =df_train["Parch"]+df_train["SibSp"]

df_test["family"] =df_test["Parch"]+df_test["SibSp"]



df_train["has_cabin"] = 1

df_train.loc[df_train["Cabin"].isna(),"has_cabin" ] = 0



df_test["has_cabin"] = 1

df_test.loc[df_test["Cabin"].isna(),"has_cabin" ] = 0



#Features:

# feature_filter=[]

# features=['PassengerId','Pclass','Sex','Embarked','Age_grp']

# cat_features=['Pclass','Sex','Embarked','Age_grp']

# target_var=['Survived']

features=['PassengerId','Pclass','Sex','Embarked','Age_grp',"family_size",'salutation',"has_cabin"]

cat_features=['Pclass','Sex','Embarked','Age_grp',"family_size",'salutation']

target_var=['Survived']



#Shuffle the data

df_train = df_train.sample(frac=1,random_state=42)



#Separate dependent and independant variables of training data:

train_data,target_data = get_predictor_target_data(df_train,features,target_var)





#Perform One Hot Encoding

train_data_OHE = pd.get_dummies(train_data,columns=cat_features,drop_first=True)



#Separate dependent and independant variables of final testing data:

test_predictor,test_target = get_predictor_target_data(df_test,features,target_var)



#Perform One Hot Encoding

test_OHE = pd.get_dummies(test_predictor,columns=cat_features,drop_first=True)



#Filter unwanted columns

train_data_OHE = train_data_OHE[[x for x in train_data_OHE.columns if filter_cols(x,feature_filter)]]



#Convert to numpy array

arr_training_data = train_data_OHE.values

arr_target_data = target_data.values.reshape(-1)





train_X, test_X,train_y,test_y = train_test_split(arr_training_data,arr_target_data,test_size=0.4,random_state=42, stratify=arr_target_data)





cv_X,test_X,cv_y,test_y = train_test_split(test_X,test_y,test_size=0.2,random_state=42, stratify=test_y)



eval_set = [(cv_X[:,1:], cv_y)]

#intialise the model

xgb_model = XGBClassifier(eta=0.01,objective="binary:logistic",random_state=42)



#fit it

xgb_model.fit(train_X[:,1:], train_y, early_stopping_rounds = 50, eval_metric = "error", eval_set = eval_set, verbose = False)



#predict on training set

training_predictions = xgb_model.predict(train_X[:,1:])

#predict on cross validation set

cv_predictions = xgb_model.predict(cv_X[:,1:])

#predict on testing set

test_predictions = xgb_model.predict(test_X[:,1:])







print("Training Metrics:")

print(confusion_matrix(train_y,training_predictions))

print(classification_report(train_y,training_predictions))

print("Train accuracy:",accuracy_score(train_y,training_predictions))

print("Train auc score:",roc_auc_score(train_y,training_predictions))



print("CV Metrics:")

#prnt confusion matrix and classification report

print(confusion_matrix(cv_y,cv_predictions))

print(classification_report(cv_y,cv_predictions))

print("CV accuracy:",accuracy_score(cv_y,cv_predictions))

print("CV auc score:",roc_auc_score(cv_y,cv_predictions))



print("Testing Metrics:")

#prnt confusion matrix and classification report

print(confusion_matrix(test_y,test_predictions))

print(classification_report(test_y,test_predictions))

print("Test accuracy:",accuracy_score(test_y,test_predictions))

print("Test auc score:",roc_auc_score(test_y,test_predictions))







print("Creating Submission...")

test_OHE = test_OHE[[x for x in test_OHE.columns if filter_cols(x,feature_filter)]]

predictions = xgb_model.predict(test_OHE.drop(['PassengerId'],axis=1).values)

# print(confusion_matrix(test_target.astype(int),predictions))

# print(classification_report(test_target.astype(int),predictions))

# print("Test accuracy:",accuracy_score(test_target.astype(int),predictions))

# print("Test auc score:",roc_auc_score(test_target.astype(int),predictions))





get_sub_mission_files()



#plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)

plt.bar( train_data_OHE.columns[1:],xgb_model.feature_importances_)

plt.xticks(train_data_OHE.columns[1:] ,rotation=90)

plt.show()